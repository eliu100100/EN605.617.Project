#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>

const int NUM_TEAMS = 20;
const int NUM_SIMS = 100000;

std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

// team structure
struct Team {
    float rating;
    int points;
    std::string name;
};

// simulate a single game
void simulate_game(Team &A, Team &B) {
    float p_draw = 0.2f;
    float p_win_total = 1.0f - p_draw;
    float total = A.rating + B.rating;
    float pA_win = p_win_total * (A.rating / total);
    float pB_win = p_win_total * (B.rating / total);

    float r = dist(rng);

    if (r < pA_win) {
        A.points += 3;
    } else if (r < pA_win + pB_win) {
        B.points += 3;
    } else {
        A.points += 1;
        B.points += 1;
    }
}

// simulate one full season
void simulate_season(std::vector<Team> &teams) {
    // reset points
    for (auto &t : teams) {
        t.points = 0;
    }

    // double round robin
    for (int i = 0; i < NUM_TEAMS; i++) {
        for (int j = i + 1; j < NUM_TEAMS; j++) {
            simulate_game(teams[i], teams[j]);
            simulate_game(teams[j], teams[i]);
        }
    }
}

// run Monte Carlo simulation
int main() {
    std::vector<Team> base_teams(NUM_TEAMS);

    // initialize base teams
    for (int i = 0; i < NUM_TEAMS; i++) {
        base_teams[i].name = "Team " + std::to_string(i);
        base_teams[i].rating = 0.5f + (float)i / NUM_TEAMS; // static ratings
        base_teams[i].points = 0;
    }

    std::vector<int> win_count(NUM_TEAMS, 0);
    std::vector<std::vector<int>> position_counts(NUM_TEAMS, std::vector<int>(NUM_TEAMS, 0));
    std::vector<int> points_sum(NUM_TEAMS, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_SIMS; t++) {
        std::vector<Team> teams = base_teams;

        simulate_season(teams);

        std::vector<int> ranking(NUM_TEAMS);
        std::iota(ranking.begin(), ranking.end(), 0);

        std::sort(ranking.begin(), ranking.end(), 
                [&](int a, int b) {
                    return teams[a].points > teams[b].points;
                });
        
        win_count[ranking[0]]++;
        for (int pos = 0; pos < NUM_TEAMS; pos++) {
            int team = ranking[pos];
            position_counts[team][pos]++;
        }
        for (int i = 0; i < NUM_TEAMS; i++) {
            points_sum[i] += teams[i].points;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    //print results
    std::cout << "Team | Win Prob | Top 4 Prob | Relegation Prob | Avg Points\n";
    for (int i = 0; i < NUM_TEAMS; i++) {
        float win_prob = (float)win_count[i] / NUM_SIMS;

        float top_4_prob = 0.0f;
        for (int pos = 0; pos < 4; pos++) {
            top_4_prob += (float)position_counts[i][pos] / NUM_SIMS;
        }

        float relegation_prob = 0.0f;
        for (int pos = 17; pos < 20; pos++) {
            relegation_prob += (float)position_counts[i][pos] / NUM_SIMS;
        }

        float avg_pts = points_sum[i] / (float)NUM_SIMS;
        std::cout << base_teams[i].name + " | "
                  << win_prob << " | "
                  << top_4_prob << " | "
                  << relegation_prob << " | "
                  << avg_pts << "\n";
    }

    std::chrono::duration<float> elapsed = end - start;

    std::cout << "\nExecution time: " << elapsed.count() << " seconds\n";
    std::cout << "Simulations per second: "
          << NUM_SIMS / elapsed.count();

    return 0;
}