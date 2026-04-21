#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>
#include "cpu_baseline.h"

const int NUM_TEAMS = 20;

std::mt19937 rng(42);

// team structure
struct Team {
    int rating;
    int points;
    int goal_diff;
    int total_goals;
    std::string name;
};

// simulate a single game
void simulate_game(Team &A, Team &B) {
    const float base_goals = 1.3f;
    const float k = 0.002f;
    const float home_adv = 0.2f;

    int diff = A.rating - B.rating;

    float lambdaA = base_goals * std::exp(k * diff) + home_adv;
    float lambdaB = base_goals * std::exp(-k * diff);

    std::poisson_distribution<int> distA(lambdaA);
    std::poisson_distribution<int> distB(lambdaB);

    int goalsA = distA(rng);
    int goalsB = distB(rng);

    A.goal_diff += (goalsA - goalsB);
    B.goal_diff += (goalsB - goalsA);
    A.total_goals += goalsA;
    B.total_goals += goalsB;

    if (goalsA > goalsB) {
        A.points += 3;
    }
    else if (goalsB > goalsA) {
        B.points += 3;
    }
    else {
        A.points += 1;
        B.points += 1;
    }
}

// simulate one full season
void simulate_season(std::vector<Team> &teams) {
    // reset points
    for (auto &t : teams) {
        t.points = 0;
        t.goal_diff = 0;
        t.total_goals = 0;
    }

    // double round robin
    for (int i = 0; i < NUM_TEAMS; i++) {
        for (int j = i + 1; j < NUM_TEAMS; j++) {
            simulate_game(teams[i], teams[j]);
            simulate_game(teams[j], teams[i]);
        }
    }
}

void display_results(std::vector<int> win_count,
                     std::vector<std::vector<int>> position_counts,
                     std::vector<int> points_sum,
                     std::vector<Team> base_teams,
                     int NUM_SIMS) {
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
}

// run Monte Carlo simulation
float run_cpu_simulation(int NUM_SIMS) {
    std::vector<Team> base_teams(NUM_TEAMS);

    // initialize base teams
    for (int i = 0; i < NUM_TEAMS; i++) {
        base_teams[i].name = "Team " + std::to_string(i);
        base_teams[i].rating = 1500 + i * 30; // static elo ratings
        base_teams[i].points = 0;
        base_teams[i].goal_diff = 0;
        base_teams[i].total_goals = 0;
    }

    std::vector<int> win_count(NUM_TEAMS, 0);
    std::vector<std::vector<int>> position_counts(NUM_TEAMS, std::vector<int>(NUM_TEAMS, 0));
    std::vector<int> points_sum(NUM_TEAMS, 0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_SIMS; t++) {
        std::vector<Team> teams = base_teams;

        simulate_season(teams);

        std::vector<int> ranking(NUM_TEAMS);
        std::iota(ranking.begin(), ranking.end(), 0);

        // sort by points, goal difference, total goals
        std::sort(ranking.begin(), ranking.end(), 
                [&](int a, int b) {
                    if (teams[a].points != teams[b].points){
                        return teams[a].points > teams[b].points;
                    }
                    if (teams[a].goal_diff != teams[b].goal_diff){
                        return teams[a].goal_diff > teams[b].goal_diff;
                    }
                    return teams[a].total_goals > teams[b].total_goals;
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

    std::chrono::duration<float> elapsed = end - start;
    display_results(win_count, position_counts, points_sum, base_teams, NUM_SIMS);
    return elapsed.count();
}