#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>
#include "cpu_baseline.h"

const int NUM_TEAMS = 20;

// team structure
struct Team {
    int rating;
    int points;
    int goal_diff;
    int total_goals;
    std::string name;
};

// simulate a single game
void simulate_game(Team &A, Team &B, std::mt19937& rng) {
    const float base_goals = 1.3f;
    const float k = 0.002f;
    const float home_adv = 0.2f;

    int diff = A.rating - B.rating;

    float lambdaA = base_goals * std::exp(k * diff) + home_adv;
    float lambdaB = base_goals * std::exp(-k * diff);

    // use poisson distribution to model goals scored
    std::poisson_distribution<int> distA(lambdaA);
    std::poisson_distribution<int> distB(lambdaB);

    int goalsA = distA(rng);
    int goalsB = distB(rng);

    // keep track of goal differential and total goals
    A.goal_diff += (goalsA - goalsB);
    B.goal_diff += (goalsB - goalsA);
    A.total_goals += goalsA;
    B.total_goals += goalsB;

    // update team points
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
void simulate_season(std::vector<Team> &teams, std::mt19937& rng) {
    // reset points, goal differential, total goals
    for (auto &t : teams) {
        t.points = 0;
        t.goal_diff = 0;
        t.total_goals = 0;
    }

    // double round robin
    for (int i = 0; i < NUM_TEAMS; i++) {
        for (int j = i + 1; j < NUM_TEAMS; j++) {
            simulate_game(teams[i], teams[j], rng);
            simulate_game(teams[j], teams[i], rng);
        }
    }
}

void display_results(std::vector<int> win_count,
                     std::vector<std::vector<int>> position_counts,
                     std::vector<int> point_sums,
                     std::vector<Team> base_teams,
                     int num_sims) {
    printf("%-10s %10s %10s %10s %10s\n", "Team", "Win%", "Top4%", 
        "Relegation%", "AvgPts");
    //std::cout << "Team | Win Prob | Top 4 Prob | Relegation Prob | Avg Points\n";
    for (int i = 0; i < NUM_TEAMS; i++) {
        float win_prob = (float)win_count[i] / num_sims;

        float top_4_prob = 0.0f;
        for (int pos = 0; pos < 4; pos++) {
            top_4_prob += (float)position_counts[i][pos] / num_sims;
        }

        float relegation_prob = 0.0f;
        for (int pos = 17; pos < 20; pos++) {
            relegation_prob += (float)position_counts[i][pos] / num_sims;
        }

        float avg_pts = point_sums[i] / (float)num_sims;

        printf("%-10s %10g %10g %10g %10.2f\n", base_teams[i].name.c_str(), 
            win_prob, top_4_prob, relegation_prob, avg_pts);
        // std::cout << base_teams[i].name + " | "
        //           << win_prob << " | "
        //           << top_4_prob << " | "
        //           << relegation_prob << " | "
        //           << avg_pts << "\n";
    }
}

// run Monte Carlo simulation
float run_cpu_simulation(int num_sims, unsigned int seed) {
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
    std::vector<std::vector<int>> position_counts(
        NUM_TEAMS, std::vector<int>(NUM_TEAMS, 0));
    std::vector<int> point_sums(NUM_TEAMS, 0);

    // initialize rng
    std::mt19937 rng(seed);

    // run + time simulation
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < num_sims; t++) {
        std::vector<Team> teams = base_teams;

        simulate_season(teams, rng);

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
            point_sums[i] += teams[i].points;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    // print results
    display_results(win_count, position_counts, point_sums, base_teams,
        num_sims);

    std::chrono::duration<float> elapsed = end - start;
    return elapsed.count();
}