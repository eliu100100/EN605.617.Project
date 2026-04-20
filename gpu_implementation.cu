#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

const int NUM_TEAMS = 20;
const int NUM_SIMS = 100000;

// initialize cuRAND state for each thread
__global__ void init_rng (unsigned int seed, curandState_t* states) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(
        seed,
        tid,
        0,
        &states[tid]);
}

__global__ void simulate_seasons(
    int* ratings,
    int* win_counts,
    int* position_counts,
    int* point_sums,
    curandState_t* states)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= NUM_SIMS) return;

    // load cuRAND state for thread
    curandState_t localState = states[tid];

    // initialize local arrays
    int points[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        points[i] = 0;
    }
    int goal_diff[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        goal_diff[i] = 0;
    }
    int total_goals[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        total_goals[i] = 0;
    }

    // simulate season
    for (int i = 0; i < NUM_TEAMS; i++) {
        for (int j = i + 1; j < NUM_TEAMS; j++) {
            // each team plays each other twice
            for (int match = 0; match < 2; match++) {
                const float base_goals = 1.3f;
                const float k = 0.002f;
                const float home_adv = 0.2f;

                int diff = ratings[i] - ratings[j];

                float lambdaA = base_goals * expf(k * diff);
                float lambdaB = base_goals * expf(-k * diff);
                if (match == 0) {
                    lambdaA += home_adv;
                }
                else {
                    lambdaB += home_adv;
                }

                int goalsA = curand_poisson(&localState, lambdaA);
                int goalsB = curand_poisson(&localState, lambdaB);

                goal_diff[i] += (goalsA - goalsB);
                goal_diff[j] += (goalsB - goalsA);
                total_goals[i] += goalsA;
                total_goals[j] += goalsB;

                if (goalsA > goalsB) {
                    points[i] += 3;
                }
                else if (goalsB > goalsA) {
                    points[j] += 3;
                }
                else {
                    points[i] += 1;
                    points[j] += 1;
                }
            }
        }
    }

    int ranking[20];
    for (int i = 0; i < NUM_TEAMS; i++) {
        ranking[i] = i;
    }

    // sort ranking based on points descending
    // use goal differential and total goals as tiebreakers
    for (int i = 1; i < NUM_TEAMS; i++) {
        int key = ranking[i];
        int j = i - 1;
        while (j >= 0) {
            int ptsJ = points[ranking[j]];
            int ptsK = points[key];
            int gdJ = goal_diff[ranking[j]];
            int gdK = goal_diff[key];
            int totalJ = total_goals[ranking[j]];
            int totalK = total_goals[key];
            bool should_swap = (ptsJ < ptsK) || (ptsJ == ptsK && gdJ < gdK) ||
                               (ptsJ == ptsK && gdJ == gdK && totalJ < totalK);
            if (!should_swap) {
                break;
            }
            ranking[j + 1] = ranking[j];
            j--;
        }
        ranking[j + 1] = key;
    }

    // update win counts, position counts, and point sums
    int winner = ranking[0];
    atomicAdd(&win_counts[winner], 1);

    for (int pos = 0; pos < NUM_TEAMS; pos++) {
        int team = ranking[pos];
        atomicAdd(&position_counts[team * NUM_TEAMS + pos], 1);
    }

    for (int i = 0; i < NUM_TEAMS; i++) {
        atomicAdd(&point_sums[i], points[i]);
    }

    states[tid] = localState;
}

void display_results(std::vector<int> win_counts, 
                     std::vector<int> position_counts, 
                     std::vector<int> point_sums,  
                     std::chrono::duration<float> elapsed) {
    std::cout << "Team | Win Prob | Top 4 Prob | Relegation Prob | Avg Points\n";
    for (int i = 0; i < NUM_TEAMS; i++) {
        float win_prob = (float)win_counts[i] / NUM_SIMS;

        float top_4_prob = 0.0f;
        for (int pos = 0; pos < 4; pos++) {
            top_4_prob += position_counts[i * NUM_TEAMS + pos];
        }
        top_4_prob /= (float)NUM_SIMS;

        float relegation_prob = 0.0f;
        for (int pos = 17; pos < 20; pos++) {
            relegation_prob += position_counts[i * NUM_TEAMS + pos];
        }
        relegation_prob /= (float)NUM_SIMS;

        float avg_pts = point_sums[i] / (float)NUM_SIMS;

        std::cout << i << " | "
                  << win_prob << " | "
                  << top_4_prob << " | "
                  << relegation_prob << " | "
                  << avg_pts << "\n";
    }

    std::cout << "\nExecution time: " << elapsed.count() << " seconds\n";
    std::cout << "Simulations per second: "
          << NUM_SIMS / elapsed.count();
}

int main() {
    // set static ratings for now 
    int h_ratings[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        h_ratings[i] = 1500 + i * 30;
    }

    // create + initialize device arrays
    int* d_ratings;
    int* d_win_counts;
    int* d_position_counts;
    int* d_point_sums;

    cudaMalloc(&d_ratings, NUM_TEAMS * sizeof(int));
    cudaMalloc(&d_win_counts, NUM_TEAMS * sizeof(int));
    cudaMalloc(&d_position_counts, NUM_TEAMS * NUM_TEAMS * sizeof(int));
    cudaMalloc(&d_point_sums, NUM_TEAMS * sizeof(int));

    cudaMemcpy(d_ratings, h_ratings, NUM_TEAMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_win_counts, 0, NUM_TEAMS * sizeof(int));
    cudaMemset(d_position_counts, 0, NUM_TEAMS * NUM_TEAMS * sizeof(int));
    cudaMemset(d_point_sums, 0, NUM_TEAMS * sizeof(int));

    curandState_t* d_states;
    cudaMalloc(&d_states, NUM_SIMS * sizeof(curandState_t));

    // set blocks + threads
    int threads = 256;
    int blocks = (NUM_SIMS + threads - 1) / threads;

    // initialize cuRAND states
    init_rng<<<blocks, threads>>>(42, d_states);

    // run + time simulation
    auto start = std::chrono::high_resolution_clock::now();
    simulate_seasons<<<blocks, threads>>>(
        d_ratings, d_win_counts, d_position_counts, d_point_sums, d_states);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // copy results into cpu vectors
    std::vector<int> win_counts(NUM_TEAMS);
    std::vector<int> position_counts(NUM_TEAMS * NUM_TEAMS);
    std::vector<int> point_sums(NUM_TEAMS);
    cudaMemcpy(win_counts.data(), d_win_counts, NUM_TEAMS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(position_counts.data(), d_position_counts, NUM_TEAMS * NUM_TEAMS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_sums.data(), d_point_sums, NUM_TEAMS * sizeof(int), cudaMemcpyDeviceToHost);

    //print results
    std::chrono::duration<float> elapsed = end - start;
    display_results(win_counts, position_counts, point_sums, elapsed);

    // free objects
    cudaFree(d_states);
    cudaFree(d_point_sums);
    cudaFree(d_position_counts);
    cudaFree(d_win_counts);
    cudaFree(d_ratings);
}