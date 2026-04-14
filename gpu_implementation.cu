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

// initialize cuRAND states
__global__ void init_rng (unsigned int seed, curandState_t* states) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(
        seed,
        tid,
        0,
        &states[tid]);
}

__global__ void simulate_seasons(
    float* ratings,
    int* win_counts,
    int* position_counts,
    int* point_sums,
    curandState_t* states,
    int num_teams,
    int num_sims)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_sims) return;

    // load cuRAND state for thread
    curandState_t localState = states[tid];

    // initialize local points array
    int points[20];
    for (int i = 0; i < num_teams; i++) {
        points[i] = 0;
    }

    // simulate season
    for (int i = 0; i < num_teams; i++) {
        for (int j = i + 1; j < num_teams; j++) {
            // each team plays each other twice
            for (int k = 0; k < 2; k++) {
                // TODO: change to use Poisson distributions
                float p_draw = 0.2f;
                float p_win_total = 1.0f - p_draw;
                float total = ratings[i] + ratings[j];
                float pA = p_win_total * (ratings[i] / total);
                float pB = p_win_total * (ratings[j] / total);

                float r = curand_uniform(&localState);

                if (r < pA) {
                    points[i] += 3;
                } else if (r < pA + pB) {
                    points[j] += 3;
                } else {
                    points[i] += 1;
                    points[j] += 1;
                }
            }
        }
    }

    int ranking[20];
    for (int i = 0; i < num_teams; i++) {
        ranking[i] = i;
    }

    // sort ranking based on points descending
    for (int i = 1; i < num_teams; i++) {
        int key = ranking[i];
        int j = i - 1;
        while (j >= 0 && points[ranking[j]] < points[key]) {
            ranking[j + 1] = ranking[j];
            j--;
        }
        ranking[j + 1] = key;
    }

    // update win counts, position counts, and point sums
    int winner = ranking[0];
    atomicAdd(&win_counts[winner], 1);

    for (int pos = 0; pos < num_teams; pos++) {
        int team = ranking[pos];
        atomicAdd(&position_counts[team * num_teams + pos], 1);
    }

    for (int i = 0; i < num_teams; i++) {
        atomicAdd(&point_sums[i], points[i]);
    }

    states[tid] = localState;
}

int main() {
    // set static ratings for now 
    float h_ratings[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        h_ratings[i] = 0.5f + (float)i / NUM_TEAMS;
    }

    // create + initialize device arrays
    float* d_ratings;
    int* d_win_counts;
    int* d_position_counts;
    int* d_point_sums;

    cudaMalloc(&d_ratings, NUM_TEAMS * sizeof(float));
    cudaMalloc(&d_win_counts, NUM_TEAMS * sizeof(int));
    cudaMalloc(&d_position_counts, NUM_TEAMS * NUM_TEAMS * sizeof(int));
    cudaMalloc(&d_point_sums, NUM_TEAMS * sizeof(int));

    cudaMemcpy(d_ratings, h_ratings, NUM_TEAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_win_counts, 0, NUM_TEAMS * sizeof(int));
    cudaMemset(d_position_counts, 0, NUM_TEAMS * NUM_TEAMS * sizeof(int));
    cudaMemset(d_point_sums, 0, NUM_TEAMS * sizeof(int));

    curandState_t* d_states;
    cudaMalloc(&d_states, NUM_SIMS * sizeof(curandState_t));

    int threads = 256;
    int blocks = (NUM_SIMS + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(42, d_states);

    auto start = std::chrono::high_resolution_clock::now();

    simulate_seasons<<<blocks, threads>>>(
        d_ratings, d_win_counts, d_position_counts, d_point_sums, d_states, NUM_TEAMS, NUM_SIMS);

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

    std::chrono::duration<float> elapsed = end - start;

    std::cout << "\nExecution time: " << elapsed.count() << " seconds\n";
    std::cout << "Simulations per second: "
          << NUM_SIMS / elapsed.count();

    // free objects
    cudaFree(d_states);
    cudaFree(d_point_sums);
    cudaFree(d_position_counts);
    cudaFree(d_win_counts);
    cudaFree(d_ratings);
}