#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>

#include <curand.h>
#include <curand_kernel.h>

#include "gpu_implementation.h"

const int NUM_TEAMS = 20;

// initialize cuRAND state for each thread
__global__ void init_rng (unsigned int seed, curandState_t* states) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(
        seed,
        tid,
        0,
        &states[tid]);
}

// simulates a match between teams i and j, updating season stat arrays
__device__ void simulate_match(
    int i, int j,
    const int* ratings,
    int* goal_diff,
    int* total_goals,
    int* points,
    curandState_t& state,
    bool homeA, bool atomicUpdate) 
{
    const float base_goals = 1.3f;
    const float k = 0.002f;
    const float home_adv = 0.2f;

    int diff = ratings[i] - ratings[j];

    float lambdaA = base_goals * expf(k * diff);
    float lambdaB = base_goals * expf(-k * diff);

    if (homeA) lambdaA += home_adv;
    else lambdaB += home_adv;

    // use poisson distribution to model goals scored
    int goalsA = curand_poisson(&state, lambdaA);
    int goalsB = curand_poisson(&state, lambdaB); 

    // keep track of goal differential + total goals
    if (atomicUpdate) {
        atomicAdd(&goal_diff[i], goalsA - goalsB);
        atomicAdd(&goal_diff[j], goalsB - goalsA);
        atomicAdd(&total_goals[i], goalsA);
        atomicAdd(&total_goals[j], goalsB);
    }
    else {
        goal_diff[i] += (goalsA - goalsB);
        goal_diff[j] += (goalsB - goalsA);
        total_goals[i] += goalsA;
        total_goals[j] += goalsB;
    }

    // update team points
    if (goalsA > goalsB) {
        if (atomicUpdate) atomicAdd(&points[i], 3);
        else points[i] += 3;
    }
    else if (goalsB > goalsA) {
        if (atomicUpdate) atomicAdd(&points[j], 3);
        else points[j] += 3;
    }
    else {
        if (atomicUpdate) {
            atomicAdd(&points[i], 1);
            atomicAdd(&points[j], 1);
        } 
        else {
            points[i] += 1;
            points[j] += 1;
        }
    }
}

// sorts the ranking array using premier league rules
// ranked by total points, goal difference, goals scored
__device__ void sort_ranking(
    int* ranking,
    const int* points,
    const int* goal_diff,
    const int* total_goals)
{
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
} 

// updates win, position, and point arrays using atomics
__device__ void accumulate_results(
    int* ranking,
    int* points,
    int* win_counts,
    int* position_counts,
    int* point_sums)
{
    int winner = ranking[0];
    atomicAdd(&win_counts[winner], 1);

    for (int i = 0; i < NUM_TEAMS; i++) {
        atomicAdd(&point_sums[i], points[i]);
    }

    for (int pos = 0; pos < NUM_TEAMS; pos++) {
        int team = ranking[pos];
        atomicAdd(&position_counts[team * NUM_TEAMS + pos], 1);
    }
}

// one group of threads per season
__global__ void simulate_seasons_hybrid(
    int* ratings,
    int* win_counts,
    int* position_counts,
    int* point_sums,
    curandState_t* states,
    int num_sims,
    int threads_per_season)
{
    const int tid = threadIdx.x;

    // number of seasons each block simulates
    const int seasons_per_block = blockDim.x / threads_per_season;
    // which season this thread is simulating
    const int season_in_block = tid / threads_per_season;
    // this thread's position within the group
    const int lane = tid % threads_per_season;

    const int global_season = blockIdx.x * seasons_per_block + season_in_block;
    if (global_season >= num_sims) return;

    // load cuRAND state for thread
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t localState = states[global_tid];

    // initialize local arrays
    int local_points[NUM_TEAMS] = {0};
    int local_goal_diff[NUM_TEAMS] = {0};
    int local_total_goals[NUM_TEAMS] = {0};

    // initialize shared memory arrays
    extern __shared__ int shared_mem[];
    const int per_season = 4 * NUM_TEAMS;
    int* season_base = shared_mem + season_in_block * per_season;
    int* points = season_base;
    int* goal_diff = points + NUM_TEAMS;
    int* total_goals = goal_diff + NUM_TEAMS;
    int* ranking = total_goals + NUM_TEAMS;

    for (int i = lane; i < NUM_TEAMS; i += threads_per_season) {
        points[i] = 0;
        goal_diff[i] = 0;
        total_goals[i] = 0;
        ranking[i] = i;
    }
    __syncthreads();

    // simulate matches
    // each thread only simulates a subset of its season
    const int TOTAL_MATCHES = NUM_TEAMS * (NUM_TEAMS - 1);
    for (int m = lane; m < TOTAL_MATCHES; m += threads_per_season) {
        int i = m / (NUM_TEAMS - 1);
        int j = m % (NUM_TEAMS - 1);
        if (j >= i) j++;

        simulate_match(i, j, ratings, local_goal_diff, local_total_goals, 
            local_points, localState, true, false);
    }

    // update shared memory
    for (int i = 0; i < NUM_TEAMS; i++) {
        atomicAdd(&points[i], local_points[i]);
        atomicAdd(&goal_diff[i], local_goal_diff[i]);
        atomicAdd(&total_goals[i], local_total_goals[i]);
    }
    __syncthreads();

    // one thread per group sorts + updates global results
    if (lane == 0) {
        sort_ranking(ranking, points, goal_diff, total_goals);
        accumulate_results(ranking, points, win_counts, position_counts, point_sums);
    }

    states[global_tid] = localState;
}

// one block per season
__global__ void simulate_seasons_block(
    int* ratings,
    int* win_counts,
    int* position_counts,
    int* point_sums,
    curandState_t* states,
    int num_sims) 
{
    int season_id = blockIdx.x;
    if (season_id >= num_sims) return;

    // load cuRAND state for thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t localState = states[tid];

    // initialize shared memory arrays
    extern __shared__ int shared_mem[];
    int* points = shared_mem;
    int* goal_diff = points + NUM_TEAMS;
    int* total_goals = goal_diff + NUM_TEAMS;
    int* ranking = total_goals + NUM_TEAMS;

    for (int i = threadIdx.x; i < NUM_TEAMS; i += blockDim.x) {
        points[i] = 0;
        goal_diff[i] = 0;
        total_goals[i] = 0;
        ranking[i] = i;
    }
    __syncthreads();

    const int TOTAL_MATCHES = NUM_TEAMS * (NUM_TEAMS - 1);

    // simulate matches
    // each thread only simulates a subset of the season
    for (int m = threadIdx.x; m < TOTAL_MATCHES; m += blockDim.x) {
        int i = m / (NUM_TEAMS - 1);
        int j = m % (NUM_TEAMS - 1);
        if (j >= i) j++;

        simulate_match(i, j, ratings, goal_diff, total_goals, points, 
            localState, true, true);
    }
    __syncthreads();

    // one thread per block sorts + updates global results
    if (threadIdx.x == 0) {
        sort_ranking(ranking, points, goal_diff, total_goals);
        accumulate_results(ranking, points, win_counts, position_counts, point_sums);
    }

    states[tid] = localState;
}

// one thread per season
__global__ void simulate_seasons(
    int* ratings,
    int* win_counts,
    int* position_counts,
    int* point_sums,
    curandState_t* states,
    int num_sims)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sims) return;

    // load cuRAND state for thread
    curandState_t localState = states[tid];

    // initialize local arrays
    int points[NUM_TEAMS] = {0};
    int goal_diff[NUM_TEAMS] = {0};
    int total_goals[NUM_TEAMS] = {0};
    int ranking[NUM_TEAMS];
    for (int i = 0; i < NUM_TEAMS; i++) {
        ranking[i] = i;
    }

    // initialize shared memory arrays
    extern __shared__ int shared_mem[];
    int* s_win_counts = shared_mem;
    int* s_point_sums = s_win_counts + NUM_TEAMS;
    int* s_position_counts = s_point_sums + NUM_TEAMS;

    for (int i = threadIdx.x; i < NUM_TEAMS; i += blockDim.x) {
        s_win_counts[i] = 0;
        s_point_sums[i] = 0;
    }
    for (int i = threadIdx.x; i < NUM_TEAMS * NUM_TEAMS; i += blockDim.x) {
        s_position_counts[i] = 0;
    }
    __syncthreads();

    // simulate season
    // each team plays each other twice
    for (int i = 0; i < NUM_TEAMS; i++) {
        for (int j = i + 1; j < NUM_TEAMS; j++) {
            for (int match = 0; match < 2; match++) {
                if (match == 0) {
                    simulate_match(i, j, ratings, goal_diff, total_goals,
                        points, localState, true, false);
                }
                else {
                    simulate_match(i, j, ratings, goal_diff, total_goals,
                        points, localState, false, false);
                }
            }
        }
    }

    sort_ranking(ranking, points, goal_diff, total_goals);

    // update shared memory
    accumulate_results(ranking, points, s_win_counts, s_position_counts, s_point_sums);
    __syncthreads();

    // use shared memory to update global results
    for (int i = threadIdx.x; i < NUM_TEAMS; i += blockDim.x) {
        atomicAdd(&win_counts[i], s_win_counts[i]);
        atomicAdd(&point_sums[i], s_point_sums[i]);
    }
    for (int i = threadIdx.x; i < NUM_TEAMS * NUM_TEAMS; i += blockDim.x) {
        atomicAdd(&position_counts[i], s_position_counts[i]);
    }

    states[tid] = localState;
}

// prints stats for each team from the overall simulation
// includes probability of winning the league, probability of finishing top 4,
// probability of finishing bottom 3 (relegation), and average number of points
void display_results(std::vector<int> win_counts,
                     std::vector<int> position_counts,
                     std::vector<int> point_sums,
                     int num_sims) 
{
    printf("%-10s %10s %10s %10s %10s\n", "Team", "Win%", "Top4%", 
        "Relegation%", "AvgPts");
    for (int i = 0; i < NUM_TEAMS; i++) {
        std::string team_name = "Team " + std::to_string(i);
        float win_prob = (float)win_counts[i] / num_sims;

        float top_4_prob = 0.0f;
        for (int pos = 0; pos < 4; pos++) {
            top_4_prob += position_counts[i * NUM_TEAMS + pos];
        }
        top_4_prob /= (float)num_sims;

        float relegation_prob = 0.0f;
        for (int pos = 17; pos < 20; pos++) {
            relegation_prob += position_counts[i * NUM_TEAMS + pos];
        }
        relegation_prob /= (float)num_sims;

        float avg_pts = point_sums[i] / (float)num_sims;

        printf("%-10s %10g %10g %10g %10.2f\n", team_name.c_str(), 
            win_prob, top_4_prob, relegation_prob, avg_pts);
    }
}

// run monte carlo simulation
float run_gpu_simulation(int num_sims, int block_size, int threads_per_season, 
                         unsigned int seed, Version v) 
{
    // set static elo ratings
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

    cudaMemcpy(d_ratings, h_ratings, NUM_TEAMS * sizeof(int), 
        cudaMemcpyHostToDevice);
    cudaMemset(d_win_counts, 0, NUM_TEAMS * sizeof(int));
    cudaMemset(d_position_counts, 0, NUM_TEAMS * NUM_TEAMS * sizeof(int));
    cudaMemset(d_point_sums, 0, NUM_TEAMS * sizeof(int));

    curandState_t* d_states;

    // set number of blocks
    int blocks;
    int seasons_per_block = block_size / threads_per_season;
    if (v == Version::ThreadPerSeason) {
        blocks = (num_sims + block_size - 1) / block_size;
    }
    else if (v == Version::Hybrid) {
        blocks = (num_sims + seasons_per_block - 1) / seasons_per_block;
    }
    else {
        blocks = num_sims;
    }

    // set shared memory size
    size_t shared_mem;
    if (v == Version::ThreadPerSeason) {
        shared_mem = (NUM_TEAMS + NUM_TEAMS + NUM_TEAMS * NUM_TEAMS) * 
            sizeof(int);
    }
    else if (v == Version::Hybrid) {
        shared_mem = (4 * NUM_TEAMS) * seasons_per_block * sizeof(int);
    }
    else {
        shared_mem = (4 * NUM_TEAMS) * sizeof(int);
    }

    // initialize cuRAND states
    cudaMalloc(&d_states, blocks * block_size * sizeof(curandState_t));
    init_rng<<<blocks, block_size>>>(seed, d_states);

    // run + time simulation
    auto start = std::chrono::high_resolution_clock::now();
    if (v == Version::ThreadPerSeason) {
        simulate_seasons<<<blocks, block_size, shared_mem>>>(
            d_ratings, d_win_counts, d_position_counts, d_point_sums, d_states,
            num_sims);
    }
    else if (v == Version::Hybrid) {
        simulate_seasons_hybrid<<<blocks, block_size, shared_mem>>>(
            d_ratings, d_win_counts, d_position_counts, d_point_sums, d_states,
            num_sims, threads_per_season);
    }
    else {
        simulate_seasons_block<<<blocks, block_size, shared_mem>>>(
            d_ratings, d_win_counts, d_position_counts, d_point_sums, d_states,
            num_sims);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // copy results into cpu vectors
    std::vector<int> win_counts(NUM_TEAMS);
    std::vector<int> position_counts(NUM_TEAMS * NUM_TEAMS);
    std::vector<int> point_sums(NUM_TEAMS);
    cudaMemcpy(win_counts.data(), d_win_counts, NUM_TEAMS * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(position_counts.data(), d_position_counts,
        NUM_TEAMS * NUM_TEAMS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(point_sums.data(), d_point_sums, NUM_TEAMS * sizeof(int),
        cudaMemcpyDeviceToHost);

    //print results
    display_results(win_counts, position_counts, point_sums, num_sims);

    // free device arrays
    cudaFree(d_states);
    cudaFree(d_point_sums);
    cudaFree(d_position_counts);
    cudaFree(d_win_counts);
    cudaFree(d_ratings);

    std::chrono::duration<float> elapsed = end - start;
    return elapsed.count();
}