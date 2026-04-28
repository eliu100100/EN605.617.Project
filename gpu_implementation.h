#pragma once

enum class Version {
    ThreadPerSeason,
    BlockPerSeason,
    Hybrid
};

float run_gpu_simulation(int num_sims, int block_size, int threads_per_season, unsigned int seed, Version v);