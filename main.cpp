#include <iostream>

#include "cpu_baseline.h"
#include "gpu_implementation.h"

#define DEFAULT_SIMS 50000
#define DEFAULT_BLOCK_SIZE 256

void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -s <int>   Number of simulations   (default: %d)\n", DEFAULT_SIMS);
    printf("  -b <int>   CUDA block size         (default: %d)\n", DEFAULT_BLOCK_SIZE);
    printf("  --no-cpu   Skip CPU simulation\n");
    printf("  -h         Show this help\n");
} 

int main(int argc, char** argv) {
    int num_sims = DEFAULT_SIMS;
    int block_size = DEFAULT_BLOCK_SIZE;
    int run_cpu = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            num_sims = atoi(argv[++i]);
            printf("Number of simulations changed to %d\n", num_sims);
        }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
            printf("Block size changed to %d\n", block_size);
        }
        else if (strcmp(argv[i], "--no-cpu") == 0) {
            run_cpu = 0;
            printf("Skipping CPU simulation\n");
        }
        else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    float cpu_time = 0.0f;;
    if (run_cpu) {
        std::cout << "Running CPU version: \n";
        float cpu_time = run_cpu_simulation(num_sims);
    }
    std::cout << "Running GPU version: \n";
    float gpu_time = run_gpu_simulation(num_sims, block_size);

    if (run_cpu) {
        std::cout << "\nCPU Execution time: " << cpu_time << " seconds\n";
        std::cout << "CPU Simulations per second: "
            << num_sims / cpu_time << "\n";
    }
    std::cout << "\nGPU Execution time: " << gpu_time << " seconds\n";
    std::cout << "GPU Simulations per second: "
          << num_sims / gpu_time << "\n";

    return 0;
}