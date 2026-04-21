#include <iostream>

#include "cpu_baseline.h"
#include "gpu_implementation.h"

int main(int argc, char** argv) {
    int NUM_SIMS = 10000;
    if (argc >= 2) {
        NUM_SIMS = atoi(argv[1]);
        std::cout << "Number of simulations changed to " << NUM_SIMS << "\n";
    }
    std::cout << "Running CPU version: \n";
    float cpu_time = run_cpu_simulation(NUM_SIMS);
    std::cout << "Running GPU version: \n";
    float gpu_time = run_gpu_simulation(NUM_SIMS);

    std::cout << "\nCPU Execution time: " << cpu_time << " seconds\n";
    std::cout << "CPU Simulations per second: "
          << NUM_SIMS / cpu_time << "\n";
    std::cout << "\nGPU Execution time: " << gpu_time << " seconds\n";
    std::cout << "GPU Simulations per second: "
          << NUM_SIMS / gpu_time << "\n";

    return 0;
}