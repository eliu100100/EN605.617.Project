all:
	nvcc main.cpp cpu_baseline.cpp gpu_implementation.cu -o sport_sim