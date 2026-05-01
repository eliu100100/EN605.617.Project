# GPU-Accelerated Monte Carlo Soccer Simulation

## Overview

This project implements a Monte Carlo simulation of a English Premier League season using CUDA. It models match outcomes probabilistically and evaluates team performance over thousands of simulated seasons.

The primary goal is to compare CPU and GPU execution strategies while demonstrating how parallel computing can dramatically accelerate certain workflows.

## Simulation Model

### Match Modeling

Each match is simulated using independent Poisson distributions:

* Goals scored in a match is modeled with the expected goals $\lambda$:
    * $\lambda$ = base rate * $e^{\text{k * rating difference}}$
    * base rate = 1.3, k = 0.002
* A minor home advantage is applied to $\lambda$ as a bias of 0.2
* Random values are generated using cuRAND

### Competition Rules

* 20 teams
* Each team plays every other team twice, once at home and once away
* Point system:
    * Win: 3 points
    * Draw: 1 point
    * Loss: 0 points
* Teams are ranked by total points, with goal difference and total goals scored used as tiebreakers.

---

## Implementations

The project explores one CPU and three GPU implementations:

### 1. CPU

* Fully sequential simulation of all seasons
* Utilizes `std` vectors 

### 2. GPU Thread Per Season 

* Each thread simulates one full season
* Minimizes atomic updates of global results via shared memory

### 3. GPU Block Per Season

* Each block simulates one full season
* Stores overall season statistics in shared memory

### 4. GPU Hybrid (Thread Group Per Season)

* Each "thread group" simulates one full season
* Each block contains multiple thread groups

---

## Requirements

| Requirement | Version |
|---|---|
| CUDA Toolkit | 13.1 (tested) |
| cuRAND | included with CUDA Toolkit |
| C++ standard | C++14 |
| Compiler | `nvcc` |

---

## Build

```
make    # compiles everything into ./sport_sim.exe
```

---

## Run

```
./sport_sim.exe [options]
```

Options:

| Flag | Description | Default |
| --- | --- | --- |
| `-s <int>` | Number of seasons to simulate | 50000 |
| `-b <int>` | CUDA block size | 256 |
| `-t <int>` | Threads per season (hybrid mode) | 32 |
| `--no-cpu` | Skip CPU baseline | |
| `-h` | Show help | |

Example:

```
./sport_sim.exe -s 100000 -b 512 -t 64
```

---

## Sample Output
![sample output](./sample%20output.PNG)  

![](./sample%20performance.PNG)