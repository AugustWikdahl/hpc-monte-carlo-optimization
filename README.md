# HPC Monte Carlo Option Pricing Optimization

## Project Overview
This project focuses on characterizing and optimizing the performance of a Monte Carlo simulation for European Call Option pricing. 

Our goal is to benchmark a baseline Python implementation and apply **High-Performance Computing (HPC)** techniques to reduce execution time and improve memory efficiency.

## Source & Attribution
The baseline logic for this project is based on the Monte Carlo implementation by **Ebrahim Pichka**.
* **Original Repository:** [mc-option-pricing](https://github.com/ebrahimpichka/mc-option-pricing)
* **Original Script:** [`main.py`](https://github.com/ebrahimpichka/mc-option-pricing/blob/main/main.py)

We use this codebase as our control variable to measure the speedup achieved through our optimizations.

## Optimization Roadmap
We are implementing three distinct optimization levels to compare against the baseline:

- **Baseline:** Pure Python implementation (Profiled with `cProfile` & `line_profiler`)
- **Level 1 (Compilation):** JIT compilation using **Cython/Numba** to reduce interpreter overhead.
- **Level 2 (GPU Acceleration):** Offloading heavy path generation to the GPU using **CUDA**.
- **Level 3 (Parallelization):** Distributed workload across CPU cores using **Multiprocessing**.

## Tools & Libraries
* **Profiling:** `cProfile`, `line_profiler`, `memory_profiler`
* **Computation:** `NumPy`, `Numba`, `CuPy`
* **Parallelism:** `multiprocessing` module

---
*Project for DD2358 High-Performance Computing at KTH Royal Institute of Technology.*