# HPC Monte Carlo Option Pricing Optimization

A high performance engineering study applying HPC optimization techniques to Monte Carlo European Call Option pricing in Python. Each optimization technique was an attempt to speed-up the bottlenecks through profiling, and all results are benchmarked against a common baseline.

## Results

Speedup measured at paths I = **100.000** and time steps M = 1000, and I = 10.000 and M = **5000**.

| Version | I=100k Time (s) | Speedup | M=5k Time (s) | Speedup |
|---|---|---|---|---|
| **Numba (Parallel prange)** | **0.2791** | **8.06x** | **0.9993** | **6.11x** |
| **GPU (CuPy Total)** | 0.4098 | 5.49x | 1.3165 | 4.64x |
| **Multiprocessing** | 0.5737 | 3.92x | 2.5378 | 2.41x |
| **Summation Trick** | 1.4706 | 1.53x | 4.0529 | 1.51x |
| **Baseline (NumPy)** | 2.2488 | 1.00x | 6.1063 | 1.00x | 
| **Numba (Serial)** | 3.5817 | 0.63x | 9.0993 | 0.67x | 
| **Cython (Serial)** | 3.8907 | 0.58x | 8.6762 | 0.70x | 

> **Key finding:** The clear winner for end-to-end execution time is Numba with `prange` for multi-threading, successfully avoiding the memory transfer bottlenecks of the GPU. Serial compiled loops (Numba/Cython without parallelism) do *not* outperform well-vectorized NumPy. The problem benefits more from parallelization when the number of paths (I) scales versus when the time steps (M) scale.

## Profiling the Baseline

`line_profiler` on the baseline (`M=1000, I=50000`) identified two dominant hotspots:

| Line | Operation | % Time |
|---|---|---|
| `np.random.standard_normal(I)` | RNG per time step | **66.9%** |
| `S[t] = S[t-1] * np.exp(...)` | GBM step computation | **33.1%** |

These two lines inside the `for t in range(M)` loop account for ~100% of execution time, making the loop body the sole optimization target.

## Optimizations

### Summation Trick (Algorithmic)
Exploits the additive property of log-returns: instead of computing `exp()` at every step for every path, the random increments are summed across time steps first, then a single `exp()` per path produces the final price. This reduces `M × I` calls to `exp()` down to `I`, and eliminates the time-step loop entirely.

### Numba JIT (Compilation)
The serial Numba version compiled scalar loops but could not beat the baseline — processing one step at a time is slower than NumPy's vectorized bulk operations. Adding `prange` for multi-threaded parallelism over the `I` paths made Numba the fastest version overall (8.06x at I=100k), outperforming even the GPU by avoiding memory transfer overhead.

### Cython (Compilation to C)
We translated our Python code directly into C and built our own custom random number generator. Just like Numba, this version was slower than the baseline because it forces the computer to calculate the math one step at a time instead of processing it in large, efficient chunks.

### Multiprocessing (Parallelism)
Splits the `I` paths across 8 worker processes using `multiprocessing.Pool`. Each worker runs the full per-step simulation on its chunk independently, and payoffs are aggregated at the end. Achieves near-linear speedup for large `I` values where the per-worker chunk size is large enough to amortize IPC overhead.

### CuPy / CUDA (GPU)
GPU-accelerated version using CuPy as a NumPy replacement, offloading array operations to the GPU. Explored in `cupy_version/cupy_version.ipynb`.

## Project Structure

```
├── baseline/              # Vectorized NumPy baseline (profiled)
│   ├── main.py
│   └── benchmark.py
├── sum/                   # Summation trick optimization
│   ├── main.py
│   └── benchmark.py
├── numba_version/         # Numba JIT compiled scalar loops
│   ├── main.py
│   └── benchmark.py
├── cython_version/        # Cython compiled to C
│   ├── main.pyx
│   ├── setup.py
│   └── benchmark.py
├── multiprocessing_version/  # Path-parallel across CPU cores
│   ├── main.py
│   └── benchmark.py
├── cupy_version/          # GPU-accelerated (CuPy/CUDA)
│   └── cupy_version.ipynb
└── logs/                  # All benchmark CSVs and profiling output
    ├── baseline/
    ├── sum/
    ├── numba_version/
    ├── multiprocessing/
    └── cython_version/
```

## Tech Stack

| Category | Tools |
|---|---|
| **Profiling** | `cProfile`, `line_profiler` |
| **Computation** | NumPy, Numba, Cython, CuPy |
| **Parallelism** | `multiprocessing.Pool` |

## Credits

Baseline Monte Carlo logic adapted from [mc-option-pricing](https://github.com/ebrahimpichka/mc-option-pricing) by **Ebrahim Pichka**.

---

*Project for DD2358 High-Performance Computing at KTH Royal Institute of Technology.*
