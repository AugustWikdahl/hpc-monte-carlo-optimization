# HPC Monte Carlo Option Pricing Optimization

A high performance engineering study applying HPC optimization techniques to Monte Carlo European Call Option pricing in Python. Each optimization technique was an attempt to speed-up the bottlenecks through profiling, and all results are benchmarked against a common baseline.

## Results

Speedup measured at paths I = **100.000** and time steps M = 1000, and I = 10.000 and M = **5000**.

| Version | I=100k Time (s) | Speedup | M=5k Time (s) | Speedup |
|---|---|---|---|---|
| **Baseline (NumPy)** | 2.2488 | 1.00x | 6.1063 | 1.00x | 
| **Cython** | 3.8907 | 0.58x | 8.6762 | 0.70x | 
| **Numba JIT** | 3.5817 | 0.63x | 9.0993 | 0.67x | 
| **Summation Trick** | 1.4706 | **1.53x** | 4.0529 | **1.51x** | 
| **Multiprocessing** | 0.5737 | **3.92x** | 2.5378 | **2.41x** | 
| **GPU (Total)** | 0.4098 | **5.49x** | 1.3165 | **4.64x** | 
| **GPU (Exec Only)** | **0.1184** | **18.99x** | **0.5823** | **10.49x** | 

> **Key finding:** Compiled scalar loops (Numba/Cython) do *not* outperform well-vectorized NumPy for this workload. NumPy delegates to optimized BLAS/LAPACK routines that already operate at near-C speed on bulk array operations. The real wins come from **algorithmic reduction** (summation trick), **parallelism** (multiprocessing), and **GPU offloading** (CuPy/CUDA achieving up to ~19x speedup).

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
We used Numba to compile our Python code into fast machine code to see if it could beat the baseline. It turns out it couldn't. Processing the math one step at a time is much slower than NumPy's ability to handle large blocks of data all at once.

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
