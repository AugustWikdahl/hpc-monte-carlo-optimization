import numpy as np
import multiprocessing as mp
from functools import partial
import importlib

def mc_step_worker(I_chunk, S0, K, r, sigma, T, M):
    np.random.seed() 
    
    dt = T / M
    S_prev = np.full(I_chunk, S0)

    for t in range(1, M + 1):
        eps = np.random.standard_normal(I_chunk)
        S_curr = S_prev * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * eps * np.sqrt(dt)
        )
        S_prev = S_curr
    
    return np.maximum(S_prev - K, 0)

def _get_worker():
    """Import mc_step_worker from the actual module file so it's picklable
    even when the script is executed via exec() (e.g. mprof run)."""
    try:
        mod = importlib.import_module("main")
    except ModuleNotFoundError:
        mod = importlib.import_module("multiprocessing_version.main")
    return mod.mc_step_worker

def mc_price_option_parallel(S0, K, r, sigma, T, M, I, pool):
    num_processes = pool._processes

    chunk_size = I // num_processes
    chunks = [chunk_size] * num_processes
    for i in range(I % num_processes):
        chunks[i] += 1

    worker_func = partial(_get_worker(), S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    results = pool.map(worker_func, chunks)

    all_payoffs = np.concatenate(results)
    C0 = np.exp(-r * T) * np.mean(all_payoffs)
    
    return C0, None

def main():
    
    S0 = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.12
    T = 0.5 # 6 months
    M = 1000
    I = 50_000

    with mp.Pool(processes=8) as pool:
        C0, S = mc_price_option_parallel(S0, K, r, sigma, T, M, I, pool)
    print(f">> Initial Stock Price: {S0}")
    print("="*28)
    print(f">> European Option with {T*12} months to maturity Value: {C0}")
    print("="*28)

    #plot_mc_paths(S, K, S0)

if __name__ == "__main__":
    main()