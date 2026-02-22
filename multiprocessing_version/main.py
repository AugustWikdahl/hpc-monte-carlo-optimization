import numpy as np
import multiprocessing as mp
from functools import partial

def mc_step_worker(I_chunk, S0, K, r, sigma, T, M):
    """Worker function to simulate a chunk of paths."""
    # Each process needs its own random seed or they will all produce the same paths
    np.random.seed() 
    
    dt = T / M
    # S_chunk only stores the current and previous step to save memory
    S_prev = np.full(I_chunk, S0)

    for t in range(1, M + 1):
        eps = np.random.standard_normal(I_chunk)
        S_curr = S_prev * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * eps * np.sqrt(dt)
        )
        S_prev = S_curr
    
    # Return payoffs to be averaged
    return np.maximum(S_prev - K, 0)

def mc_price_option_parallel(S0, K, r, sigma, T, M, I, pool):
    """Parallelized Monte Carlo using a persistent pool."""
    num_processes = pool._processes

    # Split I into chunks for the cores
    chunk_size = I // num_processes
    chunks = [chunk_size] * num_processes
    for i in range(I % num_processes):
        chunks[i] += 1

    # Partial function to fix the constant arguments
    worker_func = partial(mc_step_worker, S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    
    # Map work to the existing pool
    results = pool.map(worker_func, chunks)

    # Aggregate results
    all_payoffs = np.concatenate(results)
    C0 = np.exp(-r * T) * np.mean(all_payoffs)
    
    return C0, None