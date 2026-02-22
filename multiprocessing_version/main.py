import numpy as np
import multiprocessing as mp
from functools import partial

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

def mc_price_option_parallel(S0, K, r, sigma, T, M, I, pool):
    num_processes = pool._processes

    chunk_size = I // num_processes
    chunks = [chunk_size] * num_processes
    for i in range(I % num_processes):
        chunks[i] += 1

    worker_func = partial(mc_step_worker, S0=S0, K=K, r=r, sigma=sigma, T=T, M=M)
    results = pool.map(worker_func, chunks)

    all_payoffs = np.concatenate(results)
    C0 = np.exp(-r * T) * np.mean(all_payoffs)
    
    return C0, None