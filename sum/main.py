import numpy as np
import time

def mc_price_option(S0, K, r, sigma, T, M, I):

    dt = T / M
    drift = (r - 0.5 * sigma**2) * T
    vol_sqrt_dt = sigma * np.sqrt(dt)
    
    eps = np.random.standard_normal((M, I))
    S_final = S0 * np.exp(drift + vol_sqrt_dt * np.sum(eps, axis=0))
    
    C0 = np.exp(-r * T) * np.sum(np.maximum(S_final - K, 0)) / I
    return C0

def main():
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.12, 0.5
    M, I = 1000, 50_000

    start = time.perf_counter()
    C0 = mc_price_option(S0, K, r, sigma, T, M, I)
    end = time.perf_counter()

    print(f">> Initial Stock Price: {S0}")
    print("="*28)
    print(f">> European Option Value: {C0:.4f}")
    print(f">> Execution Time: {end - start:.4f}s")
    print("="*28)

if __name__ == "__main__":
    main()