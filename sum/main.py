import numpy as np
import time

def mc_price_option(S0, K, r, sigma, T, M, I):
    """ 
    NumPy Version using Summation Trick (Strength Reduction).
    Reduces exp() calls from M*I to just I.
    """
    dt = T / M
    # Pre-calculate drift and volatility components
    drift = (r - 0.5 * sigma**2) * T # Note: T used here because we sum all dt
    vol_sqrt_dt = sigma * np.sqrt(dt)
    
    # Generate all random numbers at once (Vectorized)
    eps = np.random.standard_normal((M, I))
    
    # The Summation Trick: sum the randoms across time steps first
    # This replaces the iterative multiplication loop
    S_final = S0 * np.exp(drift + vol_sqrt_dt * np.sum(eps, axis=0))
    
    # Payoff calculation
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