import numpy as np
from numba import jit, prange
import math

@jit(nopython=True, fastmath=True)
def step_math(S, drift, vol):
    z = np.random.standard_normal()
    return S * math.exp(drift + vol * z)

@jit(nopython=True, fastmath=True, parallel=True)
def mc_price_numba(S0, K, r, sigma, T, M, I):
    dt = T / M
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    
    total_payoff = 0.0
    
    for i in prange(I):
        S = S0
        for t in range(M):
            S = step_math(S, drift, vol)
            
        payoff = S - K
        if payoff > 0:
            total_payoff += payoff
            
    return (math.exp(-r * T) * total_payoff) / I

def main():
    
    S0 = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.12
    T = 0.5 # 6 months
    M = 1000
    I = 50_000

    C0= mc_price_numba(S0, K, r, sigma, T, M, I)
    print(f">> Initial Stock Price: {S0}")
    print("="*28)
    print(f">> European Option with {T*12} months to maturity Value: {C0}")
    print("="*28)

    #plot_mc_paths(S, K, S0)

if __name__ == "__main__":
    main()