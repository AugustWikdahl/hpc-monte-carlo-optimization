import numpy as np
from numba import jit
import math

@jit(nopython=True, fastmath=True)
def step_math(S, drift, vol):
    z = np.random.standard_normal()
    return S * math.exp(drift + vol * z)

@jit(nopython=True, fastmath=True)
def mc_price_numba(S0, K, r, sigma, T, M, I):
    dt = T / M
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    
    total_payoff = 0.0
    
    for i in range(I):
        S = S0
        for t in range(M):
            S = step_math(S, drift, vol)
            
        payoff = S - K
        if payoff > 0:
            total_payoff += payoff
            
    return (math.exp(-r * T) * total_payoff) / I