# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, cos, M_PI
from libc.stdlib cimport rand, srand, RAND_MAX
import time

# 1. C-Level Helper Function (The Box-Muller Transform)
# This turns raw C random numbers into Gaussian distribution.
# 'cdef' means this function is invisible to Python and has 0 call overhead.
cdef double c_gaussian() noexcept:
    cdef double u1 = rand() / (RAND_MAX + 1.0)
    cdef double u2 = rand() / (RAND_MAX + 1.0)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

# 2. Main Pricing Function
def mc_price_cython(double S0, double K, double r, double sigma, double T, int M, int I):
    
    # Seed the C RNG so it's not the same every time
    srand( <unsigned int> time.time() )
    
    cdef double dt = T / M
    cdef double drift = (r - 0.5 * sigma**2) * dt
    cdef double vol = sigma * sqrt(dt)
    cdef double total_payoff = 0.0
    cdef double S, payoff, z
    cdef int i, t
    
    # 3. The Pure C Loop
    # Now, NOTHING inside this loop touches Python. It is 100% Machine Code.
    for i in range(I):
        S = S0
        for t in range(M):
            # We call our C helper, not NumPy
            z = c_gaussian()
            S = S * exp(drift + vol * z)
        
        payoff = S - K
        if payoff > 0:
            total_payoff += payoff
            
    return (exp(-r * T) * total_payoff) / I