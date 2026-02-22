# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, log, cos, M_PI
from libc.stdlib cimport rand, srand, RAND_MAX
import time

cdef double c_gaussian() noexcept:
    cdef double u1 = rand() / (RAND_MAX + 1.0)
    cdef double u2 = rand() / (RAND_MAX + 1.0)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

def mc_price_cython(double S0, double K, double r, double sigma, double T, int M, int I):
    srand( <unsigned int> time.time() )
    
    cdef double dt = T / M
    cdef double drift = (r - 0.5 * sigma**2) * dt
    cdef double vol = sigma * sqrt(dt)
    cdef double total_payoff = 0.0
    cdef double S, payoff, z
    cdef int i, t
    
    for i in range(I):
        S = S0
        for t in range(M):
            z = c_gaussian()
            S = S * exp(drift + vol * z)
        
        payoff = S - K
        if payoff > 0:
            total_payoff += payoff
            
    return (exp(-r * T) * total_payoff) / I