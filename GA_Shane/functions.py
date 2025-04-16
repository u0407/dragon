import numba
import numpy as np
import pylab # Still needed for fast_fracdiff as currently written

"""
Core calculation functions (mostly Numba-optimized) and a stride-based
rolling window template.
"""

import numpy as np

def rolling_apply(func, arr, window_size=10, *args):
    w_arr = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    results = [np.nan]*(window_size-1) + [func(window, *args) for window in w_arr]
    return np.array(results)

def rolling_apply_last(func, arr, window_size=10, *args):
    # this function supports when the calculation result of func is an array, will retrieve last value of the array
    w_arr = np.lib.stride_tricks.sliding_window_view(arr, window_size)
    results = [np.nan]*(window_size-1) + [func(window, *args)[-1] for window in w_arr]
    return np.array(results)


# ---- Base Operators ---- 
@numba.njit(fastmath=True, cache=True)
def mean(arr,):
    return np.mean(arr)

@numba.njit(fastmath=True, cache=True)
def std(arr,):
    return np.std(arr)

@numba.njit(fastmath=True, cache=True)
def range_(arr,):
    return np.max(arr) - np.min(arr)

@numba.njit(fastmath=True, cache=True)
def sum(arr,):
    return np.sum(arr)

@numba.njit(fastmath=True, cache=True)
def prod(arr,):
    return np.prod(arr)





# --- Non-Numba FracDiff ---
# (Keeping original for completeness)
import pylab 
def fast_fracdiff(arr, d):
    T = len(arr)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(arr) + z
    dx = pylab.ifft(pylab.fft(z1) * pylab.fft(z2))
    return  np.real(dx[0:T]) 

# --- Entropy ---
@numba.njit(fastmath=True, cache=True)
def _maxdist_numba(x_i, x_j):
    max_val = 0.0
    for i in numba.prange(len(x_i)):
        diff = abs(x_i[i] - x_j[i])
        if diff > max_val:
            max_val = diff
    return max_val

@numba.njit(fastmath=True, cache=True)
def _phi_numba(x, m, tolerance):
    n = len(x) - m + 1
    if n <= 1:
        print(" n <= 1")
        return np.nan
    
    C = np.zeros(n)
    for i in numba.prange(n):
        x_i = x[i:i+m]
        count = 0
        for j in numba.prange(n):
            if i != j:
                x_j = x[j:j+m]
                if _maxdist_numba(x_i, x_j) <= tolerance:
                    count += 1
        C[i] = count
    
    denominator = n * (n - 1)
    if denominator == 0:
        print("denominator == 0")
        return np.nan
    
    
    result = np.sum(C) / denominator
    if result == 0:
        print("result == 0")
        return np.nan
    return result

@numba.njit(fastmath=True, cache=True)
def sample_entropy_numba(x, emb_dim, tolerance):
    if len(x) < emb_dim + 2:
        return np.nan
    
    phi_emb_dim = _phi_numba(x, emb_dim, tolerance)
    phi_emb_dim_plus_one = _phi_numba(x, emb_dim + 1, tolerance)
    
    if phi_emb_dim == 0 or phi_emb_dim_plus_one == 0:
        return 0
    
    return -np.log(phi_emb_dim_plus_one / phi_emb_dim)

@numba.njit(fastmath=True, cache=True)
def dynamic_sample_entropy_numba(x, m, r_ratio, use_std):
    if use_std:
        std_x = np.std(x)
        if std_x < 1e-6:
            return 0
        r = r_ratio * std_x
    else:
        range_x = np.max(x) - np.min(x)
        if range_x < 1e-6:
            return 0
        r = r_ratio * range_x
    
    entropy =  sample_entropy_numba(x, m, r)
    if np.isnan(entropy):
        return 0
    return entropy