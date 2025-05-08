from .functions import *

@numba.njit(fastmath=True, cache=True)
def inside_bar_ret_min(arr):
    if len(arr) <= 1:
        return 0.0
    _ret = log_ret(arr)
    return min_(_ret)/np.mean(_ret)

@numba.njit(fastmath=True, cache=True)
def inside_bar_ret_max(arr):
    if len(arr) <= 1:
        return 0.0
    _ret = log_ret(arr)
    return max_(_ret)/np.mean(_ret)

@numba.njit(fastmath=True, cache=True)
def inside_bar_cv(arr):
    if len(arr) == 1:
        return 0.0
    _ret = log_ret(arr)
    return np.std(_ret)/np.mean(_ret)

@numba.njit(fastmath=True, cache=True)
def inside_bar_skew(arr):
    if len(arr) <= 1:
        return 0.0
    _ret = log_ret(arr)
    return skewness(_ret)

@numba.njit(fastmath=True, cache=True)
def inside_bar_kurt(arr):
    if len(arr) <= 1:
        return 0.0
    _ret = log_ret(arr)
    return kurtosis(_ret)

# @numba.njit(fastmath=True, cache=True)
def inside_bar_maxdd(arr):
    if len(arr) <= 1:
        return 0.0
    return maxdd(arr)


def inside_bar_sign_entropy(arr):
    if len(arr) <= 1:
        return 0.0
    return simple_sign_entropy(arr)
    

# @numba.njit(fastmath=True, cache=True)
def inside_bar_rank_corr_a_b(arr_a, arr_b):
    if len(arr_a) <= 2: 
        return 0.0
    return rank_corr(arr_a, arr_b)

@numba.njit(fastmath=True, cache=True)
def inside_bar_ret_auto_corr(arr):
    if len(arr) <= 2: 
        return 0.0
    _ret = log_ret(arr)
    return corr(_ret[:-1], _ret[1:])

@numba.njit(fastmath=True, cache=True)
def inside_bar_price_auto_corr(arr):
    if len(arr) <= 2: 
        return 0.0
    return corr(arr[:-1], arr[1:])

def inside_bar_zero_cross(arr):
    if len(arr) <=1 :
        return 0.0
    arr = arr - arr[0]
    return float(ant.num_zerocross(arr))

def inside_bar_sample_entropy(arr,size=30,r_ratio=1):
    if len(arr) <= 1: 
        return 0.0
    arr = arr / arr[0]
    arr = reshape(arr,size)
    return dynamic_sample_entropy_numba(arr, m=2, r_ratio=r_ratio, use_std=True)

def inside_bar_spectral_entropy(arr,size=30):
    if len(arr) <= 1: 
        return 0.0
    arr = arr / arr[0]
    arr = reshape(arr,size)
    return ant.spectral_entropy(arr, sf=100, method='welch', normalize=True)



