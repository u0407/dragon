import numba
import numpy as np
import pylab # Still needed for fast_fracdiff as currently written
import antropy as ant

"""
Core calculation functions (mostly Numba-optimized) and arr stride-based
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

def rolling_apply_matrix(func, arrs, window_size=10, *args):
    # arr must in the order of [open,high,low,close,volume,total_turnover,open_interest]
    w_arr = np.lib.stride_tricks.sliding_window_view(arrs, (window_size, 1))
    res_arr = [func(window[:,:,0].T, *args) for window in w_arr]
    if isinstance(res_arr[0], float):
        pass
    elif len(res_arr[0].shape) == 1: # if is a float
        res_arr = [res[0] for res in res_arr]
    elif len(res_arr[0].shape) == 2: # if is a (float, ) in a array
        res_arr = [res[-1][0] for res in res_arr]
    results = [np.nan]*(window_size-1) + res_arr
    return np.array(results)

def rolling_apply_last_matrix(func, arrs, window_size=10, *args):
    # arr must in the order of [open,high,low,close,volume,total_turnover,open_interest]
    w_arr = np.lib.stride_tricks.sliding_window_view(arrs, (window_size, 1))
    results = [np.nan]*(window_size-1) + [func(window, *args)[-1][0] for window in w_arr]
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

@numba.njit(fastmath=True, cache=True)
def min_(arr,):
    return np.min(arr)

@numba.njit(fastmath=True, cache=True)
def max_(arr,):
    return np.max(arr)

@numba.njit(fastmath=True, cache=True)
def delay(arr, n=1):
    return np.concatenate((np.full(n, np.nan), arr[:-n]))

@numba.njit(fastmath=True, cache=True)
def delta(arr, n=1):
    return np.concatenate((np.full(n, np.nan), arr[n:] - arr[:-n]))

@numba.njit(fastmath=True, cache=True)
def pchg(arr, n=1):
    return np.concatenate((np.full(n, np.nan), (arr[n:] - arr[:-n]) / arr[:-n]))

@numba.njit(fastmath=True, cache=True)
def log_ret(arr, n=1):
    log_arr = np.log(arr)
    ret = log_arr[n:] - log_arr[:-n]
    return ret

@numba.njit(fastmath=True, cache=True)
def argmin(arr,):
    return np.argmin(arr)

@numba.njit(fastmath=True, cache=True)
def argmax(arr,):
    return np.argmax(arr)

# @numba.njit(fastmath=True, cache=True)
def corr(arr1, arr2):
    return np.corrcoef(arr1, arr2)[0, 1]

def rank(arr):
    n = len(arr)
    sorted_indices = np.argsort(arr)
    ranks = np.empty(n, dtype=np.int32)
    ranks[sorted_indices] = np.arange(1, n + 1)
    return ranks

# @numba.njit(fastmath=True, cache=True)
def rank_corr(arr1, arr2):
    rank1 = rank(arr1.flatten())
    rank2 = rank(arr2.flatten())
    return np.corrcoef(rank1, rank2)[0, 1]

@numba.njit(fastmath=True, cache=True)
def scale(arr):
    return (arr-np.mean(arr)) / np.std(arr)

@numba.njit(fastmath=True, cache=True)
def sign(arr):
    return np.sign(arr)

@numba.njit(fastmath=True, cache=True)
def signpow(arr, n=1):
    return np.sign(arr) * np.abs(arr) ** n



@numba.njit(fastmath=True, cache=True)
def std(arr):
    mean = np.sum(arr) / len(arr)
    sum_sq = 0.0
    for x in arr:
        sum_sq += (x - mean) ** 2
    n = len(arr)
    if n == 1:
        return 0
    std = np.sqrt(sum_sq / (n - 1))
    if std == 0:
        return 0.0 
    return std

@numba.njit(fastmath=True, cache=True)
def skewness(arr):
    mean = np.sum(arr) / len(arr)
    sum_cubed = 0.0
    sum_sq = 0.0
    n = len(arr)
    for x in arr:
        dev = x - mean
        sum_sq += dev ** 2
        sum_cubed += dev ** 3
    if n <= 1 or sum_sq == 0:
        return 0.0
    variance = sum_sq / (n - 1)
    std = np.sqrt(variance)
    skew = (sum_cubed / n) / (std ** 3)
    if n >= 3:
        skew *= (n * (n - 1)) ** 0.5 / (n - 2)
    return skew

@numba.njit(fastmath=True, cache=True)
def kurtosis(arr):
    mean = np.sum(arr) / len(arr)
    sum_quad = 0.0
    sum_sq = 0.0
    n = len(arr)
    for x in arr:
        dev = x - mean
        dev_sq = dev ** 2
        sum_sq += dev_sq
        sum_quad += dev_sq ** 2
    if n <= 1 or sum_sq == 0:
        return 0.0
    variance = sum_sq / (n - 1)
    kurt = (sum_quad / n) / (variance ** 2) - 3
    if n >= 4:
        kurt = ((n + 1) * kurt + 6) * (n - 1) / ((n - 2) * (n - 3))
    return kurt

@numba.njit(fastmath=True, cache=True)
def momentum(arr,n=5):
    
    arr = (arr - np.nanmean(arr)) / std 
    momentum_5 = np.nanmean((arr-np.nanmean(arr))**n) / std**n
    return momentum_5

@numba.njit(fastmath=True, cache=True)
def corr(arr1, arr2):
    return np.corrcoef(arr1, arr2)[0, 1]

@numba.njit(fastmath=True, cache=True)
def value_at_risk(a, alpha=0.05):
    a = (a - np.mean(a)) / np.std(a)
    var = np.percentile(a, alpha * 100)
    return var

# @numba.njit(fastmath=True, cache=True)
def maxdd(arr):
    running_max_y = np.maximum.accumulate(arr)
    dd = arr / running_max_y - 1 
    return - np.min(dd)


# --- TIme Series ---
def ts_rank(x, d):
    return rolling_apply(rank, x, d)

def ts_argmax(x, d):
    return rolling_apply(argmax, x, d)

def ts_argmin(x, d):
    return rolling_apply(argmin, x, d)
    
def ts_max(x, d):
    return rolling_apply(max_, x, d)

def ts_min(x, d):
    return rolling_apply(min_, x, d)

def pearson(x, y, d):
    return rolling_apply(pearson, x, y, d)

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
def dynamic_sample_entropy_numba(x, m=1, r_ratio=1, use_std=True):
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
    if np.isinf(entropy):
        return 5
    return entropy


@numba.njit(fastmath=True, cache=True)
def simple_sign_entropy(arr):
    signs = np.sign(arr)
    pos_count = np.sum(signs == 1)
    neg_count = np.sum(signs == -1)
    zero_count = np.sum(signs == 0)
    total = len(arr)
    if total == 0:
        return 0.0
    p = np.array([pos_count, neg_count, zero_count], dtype=np.float64) / total
    log_p = np.zeros(3)
    for i in range(3):
        if p[i] > 0:
            log_p[i] = np.log(p[i])
    entropy = -np.sum(p * log_p)
    return entropy / np.log(3)

# LPCC

import scipy 
from scipy.fft import fft, ifft
import numpy as np

def create_symmetric_matrix(acf, order=11):
    """Computes arr symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which arr symmetric matrix is computed
    order : int
        Order

    Returns
    -------
    nd-array
        Symmetric Matrix
    """

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix

def lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients
    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have arr length >= n_coeff")

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = scipy.signal.correlate(signal, signal, "full")

    r = np.zeros(order + 1, "float32")
    # Assuring that works for all type of input lengths
    nx = np.min([order + 1, len(signal)])
    r[:nx] = acf[len(signal) - 1 : len(signal) + order]

    smatrix = create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order + 1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.0], lpc_coeffs)))

def lpcc(signal, n_coeff=12,i=0):
    """Computes linear prediction cepstral coefficients using scipy.fft"""
    if len(signal)<n_coeff:
        return np.nan
    
    lpc_coeffs = lpc(signal, n_coeff)
    if np.allclose(lpc_coeffs, 0): return tuple(np.zeros(n_coeff))
    
    power_spectrum = np.abs(fft(lpc_coeffs, n=2*n_coeff)) ** 2
    log_spectrum = np.log(np.maximum(power_spectrum, 1e-10))  # Avoid log(0)
    lpcc_coeff = ifft(log_spectrum)[:n_coeff]
    
    return tuple(np.abs(lpcc_coeff.real))[i]


import numpy as np

def reshape(arr, w):
    """
    Reshapes an array to match the target window size w.
    
    Parameters:
    - arr: Input array (list or numpy array)
    - w: Target window size (int)
    
    Returns:
    - Reshaped array with length equal to w
    """
    current_len = len(arr)
    
    if current_len == w:
        return np.array(arr)
    
    # Case 1: Array is smaller than window size - stretch by repeating elements
    elif current_len < w:
        # Calculate stretch factors
        stretch_factors = np.linspace(0, current_len-1, w)
        # Interpolate by selecting nearest elements
        stretched_arr = [arr[int(round(i))] for i in stretch_factors]
        return np.array(stretched_arr)
    
    # Case 2: Array is larger than window size - shrink by aggregating
    else:
        # Calculate how many elements to aggregate for each new element
        agg_size = current_len / w
        shrunk_arr = []
        
        for i in range(w):
            start = int(round(i * agg_size))
            end = int(round((i + 1) * agg_size))
            # Take mean of the aggregated elements
            shrunk_arr.append(np.mean(arr[start:end]))
        
        return np.array(shrunk_arr)