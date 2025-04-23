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

# LPCC

import scipy 
from scipy.fft import fft, ifft
import numpy as np

def create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
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
        raise ValueError("Input signal must have a length >= n_coeff")

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
    lpc_coeffs = lpc(signal, n_coeff)
    if np.allclose(lpc_coeffs, 0): return tuple(np.zeros(n_coeff))
    
    power_spectrum = np.abs(fft(lpc_coeffs, n=2*n_coeff)) ** 2
    log_spectrum = np.log(np.maximum(power_spectrum, 1e-10))  # Avoid log(0)
    lpcc_coeff = ifft(log_spectrum)[:n_coeff]
    
    return tuple(np.abs(lpcc_coeff.real))[i]