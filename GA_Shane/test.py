# File: test.py

import numpy as np
import time
import numba # Needed for Numba types like NaN potentially

# --- Import Original Python Code ---
# (Copy the original sample_entropy, _maxdist, _phi, dynamic_sample_entropy here)
# Note: Ensure original code uses numpy for consistency where possible

def _maxdist_py(x_i, x_j):
    # Using numpy for potential minor speedup even in pure python version
    return np.max(np.abs(np.array(x_i) - np.array(x_j)))
    # Original list comprehension version:
    # return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

def _phi_py(x, m, tolerance):
    x_array = np.array(x) # Convert to NumPy array
    n = len(x_array) - m + 1
    if n <= 1:
        return np.nan

    # Create templates using slicing
    new_x = [x_array[i:i + m] for i in range(n)]

    # Calculate matches using list comprehension and the Python _maxdist_py
    C = [sum([1 for j in range(n) if i != j and _maxdist_py(new_x[i], new_x[j]) <= tolerance])
         for i in range(n)]

    denominator = n * (n - 1)
    if denominator == 0:
        return np.nan # Should not happen if n > 1

    # Calculate the sum of counts
    total_matches = sum(C)

    # Calculate the probability
    result = total_matches / denominator
    if result == 0: # Handle case where no matches were found
         return np.nan # Return NaN if probability is 0, to match Numba version's potential NaN outcome before log
         # Alternative: return 0.0, but this would cause log(0) later

    return result # Return the probability P(m)

def sample_entropy_py(x, emb_dim, tolerance):
    """Original Pure Python Sample Entropy calculation."""
    # Ensure input is numpy array
    x = np.asarray(x)
    if x.ndim != 1:
         raise ValueError("Input array must be 1D")

    # Check minimum length requirement
    if len(x) < emb_dim + 2:
         # print(f"PY: Window length {len(x)} too short for m={emb_dim}.")
        return np.nan

    phi_emb_dim = _phi_py(x, emb_dim, tolerance)
    phi_emb_dim_plus_one = _phi_py(x, emb_dim + 1, tolerance)

    # print(f"PY phi({emb_dim}) = {phi_emb_dim}, phi({emb_dim}+1) = {phi_emb_dim_plus_one}")

    # Check for NaN or zero results from _phi_py
    # Important: Check for NaN specifically using np.isnan
    if np.isnan(phi_emb_dim) or np.isnan(phi_emb_dim_plus_one):
        return np.nan
    # Avoid division by zero or log(0)
    if phi_emb_dim <= 1e-10 or phi_emb_dim_plus_one <= 1e-10: # Using epsilon similar to Numba version
        return np.nan

    return -np.log(phi_emb_dim_plus_one / phi_emb_dim)

def dynamic_sample_entropy_py(x, m, r_ratio, use_std):
    """Original Pure Python Dynamic Sample Entropy calculation."""
    x = np.asarray(x) # Ensure numpy array
    if x.ndim != 1: raise ValueError("Input must be 1D")

    if use_std:
        std_x = np.std(x)
        if std_x < 1e-9: # Match epsilon with Numba version
            # print(f"PY: Std dev too small ({std_x}).")
            return np.nan
        r = r_ratio * std_x
    else:
        min_x = np.min(x)
        max_x = np.max(x)
        range_x = max_x - min_x
        if range_x < 1e-9: # Match epsilon with Numba version
            # print(f"PY: Range too small ({range_x}).")
            return np.nan
        r = r_ratio * range_x

    return sample_entropy_py(x, emb_dim=m, tolerance=r)

# --- Import Numba Implementation ---
from functions import dynamic_sample_entropy_numba, rolling_apply

# --- Test Setup ---
print("Setting up test data...")
# Generate synthetic data
data_size = 2000
np.random.seed(42) # for reproducibility
data = np.random.rand(data_size) * 10 + np.sin(np.linspace(0, 10 * np.pi, data_size)) * 2

# Parameters
window_size = 20
m = 2
r_ratio = 0.2
use_std = True # Use standard deviation for tolerance

args_tuple = (m, r_ratio, use_std) # Arguments for the entropy function

print(f"Data Size: {data_size}")
print(f"Window Size: {window_size}")
print(f"Embedding Dim (m): {m}")
print(f"Tolerance Ratio (r_ratio): {r_ratio}")
print(f"Use Standard Deviation: {use_std}")
print("-" * 30)

# # --- Run Original Python Version ---
print("Running original Python version...")
start_time_py = time.time()
results_py = rolling_apply(dynamic_sample_entropy_py, data, window_size,  m, r_ratio, use_std)
end_time_py = time.time()
time_py = end_time_py - start_time_py
print(f"Python version took: {time_py:.4f} seconds")
# print("First 5 results (Python):", results_py[:5])
print("-" * 30)

# --- Run Numba Version ---
# Warm-up Numba compilation (optional but good practice for fair timing)
print("Warming up Numba (compilation)...")

dynamic_sample_entropy_numba(data[:window_size], m, r_ratio, use_std)

_ = rolling_apply(dynamic_sample_entropy_numba, data, window_size, m, r_ratio, use_std)
print("Numba warmed up.")

print("Running Numba version...")
start_time_nb = time.time()
# Use the dedicated rolling wrapper or call rolling_func directly
# results_nb = sample_entropy_numba_rolling(data, window_size, m, r_ratio, use_std)
results_nb = rolling_apply(dynamic_sample_entropy_numba, data, window_size, m, r_ratio, use_std)
end_time_nb = time.time()
time_nb = end_time_nb - start_time_nb
print(f"Numba version took: {time_nb:.4f} seconds")
# print("First 5 results (Numba):", results_nb[:5])
print("-" * 30)

# --- Comparison ---
print("Comparing results...")
# Check if lengths match (they should if no errors occurred)
if len(results_py) != len(results_nb):
    print(f"ERROR: Result lengths differ! Python: {len(results_py)}, Numba: {len(results_nb)}")
else:
    # Compare using np.allclose, allowing for small float differences and treating NaNs as equal
    tolerance_comparison = 1e-7 # Tolerance for comparing float results
    are_close = np.allclose(results_py, results_nb, rtol=tolerance_comparison, atol=tolerance_comparison, equal_nan=True)
    print(f"Results are close (within {tolerance_comparison}, NaNs equal): {are_close}")

    if not are_close:
        # Find where they differ
        diff_indices = np.where(~np.isclose(results_py, results_nb, rtol=tolerance_comparison, atol=tolerance_comparison, equal_nan=True))[0]
        print(f"Differences found at {len(diff_indices)} indices.")
        if len(diff_indices) > 0:
            print(f"First few differing indices: {diff_indices[:10]}")
            for i in diff_indices[:5]: # Print first 5 differences
                 print(f"  Index {i}: Python={results_py[i]:.6f}, Numba={results_nb[i]:.6f}, Diff={abs(results_py[i]-results_nb[i]):.2e}")


# --- Speed Improvement ---
if time_nb > 0:
    speedup = time_py / time_nb
    print(f"\nSpeedup factor (Python Time / Numba Time): {speedup:.2f}x")
else:
    print("\nNumba version was too fast to measure reliably.")

print("=" * 30)
print("Test finished.")


print(len(data))
print(len(results_py))
print(len(results_nb))