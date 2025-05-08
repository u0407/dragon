import re 
import time
import polars as pl
import numpy as np
import pandas as pd
from functools import wraps

def timed_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[Timing] {func.__name__} executed in {elapsed:.3f} seconds")
        return result
    return wrapper

def fn(X,expression):
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    reserved = {'max', 'min', 'sign','sin','cos','log','exp', 'tanh', 'abs', 'relu'}
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

    def replacer(match):
        var_name = match.group(1)
        return var_name if var_name in reserved else f"X['{var_name}']"
    expr = pattern.sub(replacer, expression)

    env = {
        'X': X,                       # DataFrame containing features
        'max': np.maximum,            # Element-wise maximum
        'min': np.minimum,            # Element-wise minimum
        'sign': np.sign,              # Element-wise sign function
        'np': np,                     # Include numpy for advanced operations
        'log':np.log,
        'exp':np.exp,
        'sin':np.sin,
        'cos':np.cos,
        'tanh': np.tanh,              # Hyperbolic tangent function
        'abs': np.abs,                # Absolute value function
        'relu': lambda x: np.maximum(0, x),  # Rectified Linear Unit function
    }
    result = eval(expr, env)
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return result.values
    else:
        return np.array(result)  # Handle scalar results
    
def split(arr,freq=30,thred=0, return_g=False):
    if not return_g:
        n = len(arr)
        arr = np.abs(arr)
        if thred ==0:
            thred = np.nanmean(arr[:180000]) * freq 
        idx_list = []
        cumsum = 0.0
        for i in range(n):
            cumsum += arr[i]
            if cumsum > thred or i == n:
                idx_list.append(i)  # 记录索引
                cumsum = 0.0
        return idx_list
    else:
        n = len(arr)
        arr = np.abs(arr)
        if thred == 0 :
            thred = np.nanmean(arr[:180000]) * freq 
        print("thread equas: ", thred)
        g_lst = []
        g = 0
        cumsum = 0.0
        for i in range(n):
            cumsum += arr[i]
            g_lst.append(g)
            if cumsum > thred or i == n:
                    cumsum = 0.0
                    g += 1 
        return g_lst

