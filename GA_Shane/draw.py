import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def kde_plot(series,save_to_path=None):

    # assert isinstance(series, pd.Series)
    if not isinstance(series, pd.Series):
        raise ValueError("Input should be a pandas Series.")

    returns_1 = series.diff().dropna()
    returns_2 = series.diff(periods=2).dropna()
    returns_3 = series.diff(periods=3).dropna()
    returns_4 = series.diff(periods=4).dropna()
    returns_5 = series.diff(periods=5).dropna()

    standard_1 = (returns_1 - returns_1.mean()) / returns_1.std()
    standard_2 = (returns_2 - returns_2.mean()) / returns_2.std()
    standard_3 = (returns_3 - returns_3.mean()) / returns_3.std()
    standard_4 = (returns_4 - returns_4.mean()) / returns_4.std()
    standard_5 = (returns_5 - returns_5.mean()) / returns_5.std()


    sns.kdeplot(standard_1, label=f"1", color='darkred')
    sns.kdeplot(standard_2, label="2", color='green')
    sns.kdeplot(standard_3, label="3", color='blue')
    sns.kdeplot(standard_4, label="4", color='orange')
    sns.kdeplot(standard_5, label="5", color='magenta')

    sns.kdeplot(np.random.normal(size=1000000), label="Normal", color='black', linestyle="--")

    # skew and kurt and mean of the series and plot in the title
    skew = standard_1.skew()
    kurt = standard_1.kurt()
    mean = standard_1.mean()

    print(f"Skew: {skew:.2f}, Kurt: {kurt:.2f}, Mean: {mean:.2f}")
    plt.xticks(range(-5, 6))
    plt.legend(loc=8, ncol=5)
    plt.title(f"Skew: {skew:.2f}, Kurt: {kurt:.2f}, Mean: {mean:.2f}", fontsize=15, fontweight="bold", fontname="Times New Roman")
    plt.xlim(-5, 5)
    plt.grid(1)
    if save_to_path:
        plt.savefig(save_to_path)
    else:
        plt.show()
    plt.close()

import math

def auto_round(x):
    min_diff = float('inf')
    closest_value = None
    for i in range(-10, 11):
        for coeff in [1, 3,5,7]:
            candidate = coeff * ( 10 ** i)
            diff = abs(candidate - x)
            if diff < min_diff:
                min_diff = diff
                closest_value = candidate
    return closest_value

import re 
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

def bar_split(arr, freq=30):
    """
    Resample Logic:
    1. Calculate the threshold using the first 5000 rows of data
    2. Accumulate the absolute diff of the factor. 
    3. If the accumulated value is greater than the threshold, record the index.
    4. Reset the accumulated value to 0.

    Expected:
    1. The length of the resampled data should be around 60000 rows.
    2. The resampled bars has Returns to  be in a normal distribution.
    """

    n = len(arr)
    arr = np.abs(arr)
    thred = np.nanmean(arr[:180000]) * freq 
    print("thread: ", thred)
    # 构造 idx_list
    idx_list = []
    cumsum = 0.0
    group_cnt = 0
    for i in range(n):
        cumsum += arr[i]
        idx_list.append(group_cnt) 
        if cumsum >= thred:
            group_cnt += 1
            idx_list.append(i)  # 记录索引
            cumsum = 0
    return idx_list



def fn_from_norm_factor(arr):
    arr = arr.dropna()

    if len(arr) < 5:
        raise Exception("Not enough data: requires at least 60000 rows.")
    # 使用前 5000 行数据计算 thred
    idx_list = []
    for i in arr.index:
        if arr[i] > 0.6 or arr[i] < -0.6: 
            idx_list.append(i)  # 记录索引
    return idx_list





# def load_df(csv_path, start_i=50000, length=-1,end_i = 120000):
#     df = pd.read_csv(csv_path)
#     if end_i>0:
#         df = df.iloc[start_i:-end_i]
#     else:
#         df = df.iloc[start_i:]

    # df.rename(columns = {'datetime':'eob'},inplace = True)
    # df['eob'] = pd.to_datetime(df['eob'])
    # df.drop(columns = ['order_book_id','trading_date'],inplace = True)
#     df['vol2oi'] = df['volume'] / df['open_interest']
#     df['avg_price'] = df['total_turnover'] / df['volume']
#     for col in df.columns:
#         if col != 'eob' and col != 'vol2oi':
#             df[col] = np.log(df[col])
#             df[f"prev_{col}"] = df[col].shift()

#     df['bar_spread'] = df['high'] - df['low']
#     df['bar_box'] = df['close'] - df['open']
#     df['bar_ret'] = df['close'] - df['prev_close']
#     df['bar_jump'] = df['open'] - df['prev_close']

#     df.replace([np.inf, -np.inf], np.nan,inplace = True)
#     df.fillna(0,inplace = True)
#     df.rename(columns = {'close':'A0'},inplace = True)
#     # df.drop(columns = ['volume','total_turnover','open_interest'],inplace = True)

#     # start from random point of the data
#     if length>0:
#         idx = np.random.randint(0, len(df) - length )
#         df = df.iloc[idx:idx+length]

#     df = df.reset_index(drop=True)

#     return df 


def load_df_version2(csv_path, start_i=50000, length=-1,end_i = 120000):
    df = pd.read_csv(csv_path)
    if end_i>0:
        df = df.iloc[start_i:-end_i]
    else:
        df = df.iloc[start_i:]

    df.rename(columns = {'datetime':'eob'},inplace = True)
    df['eob'] = pd.to_datetime(df['eob'])
    df.drop(columns = ['order_book_id','trading_date'],inplace = True)
    df['vol2oi'] = df['volume'] / df['open_interest']
    df['avg_price'] = df['total_turnover'] / df['volume']
    df['top'] = np.maximum(df['open'], df['close'])
    df['bot'] = np.minimum(df['open'], df['close'])
           
    for col in df.columns:
        if col != 'eob' and col != 'vol2oi':
            df[col] = np.log(df[col])
            df[f"prev_{col}"] = df[col].shift()

    
    df.replace([np.inf, -np.inf], np.nan,inplace = True)
    df.fillna(0,inplace = True)
    df.rename(columns = {'close':'A0'},inplace = True)

    if length>0:
        idx = np.random.randint(0, len(df) - length )
        df = df.iloc[idx:idx+length]

    df = df.reset_index(drop=True)

    return df 




def load_df_version2(csv_path, start_i=50000, length=-1,end_i = 120000):
    df = pd.read_csv(csv_path)
    if end_i>0:
        df = df.iloc[start_i:-end_i]
    else:
        df = df.iloc[start_i:]

    df.rename(columns = {'datetime':'eob'},inplace = True)
    df['eob'] = pd.to_datetime(df['eob'])
    df.drop(columns = ['order_book_id','trading_date'],inplace = True)
    df['vol2oi'] = df['volume'] / df['open_interest']
    df['top'] = np.maximum(df['open'], df['close'])
    df['bot'] = np.minimum(df['open'], df['close'])
           
    for col in df.columns:
        if col != 'eob' and col != 'vol2oi':
            df[col] = np.log(df[col])
            df[f"prev_{col}"] = df[col].shift()
    


    
    df.replace([np.inf, -np.inf], np.nan,inplace = True)
    df.fillna(0,inplace = True)
    df.rename(columns = {'close':'A0'},inplace = True)

    if length>0:
        idx = np.random.randint(0, len(df) - length )
        df = df.iloc[idx:idx+length]

    df = df.reset_index(drop=True)

    return df 



def load_df_version3(csv_path, start_i=50000, length=-1,end_i = 120000,cached=False):
    if not cached:
        df = pd.read_csv(csv_path)
        
        df.rename(columns = {'datetime':'eob'},inplace = True)
        df['eob'] = pd.to_datetime(df['eob'])
        df.drop(columns = ['order_book_id','trading_date'],inplace = True)
        df['vol2oi'] = df['volume'] / df['open_interest']
        df['top'] = np.maximum(df['open'], df['close'])
        df['bot'] = np.minimum(df['open'], df['close'])
        df['mid3'] = (df['high'] + df['low']+df['close']) / 3
        df['vwap_60'] = (df['mid3']*df['volume']).rolling(60).sum() / df['volume'].rolling(60).sum()
            
        for col in df.columns:
            if col != 'eob' and col != 'vol2oi':
                df[col] = np.log(df[col])
                df[f"prev_{col}"] = df[col].shift()
        
        df['std_ret_close_short'] = (df['close'] - df['prev_close']).rolling(60).std() 
        df['std_ret_mid3_short'] = (df['close'] - df['mid3']).rolling(60).std() 
        df['std_ret_high_low_short'] = (df['high'] - df['low']).rolling(60).std() 
        df['std_ret_top_bot_short'] = (df['top'] - df['bot']).rolling(60).std() 

        # window_size=20
        # df['entropy_close_short'] = [np.nan] * (window_size - 1) + calculate_entropy_on_windows(df['close'], window_size, m=2, r_ratio=0.2, use_std=True)   
        # df['entropy_mid_short'] = [np.nan] * (window_size - 1) + calculate_entropy_on_windows(df['mid3'], window_size, m=2, r_ratio=0.2, use_std=True)   
        # df['entropy_vol2oi_short'] = [np.nan] * (window_size - 1) + calculate_entropy_on_windows(df['vol2oi'], window_size, m=2, r_ratio=0.2, use_std=True)   

        df.replace([np.inf, -np.inf], np.nan,inplace = True)
        df.fillna(0,inplace = True)
        df.rename(columns = {'close':'A0'},inplace = True)
    else:
        df = pd.read_csv(r'E:\dragon\GA_Shane\df.csv',index_col=[0])

    if end_i>0:
        df = df.iloc[start_i:-end_i]
    
    if length>0:
        idx = np.random.randint(0, len(df) - length )
        df = df.iloc[idx:idx+length]

    df = df.reset_index(drop=True)

    if not cached:
        df.to_csv('./df.csv')
    return df 



import pandas as pd

def sample_entropy(x, emb_dim, tolerance):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x_array = np.array(x)
        n = len(x_array) - m + 1
        if n <= 1:
            return np.nan
        
        new_x = [x_array[i:i + m] for i in range(n)]
        C = [sum([1 for j in range(n) if i != j and _maxdist(new_x[i], new_x[j]) <= tolerance])
             for i in range(n)]
        
        denominator = n * (n - 1)
        if denominator == 0:
            return np.nan
        
        result = sum(C) / denominator
        if result == 0:
            return np.nan
        return result

    if len(x) < emb_dim + 2:
        return np.nan
    
    phi_emb_dim = _phi(emb_dim)
    phi_emb_dim_plus_one = _phi(emb_dim + 1)
    
    if phi_emb_dim == 0 or phi_emb_dim_plus_one == 0:
        return np.nan
    
    return -np.log(phi_emb_dim_plus_one / phi_emb_dim)



### 动态计算样本熵，使用标准差和极差两种方式来计算容差阈值

def dynamic_sample_entropy(x, m, r_ratio, use_std):

    if use_std:
        std_x = np.std(x)
        if std_x < 1e-6:
            return np.nan     ### 标准差过小，无法使用，结果为NaN
        r = r_ratio * std_x
    else:
        range_x = np.max(x) - np.min(x)
        if range_x < 1e-6:
            return np.nan      ### 极差过小，无法使用，结果为NaN
        r = r_ratio * range_x
    
    return sample_entropy(x, emb_dim=m, tolerance=r)



### 确保窗口只包含历史数据
from tqdm import tqdm 
def calculate_entropy_on_windows(data, window_size=20, m=2, r_ratio=0.2, use_std=True):

    entropy_values = []
    for i in tqdm(range(window_size - 1, len(data))):
        window = data[max(0, i - window_size + 1):i + 1]
        entropy = dynamic_sample_entropy(window, m, r_ratio, use_std)
        entropy_values.append(entropy)
        
    return entropy_values


from functions import *
import polars as pl 

def slice_df(df, start_i=50000, length=-1, end_i=120000):

    if isinstance(df,pl.DataFrame):
        df = df.to_pandas()

    if end_i>0:
        df = df.iloc[start_i:-end_i]

    if length>0:
        df = df.iloc[start_i:start_i+length]

    df = df.reset_index(drop=True)
    print('start from {} to {}'.format(df['eob'].iloc[0], df['eob'].iloc[-1]))
    return df 


def transform(df):
    """
    This read from raw excel fetched from data api.
    """
    if isinstance(df,pd.DataFrame):
        df = pl.DataFrame(df)
        
    if 'eob' not in df.columns:
        df = df.rename({'datetime':'eob'})
        df = df.drop(['order_book_id','trading_date'])
    df = df.sort('eob')
    df = df.with_columns([pl.arange(0, df.height).alias('index')])
    df = df.select([
        pl.col('index'),
        pl.col('eob'),
        pl.exclude(['index','eob'])
    ])
    return df 
