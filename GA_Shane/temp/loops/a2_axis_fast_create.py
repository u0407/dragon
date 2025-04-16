import sympy
import warnings 
warnings.filterwarnings('ignore')  
import sys 
sys.path.append("./GA_Shane") 
import pandas as pd
import polars as pl 
import numpy as np 
from draw import * 
import os
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import random

code = 'RB'
suffix = f'{code}99_1m'
csv_path =   f"./{suffix}.csv"
part = '20250411_handmade'
model_pth =  rf'E:\dragon\GA_Shane\outputs\{part}\hall_of_fame.csv'
project_pth = rf'E:\dragon\GA_Shane\outputs\{part}'
os.makedirs(project_pth,exist_ok=True)

freq = 45 
size = None

df = pl.read_csv(csv_path)
df = df.drop(['order_book_id','trading_date'])
df = df.rename({'datetime':'eob'})

def transform(df):
    df = df.sort('eob')
    df = df.with_columns([pl.arange(0, df.height).alias('index')])
    df = df.select([
        pl.col('index'),
        pl.col('eob'),
        pl.exclude(['index','eob'])
    ])

    return df 

df = transform(df)




freq = 45 
# 
cols = [ col for col in df.columns  if col not in ['index','eob']]

df = df.with_columns([
    (pl.col(col)+1e-6).alias(col) for col in ['volume','open_interest','total_turnover']
])

df = df.with_columns([
    pl.col(col).log().alias(col) for col in cols
])

df = df.with_columns([
    pl.max_horizontal(pl.col('open'), pl.col('close')).alias('top'),
    pl.min_horizontal(pl.col('open'), pl.col('close')).alias('bot'),
])

cols.extend(['top','bot'])

df = df.with_columns([
    pl.col(col).shift().alias(f"prev_{col}") for col in cols
])

cols = [ col for col in df.columns  if col not in ['index','eob']]

print(cols)

today = ['open', 'high', 'low', 'close', 'top', 'bot', 'total_turnover', 'volume', 'open_interest']
yesterday = [f"prev_{col}" for col in today]
# Run the random sample experiment 1000 times and collect the results.
results = []
best_kurt = float("inf")
for _ in tqdm(range(1000)):
    # Randomly choose two distinct columns from today's list for var_a and var_c
    var_a, var_c = 'low', random.choice(today+yesterday)
    var_b, var_d = random.choice(yesterday+today), random.choice(yesterday+today)
    var_e = random.choice(today+yesterday)
    var_f = random.choice(today+yesterday)


    combined_key = f"{var_a}_{var_b}_{var_c}_{var_d}_{var_e}"
    
    # Create a new factor column based on the random selection
    df_temp = df.with_columns([
        (
            (pl.col(var_a) - pl.col(var_b)) *
            (pl.col(var_c) - pl.col(var_d)) *
            pl.col(var_e) / pl.col(var_f)
        ).alias('factor')
    ])

    _df = df_temp[['eob', 'close', 'factor']]
    _df = _df.with_columns(pl.col('factor').abs().alias('factor'))
    _df = _df.with_columns([
        pl.col('factor').cum_sum().alias('pos_sum'),
    ])

    size_val = _df['factor'].to_pandas().iloc[:180000].mean() * freq
    size_val = auto_round(size_val)

    _df = _df.with_columns([
        (pl.col('pos_sum') // size_val).cast(pl.Int64).alias('group')
    ])

    if isinstance(_df, pd.DataFrame):
        _df = pl.DataFrame(_df)

    _df = _df.group_by('group').agg([
        pl.col('eob').last().alias('eob'),
        pl.col('close').exp().last().alias('close'),
    ])

    _df = transform(_df)

    closes = _df['close'].to_numpy()
    if len(closes) < 2:
        # Not enough data to calculate returns
        z_skewness = float("nan")
        z_kurt = float("nan")
    else:
        rets = closes[1:] - closes[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 1:
            z_rets = (rets - np.mean(rets)) / np.std(rets)
            z_skewness = skew(z_rets)
            z_kurt = kurtosis(z_rets)
        else:
            z_skewness = float("nan")
            z_kurt = float("nan")
    
    # Print if a new smallest kurt is found.
    if z_kurt < best_kurt:
        best_kurt = z_kurt
        print(f"New smallest kurt found: {best_kurt:.4f} with key {combined_key}")
    
    results.append({
        "key": combined_key,
        "kurt": z_kurt,
        "skewness": z_skewness,
        "len": df.height//_df.height
    })

result_df = pd.DataFrame(results).sort_values('kurt')
result_df.to_csv("random_sample_experiment.csv", index=False)
print("Experiment completed and results written to random_sample_experiment.csv")
