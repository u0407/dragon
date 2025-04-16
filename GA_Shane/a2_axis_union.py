import glob 
import sympy
import warnings 
warnings.filterwarnings('ignore')   
import pandas as pd
import polars as pl 
import numpy as np 
from draw import * 
import os
from tqdm import tqdm
from scipy.stats import skew, kurtosis

code = 'RB'
suffix = f'{code}99_1m'
csv_path =   f"./{suffix}.csv"

parts = ['20250411_handmade','20250411_vol2oi']

merged_factors = []

for part in parts:
    project_pth = rf'E:/dragon/GA_Shane/outputs/{part}'
    factor_files = glob.glob(project_pth + '/*factor*.csv')
    
    for factor_file in factor_files:
        df = pd.read_csv(factor_file).set_index('eob')
        merged_factors.append(df)

# Combine all factor DataFrames into one
if merged_factors:
    merged_df = pd.concat(merged_factors, axis=1)
    print(merged_df.head())
else:
    print("No factor files found.")


freq = 45
size_dict = {}
for f in df.columns:
    size = df[f].iloc[:180000].mean() * freq
    size = auto_round(size)
    size_dict.update({f:size})



group = []
current_group = 0
cumsum_dict = {f: 0 for f in size_dict.keys()}

exceed_cols = 0

for _, row in tqdm(df.iterrows()):

    for f in size_dict.keys():
        cumsum_dict[f] += row[f]
        if cumsum_dict[f] >= size_dict[f]: exceed_cols += 1

    if exceed_cols == len(df.columns) :
        current_group += 1
        cumsum_dict = {f: 0 for f in size_dict.keys()}
    
    group.append(current_group)

df['group'] = group

print(df['group'])

