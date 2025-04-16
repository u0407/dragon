import seaborn as sns 
import numpy as np
import pandas as pd
from pysr import PySRRegressor,TemplateExpressionSpec
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import warnings
from draw import *
import os
import sympy 


os.makedirs('./GA_Shane/temp',exist_ok=True)
os.chdir('./GA_Shane/')

version = 4 # from 1 to 3
freq = 30 
length = 180000
csv_path = "../RB99_1m.csv"
niterations = 10
population_size = 100
start_i = 60000 
part = '20250412_215350_x8oBIk_copy'
warm_start = 0

# 1. Fitness function for PySR
"""
Loss
v1: Closeness between Normal distribution and the generated y_pred
v2. Resample by the breakpoints of y_pred, then apply Shapiro-Wilk test to the diff of the resampled Close
v3. GMM fitness mixed with normality 
"""
with open(f"./fitness_v{version}.jl", "rb") as f:
    elementwise_loss = f.read().decode("utf-8")

model = PySRRegressor(
    niterations=niterations,
    population_size=population_size,
    # Operators
    binary_operators=["+", "-", "*", "/",'max','min'],
    unary_operators=["sign",],
    # Model complexity control
    maxsize=20,
    parsimony=0.1,

    # Optimization settings    
    # constraints={"constant_optimization": False},

    # Runtime settings
    verbosity=2,
    procs=16,  # Run 8 cores
    deterministic=False,
    random_state=2,
    parallelism="serial",
    # Dimensionality reduction
    dimensionless_constants_only = True, # constants are dimensionless
    loss_function=elementwise_loss,
    dimensional_constraint_penalty=10000,
)


# 4. Fit the model

"""
Assign units for each column in X, to ensure the generated equation has no units.
Then it will return no unit operators.


    - output/kde_original.png
    - output/kde_new_axis.png
    - output/new_axis.csv

"""
X_unit_dict = {
    'low': 'm',
    'open_interest': 'kg',
    'high': 'm',
    'total_turnover': 'm * kg',
    'volume': 'kg',
    'A0': 'm',
    'open': 'm',
    'vol2oi': '',
    'avg_price': 'm',
    'prev_low': 'm',
    'prev_open_interest': 'kg',
    'prev_high': 'm',
    'prev_total_turnover': 'kg*m',
    'prev_volume': 'm',  # amt is unit of kg*m, assuming this is prev_volume
    'prev_close': 'm',
    'prev_open': 'm',
    'prev_avg_price': 'm',
    'bar_spread':'',
    'bar_box':'',
    'bar_ret':'',
    'bar_jump':'',
    'bot':'m',
    'top':'m',
    'prev_bot':'m',
    'prev_top':'m',
    'prev_top':'m',
    'prev_top':'m',
    'mid3':'m',
    'prev_mid3':'m',
    'vwap_60':'m',
    'prev_vwap_60':'m',
    'std_ret_close':'',
    'std_ret_mid3':'',
    'std_ret_close_short':'',
    'std_ret_mid3_short':'',
    'std_ret_high_low_short':'',
    'std_ret_top_bot_short':'',
    'entropy_close_short':'',
    'entropy_mid_short':'',
    'entropy_vol2oi_short':''
}

print("Init data ... ")
# This will contain the original data, but convert into log price and diffs. 
df = load_df_version3(csv_path, 
             start_i=start_i, 
             length=length,
             cached=True)
# df = load_df_version3(csv_path, 
#              start_i=start_i,
#              end_i=-1, 
#              length=-1)

X = df.drop(columns=['eob'])
y = df['A0'].values
X_units = [ X_unit_dict[x] for x in X.columns]

if warm_start == True:

    model = PySRRegressor.from_file(run_directory=rf'E:\dragon\GA_Shane\outputs\{part}')
    model.warm_start = True 
    print('warm start')

model.fit(X, y, X_units = X_units, y_units='')



















model = PySRRegressor(
    niterations=niterations,
    population_size=population_size,
    # Operators
    binary_operators=["+", "-", "*", "/",'max','min'],
    unary_operators=["sign","exp","log","cos"],

    extra_sympy_mappings={ 
                          },
    # Model complexity control
    maxsize=20,
    parsimony=0.1,

    # Optimization settings    
    # constraints={"constant_optimization": False},

    # Runtime settings
    verbosity=2,
    procs=16,  # Run 8 cores
    deterministic=False,
    random_state=2,
    parallelism="serial",
    # Dimensionality reduction
    dimensionless_constants_only = True, # constants are dimensionless
    loss_function=elementwise_loss,
    dimensional_constraint_penalty=10000,
)


# 4. Fit the model

"""
Assign units for each column in X, to ensure the generated equation has no units.
Then it will return no unit operators.


    - output/kde_original.png
    - output/kde_new_axis.png
    - output/new_axis.csv

"""
X_unit_dict = {
    'low': 'm',
    'open_interest': 'kg',
    'high': 'm',
    'total_turnover': 'm * kg',
    'volume': 'kg',
    'A0': 'm',
    'open': 'm',
    'vol2oi': '',
    'avg_price': 'm',
    'prev_low': 'm',
    'prev_open_interest': 'kg',
    'prev_high': 'm',
    'prev_total_turnover': 'kg*m',
    'prev_volume': 'm',  # amt is unit of kg*m, assuming this is prev_volume
    'prev_close': 'm',
    'prev_open': 'm',
    'prev_avg_price': 'm',
    'bar_spread':'',
    'bar_box':'',
    'bar_ret':'',
    'bar_jump':'',
    'bot':'m',
    'top':'m',
    'prev_bot':'m',
    'prev_top':'m',
    'prev_top':'m',
    'prev_top':'m',
    'mid3':'m',
    'prev_mid3':'m',
    'vwap_60':'m',
    'prev_vwap_60':'m',
    'std_ret_close':'',
    'std_ret_mid3':'',
    'std_ret_close_short':'',
    'std_ret_mid3_short':'',
    'std_ret_high_low_short':'',
    'std_ret_top_bot_short':'',
}

print("Init data ... ")
df = load_df_version3(csv_path, 
             start_i=start_i, 
             length=length)

X = df.drop(columns=['eob'])
y = df['A0'].values
X_units = [ X_unit_dict[x] for x in X.columns]

if warm_start == True:

    model = PySRRegressor.from_file(run_directory=rf'E:\dragon\GA_Shane\outputs\{part}')
    model.warm_start = True 
    print('warm start')

model.fit(X, y, X_units = X_units, y_units='')
