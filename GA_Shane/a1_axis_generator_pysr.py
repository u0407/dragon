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
import pickle
import pysr 
pysr.install()

os.makedirs('./GA_Shane/temp',exist_ok=True)
os.chdir('./GA_Shane/')

"""
7 - Jarque Bera Test
"""
version = 7 
freq = 20
L = 7
use_minute_bar_of_freq_n = 3
print(f'version {version} ; freq {freq}; L {L}; use {use_minute_bar_of_freq_n} min bar')

length = 300000
length = length // use_minute_bar_of_freq_n
code = 'RB'
niterations = 20
population_size = 500
start_i = 10000 
start_i = start_i // use_minute_bar_of_freq_n



if use_minute_bar_of_freq_n == 1:
    csv_cache_path = f"../{code}99_1m.csv.cache.exp.csv"
else:
    csv_cache_path = f"../{code}99_1m.csv.{use_minute_bar_of_freq_n}m.csv.cache.exp.csv"

# 1. Fitness function for PySR
"""
Loss
v1: Closeness between Normal distribution and the generated y_pred
v2. Resample by the breakpoints of y_pred, then apply Shapiro-Wilk test to the diff of the resampled Close
v3. GMM fitness mixed with normality 
"""
# with open(f"./fitness_v{version}.jl", "rb") as f:
#     elementwise_loss = f.read().decode("utf-8")
    
with open(f"./fitness_v{version}.jl", "rb") as f:
    elementwise_loss = f.read().decode("utf-8")
elementwise_loss = elementwise_loss.replace("freq = 60", f"freq = {freq}") \
                               .replace("n_diff_of_gmm = 7", f"n_diff_of_gmm = {L}")

model = PySRRegressor(
    niterations=niterations,
    population_size=population_size,

    # Operators
    binary_operators=["+", "-", "*", "/",],
    unary_operators=["sign","exp","log","sin",'cos','abs', 'relu','tanh'],

    # Model complexity control
    maxsize=20,
    parsimony=0.1,

    # Runtime settings
    verbosity=2,
    procs=16,  # Run 8 cores
    deterministic=False,
    random_state=0,
    parallelism="multithreading",
    # Dimensionality reduction
    dimensionless_constants_only = True, # constants are dimensionless
    loss_function=elementwise_loss,
    dimensional_constraint_penalty=10000,
    # Optimization settings    
    # constraints={"constant_optimization": False},
)


# 4. Fit the model

"""
Assign units for each column in X, to ensure the generated equation has no units.
Then it will return no unit operators.


    - output/kde_original.png
    - output/kde_new_axis.png
    - output/new_axis.csv

"""
print("Init data ... ")

df = pl.read_csv(csv_cache_path)

with open('/home/dragon/RB99_1m.csv.cache.pkl', 'rb') as f:
    X_units_dict = pickle.load(f)

df = slice_df(df, start_i=start_i*2, length = length, end_i=(120000//freq))

X_units = [ X_units_dict[x] for x in df.columns if x in X_units_dict.keys()]
X = df.dropna()[X_units_dict.keys()]
y = np.log(df.dropna()['A0']).values

print("X shape \n",X.shape)
print("X sample \n",X.head())
print("X columns: \n",X.columns)


model.fit(X, y, X_units = X_units, y_units='')
