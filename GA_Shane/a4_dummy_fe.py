import os 
import numpy as np
import pandas as pd
import polars as pl 
import warnings 

import glob
import tsfel
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

from functions import *

warnings.filterwarnings('ignore')

code = 'RB'
roll = 5
label = 30
test_start = '2023-01-01'
fix = 513

part = '20250411_221021_UG9qRH'
dir = f'E:/dragon/GA_Shane/outputs/{part}'

window_size = 10 
cfg = tsfel.get_features_by_domain(['statistical'])

""" 
Data 
"""
file_name_1 = glob.glob(os.path.join(dir, '*_output_axis.csv'))[0]
df = pd.read_csv(file_name_1).sort_values('eob')

file_name_2 = glob.glob(os.path.join(dir, f'*_output_axis_Label_{str(fix)}.csv'))[0]
df_label = pd.read_csv(file_name_2)


df = df.merge(df_label[['datelist','state']], left_on='eob',right_on='datelist',how='left')
df_label = df[['eob','state']]
df.drop(columns=['datelist'],inplace=True)

""" 
Feature Generation
"""
features = ['open','high','low','close','volume']
X = df.copy(deep=True)[['state']+features]
X[features]=X[features].shift(1) # 因为是需要预测当期的状态，所以需要将X shift1
X = X.dropna()
print(X.head())
print("Original X.shape: ", X.shape)

price_features = ['open','high','low','close']
volume_features = ['volume',]

print('base feature :', features)

for feature in volume_features:
    X[feature] = X[feature] + 1e-5 # avoid 0 volume

for feature in features:
    X[feature] = np.log(X[feature])

print('**** DUMMY ****')
for i in range(1,10,2):
    for feature in features:
        X[f'{feature}_diff_{i}'] = X[feature].diff(i)

for i in range(1,10,2):
    for feature in price_features:
        for feature2 in price_features:
            if feature!=feature2:
                X[f'{feature}_+_{feature2}_diff_{i}'] = (X[feature]+X[feature2].shift(i))/2

X['h2l'] = X['high'] - X['low']
X['o2c'] = X['open'] - X['close']
X['t2p'] = np.maximum(X['close'],X['open']) - np.minimum(X['close'],X['open'])

for i in range(1,10,2):
    for feature in ['h2l','o2c','t2p']:
        X[f'{feature}_mean_{i}'] = X[feature].rolling(i*3).mean()

# ---- Fractiontion
print('**** Fractional ****')
d = 0.9
w = 1000
for feature in price_features:
    res = rolling_apply_last(fast_fracdiff,X[feature], w, d)
    X[f'{feature}_frac'] = res 

# ---- Entropy
print('**** Entropy ****')
window_size = 20
m = 2
r_ratio = 1
use_std = True 

for feature in price_features:
    X[f'{feature}_sample_entropy'] = rolling_apply(dynamic_sample_entropy_numba, X[feature], window_size, m, r_ratio, use_std)


generated_features = [col for col in X.columns if col not in features]
X = X[generated_features]
print("Generated X.shape: ", X.shape)
print('A0 in X' , 'A0' in X.columns)
print("Null Rows count: ",(len(X.dropna())- len(X)))

# Feature shifting
shift_params = [1,3,9]

for shift in shift_params:
    X_shift = X[generated_features].shift(shift)
    X_shift.columns = [f'{col}_shift_{shift}' for col in generated_features]
    X = pd.concat([X, X_shift], axis=1)
print("Generated X.shape with {} shifts : {}".format(len(shift_params),X.shape))
print("Null Rows count: ",(len(X.dropna())- len(X)))
X = X.dropna()
inf_columns = X.columns[np.isinf(X).any()].tolist()
X = X.rename(columns={'state':'A0'})

# Train Test Splitting
is_test = df['eob'] > test_start
X_train = X[~is_test]
X_test = X[is_test].drop(columns=['A0'])

print(X_train.tail())
print(X_test.tail())
print(df_label.tail())

print('X train shape: ',X_train.shape)
print('X test shape: ', X_test.shape)
print('X shape: ', X.shape)
print('label shape: ', df_label.shape)


# File Storage 
Test = len(X_test)
Train = len(X_train) 

file_Train = file_name_1 + "_Train_" + str(fix) +  "_" + str(Train) + '.csv'
file_Test = file_name_1 + "_Test_"  + str(fix) +  "_"  + str(Test) + '.csv'

pl.DataFrame(X_train).write_csv(file_Train)
pl.DataFrame(X_test).write_csv(file_Test)

file_Test_PCA = file_name_1 + "_Test_"  + str(fix) +  "_" + str(Test) + '_PCA' + '.csv'
tmp_df = X_train.drop('A0', axis=1)   ### 删除A0这列

X_test_PCA = pd.concat([tmp_df, X_test], axis=0, ignore_index=True)
X_test_PCA = pd.concat([X_test_PCA, tmp_df], axis=0, ignore_index=True)

pl.DataFrame(X_test_PCA).write_csv(file_Test_PCA)

print('local dir:', os.getcwd())
print("Write to : ",file_Train)
print("Write to : ",file_Test)
print("Write to : ",file_Test_PCA)
