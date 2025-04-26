
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
from sklearn.preprocessing import StandardScaler

from functions import *

warnings.filterwarnings('ignore')


part = '20250426_100441_OMBx2l'
dir = f'/home/dragon/GA_Shane/outputs/{part}'

code = 'RB'
test_start = '2023-01-01'

fix = 713

shift_params = [1,3,5,9]
window_size = 10 
cfg = tsfel.get_features_by_domain(['statistical'])

""" 
Data 
"""
file_name_1 = glob.glob(os.path.join(dir, '*_output_axis.csv'))[0]
df = pd.read_csv(file_name_1).sort_values('eob')

file_name_2 = glob.glob(os.path.join(dir, f'*_output_axis_Label_{str(fix)}.csv'))[0]
df_label = pd.read_csv(file_name_2)


df = df.merge(df_label[['datelist','state','state_p']], left_on='eob',right_on='datelist',how='left')
df_label = df[['eob','state','state_p']]
df.drop(columns=['datelist'],inplace=True)

"""
Function
"""
def assert_equal(a, b, ):
    if a != b:
        raise ValueError(f"Bars not equals {a} != {b}")

""" 
Feature Generation
"""
features = ['open','high','low','close','volume']
X = df.copy(deep=True)[['eob','state','state_p']+features]
X.loc[:,['state','state_p']] = X.loc[:,['state','state_p']].shift(-1)
end_bar = X['eob'].iloc[-1]

X = X.dropna(subset=features)
assert_equal(end_bar, X['eob'].iloc[-1])

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
                X[f'{feature}_+_{feature2}_diff_{i}'] = np.log((np.exp(X[feature])+np.exp(X[feature2]))/2).diff(i)

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

# Remove features that is raw price or volume
generated_features = [col for col in X.columns if col not in features and col not in ['eob','state','state_p']]
X = X[['eob','state','state_p']+generated_features]

# ---- Feature Shifting
print('**** Feature Shifting ****')

for shift in shift_params:
    X_shift = X[generated_features].shift(shift)
    X_shift.columns = [f'{col}_shift_{shift}' for col in generated_features]
    X = pd.concat([X, X_shift], axis=1)


# column_inf_counts = X.apply(lambda x: np.isinf(x).sum()/len(x), axis=1)
X = X.replace([np.inf, -np.inf], np.nan)

features = [ col for col in X.columns if col not in ['eob','state','state_p']]
# print("columns with inf: ", column_inf_counts[column_inf_counts>0])
print("Null Rows count: ",(len(X.dropna())- len(X)))

X = X.dropna(subset=features)

assert_equal(end_bar, X['eob'].iloc[-1])

# Train Test Splitting
print('**** Train Test Splitting ****')
X = X.rename(columns={'state':'A0','state_p':'A0_p'})

print("X.shape ", X.shape)
print("Feature #: ", len(X.columns))
print(X.tail())


is_test = df['eob'] > test_start
X_train = X[~is_test]
X_test = X[is_test].drop(columns=['A0','A0_p'])

# features = X_test.columns
features =  [ col for col in X_test.columns if 'entropy' in col]
scaler = StandardScaler().fit(X_train[features])
X_train[features] = scaler.transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

print(X_train.tail())
assert_equal(X_test['eob'].iloc[-1],df_label['eob'].iloc[-1])

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
tmp_df = X_train.drop(['A0','A0_p'], axis=1)   ### 删除A0这列

X_test_PCA = pd.concat([tmp_df, X_test], axis=0, ignore_index=True)
X_test_PCA = pd.concat([X_test_PCA, tmp_df], axis=0, ignore_index=True)

pl.DataFrame(X_test_PCA).write_csv(file_Test_PCA)

print('local dir:', os.getcwd())
print("Write to : ",file_Train)
print("Write to : ",file_Test)
print("Write to : ",file_Test_PCA)
