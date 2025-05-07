
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
from dotenv import load_dotenv

from functions import *

warnings.filterwarnings('ignore')

with open('./part.txt', 'r', encoding='utf-8') as file:
    part = file.read()

dir = f'c:/Users/shen_/Code/dragon/GA_Shane/outputs/{part}'

# Check if directory exists first
if os.path.exists(dir):
    for file in os.listdir(dir):
        if "Test" in file or "Train" in file:
            file_path = os.path.join(dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
else:
    print(f"Directory does not exist: {dir}")

code = 'RB'
test_start = '2023-01-01'

print('Init with :', part)

"""
Function
"""
def assert_equal(a, b, ):
    if a != b:
        raise ValueError(f"Bars not equals {a} != {b}")

###********************* this is key to training and avoiding overfit. ************
fix = 513
# label_shift = 8
label_shift = 6


shift_params = [10,20,30]
window_size = 10 

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
price_features = ['open','high','low','close']
volume_features = ['volume','total_turnover','open_interest']
between_bar_features = price_features + volume_features
inside_bar_features = [col for col in df.columns if 'inside_bar' in col]
X = df.copy(deep=True)[['eob','state']+between_bar_features+inside_bar_features]
X.loc[:,['state']] = X.loc[:,['state']].shift(-label_shift)
print(f'label is shifted for {label_shift}')

end_bar = X['eob'].iloc[-1]

X = X.dropna(subset=between_bar_features)
X = X.dropna(subset=inside_bar_features)

assert_equal(end_bar, X['eob'].iloc[-1])


print('base feature :', between_bar_features)
print('inside bar feature :', inside_bar_features)


for feature in volume_features:
    X[feature] = X[feature] 
    if feature != 'open_interest':
        X[feature] = X[feature].rolling(20).sum()  + 1e-5 # avoid 0 volume

for feature in between_bar_features + inside_bar_features:
    if feature != 'bar_cnt':
        X[feature] = np.log(X[feature].abs()+1E-5)*np.sign(X[feature])

print('**** DUMMY ****')
for i in range(1,10,2):
    for feature in between_bar_features:
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

for i in [5,]:
    for feature in inside_bar_features:
        X[f'{feature}_mean_{i}'] = X[feature].rolling(i).mean()
        X[f'{feature}_cov_{i}'] = X[feature].rolling(i).cov()

# ---- Fractiontion
print('**** Fractional ****')
d = 0.9
w = 1000
for feature in price_features:
    res = rolling_apply_last(fast_fracdiff,X[feature], w, d)
    X[f'{feature}_frac'] = res 

for feature in volume_features:
    res = rolling_apply_last(fast_fracdiff,X[feature], w, d)
    X[f'{feature}_frac'] = res 

# ---- Entropy
import antropy as ant

print('**** Entropy ****')
w = 10

x1 = np.array((X['high']+X['high'].diff().rolling(5).std()))
x2 = np.array((X['low']+X['low'].diff().rolling(5).std()))
x3 = np.array(X['high'].rolling(3).max())
x4 = np.array(X['low'].rolling(3).min())
x5 = np.array((X['high'].rolling(5).max()).diff())
x6 = np.array((X['low'].rolling(5).min()).diff())
oi = np.array(X['open_interest'].diff())

X['perm_entropy_high'] = rolling_apply(ant.perm_entropy,x1,10, 3,1,True)
X['perm_entropy_low'] = rolling_apply(ant.perm_entropy,x2,10, 3,1,True)

X['spectral_entropy_high'] = rolling_apply(ant.spectral_entropy,x1, 10, 1,'welch')
X['spectral_entropy_low'] = rolling_apply(ant.spectral_entropy,x2, 10, 1,'welch')
X['spectral_entropy_open_interest'] = rolling_apply(ant.spectral_entropy,oi, 10, 1,'welch')

X['num_zerocross_high'] = rolling_apply(ant.num_zerocross,x5,w,)
X['num_zerocross_low'] = rolling_apply(ant.num_zerocross,x6,w,)

x8 = np.array(X['close']/X['close'].shift(20))
X['petrosian_fd_close'] =  rolling_apply(ant.petrosian_fd,x8,w,)
X['katz_fd_close'] =  rolling_apply(ant.katz_fd,x8,w,)
X['katz_fd_close'] =  rolling_apply(ant.katz_fd,x8,w,)
# X['detrended_fluctuation_close'] =  rolling_apply(ant.detrended_fluctuation,x8,w,) s

# stats 
w = 5
x7 = np.array(X['close'].diff(9))
x8 = np.array(X['close'].diff(7))

X['skew_close_diff9'] = rolling_apply(scipy.stats.skew,x7,w,)
X['kstat_close_diff9'] = rolling_apply(scipy.stats.kstat,x7,w,3)
X['rank_close_diff7'] = rolling_apply_last(scipy.stats.rankdata,x8, w,)

# auto corr 

def auto_corr(arr):
    _ret = log_ret(arr)
    return corr(_ret[:-1], _ret[1:])
w = 10
X['auto_corr_close'] = rolling_apply(auto_corr,X['close'].to_numpy(),w,)
X['auto_corr_high'] = rolling_apply(auto_corr,X['high'].to_numpy(),w,)
X['auto_corr_close'] = rolling_apply(auto_corr,X['low'].to_numpy(),w,)
X['auto_corr_open'] = rolling_apply(auto_corr,X['open'].to_numpy(),w,)
X['auto_corr_volume'] = rolling_apply(auto_corr,X['volume'].to_numpy(),w,)
X['auto_corr_total_turnover'] = rolling_apply(auto_corr,X['total_turnover'].to_numpy(),w,)
X['auto_corr_open_interest'] = rolling_apply(auto_corr,X['open_interest'].to_numpy(),w,)


# Remove features that is raw price or volume
generated_features = [col for col in X.columns if col not in between_bar_features and col not in ['eob','state']]
generated_features = list(set(generated_features))
X = X[['eob','state']+generated_features]




# ---- Feature Shifting
print('**** Feature Shifting ****')

for shift in shift_params:
    print('debugging X',X[generated_features].shape)
    X_shift = X[generated_features].shift(shift)
    print('debugging X_shift',X_shift[generated_features].shape)
    
    X_shift.columns = [f'{col}_shift_{shift}' for col in X_shift.columns]
    X = pd.concat([X, X_shift], axis=1)


# column_inf_counts = X.apply(lambda x: np.isinf(x).sum()/len(x), axis=1)X
X = X.replace([np.inf, -np.inf], np.nan)

features = [ col for col in X.columns if col not in ['eob','state',]]
# print("columns with inf: ", column_inf_counts[column_inf_counts>0])
print("Null Rows count: ",(len(X.dropna())- len(X)))

X = X.dropna(subset=features)

assert_equal(end_bar, X['eob'].iloc[-1])


# Train Test Splitting
print('**** Train Test Splitting ****')
X = X.rename(columns={'state':'A0',})

print("X.shape ", X.shape)
print("Feature #: ", len(X.columns))
print(X.tail())


is_test = df['eob'] > test_start
X_train = X[~is_test]
X_test = X[is_test].drop(columns=['A0'])

# features = X_test.columns
features =  [ col for col in X_test.columns if col not in ['eob','state']]
if len(features)>0:
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
tmp_df = X_train.drop(['A0'], axis=1)   ### 删除A0这列

X_test_PCA = pd.concat([tmp_df, X_test], axis=0, ignore_index=True)
X_test_PCA = pd.concat([X_test_PCA, tmp_df], axis=0, ignore_index=True)

pl.DataFrame(X_test_PCA).write_csv(file_Test_PCA)

print('local dir:', os.getcwd())
print("Write to : ",file_Train)
print("Write to : ",file_Test)
print("Write to : ",file_Test_PCA)