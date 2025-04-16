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

warnings.filterwarnings('ignore')

code = 'RB'
roll = 5
label = 30
test_start = '2023-01-01'
fix = 513

part = '20250411_165242_sftNYO'
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
X = df.copy(deep=True)[features]
X = X.shift(1) # 因为是需要预测当期的状态，所以需要将X shift1
print("Original X.shape: ", X.shape)

price_features = ['open','high','low','close']
volume_features = ['volume',]

print('base feature :', features)

for feature in volume_features:
    X[feature] = X[feature] + 1e-5 # avoid 0 volume

for feature in features:
    X[feature] = np.log(X[feature])

X['mid'] = np.log((np.exp(X['open']) + np.exp(X['close']))/2)
X['mid2'] = np.log((np.exp(X['high']) + np.exp(X['low']))/2)
price_features = ['open','high','low','close','mid','mid2']


for i in range(1,10,2):
    for feature in features:
        X[f'{feature}_diff_{i}'] = X[feature].diff(i)



"""
Fractional Diffrencing
"""
import pylab as pl
def fast_fracdiff(arr, d):
    T = len(arr)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(arr) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return  np.real(dx[0:T]) 

d = 0.8
for feature in price_features:
    X[f'{feature}_frac'] = fast_fracdiff(X[feature], d=d)

d = 0.1
for feature in volume_features:
    X[f'{feature}_frac'] = fast_fracdiff(X[feature].cumsum,d=d)


def cid_ce(x, normalize):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))