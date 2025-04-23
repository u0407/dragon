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

part = '20250421_155054_2XCKt6'
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


df = df.merge(df_label[['datelist','state','state_p']], left_on='eob',right_on='datelist',how='left')
df_label = df[['eob','state','state_p']]
df.drop(columns=['datelist'],inplace=True)

""" 
Feature Generation
"""
features = ['open','high','low','close','volume']
X = df.copy(deep=True)[['state','state_p']+features]
X.loc[:,['state','state_p']] = X.loc[:,['state','state_p']].shift(-1)
X = X.dropna()
print(X.head())
print("Original X.shape: ", X.shape)

price_features = ['open','high','low','close']
volume_features = ['volume',]

for feature in volume_features:
    X[feature] = X[feature] + 1e-5 # avoid 0 volume

for feature in features:
    X[feature] = np.log(X[feature])

"""
Feature Generation
"""
import tsfel 
from joblib import Parallel, delayed
import warnings 
warnings.filterwarnings('ignore')

def extract_features_window(i, data_array, cfg, window_size):
    window_data = data_array[i-window_size:i, :]
    features = tsfel.time_series_features_extractor(
        cfg, window_data,  window_size=len(window_data), verbose=0
    )
    return i, features


window_size = 20
data_array = X[features].to_numpy()
indices = range(window_size, len(data_array))
cfg = tsfel.get_features_by_domain()

with Parallel(n_jobs=-1, verbose=1, batch_size=100) as parallel:
    train_results = parallel(
        delayed(extract_features_window)(i, data_array, cfg, window_size) for i in indices
    )
F = pd.concat(
    [feat for _, feat in sorted(
        [(idx, feat) for idx, feat in train_results if feat is not None],
        key=lambda x: x[0]
    )],
    axis=0, ignore_index=True
)

F.index = X.index[window_size:]

print('X:', data_array.shape)
print(X.index)
print('F:', F.shape)
print(F.index)

# Now concatenate properly
X = pd.concat([X, F], axis=1)

print("Generated X.shape: ", X.shape)
print('A0 in X' , 'A0' in X.columns)
print("Null Rows count: ",(len(X.dropna())- len(X)))

generated_features = [col for col in X.columns if col not in features]
X = X[generated_features]

# # Feature shifting
# shift_params = [1,3,5,9]

# for shift in shift_params:
#     X_shift = X[generated_features].shift(shift)
#     X_shift.columns = [f'{col}_shift_{shift}' for col in generated_features]
#     X = pd.concat([X, X_shift], axis=1)

# print("Generated X.shape with {} shifts : {}".format(len(shift_params),X.shape))
print("Null Rows count: ",(len(X.dropna())- len(X)))
# Get columns with all null values
null_columns = X.columns[X.isnull().all()]
print("Columns with no valid values:")
print(null_columns.tolist())


X = X.dropna()
inf_columns = X.columns[np.isinf(X).any()].tolist()
X = X.rename(columns={'state':'A0','state_p':'A0_p'})

# Train Test Splitting
is_test = df['eob'] > test_start
X_train = X[~is_test]
X_test = X[is_test].drop(columns=['A0','A0_p'])

from sklearn.preprocessing import StandardScaler
features = X_test.columns
scaler = StandardScaler().fit(X_train[features])
X_train[features] = scaler.transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])


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
tmp_df = X_train.drop(['A0','A0_p'], axis=1)   ### 删除A0这列

X_test_PCA = pd.concat([tmp_df, X_test], axis=0, ignore_index=True)
X_test_PCA = pd.concat([X_test_PCA, tmp_df], axis=0, ignore_index=True)

pl.DataFrame(X_test_PCA).write_csv(file_Test_PCA)

print('local dir:', os.getcwd())
print("Write to : ",file_Train)
print("Write to : ",file_Test)
print("Write to : ",file_Test_PCA)
