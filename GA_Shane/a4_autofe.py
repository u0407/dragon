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
reduncy_rows = 50
fix = 513

part = '20250408_183923_tf0A9D'
dir = rf'E:/dragon/GA_Shane/outputs/{part}'

# Tsfel
window_size = 10 
cfg = tsfel.get_features_by_domain(['statistical'])
file_name_1 = glob.glob(os.path.join(dir, '/*_output_axis.csv'))[0]
df = pd.read_csv(file_name_1)

file_name_2 = glob.glob(os.path.join(dir, '/*_output_axis_{fix}.csv'))[0]
df_label = pd.read_csv(file_name_2)

df = df.iloc[roll:]

print("shape df",df.shape)
print("shape df_label",df_label.shape)


# Preprocessing
df = df.set_index('eob')

df['vol2oi'] = df['volume'] / df['open_interest']

df = df[['open', 'high', 'low', 'close','y_pred']]

for col in df.columns:
    df[col] = df[col].astype('float64')

df['y_pred'] = np.sqrt(np.abs(df['y_pred'])) * np.sign(df['y_pred'])

# Crossing

df['top'] = np.maximum(df['open'],df['close'])
df['bot'] = np.minimum(df['open'],df['close'])

df['high_-_bot'] = df['high'] - df['bot']
df['high_-_bot_1'] = df['high'] - df['bot'].shift(1)

df['low_-_top'] = df['low'] - df['top']
df['low_-_top_1'] = df['low'] - df['top'].shift(1)

df['high_-_low'] = df['high'] - df['low']
df['high_-_low_1'] = df['high'] - df['low'].shift(1)

df['open_-_close'] = (df['close'] - df['open'])
df['open_-_close_1'] = (df['close'] - df['open'].shift(1))

df['high_-_close'] = df['high'] - df['close']
df['low_-_close'] = df['low'] - df['close']
df['high_-_close_1'] = df['high'] - df['close'].shift(1)

df['open_+_close'] = (df['open'] + df['close']) / 2
df['high_+_top'] = (df['high']+df['top'])/2
df['low_+_bot'] = (df['low']+df['bot'])/2

df2 = df.copy()

for col in df.columns:
    if col in ['open', 'high', 'low', 'close','open_+_close', 'high_+_top', 'low_+_bot']:
        for i in [2, 5, 9]:
            df2[col+f'_diff_{i}'] = df[col].diff(i)

    else:
        df2[col] = df[col]

df2['y_pred'] = df['y_pred']

df = df2.copy()

df = df.reset_index()

df = pl.DataFrame(df)

df.write_csv('./RB99_1m_features_1.csv')

df = df.to_pandas()



############ Train Test Splitting ############

# df = df[['eob','high_diff_2']]

date = df['eob'] > test_start

is_train = df[~date]
train_start_idx = is_train.index[0] + reduncy_rows
train_end_idx = is_train.index[-1]+1
X_train = df.iloc[train_start_idx:train_end_idx]

# Append more rows for testing data
is_test = df[date]
test_start_idx = is_test.index[0] - window_size  - reduncy_rows
test_end_idx = is_test.index[-1]+1
X_test = df.iloc[test_start_idx:test_end_idx]

train_date_lst = X_train['eob']
test_date_lst = X_test['eob']

label_train = df_label[df_label['datelist'].isin(train_date_lst)].iloc[window_size:]['state'].values

print('shape of X_test', X_test.shape)
print('shape of df_label', df_label.shape)
print('shape of df', df.shape)


# # Generate features
X_train.drop(columns=['eob'],inplace=True)
X_test.drop(columns=['eob'],inplace=True)


def extract_features_window(i, data_array, cfg, window_size):
    try:
        window_data = data_array[i-window_size:i, :]
        features = tsfel.time_series_features_extractor(
            cfg, window_data, window_size=window_size, overlap=0, verbose=0, fs=1
        )
        return i, features
    except Exception as e:
        print(f"Error at window {i}: {str(e)}")
        return i, None

# Feature extraction for training set
data_array = X_train.to_numpy()
indices = range(window_size, len(data_array))

# Process training data in parallel
with Parallel(n_jobs=-1, verbose=1, batch_size=100) as parallel:
    train_results = parallel(
        delayed(extract_features_window)(i, data_array, cfg, window_size) for i in indices
    )

# Filter, sort and concatenate training features
F_train = pd.concat(
    [feat for _, feat in sorted(
        [(idx, feat) for idx, feat in train_results if feat is not None],
        key=lambda x: x[0]
    )],
    axis=0, ignore_index=True
)
# <Task> add X_test, original features also 
F_train = pd.concat([F_train, X_train.iloc[window_size:].reset_index(drop=True)], axis=1)


print('&&&& Success features F_train {} &&&&'.format(F_train.shape))


# Process test data
data_array = X_test.to_numpy()
indices = range(window_size, len(data_array))

# Process test data in parallel
with Parallel(n_jobs=-1, verbose=1, batch_size=100) as parallel:
    test_results = parallel(
        delayed(extract_features_window)(i, data_array, cfg, window_size) for i in indices
    )

# Filter, sort and concatenate test features
F_test = pd.concat(
    [feat for _, feat in sorted(
        [(idx, feat) for idx, feat in test_results if feat is not None],
        key=lambda x: x[0]
    )],
    axis=0, ignore_index=True
)
F_test = pd.concat([F_test, X_test.iloc[window_size:].reset_index(drop=True)], axis=1)

# <Task> add X_test, original features also 


print('&&&& Success features F_test {} &&&&'.format(F_test.shape))

# Feature selection and preprocessing
corr_features, F_train = tsfel.correlated_features(F_train, drop_correlated=True)
F_test = F_test.drop(corr_features, axis=1)

print('&&&& Success Corr F_train {} &&&&'.format(F_train.shape))
print('&&&& Success Corr F_test {} &&&&'.format(F_test.shape))

selector = VarianceThreshold()
F_train = pd.DataFrame(
    selector.fit_transform(F_train),
    columns=[f'A{i+1}' for i in range(selector.transform(F_train).shape[1])]
)
F_test = pd.DataFrame(
    selector.transform(F_test),
    columns=F_train.columns
)

print('&&&& Success Variance F_train {} &&&&'.format(F_train.shape))
print('&&&& Success Variance F_test {} &&&&'.format(F_test.shape))

# Standardization
scaler = preprocessing.StandardScaler()
F_train = pd.DataFrame(
    scaler.fit_transform(F_train),
    columns=F_train.columns
)
F_test = pd.DataFrame(
    scaler.transform(F_test),
    columns=F_train.columns
)

print('&&&& Success STD F_train {} &&&&'.format(F_train.shape))
print('&&&& Success STD F_test {} &&&&'.format(F_test.shape))


# Feature shifting
shift_params = [1, 2, 3, 5]
original_cols = F_train.columns

for shift in shift_params:
    # Training data shifts
    shift_cols = F_train[original_cols].shift(shift)
    shift_cols.columns = [f'{col}_{shift}' for col in original_cols]
    F_train = pd.concat([F_train, shift_cols], axis=1)
    
    # Test data shifts
    shift_cols = F_test[original_cols].shift(shift)
    shift_cols.columns = [f'{col}_{shift}' for col in original_cols]
    F_test = pd.concat([F_test, shift_cols], axis=1)



# 73340
F_train = F_train.iloc[reduncy_rows:]
F_test = F_test.iloc[reduncy_rows:]
y_train = label_train[reduncy_rows:]

print('&&&& Success Shift F_train {} &&&&'.format(F_train.shape))
print('&&&& Success Shift F_test {} &&&&'.format(F_test.shape))


F_train['A0'] = y_train
cols = F_train.columns.tolist()
cols.insert(0, cols.pop(cols.index('A0')))
F_train = F_train[cols]

# File Storage 
Test = len(F_test)
Train = len(F_train) 

file_Train = file_name_1 + "_Train_" + str(Train) + '.csv'
file_Test = file_name_1 + "_Test_" + str(Test) + '.csv'

pl.DataFrame(F_train).write_csv(file_Train)
pl.DataFrame(F_test).write_csv(file_Test)

file_Test_PCA = file_name_1 + "_Test_" + str(Test) + '_PCA' + '.csv'
tmp_df = F_train.drop('A0', axis=1)   ### 删除A0这列

F_test_PCA = pd.concat([tmp_df, F_test], ignore_index=True)
F_test_PCA = pd.concat([F_test_PCA, tmp_df], ignore_index=True)

print('length of F_test_PCA', len(F_test_PCA))

pl.DataFrame(F_test_PCA).write_csv(file_Test_PCA)

