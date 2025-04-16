import sympy
import warnings 
import polars as pl
warnings.filterwarnings('ignore')   
import pandas as pd
import numpy as np 
from draw import * 
from pysr import PySRRegressor
import os
from tqdm import tqdm

code = 'RB'
suffix = f'{code}99_1m'
csv_path =   f"./{suffix}.csv"
part = '20250412_103114_rxl4ht'
model_pth =  rf'E:\dragon\GA_Shane\outputs\{part}\hall_of_fame.csv'
project_pth = rf'E:\dragon\GA_Shane\outputs\{part}'
os.makedirs(project_pth,exist_ok=True)

freq = 45

"""
Assume you have everything done. Now create new axis from current.
Init PySR from local ,Predict
"""

mdl = PySRRegressor.from_file(run_directory=rf'E:\dragon\GA_Shane\outputs\{part}')

sympy_str =  suffix + "_"

df_pysr = load_df_version3(csv_path, start_i=0,end_i=0,cached=False)
df_pysr = df_pysr.reset_index(drop=True)
df_pysr = df_pysr.sort_values('eob')


X = df_pysr.drop(columns=['eob'])
print(X.shape)
for i in range(len(mdl.equations_)):
    if i != len(mdl.equations_) -1 : continue
    y_pred = mdl.predict(X,i)

print(mdl.equations_)
print("Equation", i, "equal to", mdl.sympy(i))

df = pd.read_csv(csv_path)
df.reset_index(inplace=True)
df.rename(columns={'datetime':'eob','index':'hang'},inplace=True)
df.drop(columns=['order_book_id','trading_date'])
df['eob'] = pd.to_datetime(df['eob'])

df['factor'] = y_pred
df['factor'] = np.abs(df['factor'])
df['factor'] = df['factor'].replace({np.inf:np.nan,-np.inf:np.nan})
df = pl.DataFrame(df)

os.chdir(project_pth)
os.makedirs('./picture',exist_ok=1)
os.makedirs('./temp',exist_ok=1)


df[['eob', 'factor']].rename({'factor': part}).write_csv(
    f'./{suffix}_factor_{part}.csv'
)


### Aggregate
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

def agg_to_axis(df):
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)

    v = df['factor'].to_pandas().iloc[:180000]
    v = v.replace({np.inf:np.nan, -np.inf:np.nan})
    size = v.mean() * freq
    size = auto_round(size)

    print('size of thred: ', size)

    df = df.group_by('group').agg([
        pl.col('eob').last().alias('eob'),
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'), 
        pl.col('close').last().alias('close'),
        pl.col('volume').sum().alias('volume'),
        (pl.col('pos_sum')%size).last().alias('pos_sum'),
        pl.col('factor').last().alias('factor'),
        pl.col('index').last().alias('hang'),
    ])
    df = transform(df)
    df = df.with_columns([
        (pl.col('hang')-pl.col('hang').shift()).alias('bar_cnt')
    ])
    return df

axis_df = agg_to_axis(df)

axis_df.write_csv(f'./{suffix}_output_axis.csv')

title_file = f"{suffix}_{len(axis_df)}_{len(axis_df.filter(pl.col('eob') > '2023-01-01'))}"
file_name1 = f'./{suffix}_output_axis'
bars = pd.read_csv(f'{file_name1}.csv')

bars.set_index("eob", inplace=True)

returns_1 = np.log(bars['close']).diff().dropna()
returns_2 = np.log(bars['close']).diff(periods=2).dropna()
returns_3 = np.log(bars['close']).diff(periods=3).dropna()
returns_4 = np.log(bars['close']).diff(periods=4).dropna()
returns_5 = np.log(bars['close']).diff(periods=5).dropna()


standard_1 = (returns_1 - returns_1.mean()) / returns_1.std()
standard_2 = (returns_2 - returns_2.mean()) / returns_2.std()
standard_3 = (returns_3 - returns_3.mean()) / returns_3.std()
standard_4 = (returns_4 - returns_4.mean()) / returns_4.std()
standard_5 = (returns_5 - returns_5.mean()) / returns_5.std()


plt.figure(figsize=(16,12))

sns.kdeplot(standard_1, label="1", color='darkred')
sns.kdeplot(standard_2, label="2", color='green')
sns.kdeplot(standard_3, label="3", color='blue')
sns.kdeplot(standard_4, label="4", color='orange')
sns.kdeplot(standard_5, label="5", color='magenta')

v = np.random.normal(size=1000000)
sns.kdeplot(v, label="Normal", color='black', linestyle="--")


skewness = skew(standard_1)
kurt = kurtosis(standard_1,fisher=0)
print('skew of axis: ', skewness, 'kurt of axis: ', kurt)

skewness = skew(v)
kurt = kurtosis(v,fisher=0)
print('skew of normal: ', skewness, 'kurt of normal: ', kurt)


plt.xticks(range(-5, 6))
plt.legend(loc=8, ncol=5)
plt.title(f"{title_file}\nSkewness: {skewness:.2f}, Kurtosis: {kurt:.2f}", loc='center', fontsize=20, fontweight="bold", fontname="Times New Roman")
plt.xlim(-5, 5)
plt.grid(1)
# plt.show()
plt.savefig(f'./picture/{suffix}_output_axis.jpg')
plt.close()


plt.figure(figsize=(12, 8))
sns.histplot(np.log(bars['bar_cnt']), bins=30, kde=True, color='skyblue')
plt.title('Distribution of bar_cnt', fontsize=16, fontweight="bold")
plt.xlabel('bar_cnt', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.savefig(f'./picture/{suffix}_bar_dist.jpg')
plt.close()

# Calculate the ratio of ranges <5, 5~10, 10~20, 20~60, 60+ and bar count
ranges = {
    '<5': (bars['bar_cnt'] < 5).sum(),
    '5~10': ((bars['bar_cnt'] >= 5) & (bars['bar_cnt'] < 10)).sum(),
    '10~20': ((bars['bar_cnt'] >= 10) & (bars['bar_cnt'] < 20)).sum(),
    '20~60': ((bars['bar_cnt'] >= 20) & (bars['bar_cnt'] < 60)).sum(),
    '60+': (bars['bar_cnt'] >= 60).sum()
}

total = bars['bar_cnt'].count()
ratios = {key: value / total for key, value in ranges.items()}

bar_cnt = len(bars)

print("bar_cnt Ratios:")
for key, ratio in ratios.items():
    print(f"{key}: {ratio:.2%}")

print(f"Total Bar Count: {bar_cnt}, Mean Bar: {bars['bar_cnt'].mean()}")



# standardization 
# df['y_pred'] = (df['y_pred'] - df['y_pred'].rolling(600).mean())/df['y_pred'].rolling(600).std()
# idx_lst = fn_from_norm_factor(df['y_pred'])

# idx_lst = fn(y_pred , freq=freq, thred = 0)

# df_new_axis = df.iloc[idx_lst]
# df_new_axis.drop(columns=['order_book_id','trading_date'],inplace=True)
# df_new_axis.to_csv(os.path.join(project_pth,sympy_str+'output_axis.csv'),index=False)

# print('Asserting : ', np.sum(df.loc[df.index.isin(df_new_axis['hang'])]['close'].values - df_new_axis['close'].values) == 0)


# print('original length:',len(df))
# print(f'original length in {freq} mins :',len(df)//freq)
# print('new axis length:',len(df_new_axis))


# """
# Plotting
# """
# kde_plot(np.log(df['close']),os.path.join(project_pth , sympy_str+'All_kde_original.png'))
# kde_plot(np.log(df_new_axis['close']),os.path.join(project_pth, sympy_str+'All_kde_new_axis.png'))

# """
# Stats of df
# """
# from scipy import stats

# v = np.log(df['close']).diff().dropna()
# v = (v-v.mean())/v.std()
# sw_stat, sw_p = stats.shapiro(v)
# print(f"Original: Shapiro-Wilk Test: stat={sw_stat:.4f}, p-value={sw_p:.4g}, Normal={sw_p > 0.05}")
# print(f"Original: Shapiro-Wilk Test: stat={sw_stat:.4f}, p-value={sw_p:.4g}, Normal={sw_p > 0.05}")

# v = np.log(df_new_axis['close']).diff().dropna()
# v = (v-v.mean())/v.std()
# sw_stat, sw_p = stats.shapiro(v)
# print(f"Transformed: Shapiro-Wilk Test: stat={sw_stat:.4f}, p-value={sw_p}, Normal={sw_p > 0.05}")
# print(f"Transformed: Shapiro-Wilk Test: stat={sw_stat:.4f}, p-value={sw_p}, Normal={sw_p > 0.05}")


