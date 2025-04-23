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
part = '20250420_handmade'
model_pth =  rf'E:\dragon\GA_Shane\outputs\{part}\hall_of_fame.csv'
project_pth = rf'E:\dragon\GA_Shane\outputs\{part}'
os.makedirs(project_pth,exist_ok=True)

freq = 20
size = None

df = pl.read_csv(csv_path + '.cache.csv').fill_nan(0)
df = transform(df)

def gen_axis_index(df):

    factor_expr = (
        (pl.col('prev_high') - pl.min_horizontal('prev_bot','open'))*
        pl.col('std_ret_mid3')*
        ((pl.col('bot')-pl.col('high')).sign())*
        pl.col('vol2oi')
    ).alias('factor')
    #  # "(((prev_high - min(prev_bot, open)) * std_ret_mid3) * sign(bot - prev_high)) * vol2oi"

    df = df.with_columns(factor_expr)
    
    return df['factor'].to_numpy()
y_pred = gen_axis_index(df)

df = pd.read_csv(csv_path)
df.reset_index(inplace=True)
df.rename(columns={'datetime':'eob','index':'hang'},inplace=True)
df.drop(columns=['order_book_id','trading_date'])
df['eob'] = pd.to_datetime(df['eob'])

df['factor'] = y_pred
df['factor'] = np.abs(df['factor'])
df = transform(pl.DataFrame(df))

def agg_to_axis(df,size,freq):
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)

    v = df['factor'].to_pandas().iloc[:180000]
    v = v.replace({np.inf:np.nan, -np.inf:np.nan})
    if size is None:
        size = v.mean() * freq
        size = auto_round(size)

    df = df.with_columns(pl.col('factor').abs().alias('factor'))

    df = df.with_columns([
        pl.col('factor').cum_sum().alias('pos_sum'),
    ])

    df = df.with_columns([
        (pl.col('pos_sum') // size).cast(pl.Int64).alias('group')
    ])
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

axis_df = agg_to_axis(df,size=size,freq=freq)
print("len axis df ",len(axis_df))
os.chdir(project_pth)
os.makedirs('./picture',exist_ok=1)
os.makedirs('./temp',exist_ok=1)

axis_df.write_csv(f'./{suffix}_output_axis.csv')
df[['eob', 'factor']].rename({'factor': part}).write_csv(
    f'./{suffix}_factor_{part}.csv'
)

title_file = f"{suffix}_{len(axis_df)}"
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
random_value = np.random.normal(size=1000000)

skewness = skew(standard_1)
kurt = kurtosis(standard_1,fisher=1)
print("skew of series, ", skewness, "kurt of series: ", kurt)

skewness2 = skew(random_value)
kurt2 = kurtosis(random_value,fisher=1)
print("skew of normal, ", skewness2, "kurt of normal: ", kurt2)

plt.figure(figsize=(16,12))

sns.kdeplot(standard_1, label="1", color='darkred')
sns.kdeplot(standard_2, label="2", color='green')
sns.kdeplot(standard_3, label="3", color='blue')
sns.kdeplot(standard_4, label="4", color='orange')
sns.kdeplot(standard_5, label="5", color='magenta')

sns.kdeplot(random_value, label="Normal", color='black', linestyle="--")

plt.xticks(range(-5, 6))
plt.legend(loc=8, ncol=5)
plt.title(f"{title_file}\nSkewness: {skewness:.2f}, Kurtosis: {kurt:.2f}", loc='center', fontsize=20, fontweight="bold", fontname="Times New Roman")
plt.xlim(-5, 5)
plt.grid(1)
# plt.show()
plt.savefig(f'./picture/{suffix}_output_axis.jpg')
plt.close()


plt.figure(figsize=(12, 8))
sns.histplot(bars['bar_cnt'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of bar_cnt', fontsize=16, fontweight="bold")
plt.xlabel('bar_cnt', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.savefig(f'./picture/{suffix}_bar_dist.jpg')
plt.close()