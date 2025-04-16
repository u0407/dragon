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
part = '20250412_handmade'
model_pth =  rf'E:\dragon\GA_Shane\outputs\{part}\hall_of_fame.csv'
project_pth = rf'E:\dragon\GA_Shane\outputs\{part}'
os.makedirs(project_pth,exist_ok=True)

freq = 45 
size = None

df = pl.read_csv(csv_path)
df = df.drop(['order_book_id','trading_date'])
df = df.rename({'datetime':'eob'})

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



def gen_axis_index(df, size=1, freq=45, apply_abs = True):
    # 
    cols = df.columns
    """((low - max(prev_bot, top)) * (bot - high)) * top"""
    df = df.with_columns([
            pl.max_horizontal(pl.col('open'), pl.col('close')).alias('p_top'),
            pl.min_horizontal(pl.col('open'), pl.col('close')).alias('p_bot'),
        ]).with_columns([
            pl.col('p_bot').shift().alias('prev_bot'),
            pl.col('low').log().alias('log_low'),
            pl.col('high').log().alias('log_high'),
            pl.col('p_top').log().alias('log_p_top'),
            pl.col('p_bot').log().alias('log_p_bot'),
        ]).with_columns([
            pl.col('prev_bot').log().alias('log_prev_bot'),
        ]).with_columns([
            (
                (pl.col('log_low') - pl.max_horizontal(pl.col('log_prev_bot'), pl.col('log_p_top'))) *
                (pl.col('log_p_bot') - pl.col('log_high')) *
                pl.col('log_p_top')
            ).alias('factor')
        ])
    
    if apply_abs:
        df = df.with_columns(pl.col('factor').abs().alias('factor'))

    df = df.with_columns([
        pl.col('factor').cum_sum().alias('pos_sum'),
    ])

    if size == None :
        size = df['factor'].to_pandas().iloc[:180000].mean() * freq
        size = auto_round(size)

    print(' size of thred: ',size)
    df = df.with_columns([
        (pl.col('pos_sum') // size).cast(pl.Int64).alias('group')
    ])
    # 
    df = df[cols+ ['factor','pos_sum','group']]
    return  df

df = gen_axis_index(df, size=size, freq=45, apply_abs=True)


def agg_to_axis(df):
    if isinstance(df, pd.DataFrame):
        df = pl.DataFrame(df)

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
closes = axis_df['close'].to_numpy()
rets = closes[1:] - closes[:-1]
rets = rets[np.isfinite(rets)]
z_rets = (rets - np.mean(rets)) / np.std(rets)
z_skewness = skew(z_rets)
print("Z-standardized returns skewness:", z_skewness)



os.chdir(project_pth)
os.makedirs('./picture',exist_ok=1)
os.makedirs('./temp',exist_ok=1)


df[['eob', 'factor']].rename({'factor': part}).write_csv(
    f'./{suffix}_factor_{part}.csv'
)

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

skewness = skew(standard_1)
kurt = kurtosis(standard_1,fisher=1)

plt.figure(figsize=(16,12))

sns.kdeplot(standard_1, label="1", color='darkred')
sns.kdeplot(standard_2, label="2", color='green')
sns.kdeplot(standard_3, label="3", color='blue')
sns.kdeplot(standard_4, label="4", color='orange')
sns.kdeplot(standard_5, label="5", color='magenta')

sns.kdeplot(np.random.normal(size=1000000), label="Normal", color='black', linestyle="--")

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