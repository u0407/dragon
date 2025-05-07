import warnings 
warnings.filterwarnings('ignore')   
from draw import * 
import os
from dotenv import load_dotenv
load_dotenv()

code = 'RB'
suffix = f'{code}99_1m'
with open('./part.txt', 'r', encoding='utf-8') as file:
    part = file.read()
model_pth =  rf'./GA_Shane/outputs/{part}/hall_of_fame.csv'
project_pth = rf'./GA_Shane/outputs/{part}/'

print(f"Your axis file is in {part}")
# n_bar = 1 
# freq = 60

n_bar = 2
freq = 30
print(f'base bar in : {n_bar} , # of bar: {freq}')
i = -1
thred = 0 

"""
Calculate factor based on a generated equation 
"""
if n_bar == 1:
    csv_path =   rf"c:/Users/shen_/Code/dragon/{suffix}.csv"
    _df = pl.read_csv(csv_path + f'.cache.exp.csv')
else:
    csv_path =   rf"c:/Users/shen_/Code/dragon/{suffix}.csv.{n_bar}m.csv"
    _df = pl.read_csv(csv_path + f'.cache.exp.csv')

mdl_csv = pd.read_csv(model_pth)
eq = mdl_csv['Equation'].values[i]
y_pred = fn(_df,eq)
y_pred = np.abs(y_pred)
y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
print("Equation ", i, ":", eq)

# unit checking, try to *100 for numerical columns of _df, then compare  fn(_df*100,equations[i]) and y_pred
numerical_cols = _df.select(pl.selectors.by_dtype(pl.NUMERIC_DTYPES)).columns 
_df[numerical_cols] = _df[numerical_cols] * 100
print('Is equation has no unit: ', np.allclose(y_pred,np.nan_to_num(np.abs(fn(_df , eq)), nan=0.0, posinf=0.0, neginf=0.0)))
del _df

"""
Examine Factor  Normality
"""

os.chdir(project_pth)
os.makedirs('./picture',exist_ok=1)
os.makedirs('./temp',exist_ok=1)

df = pl.read_csv(csv_path).to_pandas()
df['factor'] = y_pred

df.rename(columns={'datetime':'eob'})[['eob', 'factor']].rename({'factor': part}).to_csv(
    f'./{suffix}_factor_{part}.csv'
)

def split(arr,freq=30,thred=0):
    n = len(arr)
    arr = np.abs(arr)
    if thred ==0:
        thred = np.nanmean(arr[:180000]) * freq 
    # print("thread eqalus: ", thred)
    # 构造 idx_list
    idx_list = []
    cumsum = 0.0
    thred_tmp = thred
    for i in range(n):
        cumsum += arr[i]
        if i == 180000:
            thred_tmp = (thred_tmp * (i-1) + arr[i-1]*(1)) / i
        if cumsum > thred or i == n:
            idx_list.append(i)  # 记录索引
            cumsum = 0.0
    return idx_list

idx_lst = split(y_pred, freq=freq, thred = thred)

df_new_axis = df.iloc[idx_lst]
df_new_axis.reset_index(inplace=True)
df_new_axis.rename(columns = {'index':'hang'},inplace = True)
kde_plot(np.log(df_new_axis['close']),'./picture/kde_new_axis.png')
print('original length:',len(df))
print(f'original length in {freq*n_bar} mins :',len(df)//freq)
print('new axis length:',len(df_new_axis))
print('Estimated Frequency of bars: ', int(len(df)/len(df_new_axis)))



"""
Groupping
"""

def split2(arr,freq=30,thred=0):
    n = len(arr)
    arr = np.abs(arr)
    if thred == 0 :
        thred = np.nanmean(arr[:180000]) * freq 
    print("thread equas: ", thred)
    # 构造 idx_list
    thred_tmp = thred
    g_lst = []
    g = 0
    cumsum = 0.0
    for i in range(n):
        cumsum += arr[i]
        g_lst.append(g)
        if i == 180000:
            thred_tmp = (thred_tmp * (i-1) + arr[i-1]*(1)) / i
        if cumsum > thred or i == n:
            cumsum = 0.0
            g += 1 
    return g_lst

g_lst = split2(df['factor'].values, freq=freq)
df['group'] =g_lst


def agg_to_axis(df):
    import functions_inside_bar as f 
    
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns={'datetime':'eob'})
        df = pl.DataFrame(df)

    df = df.with_columns([pl.arange(0, df.height).alias('index')])

    df = df.with_columns(
        (pl.col('close')-pl.col('open')).alias('box'),
        # (pl.col('close')-pl.col('close').shift(1)).alias('jump_c'),
        # (pl.col('open')-pl.col('open').shift(1)).alias('jump_o'),
        # (pl.max_horizontal("open","close").alias('top')),
        # (pl.min_horizontal("open","close").alias('bot')),
    )

    df = df.group_by('group').agg([
        pl.col('eob').last().alias('eob'),
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'), 
        pl.col('close').last().alias('close'),
        # pl.col('top').last().alias('top'),
        # pl.col('bot').last().alias('bot'),
        pl.col('volume').sum().alias('volume'),
        pl.col('total_turnover').sum().alias('total_turnover'),
        pl.col('open_interest').last().alias('open_interest'),
        pl.col('factor').last().alias('factor'),
        pl.col('close').map_elements(lambda s: f.inside_bar_cv(s.to_numpy())).alias('inside_bar_cv_ret_close'),
        pl.col('close').map_elements(lambda s: f.inside_bar_ret_min(s.to_numpy())).alias('inside_bar_min_ret_close'),
        pl.col('close').map_elements(lambda s: f.inside_bar_ret_max(s.to_numpy())).alias('inside_bar_max_ret_close'),
        pl.col('close').map_elements(lambda s: f.inside_bar_maxdd(s.to_numpy())).alias('inside_bar_maxdd_close'),
        pl.col('box').map_elements(lambda s: f.inside_bar_sign_entropy(s.to_numpy())).alias('inside_bar_sign_entropy_box'),
        # pl.col('jump_c').map_elements(lambda s: f.inside_bar_sign_entropy(s.to_numpy())).alias('inside_bar_sign_entropy_jump_c'),
        # pl.col('jump_o').map_elements(lambda s: f.inside_bar_sign_entropy(s.to_numpy())).alias('inside_bar_sign_entropy_jump_o'),
        pl.col('close').map_elements(lambda s: f.inside_bar_sample_entropy(s.to_numpy(), size=int(freq))).alias('inside_bar_sample_entropy_close'),
        pl.col('open_interest').map_elements(lambda s: f.inside_bar_sample_entropy(s.to_numpy(), size=int(freq))).alias('inside_bar_sample_entropy_oi'),

        pl.col('index').last().alias('hang'),
    ])
    df = transform(df)
    df = df.fill_nan(0)
    df = df.with_columns([
        (pl.col('hang')-pl.col('hang').shift()).alias('bar_cnt')
    ])
    return df

axis_df = agg_to_axis(df)
axis_df.write_csv(f'./{suffix}_output_axis.csv')

print(axis_df.tail(1))