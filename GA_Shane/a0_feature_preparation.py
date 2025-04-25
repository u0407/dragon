import polars as pl 
import pickle 
from functions import *
from draw import transform 
import pandas as pd 


# csv_path = './RB99_1m.csv'
# df = pl.read_csv(csv_path)
# df = df.rename({'datetime':'eob'})
# df = df.drop(['order_book_id','trading_date'])
# df = transform(df)
# df, X_units_dict = gen_feature(df)
# df.write_csv(csv_path + '.cache.csv')
# with open(csv_path + '.cache.pkl', 'wb') as f:
#     pickle.dump(X_units_dict, f)
    
    
    
    

def gen_feature(df):
    """
    Generate benchmark features for symbolic regression.
    """
    df = df.with_columns(pl.col('close').alias('A0'))
    set1 = ['open','high','low','close','top','bot','mid2','mid4','mid_hl','hbar','lbar','mid3']
    set2 = ['volume','open_interest','total_turnover']
    set3 = ['vol2oi','bar_spread','bar_box','bar_up','close_etp','t2b_etp']

    df = df.with_columns([
        pl.max_horizontal(pl.col('open'), pl.col('close')).alias('top'),
        pl.min_horizontal(pl.col('open'), pl.col('close')).alias('bot'),
        (pl.col('volume')/(pl.col('open_interest')+1e-6)).alias('vol2oi'),

    ])

    df = df.with_columns([
        ((pl.col('open')+pl.col('close'))/2).alias('mid2'),
        ((pl.col('open')+pl.col('close')+pl.col('high')+pl.col('low'))/4).alias('mid4'),
        ((pl.col('close')+pl.col('high')+pl.col('low'))/3).alias('mid3'),
        ((pl.col('high')+pl.col('low'))/2).alias('mid_hl'),
        ((pl.col('high')+pl.col('bot'))/2).alias('hbar'),
        ((pl.col('low')+pl.col('top'))/2).alias('lbar'),
    ])

    # log
    df = df.with_columns([
        (pl.col(c)+(1e-6)).log().alias(f'{c}') for c in set2
    ])
    
    # Entropy of kline 
    window_size = 60
    m = 2
    r_ratio = 1
    use_std = True 
    print('close entropy')
    entropyfunc = pl.Series("close_etp",rolling_apply(dynamic_sample_entropy_numba, df['close'].to_numpy(), window_size, m, r_ratio, use_std))
    df = df.insert_column(-1, entropyfunc)
    print('top to bot entropy')
    entropyfunc = pl.Series("t2b_etp",rolling_apply(dynamic_sample_entropy_numba, (df['top']/df['bot']).to_numpy(), window_size, m, r_ratio, use_std))
    df = df.insert_column(-1, entropyfunc)


    # Previous shift of every feature.
    df = df.with_columns([
        pl.col(c).shift(1).alias(f'prev_{c}') for c in set1+set2+set3 if c in df.columns
    ])

    X_units_dict = {}
    for col in set1:
        if col in df.columns:
            X_units_dict[col] = 'm'
            X_units_dict[f'prev_{col}'] = 'm'
    for col in set2:
        if col in df.columns:
            unit = 'm*kg' if col == 'total_turnover' else 'kg'
            X_units_dict[col] = unit
            X_units_dict[f'prev_{col}'] = unit
    for col in set3:
        if col in df.columns:
            X_units_dict[col] = ''
            X_units_dict[f'prev_{col}'] = ''
    
    return df, X_units_dict


csv_path = './RB99_1m.csv'
df = pl.read_csv(csv_path)
df = df.rename({'datetime':'eob'})
df = df.drop(['order_book_id','trading_date'])
df = transform(df)
print(df.head())
df, X_units_dict = gen_feature(df)
df.write_csv(csv_path + '.cache.exp.csv')
with open(csv_path + '.cache.pkl', 'wb') as f:
    pickle.dump(X_units_dict, f)
     
    