"""
### 实盘推理的步骤

1. 获取实时数据

2. 实时数据生成轴所需要的算子

3. 算子计算轴，生成轴文件

4. 根据轴文件生成特征

5. 对X预测

6. 发送信号
"""

import polars as pl 
import numpy as np
from infer.functions import *
import GA_Shane.functions_inside_bar as f 

class PipelineAbstract:
    dna_code = "RB99"
    dna_id = "1"
    dna_expr = "volume"


    def __init__(self):
        pass

    def get_data(self):
        pass 

    def to_base_feature_df(self):
        # generate base features from raw data for axis creation
        pass
        
    def get_axis_factor(self, df):
        # generate axis factor from base features
        pass

    def save_factor(self, df):
        # save axis factor to file
        pass

    def get_split_index(self, df):
        # get split index for the data
        pass

    def to_axis_df(self,df):
        # generate axis df from base features
        pass

    def to_feature_df(self):
        # generate feature df from axis df
        pass

    def predict(self):
        # predict using the feature df
        pass

    def send_signal(self):
        # send signal to the trading system
        pass


class Pipeline(PipelineAbstract):
    def __init__(self):
        pass

    def get_data(self,):
        df = pl.read_csv(f"./infer/data/data_1m_{self.dna_code}.csv")
        df = df.rename({'datetime':'eob'})
        df = df.with_columns([pl.arange(0, df.height).alias('index')])
        df = df.select([
            pl.col('index'),
            pl.col('eob'),
            pl.exclude(['index','eob'])
        ])
        df = df.sort('eob')
        return df 

    def to_agg_minute_df(self,df, n=2):
        if n == 1:
            return df
        
        df = df.with_columns(
            pl.col("index").floordiv(n).alias("group")
        )
        df = df.group_by("group").agg(
            pl.last("index"),
            pl.last("eob"),
            pl.first("open"),
            pl.max("high"),
            pl.min("low"),
            pl.last("close"),
            pl.sum("total_turnover"),
            pl.sum("volume"),
            pl.last("open_interest"),
        ).sort("eob")
            
        return df
    
    def get_axis_factor(self, df, eq):
        y_pred = fn(df.to_pandas(),eq)
        y_pred = np.abs(y_pred)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        assert len(y_pred) == len(df), "y_pred length is not equal to df length"
        
        return y_pred
    
    def to_axis_df(self, df, axis_factor):
        pass

    def aggregate_time_axis_to_new_axis_df(self, df):
        pass

    def save_factor(self, df):
        df = df.select(['eob','factor'])
        df.write_csv(f"./infer/data/factor_{self.dna_code}.csv")
        print(f"factor_{self.dna_code} saved successfully.")



class Pipeline_No_1(Pipeline):

    dna_code = "RB99"
    dna_id = "20250424_003413_oMFt2l"
    dna_expr = "((((prev_hbar - hbar) * (((hbar - prev_hbar) / close) / prev_total_turnover)) / mid2) / prev_total_turnover) / volume"
    dna_cnt_of_bars = 30
    dna_n_kline = 1
    dna_thred = 3.822308322818339e-09
    dna_skip_n_rows_to_align_with_trainset = 1


    def __init__(self):
        pass

    def get_data(self):
        return super().get_data()
    
    def to_agg_minute_df(self, df, n=1):
        return super().to_agg_minute_df(df, n)

    def to_base_feature_df(self,df):
        tmp_df = df.clone()
        set2 = ['volume','open_interest','total_turnover']
        tmp_df = tmp_df.with_columns([
            (pl.col(c)+(1e-6)).log().alias(f'{c}') for c in set2
        ])

        tmp_df = tmp_df.with_columns([
            pl.max_horizontal(pl.col('open'), pl.col('close')).alias('top'),
            pl.min_horizontal(pl.col('open'), pl.col('close')).alias('bot'),
        ])

        tmp_df = tmp_df.with_columns([
            ((pl.col('high')+pl.col('bot'))/2).alias('hbar'),
            ((pl.col('open')+pl.col('close'))/2).alias('mid2'),
        ])

        tmp_df = tmp_df.with_columns([
            pl.col('hbar').shift(1).alias('prev_hbar'),
            pl.col('total_turnover').shift(1).alias('prev_total_turnover'),
        ])
        return tmp_df
    
    def get_axis_factor(self, df):
        tmp_df = self.to_base_feature_df(df)
        return super().get_axis_factor(tmp_df,self.dna_expr)
    
    def add_group_index_of_axis(self, df, axis_factor):
        axis_factor = np.nan_to_num(axis_factor, nan=0.0, posinf=0.0, neginf=0.0)
        df = df.with_columns([
            pl.Series(name='factor', values=axis_factor)
        ])
        df = df.with_columns([
            pl.col('factor').fill_nan(0.0).alias('factor')
        ])

        # index_list = split(df['factor'].to_numpy(), freq=30, thred=self.dna_thred)

        g_lst = split(df['factor'].to_numpy(), thred=self.dna_thred, return_g=True)
        
        df = df.with_columns([
            pl.Series(name='group', values=g_lst)
        ])
        
        return df

    def aggregate_time_axis_to_new_axis_df(self,df):

        
        df = df.with_columns([pl.arange(0, df.height).alias('index')])

        df = df.with_columns(
                (pl.col('close')-pl.col('open')).alias('box'),
        )

        df = df.group_by('group').agg([
                pl.col('eob').last().alias('eob'),
                pl.col('open').first().alias('open'),
                pl.col('high').max().alias('high'),
                pl.col('low').min().alias('low'), 
                pl.col('close').last().alias('close'),
                pl.col('volume').sum().alias('volume'),
                pl.col('total_turnover').sum().alias('total_turnover'),
                pl.col('open_interest').last().alias('open_interest'),
                pl.col('factor').std().alias('factor_std'),
                pl.col('factor').mean().alias('factor_mean'),
                pl.col('close').map_elements(lambda s: f.inside_bar_cv(s.to_numpy()),return_dtype=pl.Float64).alias('inside_bar_cv_ret_close'),
                pl.col('close').map_elements(lambda s: f.inside_bar_ret_min(s.to_numpy()),return_dtype=pl.Float64).alias('inside_bar_min_ret_close'),
                pl.col('close').map_elements(lambda s: f.inside_bar_ret_max(s.to_numpy()),return_dtype=pl.Float64).alias('inside_bar_max_ret_close'),
                pl.col('close').map_elements(lambda s: f.inside_bar_maxdd(s.to_numpy()),return_dtype=pl.Float64).alias('inside_bar_maxdd_close'),
                pl.col('box').map_elements(lambda s: f.inside_bar_sign_entropy(s.to_numpy()),return_dtype=pl.Float64).alias('inside_bar_sign_entropy_box'),
                pl.col('close').map_elements(lambda s: f.inside_bar_sample_entropy(s.to_numpy(), size=int(self.dna_cnt_of_bars)),return_dtype=pl.Float64).alias('inside_bar_sample_entropy_close'),
                pl.col('open_interest').map_elements(lambda s: f.inside_bar_sample_entropy(s.to_numpy(), size=int(self.dna_cnt_of_bars)),return_dtype=pl.Float64).alias('inside_bar_sample_entropy_oi'),
                pl.col('index').last().alias('hang'),
        ])
        df = df.sort('eob')
        df = df.fill_nan(0)
        df = df.with_columns([
                (pl.col('hang')-pl.col('hang').shift()).alias('bar_cnt')
        ])
        return df


@timed_execution
def test():
    pipeline = Pipeline_No_1()
    df = pipeline.get_data()
    df = pipeline.to_agg_minute_df(df, n=1)
    axis_factor = pipeline.get_axis_factor(df)
    df = pipeline.add_group_index_of_axis(df,axis_factor)
    print(df)
    df = pipeline.aggregate_time_axis_to_new_axis_df(df)
    print(df)
    # df = pipeline.to_feature_df(df)
    # pipeline.predict(df)
    # pipeline.send_signal(df)

if __name__ == "__main__":
    test()