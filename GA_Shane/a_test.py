import polars as pl 
import pickle 
from functions import *
from draw import transform 
import pandas as pd 
import polars as pl

def agg_minute(df, n=5):
    """
    Aggregate data into n-minute frequency bars based on count of rows.
    Each bar represents 'n' rows aggregated together. Last bar is skipped if count < n.
    
    Args:
        df: Input DataFrame
        n: Number of rows to aggregate per bar
    
    Returns:
        Aggregated DataFrame with the same schema
    """
    # Create a group column (integer division of index by n)
    df = df.with_columns(
        pl.col("index").floordiv(n).alias("group")
    )
    
    # Perform aggregation by group
    result = df.group_by("group").agg(
        pl.last("index"),
        pl.last("eob"),
        pl.first("open"),
        pl.max("high"),
        pl.min("low"),
        pl.last("close"),
        pl.sum("total_turnover"),
        pl.sum("volume"),
        pl.last("open_interest"),
    ).sort("group")
    
    # Drop the group column
    result = result.drop("group")
    
    return result

csv_path = './RB99_1m.csv'
df = pl.read_csv(csv_path)
df = df.rename({'datetime':'eob'})
df = df.drop(['order_book_id','trading_date'])
df = transform(df)
print(df.schema)
df[0:10].write_csv('./first_row.csv')

agg_minute(df[0:10],2).write_csv('./agg.csv')
# df, X_units_dict = gen_feature(df)
# df.write_csv(csv_path + '.cache.exp.csv')
# with open(csv_path + '.cache.pkl', 'wb') as f:
#     pickle.dump(X_units_dict, f)
     
    