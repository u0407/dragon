
import os
import pandas as pd
from dotenv import load_dotenv
import clickhouse_connect

# Load environment variables
load_dotenv()

# --- ClickHouse Configuration ---
CH_HOST = os.getenv('CH_HOST', '127.0.1.1')
CH_PORT = int(os.getenv('CH_PORT', '18123')) # Often 8123 for HTTP, 9000 for native. Check your setup.
CH_USER = os.getenv('CH_USER', 'default')
CH_PASSWORD = os.getenv('CH_PASSWORD', '') # Default to empty string if not set
CH_DATABASE = os.getenv('CH_DATABASE', 'default') # Optional: specify default database

# --- ClickHouse Connection ---
def get_clickhouse_client():
    """
    Creates and returns a ClickHouse client instance.
    Raises ConnectionError if connection fails immediately.
    """
    # Recommended: Use database parameter during client creation
    client = clickhouse_connect.get_client(
        host='localhost',
        port='18123',
        user='default',
        password='chdefault!9',
        database='cn_futures', # Connect directly to the default DB if needed
    )
    client.query_arrow("SELECT 1")
    print(f"✅ ClickHouse client created and connection tested successfully to {CH_HOST}:{CH_PORT}")
    return client

class QueryHelper:
    # this class stores useful scripts for different db server and different use cases.

    def __init__(self):
        pass
    
    def query_cnt_records(db_name, table_name):
        return f"SELECT count(*) FROM {db_name}.{table_name}"

    @staticmethod
    def query_agg_bar_1m_at_date(code, date):
        # assert date os a int like %Y%m%d
        assert isinstance(date, int)
        assert isinstance(code, str)
        assert len(str(date)) == 8

        return f""" 
        SELECT
            datetime,
            date,
            instrument,
            sumMerge(sum_open_oi_state) / sumMerge(sum_oi_state) AS open,
            sumMerge(sum_high_oi_state) / sumMerge(sum_oi_state) AS high,
            sumMerge(sum_low_oi_state) / sumMerge(sum_oi_state) AS low,
            sumMerge(sum_close_oi_state) / sumMerge(sum_oi_state) AS close,
            sumMerge(sum_volume_state) AS volume,
            sumMerge(sum_turnover_state) AS total_turnover,
            sumMerge(sum_oi_state) AS open_interest 
        FROM cn_futures.agg_bars_1m
        where date={date} and instrument = '{code}'
        GROUP BY datetime, date, instrument 
        ORDER BY datetime DESC, instrument ASC
        ; 
        """
    
    @staticmethod
    def query_agg_bar_from_start_date(code,date):
        assert isinstance(date, int)
        assert isinstance(code, str)
        assert len(str(date)) == 8

        return f""" 
        SELECT
            datetime,
            date,
            instrument,
            sumMerge(sum_open_oi_state) / sumMerge(sum_oi_state) AS open,
            sumMerge(sum_high_oi_state) / sumMerge(sum_oi_state) AS high,
            sumMerge(sum_low_oi_state) / sumMerge(sum_oi_state) AS low,
            sumMerge(sum_close_oi_state) / sumMerge(sum_oi_state) AS close,
            sumMerge(sum_volume_state) AS volume,
            sumMerge(sum_turnover_state) AS total_turnover,
            sumMerge(sum_oi_state) AS open_interest
        FROM cn_futures.agg_bars_1m
        where date>={date} and instrument = '{code}'
        GROUP BY datetime, date, instrument
        ORDER BY datetime ASC
        ; 
        """
    
    @staticmethod
    def query_agg_bar_1m_between_dates(code, start_date, end_date):
        assert isinstance(start_date, int)
        assert isinstance(end_date, int)
        assert isinstance(code, str)
        assert len(str(start_date)) == 8
        assert len(str(end_date)) == 8

        return f""" 
        SELECT
            datetime,
            date,
            instrument,
            sumMerge(sum_open_oi_state) / sumMerge(sum_oi_state) AS open,
            sumMerge(sum_high_oi_state) / sumMerge(sum_oi_state) AS high,   
            sumMerge(sum_low_oi_state) / sumMerge(sum_oi_state) AS low,
            sumMerge(sum_close_oi_state) / sumMerge(sum_oi_state) AS close,
            sumMerge(sum_volume_state) AS volume,
            sumMerge(sum_turnover_state) AS total_turnover,
            sumMerge(sum_oi_state) AS open_interest  
        FROM cn_futures.agg_bars_1m
        where date>={start_date} and date<={end_date} and instrument = '{code}'
        GROUP BY datetime, date, instrument
        ORDER BY datetime ASC
        ; 
        """

# Example usage
if __name__ == "__main__":
    # query = QueryHelper.query_agg_bar_1m_at_date('RB2510', 20250506)
    # print(query)
    get_clickhouse_client()