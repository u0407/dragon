import os
import rqdatac
rqdatac.init()
import datetime
import polars as pl 
from dotenv import load_dotenv
load_dotenv()




class Config:
    global auto_trading
    auto_trading = 0

    global close_1m
    global data_1m

    code = 'RB99'
    start_date = '2023-05-24'

    path = './infer/data'
    if os.path.exists(path) == False:
        os.makedirs(path)

#################################################################################################################
# 下载1分钟数据
#################################################################################################################


if auto_trading == 0:

    y_mode = int(datetime.datetime.now().strftime('%Y'))
    m_mode = int(datetime.datetime.now().strftime('%m'))
    day_mode = int(datetime.datetime.now().strftime('%d'))


    print("mode=0，白天   mode=1，平日晚上   mode=3，周五晚上   mode=4，自定义")
    mode = int(input("mode = "))

    if mode == 0:
        print('*** 白天 OK ***')

    if mode == 1:
        day_mode = day_mode + 1
        print('*** 平日晚上 OK ***')

    if mode == 3:
        day_mode = day_mode + 3
        print('*** 周五晚上 OK ***')

    if mode == 4:
        y_in = int(input("年 = "))
        m_in = int(input("月 = "))
        d_in = int(input("日 = "))
        y_mode = y_in
        m_mode = m_in
        day_mode = d_in
        print('*** 自定义 OK ***')

if os.getenv('REALTIME_MODE')=='LOCAL':

    data_1m = rqdatac.get_price(Config.code, start_date=Config.start_date, end_date=str(y_mode)+'-'+str(m_mode)+'-'+str(day_mode), frequency='1m')
    pl.DataFrame(data_1m.reset_index()).write_csv(os.path.join(Config.path, f'data_1m_{Config.code}.csv'))
    
    print(f'data_1m_{Config.code}____OK')
    
    close_1m = data_1m['close'].iloc[-1]
