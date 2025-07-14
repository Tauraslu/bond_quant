import numpy as np
import pandas as pd
import datetime
from pandas_datareader.data import DataReader

def get_yield_curve(start: str, end: str) -> pd.DataFrame:
    maturity_codes = ['GS1', 'GS2', 'GS3', 'GS5', 'GS7', 'GS10', 'GS20', 'GS30']
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")

    try:
        df = DataReader(maturity_codes, 'fred', start_dt, end_dt)
        df = df.dropna()
        df = df.asfreq('B')  # 强制设置为 Business Day frequency（工作日）
        df = df.interpolate(method='time')  # 用时间插值填补缺口
    except Exception as e:
        raise RuntimeError(f"FRED data fetch failed: {e}")
    
    return df

if __name__ == "__main__":
    start = "2020-01-01"
    end = datetime.datetime.today().strftime("%Y-%m-%d")
    curve_data = get_yield_curve(start, end)
    curve_data.to_csv("curve_data_daily.csv")
    print(curve_data.tail())
