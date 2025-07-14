import numpy as np
import pandas as pd
import datetime
from scipy.optimize import curve_fit
from get_yield_curve import get_yield_curve  
from ns_core import nelson_siegel

# 设置固定到期期限
maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)

# 第一步：拉取收益率曲线数据
start = "2020-01-01"
end = datetime.datetime.today().strftime("%Y-%m-%d")
yield_data = get_yield_curve(start, end)

# 第二步：滚动窗口 NS 拟合
window_size = 30
initial_guess = [3.0, -1.0, 1.0, 0.5]
beta_list = []

dates = yield_data.index

for i in range(window_size - 1, len(dates)):
    window_dates = dates[i - window_size + 1 : i + 1]
    window_data = yield_data.loc[window_dates]

    X = np.tile(maturities, len(window_data))
    y = window_data.values.flatten()

    try:
        params, _ = curve_fit(nelson_siegel, X, y, p0=initial_guess, maxfev=20000)

        
        if all(np.abs(params[:3]) < 10) and 0 < params[3] < 10:
            beta_list.append([dates[i], *params])
        else:
            print(f"Skipped extreme beta at {dates[i]}: {params}")
            beta_list.append([dates[i], np.nan, np.nan, np.nan, np.nan])

    except Exception as e:
        print(f"Fit failed on {dates[i]}: {e}")
        beta_list.append([dates[i], np.nan, np.nan, np.nan, np.nan])

beta_df_rolling = pd.DataFrame(beta_list, columns=['date', 'beta0', 'beta1', 'beta2', 'lambda'])
beta_df_rolling.set_index('date', inplace=True)

for col in ['beta0', 'beta1', 'beta2', 'lambda']:
    beta_df_rolling[f'{col}_smooth'] = beta_df_rolling[col].rolling(window=5, center=True, min_periods=1).median()

beta_df_rolling.to_csv("beta_df_rolling1.csv")
print("✅ Done. beta_df_rolling with smoothed values saved.")
print(beta_df_rolling.tail())
