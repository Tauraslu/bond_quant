import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# === 导入你定义的风险指标函数 ===
from NS_risk_metrics import price_bond, modified_duration, convexity, dv01

# ==== 配置 ====
LAGS = 3           # 滞后阶数
FUTURE_STEPS = 30  # 未来预测步数

# ==== 读取 + 风险指标特征添加 ====
df = pd.read_csv("beta_df_rolling.csv", parse_dates=["date"], index_col="date")
df = df[["beta1_smooth", "beta0", "beta2", "lambda"]].dropna()

# === 添加风险指标 ===
df["price"]     = df.apply(lambda row: price_bond(row["beta0"], row["beta1_smooth"], row["beta2"], row["lambda"]), axis=1)
df["duration"]  = df.apply(lambda row: modified_duration(row["beta0"], row["beta1_smooth"], row["beta2"], row["lambda"]), axis=1)
df["dv01"]      = df.apply(lambda row: dv01(row["beta0"], row["beta1_smooth"], row["beta2"], row["lambda"]), axis=1)
df["convexity"] = df.apply(lambda row: convexity(row["beta0"], row["beta1_smooth"], row["beta2"], row["lambda"]), axis=1)

# === 构造滞后特征 ===
for lag in range(1, LAGS + 1):
    for var in ["beta1_smooth", "beta0", "beta2"]:
        df[f"{var}_lag{lag}"] = df[var].shift(lag)

df.dropna(inplace=True)

# === 构造训练数据 ===
feature_cols = (
    [f"{var}_lag{lag}" for var in ["beta1_smooth", "beta0", "beta2"] for lag in range(1, LAGS + 1)]
    + ["price", "duration", "dv01", "convexity"]
)
X = df[feature_cols]
y = df["beta1_smooth"]

# === 划分训练/测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 训练 XGBoost 模型 ===
model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)

# === 测试集预测 ===
y_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📊 Test RMSE: {test_rmse:.6f}")

# === 结果可视化 ===
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label="Train", color="black")
plt.plot(y_test.index,  y_test,  label="Test",  color="blue", alpha=0.6)
plt.plot(y_test.index,  y_pred,  label="XGB Test Forecast", color="orange", linestyle="--")
plt.title(f"β₁ Forecast with XGBoost + Risk Metrics (Test RMSE: {test_rmse:.4f})")
plt.xlabel("Date")
plt.ylabel("β₁")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("xgb_risk_test_forecast.png")
plt.close()

# === 多步递归预测未来 β₁ ===
window = df.iloc[-LAGS:].copy()
future_preds = []
future_dates = pd.date_range(start=window.index[-1] + pd.Timedelta(days=1), periods=FUTURE_STEPS)

for step in range(FUTURE_STEPS):
    # 准备当前步的特征
    vals = []
    for var in ["beta1_smooth", "beta0", "beta2"]:
        for lag in range(1, LAGS + 1):
            vals.append(window[var].iloc[-lag])
    # 风险特征取最近可得那一行
    risk_vals = window.iloc[-1][["price", "duration", "dv01", "convexity"]].tolist()
    
    X_step = pd.DataFrame([vals + risk_vals], columns=feature_cols)
    
    pred = model.predict(X_step)[0]
    future_preds.append(pred)

    # 用预测更新窗口
    new_row = {
        "beta1_smooth": pred,
        "beta0":        window["beta0"].iloc[-1],
        "beta2":        window["beta2"].iloc[-1],
        "price":        window["price"].iloc[-1],
        "duration":     window["duration"].iloc[-1],
        "dv01":         window["dv01"].iloc[-1],
        "convexity":    window["convexity"].iloc[-1],
    }
    for lag in range(1, LAGS + 1):
        new_row[f"beta1_smooth_lag{lag}"] = window[f"beta1_smooth_lag{lag}"].iloc[-1]
        new_row[f"beta0_lag{lag}"]        = window[f"beta0_lag{lag}"].iloc[-1]
        new_row[f"beta2_lag{lag}"]        = window[f"beta2_lag{lag}"].iloc[-1]

    new_df = pd.DataFrame(new_row, index=[future_dates[step]])
    window = pd.concat([window, new_df]).iloc[1:]

# === 绘制未来预测 ===
future_series = pd.Series(future_preds, index=future_dates)
plt.figure(figsize=(12, 6))
plt.plot(y_test.index,        y_test,        label="Test β₁", color="blue")
plt.plot(y_test.index,        y_pred,        label="XGB Test Forecast", color="orange", linestyle="--")
plt.plot(future_series.index, future_series, label="XGB 30-Day Recursive Forecast", color="red", linestyle=":")
plt.title("β₁ 30-Day Recursive Forecast with XGBoost + Risk Metrics")
plt.xlabel("Date")
plt.ylabel("β₁")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("xgb_risk_recursive_forecast.png")
plt.close()