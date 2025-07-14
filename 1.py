from NS_risk_metrics import price_bond, modified_duration, dv01, convexity
import pandas as pd

# 加载原始数据
df = pd.read_csv("beta_df_rolling.csv", parse_dates=["date"], index_col="date")
df = df[["beta0", "beta1", "beta2", "lambda"]].dropna()

# 计算风险指标
df["price"]     = df.apply(lambda row: price_bond(row["beta0"], row["beta1"], row["beta2"], row["lambda"]), axis=1)
df["duration"]  = df.apply(lambda row: modified_duration(row["beta0"], row["beta1"], row["beta2"], row["lambda"]), axis=1)
df["dv01"]      = df.apply(lambda row: dv01(row["beta0"], row["beta1"], row["beta2"], row["lambda"]), axis=1)
df["convexity"] = df.apply(lambda row: convexity(row["beta0"], row["beta1"], row["beta2"], row["lambda"]), axis=1)

# 读取原始 CSV 并合并风险指标
full_df = pd.read_csv("beta_df_rolling.csv", parse_dates=["date"], index_col="date")
full_df = full_df.join(df[["price", "duration", "dv01", "convexity"]])

# 另存为新文件，防止覆盖
full_df.to_csv("beta_df_rolling.csv")
print("✅ Saved to beta_df_rolling.csv")
