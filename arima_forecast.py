import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Step 1: Load rolling beta1 data
df = pd.read_csv("beta_df_rolling.csv", parse_dates=["date"], index_col="date")
beta1 = df["beta1_smooth"].dropna()

# Step 2: Train-test split
train_size = int(len(beta1) * 0.8)
train, test = beta1[:train_size], beta1[train_size:]

# Step 3: Fit ARIMA model (can tune order here)
order = (3, 0, 2)  # ARIMA(p,d,q)
model = ARIMA(train, order=order)
model_fit = model.fit()

# Step 4: Forecast
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Step 5: Plot
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train", color="black")
plt.plot(test.index, test, label="Test", color="blue", alpha=0.6)
plt.plot(forecast.index, forecast, label="ARIMA Forecast", color="red", linestyle="--")
plt.title("β₁ Forecast with ARIMA (p=3, d=0, q=2)")
plt.xlabel("Date")
plt.ylabel("β₁")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("beta1_arima_forecast.png")
plt.show()

# Step 6: Print test RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA Test RMSE: {rmse:.6f}")
