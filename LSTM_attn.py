import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from NS_risk_metrics import price_bond, modified_duration, dv01, convexity

# ==== Device ====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==== Load & Smooth Data ====
df = pd.read_csv("beta_df_rolling.csv", parse_dates=["date"], index_col="date")
features = df[["beta1", "beta0", "beta2"]]\
    .rolling(window=7, center=True, min_periods=1).median()\
    .dropna()
vals = features.values

# ==== Normalize ====
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(vals)

# ==== Dataset ====
seq_len    = 20
train_size = int(len(scaled) * 0.8)

class BetaMultiDataset(Dataset):
    def __init__(self, data, seq_len):
        self.xs, self.ys = [], []
        for i in range(len(data) - seq_len):
            self.xs.append(data[i:i+seq_len])
            self.ys.append(data[i+seq_len][0])  # forecast β₁
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return (torch.tensor(self.xs[idx], dtype=torch.float32),
                torch.tensor(self.ys[idx], dtype=torch.float32))

train_data   = scaled[:train_size]
test_data    = scaled[train_size - seq_len:]
train_ds     = BetaMultiDataset(train_data, seq_len)
test_ds      = BetaMultiDataset(test_data,  seq_len)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=16)

# ==== Model: Shallow LSTM + Dot-Product Attention ====
class LSTMAttn(nn.Module):
    def __init__(self, input_size=3, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size * 2, 1)
    def forward(self, x):
        outputs, (h_n, _) = self.lstm(x)       # outputs: [B, T, H]; h_n: [1,B,H]
        h_n = h_n.squeeze(0)                   # [B, H]
        # attention scores: dot(outputs, h_n)
        scores  = torch.bmm(outputs, h_n.unsqueeze(2)).squeeze(2)  # [B, T]
        weights = torch.softmax(scores, dim=1)                     # [B, T]
        # context vector: weighted sum of outputs
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # [B, H]
        combined = torch.cat([context, h_n], dim=1)  # [B, 2H]
        return self.fc(combined)                     # [B, 1]

model     = LSTMAttn(input_size=3, hidden_size=32).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== Training ====
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred   = model(xb).squeeze(1)
        loss   = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ==== In-Sample Forecast & RMSE ====
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().reshape(-1)
        preds.extend(out)
        trues.extend(yb.numpy())

# inverse-scale β₁ only
idxs      = features.index[train_size:]
preds_arr = np.array(preds).reshape(-1, 1)
trues_arr = np.array(trues).reshape(-1, 1)
preds_inv = scaler.inverse_transform(
    np.hstack([preds_arr, np.zeros_like(preds_arr), np.zeros_like(preds_arr)])
)[:, 0]
trues_inv = scaler.inverse_transform(
    np.hstack([trues_arr, np.zeros_like(trues_arr), np.zeros_like(trues_arr)])
)[:, 0]

rmse_val = np.sqrt(mean_squared_error(trues_inv, preds_inv))
print(f"✅ In-sample RMSE: {rmse_val:.6f}")

# ---- Save in-sample plot ----
plt.figure(figsize=(12,6))
plt.plot(idxs, trues_inv, label="True β₁")
plt.plot(idxs, preds_inv, label="LSTM+Attn Forecast", linestyle="--")
plt.title(f"β₁ In-Sample Forecast (RMSE={rmse_val:.4f})")
plt.xlabel("Date"); plt.ylabel("β₁")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("beta1_lstm_attn_insample.png")
plt.close()

# ==== Linear-Regression Future β₀, β₂ ====
def forecast_future(series, steps=30):
    X = np.arange(len(series)).reshape(-1,1)
    y = series.values
    lr = LinearRegression().fit(X, y)
    return lr.predict(np.arange(len(series), len(series)+steps).reshape(-1,1))

future_steps   = 30
beta0_series   = features["beta0"]
beta2_series   = features["beta2"]
future_beta0   = forecast_future(beta0_series, future_steps)
future_beta2   = forecast_future(beta2_series, future_steps)

# ==== Recursive Multi-Step β₁ Forecast with Adaptive Clip ====
window       = test_data[-seq_len:].copy()
future_preds = []
model.eval()
with torch.no_grad():
    for i in range(future_steps):
        x_in    = torch.tensor(window[np.newaxis,:,:], dtype=torch.float32).to(device)
        raw_out = model(x_in).detach().cpu().numpy().reshape(-1)[0]

        prev_val = window[-1, 0]
        # adaptive threshold based on recent volatility
        diffs     = window[1:,0] - window[:-1,0]
        sigma     = np.std(diffs)
        max_delta = 3 * sigma                 # allow ±3σ daily move
        delta     = np.clip(raw_out - prev_val, -max_delta, +max_delta)
        next_pred = prev_val + delta

        future_preds.append(next_pred)

        # roll window forward
        arr    = window[1:,:].copy()
        arr    = np.vstack([arr, [next_pred, future_beta0[i], future_beta2[i]]])
        window = arr

# inverse-scale multi-step
fp_arr   = np.array(future_preds).reshape(-1,1)
fp_inv   = scaler.inverse_transform(
    np.hstack([fp_arr, fp_arr, fp_arr])
)[:,0]
future_dates = pd.date_range(start=features.index[-1]+pd.Timedelta(1,'D'),
                             periods=future_steps)

# ---- Save future plot ----
plt.figure(figsize=(12,6))
plt.plot(idxs[-60:], trues_inv[-60:], label="Recent True β₁", color="black")
plt.plot(future_dates, fp_inv, label="30-Day LSTM+Attn Forecast", linestyle=":", color="red")
plt.title("β₁ 30-Day Recursive Forecast with LSTM+Attention (Adaptive Clip)")
plt.xlabel("Date"); plt.ylabel("β₁")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("beta1_lstm_attn_future_adaptive.png")
plt.close()

