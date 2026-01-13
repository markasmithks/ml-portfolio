import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_daily_with_trend_usage.csv"

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

# Train / test split
train = df.loc[: "2025-10-31"]
test = df.loc["2025-11-01":]

# ✅ Seasonal naive forecast (correct)
forecast = df["synthetic_usage"].shift(365)
forecast = forecast.loc[test.index]

# Align safely
valid = test["synthetic_usage"].notna() & forecast.notna()

y_true = test.loc[valid, "synthetic_usage"]
y_pred = forecast.loc[valid]

print("Valid sample count:", len(y_true))

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("Seasonal Naive Baseline Performance")
print("----------------------------------")
print(f"MAE : {mae:,.0f} kWh")
print(f"RMSE: {rmse:,.0f} kWh")
