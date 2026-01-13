import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_daily_with_trend_usage.csv"

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

# Compute annual means
annual_means = df["synthetic_usage"].resample("Y").mean()

# Estimate growth rate (average year-over-year change)
growth_rates = annual_means.pct_change().dropna()
annual_growth = growth_rates.mean()

# Train / test split
train = df.loc[: "2025-10-31"]
test  = df.loc["2025-11-01":]

# Seasonal naive forecast
forecast = df["synthetic_usage"].shift(365)
forecast = forecast.loc[test.index]

# Apply trend adjustment
forecast_adj = forecast * (1 + annual_growth)

# Align
valid = test["synthetic_usage"].notna() & forecast_adj.notna()
y_true = test.loc[valid, "synthetic_usage"]
y_pred = forecast_adj.loc[valid]

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("Trend-Adjusted Seasonal Naive Performance")
print("-----------------------------------------")
print(f"Estimated annual growth: {annual_growth:.2%}")
print(f"MAE : {mae:,.2f} kWh")
print(f"RMSE: {rmse:,.2f} kWh")
