import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_daily_usage.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

# Train / test split
train = df.loc[: "2025-10-31"]
test = df.loc["2025-11-01":]

# Seasonal naive forecast (365-day lag)
forecast = train["synthetic_usage"].shift(365).reindex(test.index)

# Plot
plt.figure(figsize=(12,5))
plt.plot(train.index, train["synthetic_usage"], label="Train")
plt.plot(test.index, test["synthetic_usage"], label="Actual")
plt.plot(test.index, forecast, label="Seasonal Naive Forecast", linestyle="--")
plt.legend()
plt.title("Seasonal Naive Baseline Forecast")
#plt.show()

# Save Plot
plt.savefig(BASE_DIR / "figures" / "seasonal_naive_forecast.png", dpi=150)
plt.close()