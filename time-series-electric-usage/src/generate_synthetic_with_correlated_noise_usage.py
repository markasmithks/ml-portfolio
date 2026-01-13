from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Define project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

PRIVATE_DATA = BASE_DIR / "private_data" / "daily_kwh_2025.csv"
OUTPUT_DATA = BASE_DIR / "data" / "synthetic_daily_with_correlated_noise_usage.csv"

# -----------------------------
# Load real data (local only)
# -----------------------------
df = pd.read_csv(
    PRIVATE_DATA,
    parse_dates=["load_date"]
)

df = (
    df.rename(columns={"load_date": "date"})
      .sort_values("date")
)

# -----------------------------
# Normalize (remove magnitude)
# -----------------------------
df["load_index"] = df["total_kwh"] / df["total_kwh"].mean()

# -----------------------------
# Extract seasonal pattern
# -----------------------------
df["day_of_year"] = df["date"].dt.dayofyear

seasonal_profile = (
    df.groupby("day_of_year")["load_index"]
      .mean()
)

# -----------------------------
# Estimate variability
# -----------------------------
df["seasonal_mean"] = df["day_of_year"].map(seasonal_profile)
residuals = df["load_index"] - df["seasonal_mean"]
resid_std = residuals.std()

# -----------------------------
# Generate multi-year synthetic data
# -----------------------------
np.random.seed(42)

years = [2023, 2024, 2025]
synthetic_rows = []

for year in years:
    # Generate correlated noise (AR(1))
    phi = 0.7
    noise = np.zeros(len(seasonal_profile))
    white_noise = np.random.normal(
        loc=0.0,
        scale=resid_std,
        size=len(seasonal_profile)
    )

    for t in range(1, len(noise)):
        noise[t] = phi * noise[t - 1] + white_noise[t]

    yearly_index = seasonal_profile.values + noise
    yearly_index = np.clip(yearly_index, 0.7, 1.4)

    dates = pd.date_range(
        start=f"{year}-01-01",
        periods=len(yearly_index),
        freq="D"
    )

    trend_factor = 1.0 + 0.01 * (year - 2023)

    synthetic_rows.append(
        pd.DataFrame({
            "date": dates,
            "synthetic_usage": trend_factor * 100 * yearly_index / yearly_index.mean()
        })
    )


synthetic_df = pd.concat(synthetic_rows, ignore_index=True)

# -----------------------------
# Save synthetic output
# -----------------------------
OUTPUT_DATA.parent.mkdir(exist_ok=True)
synthetic_df.to_csv(OUTPUT_DATA, index=False)

print("Synthetic data written to:")
print(OUTPUT_DATA)

