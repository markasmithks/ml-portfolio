from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Define project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

PRIVATE_DATA = BASE_DIR / "private_data" / "daily_kwh_2025.csv"
OUTPUT_DATA = BASE_DIR / "data" / "synthetic_daily_usage.csv"

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
# Generate synthetic data
# -----------------------------
np.random.seed(42)

synthetic_index = seasonal_profile + np.random.normal(
    loc=0.0,
    scale=resid_std,
    size=len(seasonal_profile)
)

synthetic_index = np.clip(synthetic_index, 0.7, 1.4)

# -----------------------------
# Build synthetic dataset
# -----------------------------
synthetic_df = pd.DataFrame({
    "date": pd.date_range(
        start="2025-01-01",
        periods=len(synthetic_index),
        freq="D"
    ),
    "synthetic_usage": 100 * synthetic_index / synthetic_index.mean()
})

# -----------------------------
# Save synthetic output
# -----------------------------
OUTPUT_DATA.parent.mkdir(exist_ok=True)

synthetic_df.to_csv(OUTPUT_DATA, index=False)

print("Synthetic data written to:")
print(OUTPUT_DATA)
