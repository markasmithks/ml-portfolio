import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_daily_usage.csv"

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["synthetic_usage"])
plt.title("Synthetic Daily Electric Usage (2025)")
plt.xlabel("Date")
plt.ylabel("Synthetic Usage Index")
plt.tight_layout()
#plt.show()

plt.savefig(BASE_DIR / "figures" / "synthetic_usage_2025.png", dpi=150)
plt.close()

