import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_daily_usage.csv"

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.set_index("date")

# STL decomposition (365-day seasonality)
stl = STL(df["synthetic_usage"], period=365)
result = stl.fit()

# Plot
fig = result.plot()
fig.set_size_inches(12, 8)
plt.suptitle("STL Decomposition of Synthetic Electric Usage", y=1.02)
#plt.show()

# Save Plot
plt.savefig(BASE_DIR / "figures" / "decomposition_usage_2025.png", dpi=150)
plt.close()