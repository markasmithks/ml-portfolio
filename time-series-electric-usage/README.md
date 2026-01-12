# Time Series Electric Usage Forecasting

## Overview
This project demonstrates time series forecasting techniques using
synthetic daily electric energy usage data. The goal is to model and
forecast seasonal demand patterns commonly observed in electric utilities.

The project emphasizes:
- Seasonality awareness
- Reproducible data generation
- Clear baseline and advanced forecasting models

---

## Data

### Synthetic Data
To preserve confidentiality, this project uses **synthetic daily usage data**
derived from real operational totals.

The synthetic dataset:
- Preserves annual seasonality
- Retains realistic day-to-day variability
- Removes all identifiable or proprietary values

Raw operational data is excluded from version control.

The synthetic dataset is generated using:

~~~text
src/generate_synthetic_usage.py
~~~

and written to:

~~~text
data/synthetic_daily_usage.csv
~~~



