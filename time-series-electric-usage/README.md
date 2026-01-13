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

## Time Series Decomposition & Baseline

Seasonal-Trend decomposition (STL) was applied to confirm that the synthetic
dataset preserves realistic annual seasonality and long-term trends.

A seasonal naïve baseline forecast was implemented using a 365-day lag.
This baseline serves as a benchmark that all subsequent models must outperform.

## Initial Analysis of Naive Data with Only Seasonality

Because the synthetic dataset was constructed to preserve a stable annual seasonal profile without long-term trend or structural change, a seasonal naïve forecast perfectly reproduces future values. This confirms that the baseline implementation is correct and establishes a lower bound on achievable error for this dataset.


