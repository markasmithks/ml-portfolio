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

## Introducing Annual Trend

To better reflect real-world electric load behavior, a modest year-over-year
trend was introduced into the synthetic data generation process. This change
breaks the perfect repeatability of the seasonal-only dataset and introduces
systematic growth similar to that observed in real utility demand.

Under this updated data-generating process, the unadjusted seasonal naïve
baseline produces small but non-zero forecast errors. The resulting MAE and
RMSE values are low and nearly identical, indicating a consistent bias due to
annual growth rather than random forecast failures.

This behavior closely mirrors real-world conditions, where strong seasonality
dominates load patterns but long-term growth leads to systematic underprediction
when trend is ignored.

## Trend-Adjusted Seasonal Naïve Baseline

A trend-adjusted seasonal naïve baseline was then implemented to explicitly
account for the estimated annual growth rate derived from historical data.
This adjustment scales the seasonal naïve forecast by the observed
year-over-year growth factor.

Under the assumptions used to generate the synthetic dataset—namely a stable
seasonal profile and a constant annual growth rate—the trend-adjusted seasonal
naïve forecast perfectly reproduces future values. As a result, MAE and RMSE
collapse to zero.

This outcome is expected and confirms:
- Correct estimation of the underlying growth trend
- Correct implementation of the trend adjustment
- Internal consistency between the data-generating process and the baseline model

The trend-adjusted seasonal naïve forecast therefore represents the optimal
forecast under these assumptions and establishes a strict upper bound on
achievable accuracy for this dataset. More complex models are evaluated against
this benchmark to assess whether additional flexibility is justified under
more realistic conditions.

