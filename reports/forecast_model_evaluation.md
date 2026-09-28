# Demand Forecasting: Model Evaluation

## Objective
Develop a daily retail demand forecasting model for RetailPulse and
compare it against a seasonal-naive baseline using the same calendar
date from the previous year.

## Methodology
A Random Forest Regressor used calendar, historical sales, rolling
average, and scheduled promotion features. Expanding-window,
chronological backtesting covered seven quarters from Q1 2024
through Q3 2025.

## Results
Random Forest had lower MAPE than the seasonal-naive baseline in all
seven evaluated quarters. Mean quarterly MAPE was 3.26% for Random
Forest and 5.24% for the baseline, a relative reduction of about 37.7%.

## Interpretation
The results suggest that historical sales and calendar features
provide useful predictive information. Performance varied by quarter.

## Limitations and Future Work
This was rolling one-day-ahead evaluation, not a fixed-origin,
multi-month forecast. Promotion-feature availability must be verified
for each forecast date. Further validation, leakage checks, and
production monitoring are required.
