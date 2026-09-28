
# RetailPulse — AI-Powered Demand Forecasting

## Project Overview

RetailPulse is a retail analytics project that uses historical sales data and machine learning to forecast daily product demand. The goal is to help retail businesses make better inventory and demand-planning decisions.

## Current Features

- Interactive Streamlit demand forecasting dashboard
- Daily sales data analysis and visualization
- Random Forest demand forecasting model
- Adjustable forecast horizon
- Forecast results displayed in a chart and table
- CSV export of generated forecasts
- Model evaluation using MAE, RMSE, and MAPE
- Historical quarterly backtesting against a seasonal-naive baseline

## Technology Stack

- Python 3.11
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- DuckDB

## Model Evaluation

The Random Forest model was evaluated on historical data from October through December 2025 using rolling one-day-ahead predictions.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 411.93 units/day |
| Root Mean Squared Error (RMSE) | 509.59 units/day |
| Mean Absolute Percentage Error (MAPE) | 2.55% |

Across seven historical quarters, the Random Forest model achieved an average quarterly MAPE of 3.26%, compared with 5.24% for the seasonal-naive baseline.

These are historical evaluation results, not guarantees of future forecast accuracy. Multi-day forecasts may perform differently because they rely on predicted rather than observed future sales.

## Project Structure

```text
RetailPulse/
├── data/
│   └── processed/
├── models/
│   ├── random_forest_demand_forecast.joblib
│   └── forecast_features.json
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── forecast_model_comparison.csv
│   ├── quarterly_forecast_backtest.csv
│   └── forecast_model_evaluation.md
├── tests/
├── dashboard/
├── app.py
└── README.md
```

## Run Locally

1. Clone or download the project repository.
2. Open a terminal in the project directory.
3. Activate the Python virtual environment.
4. Install the required dependencies.
5. Run the Streamlit application.

```bash
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn joblib
streamlit run app.py
```

Ensure the required sales dataset, trained model, and feature configuration files exist at the paths expected by `app.py`.

## Data Quality

The sales data was explored using DuckDB to handle the large transaction file efficiently. Exact duplicate transaction rows were identified, and a deduplicated daily sales series was used for the forecasting experiments. This assumes identical full-row records represent repeated records; that assumption should be validated against the source system where possible.

## Limitations and Future Work

- Validate forecast performance with additional rolling-origin backtests.
- Verify that all forecasting features would be available at prediction time.
- Investigate demand peaks that the model underestimates.
- Extend the platform with customer segmentation and churn prediction.
- Explore inventory optimization and anomaly detection.
- Add automated tests, model monitoring, and deployment configuration.

## Disclaimer

This is an ongoing project. Reported results reflect the dataset and evaluation procedure used during development. Additional validation is required before operational or business-critical use.
# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is a retail analytics platform designed to analyze sales trends and forecast daily product demand using historical retail transaction data and machine learning.

## Project Objectives

- Forecast daily retail demand.
- Analyze historical sales patterns and seasonality.
- Evaluate forecasting models against a seasonal-naive baseline.
- Build an interactive dashboard for generating and exploring demand forecasts.
- Establish a foundation for customer analytics, churn prediction, inventory optimization, and model monitoring.

## Technology Stack

- Python 3.11
- Pandas and NumPy
- DuckDB for large-dataset analysis
- Scikit-learn
- Prophet
- Matplotlib
- Streamlit
- Joblib

## Dataset

The project uses retail transaction data covering January 2022 through December 2025.

The original sales dataset contains approximately 9.97 million transaction rows. Exact duplicate rows were removed for the cleaned forecasting series, subject to the assumption that identical full-row records represent duplicate transactions.

The daily forecasting dataset contains 1,461 calendar dates.

## Demand Forecasting

A Random Forest regression model was developed using calendar variables, historical sales lags, rolling averages, and promotion-calendar features.

A seasonal-naive model, which uses sales from the corresponding date in the previous year, serves as the baseline.

### Historical Model Evaluation

The Random Forest model was evaluated on October–December 2025 using rolling one-day-ahead predictions.

| Metric | Random Forest |
|---|---:|
| Mean Absolute Error (MAE) | 411.93 units/day |
| Root Mean Squared Error (RMSE) | 509.59 units/day |
| Mean Absolute Percentage Error (MAPE) | 2.55% |

### Quarterly Backtesting

Across seven historical quarters from 2024 Q1 through 2025 Q3:

- Mean quarterly Random Forest MAPE: 3.26%
- Mean quarterly seasonal-naive MAPE: 5.24%

These are averages of quarterly MAPE values, not a pooled MAPE. The backtest results are historical measurements and do not guarantee future forecasting accuracy.

The October–December 2025 evaluation uses actual previous-day sales for lag features. A multi-day forecast that feeds its own predictions back into the model may perform differently.

## Interactive Dashboard

The Streamlit dashboard provides:

- A selectable forecast horizon.
- Daily demand forecasts.
- Forecast visualizations and summary metrics.
- A downloadable forecast CSV.
- Historical model-performance metrics.
- Quarterly comparison of the forecasting model and baseline.

## Project Structure

```text
RetailPulse/
├── dashboard/
├── data/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
├── src/
├── tests/
├── app.py
└── README.md
```

## Running the Dashboard

Activate the project virtual environment:

```bash
source .venv/bin/activate
```

Install the dependencies if they are not already installed:

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib
```

Start the dashboard from the project root:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Limitations and Future Work

- Evaluate forecasting with additional rolling-origin backtests and an untouched final test period.
- Investigate the business validity of exact duplicate removal.
- Verify that promotion features would be available at the time a forecast is made.
- Extend the platform with customer segmentation and churn prediction.
- Add inventory anomaly detection and inventory optimization.
- Introduce automated tests, model monitoring, drift detection, and deployment workflows.

## Disclaimer

Forecast metrics describe performance on the historical evaluation periods and should not be interpreted as guaranteed future accuracy. Results should be independently validated before use in retail operational decisions.