# RetailPulse — AI-Powered Retail Analytics & Demand Forecasting

RetailPulse is a retail analytics project that combines machine learning, historical sales analysis, customer analytics, inventory optimization, and data-quality checks in an interactive Streamlit dashboard.

**Live dashboard:** https://retailpulse-venu.streamlit.app
**GitHub repository:** https://github.com/mudikevenu/RetailPulse

## Project Objectives

- Analyze historical retail sales patterns.
- Forecast daily unit demand using machine learning.
- Compare forecasting models using consistent evaluation periods.
- Segment customers and analyze retention indicators and spending declines.
- Generate demand-based inventory recommendations.
- Identify data-quality issues and communicate limitations transparently.
- Present analytical results through an interactive dashboard.

## Dashboard Features

RetailPulse contains five main pages:

### 1. Dashboard
Provides a consolidated view of historical sales and key retail analytics.

### 2. Forecast
Displays predicted daily demand, forecast charts, prediction tables, downloadable results, model evaluation metrics, and quarterly backtesting comparisons.

### 3. Customers
Includes customer segmentation, retention monitoring, and historical spending-decline analysis. These indicators support customer analysis but should not be interpreted as confirmed churn predictions.

### 4. Inventory
Displays store–SKU inventory recommendations, stock-risk categories, suggested order quantities, and supporting summaries.

### 5. Data Quality
Reports automated data-quality checks, including missing columns, missing values, invalid values, sales-value mismatches, and exact duplicate-row findings.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy, DuckDB
- **Machine learning:** scikit-learn
- **Forecasting:** Random Forest and experimental LSTM/ensemble models
- **Visualization and dashboard:** Streamlit, Matplotlib
- **Model persistence:** joblib and Keras
- **Development and version control:** Jupyter Notebook, Git, GitHub

The presence of a model or library in the repository does not by itself establish production deployment, continuous retraining, or live monitoring.

## Dataset and Data Preparation

The historical sales data covers **January 1, 2022, through December 31, 2025**. The original transaction source contains approximately 9.97 million rows. DuckDB was used to process the large dataset and aggregate transaction-level data into a daily time series.

The processed daily sales dataset contains **1,461 daily records**.

### Data-quality findings

The automated report identifies:

- **9,972,038** source sales records scanned.
- **13,019** excess exact duplicate rows.
- **0** missing required columns.
- **0** missing values.
- **0** invalid values in the configured checks.
- **0** sales-value mismatches.

The estimated sales value associated with excess exact duplicate rows is **₹14,828,212.66**. This is a diagnostic estimate, not confirmed financial loss.

Exact duplicate rows are excluded from the forecasting aggregation. However, identical records may represent repeated legitimate transactions in some source systems. The duplicate findings should therefore be validated against authoritative transaction identifiers and source-system guidance before permanent deletion from the raw data.

## Demand Forecasting

The primary forecasting model is a Random Forest regressor using historical daily unit sales as its target. Features include calendar attributes, lagged sales, rolling averages, and promotion-related variables.

The repository also contains LSTM and ensemble forecasting experiments. These models are compared using saved evaluation artifacts; their existence does not imply that they are the model used by every dashboard workflow.

### Final 30-day holdout comparison

The saved final comparison evaluates the period **December 2–31, 2025**, using the same 30-day holdout for each listed model.

| Model | MAE (units/day) | RMSE (units/day) | MAPE |
|---|---:|---:|---:|
| Random Forest | 169.04 | 218.63 | 0.93% |
| Seasonal-naive | 574.97 | 672.10 | 3.14% |
| Prophet + LSTM ensemble | 991.50 | 1,263.53 | 5.42% |
| Prophet | 953.06 | 1,320.45 | 5.23% |
| Multivariate LSTM | 1,227.42 | 1,345.53 | 6.72% |

**Best model in this comparison:** Random Forest, based on the lowest MAE, RMSE, and MAPE among the evaluated candidates.

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **MAPE:** Mean Absolute Percentage Error.

These results apply to the recorded historical evaluation period. They do not guarantee future performance.

### Quarterly backtesting

The saved quarterly backtesting report covers seven expanding-window quarterly evaluations from Q1 2024 through Q3 2025.

| Model | Mean quarterly MAPE |
|---|---:|
| Random Forest | 3.26% |
| Seasonal-naive baseline | 5.24% |

The reported Random Forest mean quarterly MAPE is approximately 37.7% lower than the seasonal-naive baseline. This describes the evaluated periods, not guaranteed future accuracy.

### Forecast limitations

The dashboard's demonstration forecast begins on January 1, 2026, immediately after the available historical sales period ending December 31, 2025. It is a historical demonstration rather than a forecast updated through the current date.

Future promotion activity is set to zero in the demonstration forecast because a validated future promotion schedule is unavailable. Multi-day recursive predictions may differ from rolling one-day-ahead evaluation because later predictions use earlier predictions as lag inputs.

## Customer Analytics

### Customer segmentation
RetailPulse includes a saved customer segmentation pipeline and segment profiles for grouping customers by historical behavior and value-related features.

### Retention monitoring
Customer retention summaries provide activity and inactivity indicators. These are monitoring categories, not verified churn probabilities.

### Spending-decline analysis
The project includes a customer spending-decline model and associated evaluation artifacts. The target is a future spending decline, not customer churn. Its scores should be interpreted in the context of the evaluation dataset and its class imbalance.

## Inventory Optimization

Inventory recommendations operate on store–SKU pairs using historical sales and the available inventory snapshot.

The current recommendations use historical sales from **October 3 through December 31, 2025**, with a configurable demonstration target of **14 days of stock cover**.

Important limitations:

- The recommendations do not represent live inventory levels.
- Incoming purchase orders and confirmed deliveries are not included.
- The 14-day stock-cover target is a demonstration assumption, not an estimate of supplier lead time.
- Store–SKU pairs with no recent demand are flagged for review instead of automatically receiving demand-based order quantities.
- Suggested order quantities should be reviewed before operational use.

## Project Structure

```text
RetailPulse/
├── app.py
├── 01_eda.ipynb
├── requirements.txt
├── data/
│   └── processed/
│       ├── customer_retention_analysis.csv
│       ├── customer_segments.csv
│       ├── customer_spending_analysis.csv
│       ├── customer_spending_decline_model_dataset.csv
│       ├── daily_sales_with_promotion_features.csv
│       └── inventory_recommendations.csv
├── models/
│   ├── customer_segmentation_pipeline.joblib
│   ├── customer_spending_decline_logistic_regression.joblib
│   ├── customer_spending_decline_model.joblib
│   ├── customer_spending_decline_random_forest.joblib
│   ├── forecast_features.json
│   ├── lstm_demand_forecast.keras
│   ├── lstm_features.json
│   ├── lstm_multivariate_demand_forecast.keras
│   ├── lstm_multivariate_scalers.json
│   ├── lstm_demand_scaler.json
│   └── random_forest_demand_forecast.joblib
├── reports/
│   ├── customer_retention_summary.csv
│   ├── customer_segment_profiles.csv
│   ├── customer_spending_decline_model_metrics.csv
│   ├── customer_spending_decline_summary.csv
│   ├── data_quality_report.json
│   ├── final_forecast_model_comparison.csv
│   ├── forecast_model_comparison.csv
│   ├── forecast_model_evaluation.md
│   ├── inventory_optimization_summary.csv
│   ├── lstm_30_day_predictions.csv
│   ├── lstm_model_evaluation.csv
│   ├── lstm_multivariate_30_day_predictions.csv
│   ├── lstm_multivariate_evaluation.csv
│   ├── prophet_lstm_ensemble_evaluation.csv
│   ├── prophet_lstm_ensemble_predictions.csv
│   └── quarterly_forecast_backtest.csv
├── src/
│   ├── customer_retention.py
│   ├── customer_segmentation.py
│   ├── customer_spending_analysis.py
│   ├── customer_spending_decline_model.py
│   ├── data_quality_checks.py
│   ├── final_forecast_comparison.py
│   ├── forecast_ensemble.py
│   ├── inventory_optimization.py
│   └── lstm_demand_forecast.py
└── tests/
    ├── test_customer_retention.py
    ├── test_customer_segmentation.py
    ├── test_customer_spending_decline_model.py
    ├── test_data_quality_checks.py
    └── test_inventory_optimization.py
```


## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mudikevenu/RetailPulse.git
cd RetailPulse
```

### 2. Create and activate a virtual environment

Python 3.11 is the development environment used for this project.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

### 5. Run automated tests

```bash
python -m py_compile app.py
python -m pytest -q
```

The latest recorded test run passed all **31 tests**. Run the tests again after future changes to confirm the current result.

The existing dashboard uses processed datasets and saved model artifacts. The original large transaction dataset is not required to launch the current dashboard.

## Production Readiness and Future Work

The current project contains analytics modules, saved models, evaluation reports, automated tests, a Streamlit deployment, and Git-based version control.

The following capabilities should be considered separate production-readiness work unless independently implemented and verified:

- Automated ETL scheduling and recovery.
- Automated model retraining and model registry workflows.
- Live model and data-drift monitoring.
- CI/CD automation.
- Production container orchestration and cloud infrastructure.
- Real-time data ingestion and inventory integration.
- Forecast refresh using current sales and promotion data.
- Validation of duplicate transactions using authoritative identifiers.

## Disclaimer

RetailPulse is a data science and analytics project. Forecasts, customer indicators, and inventory recommendations are based on historical data and modeling assumptions. They are not guaranteed outcomes and should not be used as the sole basis for business decisions.
