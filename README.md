# RetailPulse — Retail Demand Forecasting & Analytics

RetailPulse is a data science project that uses historical retail sales data and machine learning to forecast daily demand. It includes a Streamlit dashboard for exploring historical sales, reviewing model performance, and viewing future demand forecasts.

**Live dashboard:** https://retailpulse-venu.streamlit.app
**GitHub repository:** https://github.com/mudikevenu/RetailPulse

## Project Objectives

- Analyze historical retail sales patterns.
- Build a machine learning model to forecast daily unit demand.
- Compare model performance against a seasonal-naive baseline.
- Present historical data, forecasts, and evaluation results through an interactive dashboard.
- Document data-quality decisions, evaluation methodology, and limitations.

## Features

- Customer segmentation and retention analysis.
- Inventory optimization with stockout alerts, demand-based replenishment suggestions, and downloadable recommendations.
- Historical daily sales visualization.
- Random Forest demand forecasting.
- Calendar, lagged-sales, rolling-average, and promotion-related features.
- Comparison with a seasonal-naive forecasting baseline.
- Quarterly expanding-window backtesting.
- Interactive forecast dashboard built with Streamlit.
- Forecast evaluation metrics and downloadable results, where available in the dashboard.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy, DuckDB
- **Machine learning:** scikit-learn
- **Forecasting:** Random Forest regression
- **Visualization and dashboard:** Matplotlib, Streamlit
- **Model persistence:** joblib
- **Development:** Jupyter Notebook, Git, GitHub

Prophet is included in the project environment, but the reported model results in this README are for the Random Forest model and seasonal-naive baseline.

## Dataset and Data Preparation

The project uses retail transaction data covering **January 1, 2022, through December 31, 2025**.

The original transaction dataset contains approximately 9.97 million rows. DuckDB was used to process the large dataset without loading the entire file into pandas.

### Data-quality handling

- Checked transaction fields and date coverage.
- Validated the relationship between quantity, unit price, discount, and transaction value.
- Identified exact full-row duplicate records.
- Removed exact full-row duplicates for the daily aggregation used in forecasting.
- Aggregated transactions into a daily sales time series.
- Created calendar, lag, rolling-average, and promotion-related features.

After exact full-row deduplication, the processed data contains **9,959,019 transaction rows** and **1,461 daily records**.

Duplicate removal is a data-preparation assumption: identical rows may sometimes represent legitimate repeated transactions. The deduplication decision should therefore be reviewed if authoritative transaction IDs or source-system guidance become available.

The resulting daily dataset contains **18,733,005 units sold** across the four-year period.

## Forecasting Methodology

The forecasting workflow uses daily unit sales as its target.

The Random Forest model uses features such as:

- Calendar information.
- Previous-day and other lagged sales.
- Rolling averages of historical sales.
- Promotion-related features.
- Year information.

The model is compared with a seasonal-naive baseline, which uses sales from the corresponding seasonal period in the past.

### Evaluation approach

The reported final-period evaluation covers **October–December 2025** using rolling one-day-ahead predictions. For each prediction, lag features use actual sales observed before that forecast date.

This is different from forecasting multiple future days recursively, where earlier predictions are used to construct lag features for later predictions.

## Model Evaluation Results

### October–December 2025

| Metric | Random Forest | Seasonal-naive baseline |
|---|---:|---:|
| MAE | 411.93 units/day | 515.37 units/day |
| RMSE | 509.59 units/day | 623.17 units/day |
| MAPE | 2.55% | 3.24% |

**Metric definitions**

- **MAE:** Mean Absolute Error; the average absolute difference between actual and predicted daily units.
- **RMSE:** Root Mean Squared Error; a metric that penalizes larger errors more heavily.
- **MAPE:** Mean Absolute Percentage Error; the average absolute percentage error.

### Quarterly backtesting

Across seven expanding-window quarterly backtests from **Q1 2024 through Q3 2025**, the recorded mean quarterly MAPE was:

| Model | Mean quarterly MAPE |
|---|---:|
| Random Forest | 3.26% |
| Seasonal-naive baseline | 5.24% |

The Random Forest model had lower MAPE than the baseline in each of the seven evaluated quarters. These results describe the evaluated historical periods and do not guarantee future performance.

The October–December 2025 period was used during model/configuration selection, so it should not be considered a completely untouched final holdout.

## Streamlit Dashboard

The dashboard presents:

- Historical daily sales.
- Forecasted demand.
- Model evaluation metrics.
- Quarterly model comparison results.
- Supporting tables and forecast outputs.

The demonstration forecast is configured to start on **January 1, 2026**, immediately after the available historical sales period.

## Project Structure

```text
RetailPulse/
├── 01_eda.ipynb
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── processed/
│       ├── daily_sales_with_promotion_features.csv
│       └── inventory_recommendations.csv
├── models/
│   ├── forecast_features.json
│   └── random_forest_demand_forecast.joblib
├── reports/
│   ├── forecast_model_comparison.csv
│   ├── forecast_model_evaluation.md
│   ├── quarterly_forecast_backtest.csv
│   └── inventory_optimization_summary.csv
├── src/
│   ├── customer_retention.py
│   ├── customer_segmentation.py
│   ├── customer_spending_analysis.py
│   └── inventory_optimization.py
└── tests/
    ├── test_customer_retention.py
    ├── test_customer_segmentation.py
    └── test_inventory_optimization.py
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mudikevenu/RetailPulse.git
cd RetailPulse
```

### 2. Create a virtual environment

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

Open the local URL displayed in your Terminal, usually:

`http://localhost:8501`

The dashboard uses the processed data and saved model included in the repository. The original large transaction dataset is not required to launch the existing dashboard.

## Limitations and Future Improvements

- Forecast accuracy may change as sales patterns and customer behavior change.
- Random Forest predictions may smooth sharp seasonal peaks.
- Historical evaluation results do not guarantee future accuracy.
- Recursive multi-day forecasts can differ from rolling one-day-ahead evaluation results.
- The available promotion schedule ends in November 2025. Future promotion activity is set to zero in the demonstration forecast because a future schedule is unavailable.
- The displayed forecast begins in January 2026 and is a historical demonstration, not a forecast updated through the current date.
- Exact-row deduplication should be reviewed against authoritative transaction identifiers where available.
- Additional work could include a validated churn-prediction model when suitable labels become available, automated model monitoring, model retraining, and deployment improvements.
- Inventory recommendations use historical sales from October 3 through December 31, 2025, and the inventory snapshot available for this project. They do not represent live inventory levels.
- The 14-day target stock cover is a configurable demonstration assumption, not an estimate of supplier lead time. Suggested order quantities do not account for incoming purchase orders or confirmed deliveries and should be reviewed before operational use.
- Store–SKU pairs with no sales during the 90-day analysis window are flagged for review rather than automatically assigned a demand-based order quantity.

## Disclaimer

RetailPulse is a data science project intended for analytical and educational use. Forecasts are estimates based on historical data and modeling assumptions. They should not be treated as guaranteed sales outcomes or used as the sole basis for business decisions.
