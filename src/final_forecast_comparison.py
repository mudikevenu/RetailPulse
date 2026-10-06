from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data/processed/daily_sales_with_promotion_features.csv"
RF_MODEL_PATH = BASE / "models/random_forest_demand_forecast.joblib"
FEATURE_PATH = BASE / "models/forecast_features.json"

LSTM_PRED_PATH = BASE / "reports/lstm_multivariate_30_day_predictions.csv"
ENSEMBLE_PRED_PATH = BASE / "reports/prophet_lstm_ensemble_predictions.csv"

RESULT_PATH = BASE / "reports/final_forecast_model_comparison.csv"


print("=" * 70)
print("RETAILPULSE FINAL FORECAST MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

TARGET = "units_sold"
TEST_DAYS = 30

test_start = df["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)

train = df[df["date"] < test_start].copy()
test = df[df["date"] >= test_start].copy()

print(f"Dataset: {DATA_PATH}")
print(f"Train: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Test observations: {len(test)}")


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0

    mape = np.mean(
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        )
    ) * 100

    return mae, rmse, mape


results = []


# ---------------------------------------------------------------------
# 1. Seasonal-naive baseline
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. SEASONAL-NAIVE BASELINE")
print("=" * 70)

history = df.set_index("date")[TARGET]

baseline_predictions = []

for date in test["date"]:
    previous_year = date - pd.DateOffset(years=1)

    if previous_year in history.index:
        baseline_predictions.append(history.loc[previous_year])
    else:
        baseline_predictions.append(np.nan)

baseline_predictions = np.asarray(baseline_predictions, dtype=float)

valid = ~np.isnan(baseline_predictions)

mae, rmse, mape = metrics(
    test.loc[valid, TARGET],
    baseline_predictions[valid]
)

results.append(
    {
        "model": "Seasonal-naive baseline",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }
)


# ---------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. RANDOM FOREST")
print("=" * 70)

rf_model = joblib.load(RF_MODEL_PATH)

import json

with open(FEATURE_PATH, "r") as f:
    rf_features = json.load(f)

rf_df = df.copy()

rf_df["day_of_week"] = rf_df["date"].dt.dayofweek
rf_df["day_of_month"] = rf_df["date"].dt.day
rf_df["month"] = rf_df["date"].dt.month
rf_df["quarter"] = rf_df["date"].dt.quarter
rf_df["is_weekend"] = (rf_df["date"].dt.dayofweek >= 5).astype(int)
rf_df["year"] = rf_df["date"].dt.year

rf_df["units_lag_1"] = rf_df[TARGET].shift(1)
rf_df["units_lag_7"] = rf_df[TARGET].shift(7)
rf_df["units_lag_364"] = rf_df[TARGET].shift(364)

rf_df["units_rolling_mean_7"] = (
    rf_df[TARGET]
    .shift(1)
    .rolling(7)
    .mean()
)

rf_df["active_promotion_count"] = (
    rf_df["active_promotion_count"].fillna(0)
)

rf_df["active_all_target_promotions"] = (
    rf_df["active_all_target_promotions"].fillna(0)
)

rf_df["active_targeted_promotions"] = (
    rf_df["active_targeted_promotions"].fillna(0)
)

rf_df["max_scheduled_discount_pct"] = (
    rf_df["max_scheduled_discount_pct"].fillna(0)
)

rf_test = rf_df[rf_df["date"] >= test_start].copy()

rf_test = rf_test.dropna(subset=rf_features)

rf_predictions = rf_model.predict(
    rf_test[rf_features]
)

rf_actual = rf_test[TARGET].to_numpy()

mae, rmse, mape = metrics(
    rf_actual,
    rf_predictions
)

results.append(
    {
        "model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }
)


# ---------------------------------------------------------------------
# 3. Prophet
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. PROPHET")
print("=" * 70)

prophet_train = train[["date", TARGET]].rename(
    columns={
        "date": "ds",
        TARGET: "y",
    }
)

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

prophet_model.fit(prophet_train)

future = test[["date"]].rename(columns={"date": "ds"})

prophet_forecast = prophet_model.predict(future)

prophet_predictions = prophet_forecast["yhat"].to_numpy()

mae, rmse, mape = metrics(
    test[TARGET],
    prophet_predictions
)

results.append(
    {
        "model": "Prophet",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }
)


# ---------------------------------------------------------------------
# 4. Multivariate LSTM
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MULTIVARIATE LSTM")
print("=" * 70)

lstm_predictions = pd.read_csv(
    LSTM_PRED_PATH,
    parse_dates=["date"]
)

lstm_predictions = test[["date", TARGET]].merge(
    lstm_predictions[
        ["date", "predicted_units"]
    ],
    on="date",
    how="left"
)

if lstm_predictions["predicted_units"].isna().any():
    raise RuntimeError(
        "Missing multivariate LSTM predictions."
    )

mae, rmse, mape = metrics(
    lstm_predictions[TARGET],
    lstm_predictions["predicted_units"]
)

results.append(
    {
        "model": "Multivariate LSTM",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }
)


# ---------------------------------------------------------------------
# 5. Prophet + LSTM Ensemble
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. PROPHET + LSTM ENSEMBLE")
print("=" * 70)

ensemble_predictions = pd.read_csv(
    ENSEMBLE_PRED_PATH,
    parse_dates=["date"]
)

ensemble_predictions = test[["date", TARGET]].merge(
    ensemble_predictions[
        ["date", "ensemble_pred"]
    ],
    on="date",
    how="left"
)

if ensemble_predictions["ensemble_pred"].isna().any():
    raise RuntimeError(
        "Missing ensemble predictions."
    )

mae, rmse, mape = metrics(
    ensemble_predictions[TARGET],
    ensemble_predictions["ensemble_pred"]
)

results.append(
    {
        "model": "Prophet + LSTM Ensemble",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    }
)


# ---------------------------------------------------------------------
# Final comparison
# ---------------------------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "MAPE_percent"
).reset_index(drop=True)

comparison.insert(
    0,
    "rank",
    range(1, len(comparison) + 1)
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

best = comparison.iloc[0]

print("\n" + "=" * 70)
print("WINNER")
print("=" * 70)

print(f"Model: {best['model']}")
print(f"MAPE:  {best['MAPE_percent']:.4f}%")
print(f"MAE:   {best['MAE']:.4f}")
print(f"RMSE:  {best['RMSE']:.4f}")

comparison.to_csv(
    RESULT_PATH,
    index=False
)

print("\nSaved:")
print(" -", RESULT_PATH)

print("\nFinal forecasting comparison completed successfully.")
