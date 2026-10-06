from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prophet import Prophet
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data/processed/daily_sales_with_promotion_features.csv"
LSTM_MODEL_PATH = BASE / "models/lstm_multivariate_demand_forecast.keras"
SCALER_PATH = BASE / "models/lstm_multivariate_scalers.json"
FEATURE_PATH = BASE / "models/lstm_features.json"

RESULT_PATH = BASE / "reports/prophet_lstm_ensemble_predictions.csv"
EVAL_PATH = BASE / "reports/prophet_lstm_ensemble_evaluation.csv"


print("=" * 70)
print("RETAILPULSE PROPHET + MULTIVARIATE LSTM ENSEMBLE")
print("=" * 70)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset: {DATA_PATH}")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

TARGET = "units_sold"
TEST_DAYS = 30
test_start = df["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)

train = df[df["date"] < test_start].copy()
test = df[df["date"] >= test_start].copy()

print(f"Train observations: {len(train)}")
print(f"Test observations: {len(test)}")
print(f"Test period: {test['date'].min().date()} to {test['date'].max().date()}")

# ---------------------------------------------------------------------
# 1. Prophet
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING PROPHET")
print("=" * 70)

prophet_train = train[["date", TARGET]].rename(
    columns={"date": "ds", TARGET: "y"}
)

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
)

prophet_model.fit(prophet_train)

prophet_future = test[["date"]].rename(columns={"date": "ds"})
prophet_forecast = prophet_model.predict(prophet_future)

prophet_pred = prophet_forecast["yhat"].to_numpy()

# ---------------------------------------------------------------------
# 2. Load multivariate LSTM
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOADING MULTIVARIATE LSTM")
print("=" * 70)

lstm_model = load_model(LSTM_MODEL_PATH)

with open(SCALER_PATH, "r") as f:
    scaler_info = json.load(f)

with open(FEATURE_PATH, "r") as f:
    lstm_features = json.load(f)

print(f"LSTM features: {len(lstm_features)}")
print("LSTM model loaded successfully.")

# ---------------------------------------------------------------------
# 3. Reproduce LSTM test predictions from saved prediction artifact
# ---------------------------------------------------------------------

lstm_prediction_path = BASE / "reports/lstm_multivariate_30_day_predictions.csv"

lstm_predictions = pd.read_csv(lstm_prediction_path, parse_dates=["date"])

lstm_predictions = lstm_predictions[
    ["date", "predicted_units"]
].copy()

lstm_predictions = lstm_predictions.sort_values("date")

merged = test[["date", TARGET]].merge(
    lstm_predictions,
    on="date",
    how="left"
)

if merged["predicted_units"].isna().any():
    raise RuntimeError("Missing LSTM predictions for some test dates.")

lstm_pred = merged["predicted_units"].to_numpy()

# ---------------------------------------------------------------------
# 4. Align Prophet and LSTM predictions
# ---------------------------------------------------------------------

merged["prophet_pred"] = prophet_pred
merged["lstm_pred"] = lstm_pred

# Simple equal-weight ensemble.
merged["ensemble_pred"] = (
    0.5 * merged["prophet_pred"]
    + 0.5 * merged["lstm_pred"]
)

# ---------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------

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


models = {
    "Prophet": merged["prophet_pred"],
    "Multivariate LSTM": merged["lstm_pred"],
    "Prophet + LSTM Ensemble": merged["ensemble_pred"],
}

evaluation_rows = []

for model_name, predictions in models.items():
    mae, rmse, mape = metrics(
        merged[TARGET],
        predictions
    )

    evaluation_rows.append(
        {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape,
        }
    )

evaluation = pd.DataFrame(evaluation_rows)

# ---------------------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION")
print("=" * 70)

print(
    evaluation.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n30-day predictions:")
print(
    merged[
        [
            "date",
            TARGET,
            "prophet_pred",
            "lstm_pred",
            "ensemble_pred",
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

merged[
    [
        "date",
        TARGET,
        "prophet_pred",
        "lstm_pred",
        "ensemble_pred",
    ]
].to_csv(RESULT_PATH, index=False)

evaluation.to_csv(EVAL_PATH, index=False)

print("\nSaved:")
print(" -", RESULT_PATH)
print(" -", EVAL_PATH)

best_model = evaluation.loc[
    evaluation["MAPE_percent"].idxmin(),
    "model"
]

best_mape = evaluation.loc[
    evaluation["MAPE_percent"].idxmin(),
    "MAPE_percent"
]

print("\nBest model:")
print(f" - {best_model}")
print(f" - MAPE: {best_mape:.4f}%")

print("\nProphet + LSTM ensemble evaluation completed successfully.")
