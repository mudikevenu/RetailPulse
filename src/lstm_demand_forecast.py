from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
LOOKBACK = 30
TEST_DAYS = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_sales_with_promotion_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_multivariate_demand_forecast.keras"
SCALER_PATH = MODEL_DIR / "lstm_multivariate_scalers.json"
FEATURE_PATH = MODEL_DIR / "lstm_features.json"
RESULT_PATH = REPORT_DIR / "lstm_multivariate_evaluation.csv"
PREDICTION_PATH = REPORT_DIR / "lstm_multivariate_30_day_predictions.csv"


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Load dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

print("=" * 70)
print("RETAILPULSE MULTIVARIATE LSTM DEMAND FORECAST")
print("=" * 70)
print("Dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

required_columns = [
    "date",
    "units_sold",
    "active_promotion_count",
    "active_all_target_promotions",
    "active_targeted_promotions",
    "max_scheduled_discount_pct",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# Prepare daily dataset
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df["units_sold"] = pd.to_numeric(
    df["units_sold"],
    errors="coerce",
)

for column in required_columns[2:]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)

df = df.dropna(
    subset=["date", "units_sold"]
)

df = (
    df.groupby("date", as_index=False)
    .agg(
        {
            "units_sold": "sum",
            "active_promotion_count": "max",
            "active_all_target_promotions": "max",
            "active_targeted_promotions": "max",
            "max_scheduled_discount_pct": "max",
        }
    )
    .sort_values("date")
    .reset_index(drop=True)
)

print("\nDataset shape:", df.shape)
print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date(),
)


# ============================================================
# Verify daily continuity
# ============================================================

full_dates = pd.date_range(
    df["date"].min(),
    df["date"].max(),
    freq="D",
)

missing_dates = full_dates.difference(
    df["date"]
)

print(
    "Missing calendar dates:",
    len(missing_dates)
)

if len(missing_dates) > 0:
    print("WARNING: Missing dates detected.")
    print(missing_dates[:10])


# ============================================================
# Feature engineering
# ============================================================
#
# IMPORTANT:
# All demand-derived features are lagged.
# Therefore, today's prediction never sees today's actual demand.
# ============================================================

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = (
    df["date"].dt.dayofweek >= 5
).astype(int)

# Historical demand only.
df["units_lag_1"] = df["units_sold"].shift(1)
df["units_lag_7"] = df["units_sold"].shift(7)
df["units_lag_14"] = df["units_sold"].shift(14)
df["units_lag_28"] = df["units_sold"].shift(28)

df["units_rolling_mean_7"] = (
    df["units_sold"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["units_rolling_mean_14"] = (
    df["units_sold"]
    .shift(1)
    .rolling(14)
    .mean()
)

df["units_rolling_mean_30"] = (
    df["units_sold"]
    .shift(1)
    .rolling(30)
    .mean()
)

FEATURES = [
    "units_sold",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "active_promotion_count",
    "active_all_target_promotions",
    "active_targeted_promotions",
    "max_scheduled_discount_pct",
    "units_lag_1",
    "units_lag_7",
    "units_lag_14",
    "units_lag_28",
    "units_rolling_mean_7",
    "units_rolling_mean_14",
    "units_rolling_mean_30",
]

TARGET = "units_sold"

df = df.dropna(
    subset=FEATURES
).reset_index(drop=True)

print("\nFeatures:")
for feature in FEATURES:
    print(" -", feature)

with open(FEATURE_PATH, "w") as f:
    json.dump(FEATURES, f, indent=2)


# ============================================================
# Train / test split
# ============================================================

test_start = len(df) - TEST_DAYS

train_df = df.iloc[:test_start].copy()
test_df = df.iloc[test_start:].copy()

print("\nTrain observations:", len(train_df))
print("Test observations:", len(test_df))

print(
    "Test period:",
    test_df["date"].min().date(),
    "to",
    test_df["date"].max().date(),
)


# ============================================================
# Scale features using TRAINING data only
# ============================================================

feature_scaler = MinMaxScaler()

train_features = feature_scaler.fit_transform(
    train_df[FEATURES]
)

# Target scaler is the same feature scaler because
# units_sold is the first feature.
target_min = feature_scaler.data_min_[0]
target_max = feature_scaler.data_max_[0]
target_scale = feature_scaler.scale_[0]
target_offset = feature_scaler.min_[0]

# Transform the full dataset using training parameters.
all_features_scaled = feature_scaler.transform(
    df[FEATURES]
)

with open(SCALER_PATH, "w") as f:
    json.dump(
        {
            "features": FEATURES,
            "data_min": feature_scaler.data_min_.tolist(),
            "data_max": feature_scaler.data_max_.tolist(),
            "scale": feature_scaler.scale_.tolist(),
            "min_offset": feature_scaler.min_.tolist(),
        },
        f,
        indent=2,
    )


# ============================================================
# Create sequences
# ============================================================

def create_sequences(
    values,
    target_index,
    lookback,
    start_index,
    end_index,
):
    X = []
    y = []

    for i in range(
        max(lookback, start_index),
        end_index,
    ):
        X.append(
            values[i - lookback:i]
        )
        y.append(
            values[i, target_index]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


target_index = FEATURES.index(TARGET)

X_all, y_all = create_sequences(
    all_features_scaled,
    target_index,
    LOOKBACK,
    0,
    test_start,
)

print(
    "\nTraining sequence shape:",
    X_all.shape,
)

print(
    "Training target shape:",
    y_all.shape,
)


# ============================================================
# Chronological validation split
# ============================================================

validation_size = max(
    30,
    int(len(X_all) * 0.15),
)

X_fit = X_all[:-validation_size]
y_fit = y_all[:-validation_size]

X_val = X_all[-validation_size:]
y_val = y_all[-validation_size:]

print(
    "Fit sequences:",
    X_fit.shape,
)

print(
    "Validation sequences:",
    X_val.shape,
)


# ============================================================
# Build model
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(
                LOOKBACK,
                len(FEATURES),
            )
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(
            32,
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16,
            activation="relu",
        ),

        tf.keras.layers.Dense(1),
    ],
    name="retailpulse_multivariate_lstm",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"],
)

print("\nModel architecture:")
model.summary()


# ============================================================
# Train
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    ),
]

print("\nTraining multivariate LSTM...")

model.fit(
    X_fit,
    y_fit,
    validation_data=(
        X_val,
        y_val,
    ),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# Recursive 30-day forecasting
# ============================================================
#
# For each future day:
# - known calendar/promotion features come from the dataset
# - demand-derived features are updated recursively
# - predicted demand becomes the next lagged demand value
#
# This prevents actual future units_sold from leaking into forecasts.
# ============================================================

history = df.iloc[
    :test_start
].copy()

predictions = []

for step in range(TEST_DAYS):

    future_row = test_df.iloc[step].copy()

    combined = pd.concat(
        [
            history,
            pd.DataFrame(
                [future_row]
            ),
        ],
        ignore_index=True,
    )

    # Recalculate demand-derived features.
    combined["units_lag_1"] = (
        combined["units_sold"].shift(1)
    )

    combined["units_lag_7"] = (
        combined["units_sold"].shift(7)
    )

    combined["units_lag_14"] = (
        combined["units_sold"].shift(14)
    )

    combined["units_lag_28"] = (
        combined["units_sold"].shift(28)
    )

    combined["units_rolling_mean_7"] = (
        combined["units_sold"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    combined["units_rolling_mean_14"] = (
        combined["units_sold"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    combined["units_rolling_mean_30"] = (
        combined["units_sold"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    current_features = combined[
        FEATURES
    ].iloc[-LOOKBACK:]

    current_scaled = feature_scaler.transform(
        current_features
    )

    X_input = current_scaled.reshape(
        1,
        LOOKBACK,
        len(FEATURES),
    ).astype(np.float32)

    prediction_scaled = model.predict(
        X_input,
        verbose=0,
    )[0, 0]

    prediction = (
        prediction_scaled - target_offset
    ) / target_scale

    prediction = max(
        0.0,
        float(prediction),
    )

    predictions.append(prediction)

    # Add prediction to history for recursive lags.
    predicted_row = future_row.copy()
    predicted_row["units_sold"] = prediction

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [predicted_row]
            ),
        ],
        ignore_index=True,
    )


# ============================================================
# Evaluation
# ============================================================

actual = test_df[
    "units_sold"
].to_numpy(
    dtype=float
)

predictions = np.asarray(
    predictions,
    dtype=float,
)

mae = mean_absolute_error(
    actual,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions,
    )
)

nonzero_mask = actual != 0

if nonzero_mask.any():
    mape = (
        np.mean(
            np.abs(
                (
                    actual[nonzero_mask]
                    - predictions[nonzero_mask]
                )
                / actual[nonzero_mask]
            )
        )
        * 100
    )
else:
    mape = np.nan


# ============================================================
# Save predictions
# ============================================================

prediction_df = pd.DataFrame(
    {
        "date": test_df["date"].values,
        "actual_units": actual,
        "predicted_units": predictions,
    }
)

prediction_df["absolute_error"] = (
    np.abs(
        prediction_df["actual_units"]
        - prediction_df["predicted_units"]
    )
)

prediction_df["absolute_percentage_error"] = (
    prediction_df["absolute_error"]
    / prediction_df["actual_units"].replace(
        0,
        np.nan,
    )
    * 100
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False,
)


# ============================================================
# Save evaluation
# ============================================================

evaluation = pd.DataFrame(
    [
        {
            "model": "Multivariate LSTM",
            "lookback_days": LOOKBACK,
            "test_days": TEST_DAYS,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "train_observations": len(train_df),
            "test_observations": len(test_df),
            "feature_count": len(FEATURES),
        }
    ]
)

evaluation.to_csv(
    RESULT_PATH,
    index=False,
)


# ============================================================
# Save model
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("MULTIVARIATE LSTM EVALUATION")
print("=" * 70)

print(
    f"MAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAPE: {mape:.4f}%"
)

print("\n30-day forecast:")

print(
    prediction_df[
        [
            "date",
            "actual_units",
            "predicted_units",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved:")
print(" -", MODEL_PATH)
print(" -", SCALER_PATH)
print(" -", FEATURE_PATH)
print(" -", RESULT_PATH)
print(" -", PREDICTION_PATH)

print(
    "\nMultivariate LSTM training completed successfully."
)
