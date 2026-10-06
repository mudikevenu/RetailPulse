import joblib
import numpy as np
import pandas as pd

from src.customer_spending_decline_model import (
    FEATURE_COLUMNS,
    precision_at_top_k,
)


MODEL_PATH = "models/customer_spending_decline_model.joblib"
DATASET_PATH = (
    "data/processed/"
    "customer_spending_decline_model_dataset.csv"
)


def test_decline_dataset_exists_and_has_expected_columns():
    df = pd.read_csv(DATASET_PATH)

    expected_columns = {
        "customer_id",
        "observation_period",
        "target_period",
        *FEATURE_COLUMNS,
        "future_spend",
        "future_spending_change_pct",
        "decline_target",
    }

    assert expected_columns.issubset(df.columns)
    assert len(df) == 70_000


def test_decline_target_is_binary():
    df = pd.read_csv(DATASET_PATH)

    assert set(df["decline_target"].unique()).issubset({0, 1})


def test_decline_target_matches_spending_change():
    df = pd.read_csv(DATASET_PATH)

    expected_target = (
        df["future_spending_change_pct"] < -25
    ).astype(int)

    assert np.array_equal(
        df["decline_target"].to_numpy(),
        expected_target.to_numpy(),
    )


def test_model_features_have_no_missing_values():
    df = pd.read_csv(DATASET_PATH)

    assert df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_precision_at_top_k():
    y_true = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.95, 0.90, 0.80, 0.20, 0.10])

    precision = precision_at_top_k(
        y_true,
        scores,
        fraction=0.40,
    )

    assert precision == 1.0


def test_saved_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict_proba")
