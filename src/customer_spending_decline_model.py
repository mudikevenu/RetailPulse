from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SALES_FILE = (
    Path.home()
    / "Downloads"
    / "archive"
    / "retail_clean_dataset"
    / "sales_transactions.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "current_spend",
    "current_active_days",
    "current_receipts",
    "current_units",
    "current_recency_days",
    "spend_per_active_day",
    "spend_per_receipt",
]


def build_historical_dataset():
    """
    Build chronological six-month customer observations.

    Features are calculated only from the current six-month period.
    The target is calculated from the immediately following six-month
    period, so future information is not used as a predictor.
    """

    query = """
    WITH sales AS (
        SELECT
            customer_id,
            CAST(date AS DATE) AS sale_date,
            CAST(quantity AS DOUBLE) AS units,
            CAST(total_value AS DOUBLE) AS spend,
            receipt_id
        FROM read_csv_auto(?, header = true)
        WHERE customer_id IS NOT NULL
          AND date IS NOT NULL
          AND receipt_id IS NOT NULL
    ),

    periodized AS (
        SELECT
            customer_id,
            sale_date,
            units,
            spend,
            receipt_id,
            CASE
                WHEN sale_date BETWEEN DATE '2022-01-01'
                    AND DATE '2022-06-30' THEN '2022_H1'
                WHEN sale_date BETWEEN DATE '2022-07-01'
                    AND DATE '2022-12-31' THEN '2022_H2'
                WHEN sale_date BETWEEN DATE '2023-01-01'
                    AND DATE '2023-06-30' THEN '2023_H1'
                WHEN sale_date BETWEEN DATE '2023-07-01'
                    AND DATE '2023-12-31' THEN '2023_H2'
                WHEN sale_date BETWEEN DATE '2024-01-01'
                    AND DATE '2024-06-30' THEN '2024_H1'
                WHEN sale_date BETWEEN DATE '2024-07-01'
                    AND DATE '2024-12-31' THEN '2024_H2'
                WHEN sale_date BETWEEN DATE '2025-01-01'
                    AND DATE '2025-06-30' THEN '2025_H1'
                WHEN sale_date BETWEEN DATE '2025-07-01'
                    AND DATE '2025-12-31' THEN '2025_H2'
            END AS period
        FROM sales
    ),

    customer_periods AS (
        SELECT
            customer_id,
            period,
            SUM(spend) AS spend,
            SUM(units) AS units,
            COUNT(DISTINCT receipt_id) AS receipts,
            COUNT(DISTINCT sale_date) AS active_days,
            MAX(sale_date) AS last_purchase_date
        FROM periodized
        WHERE period IS NOT NULL
        GROUP BY customer_id, period
    ),

    period_features AS (
        SELECT
            *,
            CASE
                WHEN active_days > 0
                THEN spend / active_days
                ELSE 0
            END AS spend_per_active_day,

            CASE
                WHEN receipts > 0
                THEN spend / receipts
                ELSE 0
            END AS spend_per_receipt,

            CASE
                WHEN period LIKE '%H1'
                THEN DATE_TRUNC(
                    'year',
                    last_purchase_date
                ) + INTERVAL '6 months'
                ELSE DATE_TRUNC(
                    'year',
                    last_purchase_date
                ) + INTERVAL '1 year'
            END AS unused_period_marker

        FROM customer_periods
    )

    SELECT
        customer_id,

        period,

        spend AS current_spend,
        active_days AS current_active_days,
        receipts AS current_receipts,
        units AS current_units,

        DATE_DIFF(
            'day',
            last_purchase_date,
            CASE
                WHEN period LIKE '%H1'
                    THEN CASE
                        WHEN period = '2022_H1' THEN DATE '2022-06-30'
                        WHEN period = '2023_H1' THEN DATE '2023-06-30'
                        WHEN period = '2024_H1' THEN DATE '2024-06-30'
                        WHEN period = '2025_H1' THEN DATE '2025-06-30'
                    END
                ELSE CASE
                    WHEN period = '2022_H2' THEN DATE '2022-12-31'
                    WHEN period = '2023_H2' THEN DATE '2023-12-31'
                    WHEN period = '2024_H2' THEN DATE '2024-12-31'
                    WHEN period = '2025_H2' THEN DATE '2025-12-31'
                END
            END
        ) AS current_recency_days,

        spend_per_active_day,
        spend_per_receipt

    FROM period_features
    WHERE period IS NOT NULL
    """

    with duckdb.connect() as connection:
        current = connection.execute(
            query,
            [str(SALES_FILE)],
        ).fetchdf()

    period_order = [
        "2022_H1",
        "2022_H2",
        "2023_H1",
        "2023_H2",
        "2024_H1",
        "2024_H2",
        "2025_H1",
        "2025_H2",
    ]

    rows = []

    for i in range(len(period_order) - 1):
        current_period = period_order[i]
        future_period = period_order[i + 1]

        current_df = current[current["period"] == current_period].copy()
        future_df = current[current["period"] == future_period].copy()

        merged = current_df.merge(
            future_df[
                ["customer_id", "current_spend"]
            ].rename(
                columns={
                    "current_spend": "future_spend"
                }
            ),
            on="customer_id",
            how="inner",
        )

        merged["future_spending_change_pct"] = (
            100
            * (
                merged["future_spend"]
                - merged["current_spend"]
            )
            / merged["current_spend"]
        )

        merged["decline_target"] = (
            merged["future_spending_change_pct"] < -25
        ).astype(int)

        merged["observation_period"] = current_period
        merged["target_period"] = future_period

        rows.append(merged)

    dataset = pd.concat(rows, ignore_index=True)

    dataset = dataset[
        [
            "customer_id",
            "observation_period",
            "target_period",
            *FEATURE_COLUMNS,
            "future_spend",
            "future_spending_change_pct",
            "decline_target",
        ]
    ]

    return dataset


def precision_at_top_k(y_true, scores, fraction=0.20):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    k = max(1, int(np.ceil(len(y_true) * fraction)))

    order = np.argsort(-scores)
    top_targets = y_true[order[:k]]

    return float(top_targets.mean())


def main():
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {SALES_FILE}"
        )

    print("Building chronological customer decline dataset...")

    dataset = build_historical_dataset()

    dataset.to_csv(
        OUTPUT_DIR / "customer_spending_decline_model_dataset.csv",
        index=False,
    )

    train_periods = [
        "2022_H1",
        "2022_H2",
        "2023_H1",
        "2023_H2",
        "2024_H1",
        "2024_H2",
    ]

    test_period = "2025_H1"

    train = dataset[
        dataset["observation_period"].isin(train_periods)
    ].copy()

    test = dataset[
        dataset["observation_period"] == test_period
    ].copy()

    X_train = train[FEATURE_COLUMNS]
    y_train = train["decline_target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["decline_target"]

    print()
    print("===== DATASET =====")
    print(f"Total observations: {len(dataset):,}")
    print(f"Training observations: {len(train):,}")
    print(f"Test observations: {len(test):,}")
    print(
        f"Training positive rate: {y_train.mean() * 100:.2f}%"
    )
    print(
        f"Test positive rate: {y_test.mean() * 100:.2f}%"
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        print()
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        precision20 = precision_at_top_k(
            y_test,
            probabilities,
            fraction=0.20,
        )

        results.append(
            {
                "model": name,
                "test_period": test_period,
                "roc_auc": round(float(auc), 4),
                "precision_at_top_20_pct": round(
                    float(precision20),
                    4,
                ),
                "test_positive_rate": round(
                    float(y_test.mean()),
                    4,
                ),
            }
        )

        joblib.dump(
            model,
            MODEL_DIR / f"customer_spending_decline_{name}.joblib",
        )

        print(f"ROC-AUC: {auc:.4f}")
        print(
            f"Precision@Top-20%: "
            f"{precision20:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "customer_spending_decline_model_metrics.csv",
        index=False,
    )

    best_model_name = results_df.loc[
        results_df["roc_auc"].idxmax(),
        "model",
    ]

    best_model = models[best_model_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "customer_spending_decline_model.joblib",
    )

    print()
    print("===== MODEL RESULTS =====")
    print(results_df.to_string(index=False))

    print()
    print(
        "Best model by test ROC-AUC:",
        best_model_name,
    )

    print()
    print("Saved:")
    print(
        OUTPUT_DIR
        / "customer_spending_decline_model_dataset.csv"
    )
    print(
        REPORT_DIR
        / "customer_spending_decline_model_metrics.csv"
    )
    print(
        MODEL_DIR
        / "customer_spending_decline_model.joblib"
    )


if __name__ == "__main__":
    main()
