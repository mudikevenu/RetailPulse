
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# RetailPulse project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path.home() / "Downloads" / "archive" / "retail_clean_dataset"

SALES_FILE = RAW_DATA_DIR / "sales_transactions.csv"
CUSTOMERS_FILE = RAW_DATA_DIR / "customer_master.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

N_CLUSTERS = 4
RANDOM_STATE = 42

FEATURES = [
    "log_recency",
    "log_frequency",
    "log_monetary",
]


def build_customer_features() -> pd.DataFrame:
    """Calculate customer purchasing features using DuckDB."""

    if not SALES_FILE.exists():
        raise FileNotFoundError(f"Sales file not found: {SALES_FILE}")

    if not CUSTOMERS_FILE.exists():
        raise FileNotFoundError(
            f"Customer master file not found: {CUSTOMERS_FILE}"
        )

    sales_path = str(SALES_FILE)
    customer_path = str(CUSTOMERS_FILE)

    # Aggregate the large transaction file inside DuckDB.
    # This avoids loading all transaction rows into pandas.
    con = duckdb.connect()

    try:
        sales = con.execute(
            """
            SELECT
                customer_id AS cust_id,
                MAX(CAST(date AS DATE)) AS last_purchase_date,
                COUNT(DISTINCT receipt_id) AS frequency,
                SUM(CAST(total_value AS DOUBLE)) AS monetary,
                SUM(CAST(quantity AS DOUBLE)) AS total_units,
                MAX(CAST(date AS DATE)) AS dataset_end_date
            FROM read_csv_auto(?, header = true)
            WHERE customer_id IS NOT NULL
            GROUP BY customer_id
            """,
            [sales_path],
        ).fetchdf()
    finally:
        con.close()

    sales["last_purchase_date"] = pd.to_datetime(
        sales["last_purchase_date"]
    )
    sales["dataset_end_date"] = pd.to_datetime(
        sales["dataset_end_date"]
    )

    # Use the dataset's last date + 1 day as the reference date.
    # This makes recency reproducible and avoids mixing in today's date.
    reference_date = sales["dataset_end_date"].max() + pd.Timedelta(days=1)

    sales["recency_days"] = (
        reference_date - sales["last_purchase_date"]
    ).dt.days

    sales = sales.drop(columns=["dataset_end_date"])

    customers = pd.read_csv(
        CUSTOMERS_FILE,
        dtype={"cust_id": "string"},
    )
    customers["cust_id"] = customers["cust_id"].astype("string")
    sales["cust_id"] = sales["cust_id"].astype("string")

    # Keep customers who have no recorded purchases, if any.
    features = customers.merge(
        sales,
        on="cust_id",
        how="left",
        validate="one_to_one",
    )

    features["frequency"] = features["frequency"].fillna(0)
    features["monetary"] = features["monetary"].fillna(0)
    features["total_units"] = features["total_units"].fillna(0)

    # A customer with no purchases is assigned a recency beyond the
    # observed history, so they remain in the segmentation dataset.
    max_observed_recency = int(sales["recency_days"].max())
    features["recency_days"] = (
        features["recency_days"].fillna(max_observed_recency + 1)
    )

    numeric_columns = [
        "recency_days",
        "frequency",
        "monetary",
        "total_units",
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    features[numeric_columns] = (
        features[numeric_columns].replace([np.inf, -np.inf], np.nan)
    )

    features = features.dropna(subset=numeric_columns).copy()

    # Log transforms reduce the effect of very large customer values.
    features["log_recency"] = np.log1p(
        features["recency_days"].clip(lower=0)
    )
    features["log_frequency"] = np.log1p(
        features["frequency"].clip(lower=0)
    )
    features["log_monetary"] = np.log1p(
        features["monetary"].clip(lower=0)
    )

    if len(features) < N_CLUSTERS:
        raise ValueError(
            f"Need at least {N_CLUSTERS} customers to create clusters."
        )

    return features


def run_segmentation() -> pd.DataFrame:
    """Fit the segmentation model and save results and cluster profiles."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    features = build_customer_features()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=N_CLUSTERS,
                    random_state=RANDOM_STATE,
                    n_init=10,
                ),
            ),
        ]
    )

    features["cluster_id"] = model.fit_predict(features[FEATURES])

    # Save customer-level results, including useful customer attributes.
    output_file = OUTPUT_DIR / "customer_segments.csv"
    features.to_csv(output_file, index=False)

    # Summarize the actual behavior of each cluster.
    profile = (
        features.groupby("cluster_id")
        .agg(
            customer_count=("cust_id", "nunique"),
            avg_recency_days=("recency_days", "mean"),
            avg_orders=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            median_spend=("monetary", "median"),
            avg_units=("total_units", "mean"),
        )
        .reset_index()
    )

    profile["customer_share_pct"] = (
        profile["customer_count"] / len(features) * 100
    ).round(2)

    profile_file = REPORT_DIR / "customer_segment_profiles.csv"
    profile.to_csv(profile_file, index=False)

    model_file = MODEL_DIR / "customer_segmentation_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "reference_date": (
                features["last_purchase_date"].max()
                + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),
        },
        model_file,
    )

    print("\nCustomer segmentation completed.")
    print(f"Customers segmented: {len(features):,}")
    print(f"Clusters created: {features['cluster_id'].nunique()}")
    print(f"Customer results: {output_file}")
    print(f"Cluster profiles: {profile_file}")
    print(f"Saved model: {model_file}")
    print("\nCluster profiles:")
    print(profile.to_string(index=False))

    return features


if __name__ == "__main__":
    run_segmentation()