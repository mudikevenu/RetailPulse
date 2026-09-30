
from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# 1. Project paths and analysis settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SALES_FILE = (
    Path.home()
    / "Downloads"
    / "archive"
    / "retail_clean_dataset"
    / "sales_transactions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_analysis.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "customer_retention_summary.csv"
)

# The sales dataset ends on December 31, 2025.
# This reference date keeps recency calculations consistent
# with the available data.
REFERENCE_DATE = "2026-01-01"


# --------------------------------------------------
# 2. Validate input and prepare output directories
# --------------------------------------------------

def validate_input():
    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {SALES_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Aggregate customer purchase behaviour
# --------------------------------------------------

def build_customer_analysis():
    query = """
    WITH customer_metrics AS (
        SELECT
            customer_id,

            MAX(CAST(date AS DATE)) AS last_purchase_date,

            COUNT(DISTINCT receipt_id) AS total_receipts,

            SUM(CAST(quantity AS DOUBLE)) AS total_units,

            SUM(CAST(total_value AS DOUBLE)) AS total_spend

        FROM read_csv_auto(?, header = true)

        WHERE customer_id IS NOT NULL

        GROUP BY customer_id
    )

    SELECT
        customer_id,
        last_purchase_date,
        total_receipts,
        total_units,
        total_spend,

        DATE_DIFF(
            'day',
            last_purchase_date,
            CAST(? AS DATE)
        ) AS recency_days

    FROM customer_metrics
    ORDER BY customer_id
    """

    with duckdb.connect() as connection:
        customers = connection.execute(
            query,
            [str(SALES_FILE), REFERENCE_DATE],
        ).fetchdf()

    if customers.empty:
        raise ValueError(
            "No customer records were found in the sales dataset."
        )

    numeric_columns = [
        "total_receipts",
        "total_units",
        "total_spend",
        "recency_days",
    ]

    if customers[numeric_columns].isna().any().any():
        raise ValueError(
            "Customer analysis contains missing numeric values."
        )

    if (customers[numeric_columns] < 0).any().any():
        raise ValueError(
            "Unexpected negative values found in customer metrics."
        )

    return customers


# --------------------------------------------------
# 4. Assign transparent inactivity categories
# --------------------------------------------------

def assign_retention_categories(customers):
    conditions = [
        customers["recency_days"] <= 3,
        customers["recency_days"].between(4, 7),
        customers["recency_days"].between(8, 14),
        customers["recency_days"] >= 15,
    ]

    categories = [
        "Recently Active",
        "Active - Monitor",
        "Inactivity Watch",
        "Extended Inactivity",
    ]

    customers["retention_category"] = pd.Series(
        pd.NA,
        index=customers.index,
        dtype="string",
    )

    for condition, category in zip(conditions, categories):
        customers.loc[condition, "retention_category"] = category

    customers["risk_interpretation"] = customers[
        "retention_category"
    ].map(
        {
            "Recently Active": "Recent purchase activity",
            "Active - Monitor": "Monitor purchase recency",
            "Inactivity Watch": "Review customer engagement",
            "Extended Inactivity": (
                "Review individually; inactivity is not proof of churn"
            ),
        }
    )

    return customers


# --------------------------------------------------
# 5. Create a summary by retention category
# --------------------------------------------------

def build_summary(customers):
    summary = (
        customers.groupby("retention_category", observed=True)
        .agg(
            customer_count=("customer_id", "nunique"),
            average_recency_days=("recency_days", "mean"),
            average_receipts=("total_receipts", "mean"),
            average_spend=("total_spend", "mean"),
            total_customer_spend=("total_spend", "sum"),
        )
        .reset_index()
    )

    summary["customer_percentage"] = (
        summary["customer_count"]
        / customers["customer_id"].nunique()
        * 100
    )

    summary["average_recency_days"] = (
        summary["average_recency_days"].round(2)
    )

    summary["average_receipts"] = (
        summary["average_receipts"].round(2)
    )

    summary["average_spend"] = (
        summary["average_spend"].round(2)
    )

    summary["total_customer_spend"] = (
        summary["total_customer_spend"].round(2)
    )

    return summary


# --------------------------------------------------
# 6. Run the complete analysis
# --------------------------------------------------

def main():
    validate_input()

    print("Building customer retention analysis...")
    print(f"Reference date: {REFERENCE_DATE}")

    customers = build_customer_analysis()

    customers = assign_retention_categories(customers)

    summary = build_summary(customers)

    customers.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n===== CUSTOMER RETENTION SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== VALIDATION =====")
    print(f"Total customers: {len(customers):,}")
    print(
        "Unique customer IDs: "
        f"{customers['customer_id'].nunique():,}"
    )
    print(
        "Missing customer IDs: "
        f"{customers['customer_id'].isna().sum():,}"
    )
    print(
        "Missing retention categories: "
        f"{customers['retention_category'].isna().sum():,}"
    )

    print(f"\nCustomer analysis saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()