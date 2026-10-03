
from pathlib import Path

import duckdb


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
    / "customer_spending_analysis.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "customer_spending_decline_summary.csv"
)

# Compare two consecutive six-month periods.
BASELINE_START = "2025-01-01"
BASELINE_END = "2025-06-30"
FOLLOWUP_START = "2025-07-01"
FOLLOWUP_END = "2025-12-31"


def run_analysis():
    """Calculate customer spending changes and summarize decline categories."""

    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {SALES_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()

    try:
        query = """
            WITH sales AS (
                SELECT
                    customer_id,
                    CAST(date AS DATE) AS sale_date,
                    CAST(total_value AS DOUBLE) AS spend
                FROM read_csv_auto(?)
                WHERE customer_id IS NOT NULL
            ),
            customer_periods AS (
                SELECT
                    customer_id,

                    SUM(spend) FILTER (
                        WHERE sale_date BETWEEN ? AND ?
                    ) AS baseline_spend,

                    SUM(spend) FILTER (
                        WHERE sale_date BETWEEN ? AND ?
                    ) AS followup_spend,

                    COUNT(DISTINCT sale_date) FILTER (
                        WHERE sale_date BETWEEN ? AND ?
                    ) AS baseline_active_days,

                    COUNT(DISTINCT sale_date) FILTER (
                        WHERE sale_date BETWEEN ? AND ?
                    ) AS followup_active_days

                FROM sales
                WHERE sale_date BETWEEN ? AND ?
                GROUP BY customer_id
            ),
            changes AS (
                SELECT
                    *,
                    CASE
                        WHEN baseline_spend > 0
                        THEN 100.0 * (
                            followup_spend - baseline_spend
                        ) / baseline_spend
                        ELSE NULL
                    END AS spending_change_pct
                FROM customer_periods
            )
            SELECT
                customer_id,
                baseline_spend,
                followup_spend,
                COALESCE(baseline_active_days, 0)
                    AS baseline_active_days,
                COALESCE(followup_active_days, 0)
                    AS followup_active_days,
                spending_change_pct,

                CASE
                    WHEN baseline_spend IS NULL
                         OR baseline_spend <= 0
                        THEN 'Insufficient Baseline'

                    WHEN followup_spend IS NULL
                         OR followup_spend <= 0
                        THEN 'No Follow-up Spend'

                    WHEN spending_change_pct >= 0
                        THEN 'Stable or Increased'

                    WHEN spending_change_pct > -25
                        THEN 'Low Decline'

                    WHEN spending_change_pct > -50
                        THEN 'Moderate Decline'

                    ELSE 'High Decline'
                END AS spending_category

            FROM changes
            ORDER BY customer_id
        """

        params = [
            str(SALES_FILE),
            BASELINE_START, BASELINE_END,
            FOLLOWUP_START, FOLLOWUP_END,
            BASELINE_START, BASELINE_END,
            FOLLOWUP_START, FOLLOWUP_END,
            BASELINE_START, FOLLOWUP_END,
        ]

        customers = connection.execute(query, params).df()

        customers.to_csv(OUTPUT_FILE, index=False)

        summary = (
            customers.groupby("spending_category", dropna=False)
            .agg(
                customer_count=("customer_id", "nunique"),
                average_baseline_spend=("baseline_spend", "mean"),
                average_followup_spend=("followup_spend", "mean"),
                average_spending_change_pct=(
                    "spending_change_pct", "mean"
                ),
            )
            .reset_index()
        )

        summary["customer_percentage"] = (
            summary["customer_count"]
            / customers["customer_id"].nunique()
            * 100
        )

        summary.to_csv(SUMMARY_FILE, index=False)

        print("\nCustomer spending analysis completed.")
        print(f"Customers analysed: {customers['customer_id'].nunique():,}")
        print(f"Customer report: {OUTPUT_FILE}")
        print(f"Summary report: {SUMMARY_FILE}")
        print("\nCategory summary:")
        print(summary.round(2).to_string(index=False))

    finally:
        connection.close()


if __name__ == "__main__":
    run_analysis()