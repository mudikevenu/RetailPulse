"""RetailPulse inventory optimization using historical sales and stock data.

The 14-day target is a configurable demonstration assumption, not a
supplier lead-time estimate. Inventory and sales data are historical.
"""

import argparse
import math
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    Path.home() / "Downloads" / "archive" / "retail_clean_dataset"
)
DEFAULT_TARGET_COVER_DAYS = 14
DEFAULT_LOW_COVER_DAYS = 7


def apply_inventory_rules(
    df,
    target_cover_days=DEFAULT_TARGET_COVER_DAYS,
    low_cover_days=DEFAULT_LOW_COVER_DAYS,
):
    """Apply demand-based inventory statuses and order quantities."""

    if target_cover_days <= 0 or low_cover_days < 0:
        raise ValueError("Cover days must be positive (low cover may be zero).")

    df = df.copy()

    df["target_cover_days"] = target_cover_days
    df["target_stock_units"] = df["avg_daily_demand"].apply(
        lambda demand: math.ceil(demand * target_cover_days)
        if pd.notna(demand) and demand > 0 else 0
    )

    df["recommended_order_qty"] = (
        df["target_stock_units"] - df["stock_on_hand"]
    ).clip(lower=0).astype(int)

    def classify(row):
        if row["units_90d"] == 0:
            return "NO_RECENT_DEMAND_REVIEW"
        if row["stock_on_hand"] == 0:
            return "STOCKOUT_URGENT"
        if row["stock_on_hand"] <= row["safety_stock"]:
            return "BELOW_SAFETY_STOCK"
        if row["stock_on_hand"] <= row["reorder_point"]:
            return "REORDER_POINT_BREACHED"
        if row["stock_cover_days"] < low_cover_days:
            return "LOW_STOCK_COVER"
        return "ADEQUATE_STOCK"

    df["recommendation_status"] = df.apply(classify, axis=1)

    triggered = (
        (df["units_90d"] > 0)
        & (
            (df["stock_on_hand"] == 0)
            | (df["stock_on_hand"] <= df["reorder_point"])
            | (
                df["stock_cover_days"].notna()
                & (df["stock_cover_days"] < low_cover_days)
            )
        )
    )

    df.loc[~triggered, "recommended_order_qty"] = 0

    df["recommendation_note"] = df.apply(
        lambda row: (
            "Review demand before ordering: no sales recorded in the 90-day window."
            if row["units_90d"] == 0
            else (
                f"Replenish toward approximately {target_cover_days} days "
                "of observed demand; verify supplier lead time and incoming stock."
                if row["recommended_order_qty"] > 0
                else "No demand-based order suggested by the current rules."
            )
        ),
        axis=1,
    )

    return df
def build_inventory_recommendations(
    data_dir=DEFAULT_DATA_DIR,
    target_cover_days=DEFAULT_TARGET_COVER_DAYS,
    low_cover_days=DEFAULT_LOW_COVER_DAYS,
):
    """Build store–SKU inventory recommendations from the last 90 days."""

    if target_cover_days <= 0 or low_cover_days < 0:
        raise ValueError("Cover days must be positive (low cover may be zero).")

    data_dir = Path(data_dir)
    required_files = [
        "sales_transactions.csv",
        "inventory_snapshot.csv",
        "sku_master.csv",
        "store_master.csv",
        "sku_inventory_flags.csv",
    ]
    missing = [name for name in required_files if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files: " + ", ".join(missing)
        )

    sales = (data_dir / "sales_transactions.csv").as_posix()
    inventory = (data_dir / "inventory_snapshot.csv").as_posix()
    sku_master = (data_dir / "sku_master.csv").as_posix()
    store_master = (data_dir / "store_master.csv").as_posix()
    flags = (data_dir / "sku_inventory_flags.csv").as_posix()

    con = duckdb.connect()
    try:
        query = f"""
        WITH recent_sales AS (
            SELECT
                store_id,
                sku_id,
                SUM(quantity) AS units_90d
            FROM read_csv_auto('{sales}')
            WHERE CAST(date AS DATE)
                  BETWEEN DATE '2025-10-03' AND DATE '2025-12-31'
            GROUP BY store_id, sku_id
        ),
        matched_flags AS (
            SELECT
                i.store_id,
                i.sku_id,
                i.stock_on_hand,
                i.reorder_point,
                i.safety_stock,
                i.last_restock_date,
                COALESCE(s.units_90d, 0) AS units_90d,
                f.flag AS inventory_flag
            FROM read_csv_auto('{inventory}') i
            LEFT JOIN recent_sales s
                ON i.store_id = s.store_id
               AND i.sku_id = s.sku_id
            LEFT JOIN read_csv_auto('{flags}') f
                ON i.sku_id = f.sku_id
               AND contains(
                    ';' || replace(f.affected_stores, ' ', '') || ';',
                    ';' || i.store_id || ';'
               )
        )
        SELECT
            i.store_id,
            st.store_name,
            st.city,
            i.sku_id,
            sk.sku_name,
            sk.category,
            sk.subcategory,
            sk.brand,
            sk.unit_price,
            sk.cost_price,
            i.stock_on_hand,
            i.reorder_point,
            i.safety_stock,
            i.last_restock_date,
            i.units_90d,
            i.units_90d / 90.0 AS avg_daily_demand,
            CASE
                WHEN i.units_90d > 0
                THEN i.stock_on_hand / (i.units_90d / 90.0)
                ELSE NULL
            END AS stock_cover_days,
            COALESCE(i.inventory_flag, 'NO_FLAG') AS inventory_flag
        FROM matched_flags i
        LEFT JOIN read_csv_auto('{sku_master}') sk
            ON i.sku_id = sk.sku_id
        LEFT JOIN read_csv_auto('{store_master}') st
            ON i.store_id = st.store_id
        """

        df = con.execute(query).df()
    finally:
        con.close()

    df = apply_inventory_rules(
        df,
        target_cover_days=target_cover_days,
        low_cover_days=low_cover_days,
    )

    # Put actionable and higher-priority records first.
    priority_order = {
        "STOCKOUT_URGENT": 0,
        "BELOW_SAFETY_STOCK": 1,
        "REORDER_POINT_BREACHED": 2,
        "LOW_STOCK_COVER": 3,
        "NO_RECENT_DEMAND_REVIEW": 4,
        "ADEQUATE_STOCK": 5,
    }
    df["priority_rank"] = df["recommendation_status"].map(priority_order)
    df = df.sort_values(
        ["priority_rank", "recommended_order_qty", "units_90d"],
        ascending=[True, False, False],
    ).drop(columns=["priority_rank"])

    output_dir = PROJECT_ROOT / "data" / "processed"
    report_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    recommendations_path = output_dir / "inventory_recommendations.csv"
    summary_path = report_dir / "inventory_optimization_summary.csv"

    df.to_csv(recommendations_path, index=False)

    summary = (
        df.groupby("recommendation_status", dropna=False)
        .agg(
            inventory_pairs=("sku_id", "size"),
            zero_stock_pairs=("stock_on_hand", lambda s: int((s == 0).sum())),
            total_units_90d=("units_90d", "sum"),
            suggested_order_units=("recommended_order_qty", "sum"),
        )
        .reset_index()
    )
    summary["target_cover_days_assumption"] = target_cover_days
    summary["demand_window"] = "2025-10-03 to 2025-12-31"
    summary.to_csv(summary_path, index=False)

    print("\n--- INVENTORY OPTIMIZATION COMPLETE ---")
    print(f"Inventory pairs analyzed: {len(df):,}")
    print(f"Target stock cover assumption: {target_cover_days} days")
    print(f"Pairs with positive order suggestions: "
          f"{int((df['recommended_order_qty'] > 0).sum()):,}")
    print(f"Total suggested order units: {df['recommended_order_qty'].sum():,}")
    print(f"\nRecommendations: {recommendations_path}")
    print(f"Summary: {summary_path}")
    print("\n--- STATUS SUMMARY ---")
    print(summary.to_string(index=False))

    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical, demand-based inventory recommendations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the clean retail CSV files.",
    )
    parser.add_argument(
        "--target-cover-days",
        type=int,
        default=DEFAULT_TARGET_COVER_DAYS,
        help="Demonstration target stock cover in days (default: 14).",
    )
    parser.add_argument(
        "--low-cover-days",
        type=int,
        default=DEFAULT_LOW_COVER_DAYS,
        help="Low-cover alert threshold in days (default: 7).",
    )
    args = parser.parse_args()

    build_inventory_recommendations(
        data_dir=args.data_dir,
        target_cover_days=args.target_cover_days,
        low_cover_days=args.low_cover_days,
    )


if __name__ == "__main__":
    main()
