
"""Data quality checks for RetailPulse sales transactions.

Uses DuckDB to analyze large CSV files without loading the full dataset
into pandas memory.
"""

from pathlib import Path
import argparse
import json

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SALES_FILE = (
    Path.home()
    / "Downloads"
    / "archive"
    / "retail_clean_dataset"
    / "sales_transactions.csv"
)

REQUIRED_COLUMNS = [
    "date",
    "receipt_id",
    "store_id",
    "sku_id",
    "customer_id",
    "quantity",
    "unit_price",
    "total_value",
    "channel",
    "discount_pct",
    "promo_id",
]


def quote_identifier(name: str) -> str:
    """Safely quote a SQL column identifier."""
    return '"' + name.replace('"', '""') + '"'


def inspect_sales_file(sales_file: Path) -> dict:
    """Check sales-file structure and calculate data-quality metrics."""
    sales_file = Path(sales_file).expanduser().resolve()

    if not sales_file.is_file():
        raise FileNotFoundError(f"Sales file not found: {sales_file}")

    con = duckdb.connect(":memory:")

    try:
        file_sql = str(sales_file).replace("'", "''")

        # Read values as strings first so malformed values can be detected.
        con.execute(
            f"""
            CREATE VIEW sales_raw AS
            SELECT *
            FROM read_csv(
                '{file_sql}',
                header = true,
                all_varchar = true,
                sample_size = -1,
                ignore_errors = false
            )
            """
        )

        schema = con.execute(
            "DESCRIBE SELECT * FROM sales_raw"
        ).fetchall()

        available_columns = [row[0] for row in schema]
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in available_columns
        ]

        if missing_columns:
            return {
                "file": sales_file.name,
                "status": "FAILED",
                "missing_columns": missing_columns,
                "message": "Required columns are missing.",
            }

        # Count blank or NULL values in required fields.
        # promo_id is optional when no promotion was applied.
        non_nullable_columns = [
            col for col in REQUIRED_COLUMNS if col != "promo_id"
        ]

        missing_sql = ", ".join(
            f"""
            COUNT(*) FILTER (
                WHERE {quote_identifier(col)} IS NULL
                   OR TRIM({quote_identifier(col)}) = ''
            ) AS {quote_identifier(col)}
            """
            for col in non_nullable_columns
        )

        missing_row = con.execute(
            f"SELECT {missing_sql} FROM sales_raw"
        ).fetchone()

        missing_counts = dict(
            zip(non_nullable_columns, missing_row)
        )

        # TRY_CAST identifies values that cannot be interpreted correctly.
        invalid_checks = {
            "invalid_dates": """
                TRY_CAST("date" AS DATE) IS NULL
            """,
            "invalid_quantities": """
                TRY_CAST("quantity" AS DOUBLE) IS NULL
                OR TRY_CAST("quantity" AS DOUBLE) <= 0
            """,
            "invalid_unit_prices": """
                TRY_CAST("unit_price" AS DOUBLE) IS NULL
                OR TRY_CAST("unit_price" AS DOUBLE) < 0
            """,
            "invalid_total_values": """
                TRY_CAST("total_value" AS DOUBLE) IS NULL
                OR TRY_CAST("total_value" AS DOUBLE) < 0
            """,
            "invalid_discounts": """
                TRY_CAST("discount_pct" AS DOUBLE) IS NULL
                OR TRY_CAST("discount_pct" AS DOUBLE) < 0
                OR TRY_CAST("discount_pct" AS DOUBLE) > 100
            """,
        }

        invalid_sql = ", ".join(
            f"COUNT(*) FILTER (WHERE {condition}) AS {name}"
            for name, condition in invalid_checks.items()
        )

        invalid_row = con.execute(
            f"SELECT {invalid_sql} FROM sales_raw"
        ).fetchone()

        invalid_counts = dict(
            zip(invalid_checks.keys(), invalid_row)
        )

        # Check total_value against quantity * price after discount.
        value_mismatches = con.execute(
            """
            SELECT COUNT(*)
            FROM sales_raw
            WHERE TRY_CAST("quantity" AS DOUBLE) IS NOT NULL
              AND TRY_CAST("unit_price" AS DOUBLE) IS NOT NULL
              AND TRY_CAST("total_value" AS DOUBLE) IS NOT NULL
              AND TRY_CAST("discount_pct" AS DOUBLE) IS NOT NULL
              AND ABS(
                    TRY_CAST("total_value" AS DOUBLE)
                    - TRY_CAST("quantity" AS DOUBLE)
                      * TRY_CAST("unit_price" AS DOUBLE)
                      * (1 - TRY_CAST("discount_pct" AS DOUBLE) / 100)
                  ) > GREATEST(
                        0.02,
                        ABS(TRY_CAST("total_value" AS DOUBLE)) * 0.000001
                  )
            """
        ).fetchone()[0]

        # Exact full-row duplicates; do not delete or modify source data.
        group_by_columns = ", ".join(
            quote_identifier(col) for col in available_columns
        )

        duplicate_count, potential_duplicate_sales_value = con.execute(
            f"""
            SELECT
                COALESCE(SUM(row_count - 1), 0),
                COALESCE(
                    SUM(TRY_CAST(total_value AS DOUBLE) * (row_count - 1)),
                    0
                )
            FROM (
                SELECT
                    {group_by_columns},
                    COUNT(*) AS row_count
                FROM sales_raw
                GROUP BY {group_by_columns}
                HAVING COUNT(*) > 1
            ) AS duplicate_groups
            """
        ).fetchone()
        total_rows = con.execute(
            "SELECT COUNT(*) FROM sales_raw"
        ).fetchone()[0]

        result = {
            "file": sales_file.name,
            "status": "COMPLETED",
            "total_rows": total_rows,
            "columns_found": len(available_columns),
            "missing_required_columns": [],
            "missing_value_counts": missing_counts,
            "invalid_value_counts": invalid_counts,
            "sales_value_mismatches": value_mismatches,
            "excess_exact_duplicate_rows": duplicate_count,
            "potential_duplicate_sales_value": round(float(potential_duplicate_sales_value), 2),
        }

        # Overall status is descriptive; investigate findings before
        # deciding whether a dataset is suitable for a particular model.
        has_findings = (
            any(missing_counts.values())
            or any(invalid_counts.values())
            or value_mismatches > 0
            or duplicate_count > 0
        )
        result["quality_result"] = (
            "REVIEW_REQUIRED" if has_findings else "NO_ISSUES_DETECTED"
        )

        return result

    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RetailPulse sales data-quality checks."
    )
    parser.add_argument(
        "--sales-file",
        type=Path,
        default=DEFAULT_SALES_FILE,
        help="Path to sales_transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_quality_report.json",
        help="Path for the JSON report",
    )
    args = parser.parse_args()

    result = inspect_sales_file(args.sales_file)

    if result["status"] == "COMPLETED":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(result, indent=2))
    if result["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
