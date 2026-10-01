
"""Tests for RetailPulse data-quality checks."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_quality_checks import REQUIRED_COLUMNS, inspect_sales_file


class TestDataQualityChecks(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sales_file = Path(self.temp_dir.name) / "sales.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_sales_csv(self, rows, columns=None):
        """Create a small temporary sales CSV for testing."""
        columns = columns or REQUIRED_COLUMNS

        with self.sales_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def valid_row(self, **overrides):
        """Return a valid example transaction."""
        row = {
            "date": "2025-12-01",
            "receipt_id": "R001",
            "store_id": "STORE01",
            "sku_id": "SKU001",
            "customer_id": "C001",
            "quantity": "2",
            "unit_price": "100",
            "total_value": "180",
            "channel": "Online",
            "discount_pct": "10",
            "promo_id": "",
        }
        row.update(overrides)
        return row

    def test_valid_data_has_no_quality_issues(self):
        self.write_sales_csv([self.valid_row()])

        result = inspect_sales_file(self.sales_file)

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["total_rows"], 1)
        self.assertEqual(result["quality_result"], "NO_ISSUES_DETECTED")
        self.assertEqual(result["sales_value_mismatches"], 0)
        self.assertEqual(result["excess_exact_duplicate_rows"], 0)

    def test_exact_duplicate_rows_are_counted(self):
        row = self.valid_row()
        self.write_sales_csv([row, row, row])

        result = inspect_sales_file(self.sales_file)

        self.assertEqual(result["total_rows"], 3)
        self.assertEqual(result["excess_exact_duplicate_rows"], 2)
        self.assertEqual(result["quality_result"], "REVIEW_REQUIRED")

    def test_blank_promotion_id_is_not_a_missing_required_value(self):
        self.write_sales_csv([
            self.valid_row(promo_id=""),
        ])

        result = inspect_sales_file(self.sales_file)

        self.assertNotIn("promo_id", result["missing_value_counts"])
        self.assertEqual(
            sum(result["missing_value_counts"].values()), 0
        )

    def test_invalid_quantity_is_detected(self):
        self.write_sales_csv([
            self.valid_row(quantity="0", total_value="0"),
        ])

        result = inspect_sales_file(self.sales_file)

        self.assertEqual(
            result["invalid_value_counts"]["invalid_quantities"], 1
        )
        self.assertEqual(result["quality_result"], "REVIEW_REQUIRED")

    def test_sales_value_mismatch_is_detected(self):
        self.write_sales_csv([
            self.valid_row(total_value="999"),
        ])

        result = inspect_sales_file(self.sales_file)

        self.assertEqual(result["sales_value_mismatches"], 1)
        self.assertEqual(result["quality_result"], "REVIEW_REQUIRED")

    def test_missing_required_column_is_reported(self):
        columns = [
            column for column in REQUIRED_COLUMNS
            if column != "customer_id"
        ]
        row = self.valid_row()
        row.pop("customer_id")
        self.write_sales_csv([row], columns=columns)

        result = inspect_sales_file(self.sales_file)

        self.assertEqual(result["status"], "FAILED")
        self.assertIn("customer_id", result["missing_columns"])

    def test_missing_sales_file_raises_error(self):
        missing_file = Path(self.temp_dir.name) / "missing.csv"

        with self.assertRaises(FileNotFoundError):
            inspect_sales_file(missing_file)


if __name__ == "__main__":
    unittest.main()
