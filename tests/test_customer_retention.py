
import unittest

import pandas as pd

from src.customer_retention import (
    assign_retention_categories,
    build_summary,
)


class TestCustomerRetention(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003", "C004"],
            "last_purchase_date": pd.to_datetime([
                "2025-12-31",
                "2025-12-27",
                "2025-12-20",
                "2025-12-10",
            ]),
            "total_receipts": [100, 80, 60, 40],
            "total_units": [250, 180, 120, 80],
            "total_spend": [50000.0, 40000.0, 30000.0, 20000.0],
            "recency_days": [1, 5, 12, 22],
        })

    def test_assigns_expected_categories(self):
        result = assign_retention_categories(
            self.customers.copy()
        )

        self.assertEqual(
            result["retention_category"].tolist(),
            [
                "Recently Active",
                "Active - Monitor",
                "Inactivity Watch",
                "Extended Inactivity",
            ],
        )

    def test_every_customer_receives_a_category(self):
        result = assign_retention_categories(
            self.customers.copy()
        )

        self.assertTrue(
            result["retention_category"].notna().all()
        )

    def test_summary_counts_all_customers(self):
        categorized = assign_retention_categories(
            self.customers.copy()
        )
        summary = build_summary(categorized)

        self.assertEqual(
            int(summary["customer_count"].sum()),
            len(self.customers),
        )

    def test_summary_percentages_total_100(self):
        categorized = assign_retention_categories(
            self.customers.copy()
        )
        summary = build_summary(categorized)

        self.assertAlmostEqual(
            summary["customer_percentage"].sum(),
            100.0,
        )

    def test_empty_customer_input_is_handled(self):
        empty_customers = self.customers.iloc[0:0].copy()
        categorized = assign_retention_categories(
            empty_customers
        )
        summary = build_summary(categorized)

        self.assertEqual(len(categorized), 0)
        self.assertEqual(len(summary), 0)


if __name__ == "__main__":
    unittest.main()