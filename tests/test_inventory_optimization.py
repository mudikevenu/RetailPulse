import unittest

import pandas as pd

from src.inventory_optimization import apply_inventory_rules


class TestInventoryOptimization(unittest.TestCase):

    def setUp(self):
        self.inventory = pd.DataFrame({
            "store_id": ["ST01", "ST02", "ST03", "ST04"],
            "sku_id": ["SKU001", "SKU002", "SKU003", "SKU004"],
            "stock_on_hand": [0, 5, 100, 0],
            "reorder_point": [10, 20, 30, 10],
            "safety_stock": [3, 8, 5, 3],
            "units_90d": [90, 90, 900, 0],
            "avg_daily_demand": [1.0, 1.0, 10.0, 0.0],
            "stock_cover_days": [0.0, 5.0, 10.0, None],
        })

    def test_stockout_gets_urgent_status_and_order(self):
        result = apply_inventory_rules(self.inventory.copy())

        self.assertEqual(
            result.loc[0, "recommendation_status"],
            "STOCKOUT_URGENT",
        )
        self.assertEqual(result.loc[0, "recommended_order_qty"], 14)

    def test_order_quantity_targets_configured_stock_cover(self):
        result = apply_inventory_rules(
            self.inventory.copy(),
            target_cover_days=14,
        )

        self.assertEqual(result.loc[1, "target_stock_units"], 14)
        self.assertEqual(result.loc[1, "recommended_order_qty"], 9)

    def test_adequate_stock_does_not_trigger_order(self):
        result = apply_inventory_rules(self.inventory.copy())

        self.assertEqual(
            result.loc[2, "recommendation_status"],
            "ADEQUATE_STOCK",
        )
        self.assertEqual(result.loc[2, "recommended_order_qty"], 0)

    def test_no_recent_demand_is_flagged_without_order(self):
        result = apply_inventory_rules(self.inventory.copy())

        self.assertEqual(
            result.loc[3, "recommendation_status"],
            "NO_RECENT_DEMAND_REVIEW",
        )
        self.assertEqual(result.loc[3, "recommended_order_qty"], 0)

    def test_invalid_target_cover_raises_error(self):
        with self.assertRaises(ValueError):
            apply_inventory_rules(
                self.inventory.copy(),
                target_cover_days=0,
            )

    def test_invalid_low_cover_raises_error(self):
        with self.assertRaises(ValueError):
            apply_inventory_rules(
                self.inventory.copy(),
                low_cover_days=-1,
            )

    def test_input_dataframe_is_not_modified(self):
        original = self.inventory.copy(deep=True)

        apply_inventory_rules(self.inventory)

        pd.testing.assert_frame_equal(self.inventory, original)

    def test_order_quantities_are_never_negative(self):
        result = apply_inventory_rules(self.inventory.copy())

        self.assertTrue((result["recommended_order_qty"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
