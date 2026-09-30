
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CUSTOMER_FILE = ROOT / "data/processed/customer_segments.csv"
PROFILE_FILE = ROOT / "reports/customer_segment_profiles.csv"
MODEL_FILE = ROOT / "models/customer_segmentation_pipeline.joblib"


class TestCustomerSegmentation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for path in (CUSTOMER_FILE, PROFILE_FILE, MODEL_FILE):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required segmentation output is missing: {path}"
                )

        cls.customers = pd.read_csv(CUSTOMER_FILE)
        cls.profiles = pd.read_csv(PROFILE_FILE)

    def test_all_customers_are_unique(self):
        self.assertEqual(len(self.customers), 10000)
        self.assertEqual(
            self.customers["cust_id"].nunique(),
            len(self.customers),
        )

    def test_customer_ids_and_clusters_are_present(self):
        self.assertFalse(self.customers["cust_id"].isna().any())
        self.assertFalse(self.customers["cluster_id"].isna().any())
        self.assertEqual(self.customers["cluster_id"].nunique(), 4)

    def test_purchase_values_are_valid(self):
        for column in (
            "recency_days",
            "frequency",
            "monetary",
            "total_units",
        ):
            self.assertTrue(
                pd.api.types.is_numeric_dtype(self.customers[column]),
                f"{column} must be numeric",
            )
            self.assertTrue(
                self.customers[column].notna().all(),
                f"{column} contains missing values",
            )
            self.assertTrue(
                (self.customers[column] >= 0).all(),
                f"{column} contains negative values",
            )

    def test_cluster_profiles_reconcile(self):
        self.assertEqual(len(self.profiles), 4)
        self.assertEqual(
            int(self.profiles["customer_count"].sum()),
            len(self.customers),
        )
        self.assertAlmostEqual(
            self.profiles["customer_share_pct"].sum(),
            100.0,
            places=1,
        )

    def test_saved_model_can_be_loaded(self):
        artifact = joblib.load(MODEL_FILE)

        self.assertIn("pipeline", artifact)
        self.assertIn("features", artifact)
        self.assertEqual(artifact["n_clusters"], 4)
        self.assertEqual(len(artifact["features"]), 3)

        predictions = artifact["pipeline"].predict(
            self.customers[artifact["features"]].head(5)
        )
        self.assertEqual(len(predictions), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)