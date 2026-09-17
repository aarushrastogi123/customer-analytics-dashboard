"""
test_pipeline.py
----------------
Unit and integration tests for the Customer Analytics pipeline.
Uses synthetic transactional data to test all modules without requiring
the full external dataset download.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import clean_data, validate_raw
from rfm import add_rfm_scores, compute_rfm
from segmentation import (
    assign_segments,
    choose_k,
    compute_elbow_silhouette,
    scale_rfm,
    train_kmeans,
)
from campaign_model import (
    build_features_target,
    evaluate_model,
    score_all_customers,
    train_model,
)
from insights import generate_insights


class TestCustomerAnalyticsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a synthetic transaction DataFrame for testing."""
        np.random.seed(42)
        base_date = datetime(2011, 1, 1)

        records = []
        customer_ids = [f"CUST_{i:03d}" for i in range(1, 51)]  # 50 customers

        for i in range(500):
            cust_id = np.random.choice(customer_ids)
            days_offset = np.random.randint(0, 350)
            inv_date = base_date + timedelta(days=days_offset, hours=np.random.randint(8, 20))
            inv_no = f"{10000 + (days_offset // 3) * 10 + int(cust_id.split('_')[1])}"
            qty = np.random.randint(1, 20)
            price = round(float(np.random.uniform(2.0, 50.0)), 2)
            desc = f"Product {np.random.randint(1, 30)}"
            country = np.random.choice(["United Kingdom", "Germany", "France"])

            records.append({
                "Invoice": inv_no,
                "StockCode": f"SKU_{np.random.randint(1, 30):03d}",
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": inv_date,
                "Price": price,
                "Customer ID": cust_id,
                "Country": country,
            })

        # Add some dirty data to test cleaning:
        # 1. Cancellation
        records.append({
            "Invoice": "C10099",
            "StockCode": "SKU_001",
            "Description": "Cancelled Item",
            "Quantity": -5,
            "InvoiceDate": base_date + timedelta(days=50),
            "Price": 10.0,
            "Customer ID": "CUST_001",
            "Country": "United Kingdom",
        })
        # 2. Missing Customer ID
        records.append({
            "Invoice": "10100",
            "StockCode": "SKU_002",
            "Description": "Guest Checkout",
            "Quantity": 2,
            "InvoiceDate": base_date + timedelta(days=60),
            "Price": 15.0,
            "Customer ID": None,
            "Country": "United Kingdom",
        })
        # 3. Non-positive Quantity
        records.append({
            "Invoice": "10101",
            "StockCode": "SKU_003",
            "Description": "Zero Qty",
            "Quantity": 0,
            "InvoiceDate": base_date + timedelta(days=70),
            "Price": 5.0,
            "Customer ID": "CUST_002",
            "Country": "United Kingdom",
        })

        cls.raw_df = pd.DataFrame(records)

    def test_01_validation(self):
        """Test data validation report on raw transactions."""
        report = validate_raw(self.raw_df)
        self.assertIn("total_rows", report)
        self.assertIn("missing_per_column", report)
        self.assertGreater(report["total_rows"], 500)
        self.assertEqual(report["missing_per_column"]["Customer ID"], 1)

    def test_02_cleaning(self):
        """Test that data cleaning filters invalid records and creates derived columns."""
        clean_df, log = clean_data(self.raw_df)
        self.assertTrue((clean_df["Quantity"] > 0).all())
        self.assertTrue((clean_df["UnitPrice"] > 0).all())
        self.assertTrue(clean_df["CustomerID"].notna().all())
        self.assertFalse(clean_df["InvoiceNo"].str.startswith("C").any())
        self.assertIn("TotalPrice", clean_df.columns)
        self.assertIn("YearMonth", clean_df.columns)
        self.assertEqual(log["missing_customer_id_removed"], 1)
        self.assertEqual(log["cancellation_rows_excluded"], 1)

    def test_03_rfm_computation(self):
        """Test RFM computation and scoring."""
        clean_df, _ = clean_data(self.raw_df)
        rfm = compute_rfm(clean_df)
        self.assertEqual(len(rfm), clean_df["CustomerID"].nunique())
        self.assertTrue((rfm["Recency"] >= 0).all())
        self.assertTrue((rfm["Frequency"] >= 1).all())
        self.assertTrue((rfm["Monetary"] > 0).all())

        scored_rfm = add_rfm_scores(rfm)
        self.assertIn("R_Score", scored_rfm.columns)
        self.assertIn("F_Score", scored_rfm.columns)
        self.assertIn("M_Score", scored_rfm.columns)
        self.assertIn("RFM_String", scored_rfm.columns)

    def test_04_segmentation(self):
        """Test K-Means clustering and segment label assignment."""
        clean_df, _ = clean_data(self.raw_df)
        rfm = compute_rfm(clean_df)
        X, scaler, features = scale_rfm(rfm)
        self.assertEqual(X.shape[0], len(rfm))

        metrics = compute_elbow_silhouette(X, k_range=range(2, 6))
        self.assertEqual(len(metrics["k"]), 4)

        best_k = choose_k(metrics, k_min=3, k_max=5)
        self.assertTrue(3 <= best_k <= 5)

        km = train_kmeans(X, k=best_k)
        rfm_seg, stats, label_map = assign_segments(rfm, km.labels_)
        self.assertIn("Segment", rfm_seg.columns)
        self.assertEqual(rfm_seg["Segment"].nunique(), best_k)

    def test_05_campaign_propensity_model(self):
        """Test feature engineering, model training, evaluation, and customer scoring."""
        clean_df, _ = clean_data(self.raw_df)
        rfm = compute_rfm(clean_df)
        X_feat, y, customers, feat_df = build_features_target(clean_df, holdout_days=60)

        self.assertGreater(len(X_feat), 0)
        self.assertEqual(len(X_feat), len(y))

        model, scaler, X_tr, X_te, y_tr, y_te = train_model(X_feat, y)
        eval_res = evaluate_model(model, X_te, y_te)

        self.assertIn("accuracy", eval_res)
        self.assertIn("roc_auc", eval_res)
        self.assertIn("confusion_matrix", eval_res)

        scored = score_all_customers(model, scaler, X_feat, customers, rfm)
        self.assertIn("Propensity", scored.columns)
        self.assertIn("Priority", scored.columns)
        self.assertTrue((scored["Propensity"] >= 0.0).all() and (scored["Propensity"] <= 1.0).all())

    def test_06_business_insights(self):
        """Test that business insights are dynamically generated without error."""
        clean_df, _ = clean_data(self.raw_df)
        rfm = compute_rfm(clean_df)
        X, scaler, _ = scale_rfm(rfm)
        km = train_kmeans(X, k=4)
        rfm_seg, _, _ = assign_segments(rfm, km.labels_)

        insights = generate_insights(clean_df, rfm_seg)
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 3)
        for ins in insights:
            self.assertIn("title", ins)
            self.assertIn("observation", ins)
            self.assertIn("recommendation", ins)
            self.assertIn("severity", ins)


if __name__ == "__main__":
    unittest.main()
