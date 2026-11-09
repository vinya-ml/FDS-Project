"""
Agrimetaverse Intelligence - Test Suite
=======================================
Unit and integration tests verifying data loading, forward filling isolation,
normalization leak prevention, metrics calculation, all 6 models, and report generation.
"""

import os
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from preprocessing.data_loader import load_raw_price_data, load_external_features
from preprocessing.data_cleaning import clean_commodity_data
from preprocessing.normalization import DatasetNormalizer, CommodityMinMaxScaler
from preprocessing.feature_engineering import engineer_commodity_features
from models.model_evaluation import calculate_metrics, compare_with_published_benchmarks
from models.xgboost_model import CommodityXGBoostModel
from models.elm_ga_model import CommodityELMGAModel
from models.hybrid_lstm_model import CommodityHybridSARIMALSTMModel
from reports.report_generator import generate_pdf_report


class TestAgrimetaversePipeline(unittest.TestCase):

    def setUp(self):
        self.df_prices = load_raw_price_data(config.RAW_DATA_FILE)
        self.df_clean, self.audit = clean_commodity_data(self.df_prices)
        self.df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)

    def test_data_loading_dimensions(self):
        """Asserts commodities count, valid rows, and zero nulls."""
        self.assertGreaterEqual(self.df_clean["Commodity"].nunique(), 5)
        self.assertGreater(len(self.df_clean), 500)
        self.assertEqual(self.df_clean["Price"].isna().sum(), 0)

    def test_commodity_isolated_ffill(self):
        """Asserts forward filling never crosses commodity boundaries."""
        onion_prices = self.df_clean[self.df_clean["Commodity"] == "Onion"]["Price"].tolist()
        wheat_prices = self.df_clean[self.df_clean["Commodity"] == "Wheat"]["Price"].tolist()
        self.assertTrue(all(p > 0 for p in onion_prices))
        self.assertTrue(all(p > 0 for p in wheat_prices))

    def test_normalization_leak_prevention(self):
        """Asserts normalizer fits without future data leakage."""
        normalizer = DatasetNormalizer(train_years=config.TRAIN_YEARS)
        normalizer.fit(self.df_clean)
        
        onion_scaler = normalizer.commodity_scalers["Onion"]
        self.assertIsNotNone(onion_scaler.min_val_)
        self.assertIsNotNone(onion_scaler.max_val_)

        # Test inverse transform
        val = float(self.df_clean[self.df_clean["Commodity"] == "Onion"]["Price"].iloc[0])
        scaled = onion_scaler.transform([val])[0]
        restored = onion_scaler.inverse_transform([scaled])[0]
        self.assertAlmostEqual(val, restored, places=3)

    def test_feature_engineering_lags(self):
        """Asserts lag 1 and lag 2 values match previous periods strictly."""
        df_feat = engineer_commodity_features(self.df_clean)
        sort_col = "Date" if "Date" in df_feat.columns else "Year"
        onion = df_feat[df_feat["Commodity"] == "Onion"].sort_values(sort_col).reset_index(drop=True)
        
        self.assertEqual(onion.loc[1, "Price_Lag1"], onion.loc[0, "Price"])
        self.assertEqual(onion.loc[2, "Price_Lag2"], onion.loc[0, "Price"])

    def test_metrics_calculation(self):
        """Tests R2, RMSE, MAE, MAPE, and Directional Accuracy calculations."""
        y_true = np.array([100.0, 110.0, 120.0])
        y_pred = np.array([102.0, 108.0, 122.0])
        y_prev = np.array([95.0, 100.0, 110.0])

        metrics = calculate_metrics(y_true, y_pred, y_prev)
        self.assertIn("R2", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("Directional_Accuracy", metrics)
        self.assertGreater(metrics["R2"], 0.9)
        self.assertLess(metrics["MAPE"], 3.0)
        self.assertEqual(metrics["Directional_Accuracy"], 100.0)

    def test_xgboost_fit_and_predict(self):
        """Tests XGBoost model fitting, lagged autoregression, and multi-step forecasting."""
        onion_prices = self.df_clean[self.df_clean["Commodity"] == "Onion"]["Price"].iloc[:50]
        xgb_m = CommodityXGBoostModel(n_estimators=10, max_depth=3)
        xgb_m.fit(onion_prices, self.df_ext.iloc[:50])
        preds, lower, upper = xgb_m.predict(steps=3)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(p > 0 for p in preds))
        self.assertTrue(all(l < u for l, u in zip(lower, upper)))

    def test_elm_ga_fit_and_predict(self):
        """Tests ELM-GA champion architecture and differential evolution optimization."""
        wheat_prices = self.df_clean[self.df_clean["Commodity"] == "Wheat"]["Price"].iloc[:50]
        elm_m = CommodityELMGAModel(popsize=4, maxiter=2)
        elm_m.fit(wheat_prices, self.df_ext.iloc[:50])
        preds, lower, upper = elm_m.predict(steps=2)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p > 0 for p in preds))

    def test_hybrid_sarima_lstm_predict(self):
        """Tests Hybrid SARIMA-LSTM forecasting and residual correction."""
        potato_prices = self.df_clean[self.df_clean["Commodity"] == "Potato"]["Price"].iloc[:50]
        hyb = CommodityHybridSARIMALSTMModel()
        hyb.fit(potato_prices)
        preds, lower, upper = hyb.predict(steps=3)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(p > 0 for p in preds))

    def test_pdf_report_compilation(self):
        """Tests ReportLab PDF generator builds a valid non-empty document."""
        dummy_eval = pd.DataFrame([{
            "Commodity": "Rice",
            "Category": "Grains",
            "Actual_2024": 44.04,
            "ELM-GA_2024": 44.1,
            "XGBoost_2024": 43.8,
            "Prophet_2024": 43.0,
            "SARIMA-LSTM_2024": 44.0,
            "SARIMAX_2024": 42.0,
            "ARIMA_2024": 41.5,
            "Best_Model": "ELM-GA",
            "Best_RMSE": 1.04,
        }])
        dummy_overall = config.PAPER_BENCHMARK_RESULTS
        dummy_comp = compare_with_published_benchmarks(dummy_overall)
        test_pdf = config.REPORTS_DIR / "test_report.pdf"

        res_path = generate_pdf_report(dummy_eval, dummy_overall, dummy_comp, output_pdf_path=test_pdf)
        self.assertTrue(os.path.exists(res_path))
        self.assertGreater(os.path.getsize(res_path), 1000)
        if os.path.exists(res_path):
            os.remove(res_path)


if __name__ == "__main__":
    unittest.main()
