"""
Test suite to verify that all dashboard modules, page renderers, and underlying functions
execute completely without KeyErrors, NameErrors, or unexpected exceptions.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from preprocessing.data_loader import (
    inspect_raw_dataset,
    load_raw_price_data,
    load_external_features,
    get_feature_availability_report,
)
from preprocessing.data_cleaning import (
    clean_commodity_data,
    clean_external_features,
    generate_data_quality_report,
)
from analysis.eda import (
    get_growth_summary,
    plot_commodity_trends,
    plot_category_indexed_comparison,
    plot_price_distribution_boxplot,
)
from analysis.external_factor_analysis import (
    compute_correlation_matrix,
    plot_correlation_heatmap,
    plot_factor_scatter_regression,
)
from forecasting.future_forecast import (
    generate_commodity_future_forecast,
    load_paper_reference_forecasts,
)
from forecasting.forecast_visualization import plot_commodity_forecast
from models.model_training import get_cached_or_train_evaluation
from reports.report_generator import generate_pdf_report


class TestDashboardAndAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_prices = load_raw_price_data(config.RAW_DATA_FILE)
        cls.df_prices, cls.audit = clean_commodity_data(cls.df_prices)
        cls.df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)
        cls.df_ext, _ = clean_external_features(cls.df_ext)
        cls.eval_df, cls.overall_metrics, cls.comp_df = get_cached_or_train_evaluation()

    def test_inspect_raw_dataset_keys(self):
        inspect = inspect_raw_dataset(config.RAW_DATA_FILE)
        self.assertTrue(inspect["exists"])
        self.assertIn("rows", inspect)
        self.assertIn("cols", inspect)
        self.assertIn("row_count", inspect)
        self.assertIn("col_count", inspect)
        self.assertGreater(inspect["rows"], 0)
        self.assertGreater(inspect["cols"], 0)

    def test_data_quality_reports(self):
        report = generate_data_quality_report(self.df_prices, self.df_ext, self.audit)
        self.assertIsInstance(report, pd.DataFrame)
        self.assertGreater(len(report), 0)

        ext_rep = get_feature_availability_report(self.df_ext)
        self.assertIsInstance(ext_rep, pd.DataFrame)
        self.assertEqual(len(ext_rep), len(config.EXTERNAL_FEATURE_COLUMNS))

    def test_eda_functions(self):
        growth = get_growth_summary(self.df_prices)
        self.assertIsInstance(growth, pd.DataFrame)
        self.assertEqual(len(growth), self.df_prices["Commodity"].nunique())

        fig_trends = plot_commodity_trends(self.df_prices)
        self.assertIsNotNone(fig_trends)

        fig_cat = plot_category_indexed_comparison(self.df_prices)
        self.assertIsNotNone(fig_cat)

        fig_box = plot_price_distribution_boxplot(self.df_prices)
        self.assertIsNotNone(fig_box)

    def test_external_factors_functions(self):
        corr = compute_correlation_matrix(self.df_prices, self.df_ext)
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertGreater(corr.shape[0], 0)
        self.assertGreater(corr.shape[1], 0)

        fig_heat = plot_correlation_heatmap(corr)
        self.assertIsNotNone(fig_heat)

        sample_comm = self.df_prices["Commodity"].iloc[0]
        sample_factor = config.EXTERNAL_FEATURE_COLUMNS[0]
        fig_scatter = plot_factor_scatter_regression(self.df_prices, self.df_ext, sample_comm, sample_factor)
        self.assertIsNotNone(fig_scatter)

    def test_all_forecasting_models(self):
        sample_comm = self.df_prices["Commodity"].iloc[0]
        models = ["ELM-GA", "XGBoost", "Prophet", "SARIMA-LSTM", "SARIMAX", "ARIMA", "Multi-Model Ensemble"]
        for m in models:
            res = generate_commodity_future_forecast(
                commodity=sample_comm,
                horizon_years=2,
                selected_model=m,
                scenario="Baseline Trend",
            )
            self.assertIn("forecast_df", res)
            self.assertEqual(len(res["forecast_df"]), 2)
            fig = plot_commodity_forecast(
                res["historical_df"],
                res["forecast_df"],
                sample_comm,
                res["unit"],
                m,
            )
            self.assertIsNotNone(fig)

    def test_reports_pdf_generation(self):
        test_pdf = config.REPORTS_DIR / "test_report.pdf"
        generate_pdf_report(self.eval_df, self.overall_metrics, self.comp_df, output_pdf_path=test_pdf)
        self.assertTrue(test_pdf.exists())
        self.assertGreater(test_pdf.stat().st_size, 1000)
        if test_pdf.exists():
            test_pdf.unlink()


if __name__ == "__main__":
    unittest.main()
