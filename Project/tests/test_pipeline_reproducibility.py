"""
Pipeline Reproducibility and Artifacts Verification Tests
Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather
"""

import os
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_generated_tables_exist():
    """Verify that all core summary tables are present in outputs/tables/."""
    required_tables = [
        "data_pipeline_stage_counts.csv",
        "temporal_gaps_summary.csv",
        "commodity_price_summary.csv",
        "eda_commodity_volatility_summary.csv",
        "eda_correlation_matrix.csv",
        "econometric_commodity_models.csv",
        "econometric_rainfall_windows.csv",
        "ml_model_evaluation_metrics.csv",
        "robustness_weather_ablation.csv"
    ]
    tables_dir = os.path.join(BASE_DIR, "outputs", "tables")
    for tbl in required_tables:
        path = os.path.join(tables_dir, tbl)
        assert os.path.exists(path), f"Missing required table: {tbl}"

def test_generated_figures_exist():
    """Verify that publication figures are present in outputs/figures/."""
    required_figures = [
        "01_commodity_price_trends.png",
        "02_commodity_price_distributions.png",
        "03_volatility_30d_distributions.png",
        "04_weather_anomaly_distributions.png",
        "06_extreme_weather_impact_comparison.png",
        "08_correlation_matrix_annotated.png",
        "09_commodity_comprehensive_summary.png",
        "data_quality_attrition_waterfall.png",
        "ml_actual_vs_predicted.png",
        "ml_feature_importance.png",
        "robustness_weather_ablation.png"
    ]
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    for fig in required_figures:
        path = os.path.join(figures_dir, fig)
        assert os.path.exists(path), f"Missing required figure: {fig}"

def test_ml_metrics_validity():
    """Verify that predictive model metrics are within mathematically valid bounds."""
    metrics_path = os.path.join(BASE_DIR, "outputs", "tables", "ml_model_evaluation_metrics.csv")
    assert os.path.exists(metrics_path), "ml_model_evaluation_metrics.csv missing!"
    df_metrics = pd.read_csv(metrics_path)
    
    assert (df_metrics["Test_MAE"] > 0).all(), "MAE must be positive!"
    assert (df_metrics["Test_RMSE"] > 0).all(), "RMSE must be positive!"
    assert (df_metrics["Test_R2"] <= 1.0).all(), "R2 cannot exceed 1.0!"
