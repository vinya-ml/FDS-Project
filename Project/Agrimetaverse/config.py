"""
Agrimetaverse Intelligence - Configuration & Constants
======================================================
Defines paths, commodity categories, canonical name normalizations,
the 13 external features, chronological time-series splits,
and published research paper reference benchmarks.
"""

import os
from pathlib import Path

# Base Paths
CONFIG_DIR = Path(__file__).resolve().parent
BASE_DIR = CONFIG_DIR

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "outputs" / "saved_models"

for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EXTERNAL_DIR, OUTPUTS_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset file paths
RAW_DATA_FILE = DATA_RAW_DIR / "agriculture_prices_2states_cleaned.csv"
SIH_RAW_DATA_FILE = DATA_RAW_DIR / "DatasetSIH1647.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "cleaned_commodity_prices.csv"
EXTERNAL_FEATURES_FILE = DATA_EXTERNAL_DIR / "enhanced_features.csv"
PAPER_FORECASTS_FILE = DATA_EXTERNAL_DIR / "paper_forecasts_2025_2029.csv"
EVALUATION_METRICS_FILE = OUTPUTS_DIR / "model_evaluation_metrics.csv"
FUTURE_FORECASTS_FILE = OUTPUTS_DIR / "all_commodities_future_forecasts.csv"

# Chronological Time-Series Splits (Daily Mandi Pricing Horizon: June 2023 - June 2025)
TRAIN_RATIO = 0.70   # First 70% of chronological observations
VAL_RATIO = 0.15     # Middle 15% validation partition
TEST_RATIO = 0.15    # Final 15% holdout test partition
FORECAST_HORIZONS = [7, 14, 30, 60, 90, 180]
DEFAULT_HORIZON_DAYS = 30
TRAIN_YEARS = list(range(2014, 2022))
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024]
ALL_HISTORICAL_YEARS = list(range(2014, 2025))
PAPER_FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029]

# Canonical Commodity Name Mapping
CANONICAL_COMMODITY_MAPPING = {
    "Onion": "Onion",
    "Potato": "Potato",
    "Tomato": "Tomato",
    "Wheat": "Wheat",
    "Rice": "Rice",
    "Onion (Maharashtra)": "Onion (Maharashtra)",
    "Potato (Maharashtra)": "Potato (Maharashtra)",
    "Tomato (Maharashtra)": "Tomato (Maharashtra)",
    "Wheat (Maharashtra)": "Wheat (Maharashtra)",
    "Rice (Maharashtra)": "Rice (Maharashtra)",
    "Onion (Karnataka)": "Onion (Karnataka)",
    "Potato (Karnataka)": "Potato (Karnataka)",
    "Tomato (Karnataka)": "Tomato (Karnataka)",
    "Wheat (Karnataka)": "Wheat (Karnataka)",
    "Rice (Karnataka)": "Rice (Karnataka)",
}

# Commodity Categories (Indian Agricultural Mandi Classification)
COMMODITY_CATEGORIES = {
    "Vegetables": [
        "Onion",
        "Potato",
        "Tomato",
        "Onion (Maharashtra)",
        "Potato (Maharashtra)",
        "Tomato (Maharashtra)",
        "Onion (Karnataka)",
        "Potato (Karnataka)",
        "Tomato (Karnataka)",
    ],
    "Grains & Cereals": [
        "Wheat",
        "Rice",
        "Wheat (Maharashtra)",
        "Rice (Maharashtra)",
        "Wheat (Karnataka)",
        "Rice (Karnataka)",
    ],
}

# Inverted mapping: Commodity -> Category
COMMODITY_TO_CATEGORY = {}
for category, items in COMMODITY_CATEGORIES.items():
    for item in items:
        COMMODITY_TO_CATEGORY[item] = category

# Standard Indian Mandi Units of measurement
DEFAULT_UNIT = "₹/Quintal"
COMMODITY_UNITS = {c: DEFAULT_UNIT for items in COMMODITY_CATEGORIES.values() for c in items}

# 13 External Agro-Climatic, Weather & Shock Indicators
EXTERNAL_FEATURE_COLUMNS = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Temp_Min",
    "Temp_Max",
    "Humidity_%",
    "Wind_Speed",
    "Solar_Radiation",
    "Drought_Index",
    "Flood_Index",
    "Rainfall_7d_Avg",
    "Rainfall_30d_Avg",
    "Temp_7d_Avg",
    "Temp_30d_Avg",
]

EXTERNAL_FEATURE_METADATA = {
    "Rainfall_mm": {"label": "Daily Precipitation", "unit": "mm", "domain": "Precipitation"},
    "Avg_Temperature_C": {"label": "Mean Surface Temperature", "unit": "°C", "domain": "Thermal"},
    "Temp_Min": {"label": "Minimum Temperature", "unit": "°C", "domain": "Thermal"},
    "Temp_Max": {"label": "Maximum Temperature", "unit": "°C", "domain": "Thermal"},
    "Humidity_%": {"label": "Relative Humidity", "unit": "%", "domain": "Atmospheric"},
    "Wind_Speed": {"label": "Surface Wind Speed", "unit": "m/s", "domain": "Atmospheric"},
    "Solar_Radiation": {"label": "Solar Insolation / Radiation", "unit": "MJ/m²", "domain": "Solar"},
    "Drought_Index": {"label": "Soil Moisture Drought Indicator", "unit": "Index (0-1)", "domain": "Agro-Climatic Shock"},
    "Flood_Index": {"label": "Excessive Rainfall Flood Indicator", "unit": "Index (0-1)", "domain": "Agro-Climatic Shock"},
    "Rainfall_7d_Avg": {"label": "7-Day Cumulative Rainfall Avg", "unit": "mm", "domain": "Weather Trend"},
    "Rainfall_30d_Avg": {"label": "30-Day Cumulative Monsoon Trend", "unit": "mm", "domain": "Weather Trend"},
    "Temp_7d_Avg": {"label": "7-Day Ambient Temperature Avg", "unit": "°C", "domain": "Thermal Trend"},
    "Temp_30d_Avg": {"label": "30-Day Ambient Temperature Avg", "unit": "°C", "domain": "Thermal Trend"},
}

# Available Algorithms in Agrimetaverse Intelligence
ALGORITHMS = [
    "ELM-GA",
    "XGBoost",
    "Prophet",
    "SARIMA-LSTM",
    "SARIMAX",
    "ARIMA",
]

ALGORITHM_LABELS = {
    "ELM-GA": "ELM-GA (Extreme Learning Machine + Genetic Algorithm) 🏆 Champion",
    "XGBoost": "XGBoost (Extreme Gradient Boosting) 🥈 Runner-up",
    "Prophet": "Prophet (Decomposable Additive Model) 🥉 3rd",
    "SARIMA-LSTM": "Hybrid SARIMA-LSTM (Deep Neural Residual Correction)",
    "SARIMAX": "SARIMAX (Seasonal Exogenous ARIMA)",
    "ARIMA": "ARIMA (AutoRegressive Integrated Moving Average)",
}

# ==============================================================================
# IEEE Conference Published Research Benchmarks (ICITSIF 2026 Paper)
# Title: Agrimetaverse Intelligence Advanced Agricultural Commodity Price Forecasting System
# ==============================================================================
IEEE_PAPER_TABLE_1 = [
    {
        "Model": "SARIMAX",
        "Mean_R2": 0.756,
        "Mean_RMSE": 12.45,
        "Mean_MAPE_%": 8.92,
        "Directional_Accuracy_%": 78.4,
        "Rank": "1st 🏆 Best Model",
        "Significance": "p = 0.012",
    },
    {
        "Model": "Prophet",
        "Mean_R2": 0.723,
        "Mean_RMSE": 14.21,
        "Mean_MAPE_%": 9.67,
        "Directional_Accuracy_%": 75.2,
        "Rank": "2nd 🥈 Runner-up",
        "Significance": "p = 0.012",
    },
    {
        "Model": "ARIMA",
        "Mean_R2": 0.689,
        "Mean_RMSE": 16.78,
        "Mean_MAPE_%": 11.34,
        "Directional_Accuracy_%": 72.1,
        "Rank": "3rd 🥉 Baseline",
        "Significance": "p = 0.012",
    },
]

IEEE_PAPER_TABLE_2 = [
    {
        "Commodity_Category": "Cereals (Rice, Wheat)",
        "SARIMAX_R2": 0.812,
        "Prophet_R2": 0.756,
        "ARIMA_R2": 0.712,
        "Best_Model": "SARIMAX",
    },
    {
        "Commodity_Category": "Pulses (Dals)",
        "SARIMAX_R2": 0.745,
        "Prophet_R2": 0.701,
        "ARIMA_R2": 0.678,
        "Best_Model": "SARIMAX",
    },
    {
        "Commodity_Category": "Oils",
        "SARIMAX_R2": 0.798,
        "Prophet_R2": 0.723,
        "ARIMA_R2": 0.695,
        "Best_Model": "SARIMAX",
    },
    {
        "Commodity_Category": "Vegetables",
        "SARIMAX_R2": 0.712,
        "Prophet_R2": 0.756,
        "ARIMA_R2": 0.682,
        "Best_Model": "Prophet",
    },
    {
        "Commodity_Category": "Spices",
        "SARIMAX_R2": 0.735,
        "Prophet_R2": 0.728,
        "ARIMA_R2": 0.671,
        "Best_Model": "SARIMAX",
    },
    {
        "Commodity_Category": "Others",
        "SARIMAX_R2": 0.751,
        "Prophet_R2": 0.723,
        "ARIMA_R2": 0.689,
        "Best_Model": "SARIMAX",
    },
]

IEEE_PAPER_P_VALUE = 0.012

# Authors' Extended GitHub Repository Benchmark Metrics (All 6 Algorithms)
PAPER_BENCHMARK_RESULTS = {
    "ELM-GA": {
        "Rank": "1st 🏆 Champion",
        "R2": 0.891,
        "RMSE": 8.12,
        "MAE": 5.94,
        "MAPE": 6.45,
        "Directional_Accuracy": 86.4,
        "Wins": "13/22",
        "Win_Rate": 59.1,
        "Significance": "p < 0.001",
    },
    "XGBoost": {
        "Rank": "2nd 🥈 Runner-up",
        "R2": 0.847,
        "RMSE": 9.84,
        "MAE": 7.10,
        "MAPE": 7.32,
        "Directional_Accuracy": 82.5,
        "Wins": "9/22",
        "Win_Rate": 40.9,
        "Significance": "p < 0.01",
    },
    "Prophet": {
        "Rank": "3rd 🥉 Consistent",
        "R2": 0.763,
        "RMSE": 13.15,
        "MAE": 9.42,
        "MAPE": 9.24,
        "Directional_Accuracy": 76.8,
        "Wins": "0/22",
        "Win_Rate": 0.0,
        "Significance": "Baseline",
    },
    "SARIMA-LSTM": {
        "Rank": "4th",
        "R2": 0.721,
        "RMSE": 14.50,
        "MAE": 10.25,
        "MAPE": 10.15,
        "Directional_Accuracy": 74.2,
        "Wins": "0/22",
        "Win_Rate": 0.0,
        "Significance": "-",
    },
    "SARIMAX": {
        "Rank": "5th",
        "R2": 0.678,
        "RMSE": 15.62,
        "MAE": 11.18,
        "MAPE": 11.20,
        "Directional_Accuracy": 71.5,
        "Wins": "0/22",
        "Win_Rate": 0.0,
        "Significance": "-",
    },
    "ARIMA": {
        "Rank": "6th",
        "R2": 0.634,
        "RMSE": 17.85,
        "MAE": 12.75,
        "MAPE": 12.90,
        "Directional_Accuracy": 68.3,
        "Wins": "0/22",
        "Win_Rate": 0.0,
        "Significance": "-",
    },
}

# Commodity-Specific Published Model Dominance
PAPER_COMMODITY_CHAMPIONS = {
    # 13 ELM-GA Dominant Commodities
    "Rice": {"Best_Model": "ELM-GA", "R2": 0.923, "Category": "Grains"},
    "Wheat": {"Best_Model": "ELM-GA", "R2": 0.908, "Category": "Grains"},
    "Atta (Wheat)": {"Best_Model": "ELM-GA", "R2": 0.887, "Category": "Grains"},
    "Gram Dal": {"Best_Model": "ELM-GA", "R2": 0.901, "Category": "Pulses"},
    "Tur/Arhar Dal": {"Best_Model": "ELM-GA", "R2": 0.895, "Category": "Pulses"},
    "Groundnut Oil (Packed)": {"Best_Model": "ELM-GA", "R2": 0.876, "Category": "Oils"},
    "Mustard Oil (Packed)": {"Best_Model": "ELM-GA", "R2": 0.889, "Category": "Oils"},
    "Soya Oil (Packed)": {"Best_Model": "ELM-GA", "R2": 0.892, "Category": "Oils"},
    "Potato": {"Best_Model": "ELM-GA", "R2": 0.885, "Category": "Vegetables"},
    "Onion": {"Best_Model": "ELM-GA", "R2": 0.878, "Category": "Vegetables"},
    "Sugar": {"Best_Model": "ELM-GA", "R2": 0.896, "Category": "Other staples"},
    "Milk": {"Best_Model": "ELM-GA", "R2": 0.901, "Category": "Other staples"},
    "Salt Pack (Iodised)": {"Best_Model": "ELM-GA", "R2": 0.867, "Category": "Other staples"},
    # 9 XGBoost Dominant Commodities
    "Urad Dal": {"Best_Model": "XGBoost", "R2": 0.854, "Category": "Pulses"},
    "Moong Dal": {"Best_Model": "XGBoost", "R2": 0.843, "Category": "Pulses"},
    "Masoor Dal": {"Best_Model": "XGBoost", "R2": 0.851, "Category": "Pulses"},
    "Vanaspati (Packed)": {"Best_Model": "XGBoost", "R2": 0.847, "Category": "Oils"},
    "Sunflower Oil (Packed)": {"Best_Model": "XGBoost", "R2": 0.849, "Category": "Oils"},
    "Palm Oil (Packed)": {"Best_Model": "XGBoost", "R2": 0.845, "Category": "Oils"},
    "Tomato": {"Best_Model": "XGBoost", "R2": 0.852, "Category": "Vegetables"},
    "Gur": {"Best_Model": "XGBoost", "R2": 0.841, "Category": "Other staples"},
    "Tea Loose": {"Best_Model": "XGBoost", "R2": 0.838, "Category": "Other staples"},
}

# Published Statistical Validation Results
PAPER_STATISTICAL_VALIDATION = {
    "ANOVA": {
        "F_statistic": 10.9536,
        "p_value": 9.08e-09,
        "significance": "Highly significant differences between model classes (p < 0.001)",
    },
    "Pairwise_T_Tests": [
        {"Comparison": "ELM-GA vs XGBoost", "t_stat": 3.207, "p_value": 0.003, "Result": "Significant (p < 0.01)"},
        {"Comparison": "ELM-GA vs Prophet", "t_stat": 6.345, "p_value": 0.000001, "Result": "Highly Significant (p < 0.001)"},
        {"Comparison": "XGBoost vs Prophet", "t_stat": 3.273, "p_value": 0.002, "Result": "Significant (p < 0.01)"},
    ],
    "Effect_Sizes": [
        {"Comparison": "ELM-GA vs XGBoost", "Cohens_d": 0.967, "Effect": "Large Effect Size"},
        {"Comparison": "XGBoost vs Prophet", "Cohens_d": 0.987, "Effect": "Large Effect Size"},
    ],
}

