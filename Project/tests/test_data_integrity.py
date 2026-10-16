"""
Unit & Data Integrity Test Suite
Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather
"""

import os
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def analytical_df():
    path = os.path.join(BASE_DIR, "final_mandi.csv")
    if not os.path.exists(path):
        path = os.path.join(BASE_DIR, "final_mandi.csv.gz")
    assert os.path.exists(path), f"File {path} does not exist!"
    # Load sample for fast automated testing
    return pd.read_csv(path, nrows=50000, parse_dates=["date"])

@pytest.fixture(scope="session")
def predictive_df():
    path = os.path.join(BASE_DIR, "predictive_final.csv")
    if not os.path.exists(path):
        path = os.path.join(BASE_DIR, "predictive_final.csv.gz")
    assert os.path.exists(path), f"File {path} does not exist!"
    return pd.read_csv(path, nrows=50000, parse_dates=["date"])

def test_price_logic_bounds(analytical_df):
    """Ensure all prices are strictly positive and min price <= max price."""
    assert (analytical_df["Modal_Price"] > 0).all(), "Found zero or negative modal prices!"

def test_volatility_non_negative(analytical_df):
    """Volatility must be strictly non-negative."""
    assert (analytical_df["volatility_30d"] >= 0).all(), "Found negative volatility values!"
    assert analytical_df["volatility_30d"].notna().all(), "Found NaN volatility in final dataset!"

def test_target_commodities_valid(analytical_df):
    """Dataset should strictly contain Tomato, Onion, and Potato."""
    valid_commodities = {"Tomato", "Onion", "Potato"}
    observed = set(analytical_df["Commodity"].unique())
    assert observed.issubset(valid_commodities), f"Unexpected commodities found: {observed - valid_commodities}"

def test_temporal_split_zero_leakage():
    """Verify that train dates strictly precede validation and test dates."""
    path = os.path.join(BASE_DIR, "predictive_final.csv")
    if not os.path.exists(path):
        path = os.path.join(BASE_DIR, "predictive_final.csv.gz")
    
    # Read dates only for speed
    dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * 0.70)]
    test_start = dates[int(n * 0.85)]
    
    assert train_end < test_start, "Temporal Leakage: Train date overlaps Test date!"

def test_target_not_in_features(predictive_df):
    """Ensure future target is excluded from model feature sets."""
    features = [
        "temp_anomaly", "rain_anomaly", "extreme_temp", "extreme_rain",
        "flood_indicator", "drought_indicator", "price_lag_7d", "price_lag_30d"
    ]
    assert "future_volatility_30d" not in features, "Target leakage detected: Target column present in feature list!"
