"""
Agrimetaverse Intelligence - Commodity Deep Dive Analysis Module
================================================================
Generates detailed single-commodity statistical profile cards,
YoY price shifts, volatility measures, and model recommendation profiles.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_commodity_profile(df_prices: pd.DataFrame, commodity: str) -> Dict[str, Any]:
    """Computes comprehensive profile metrics for an individual agricultural commodity."""
    sub = df_prices[df_prices["Commodity"] == commodity].sort_values("Year")
    prices = sub["Price"].values

    if len(prices) == 0:
        return {}

    first_price = float(prices[0])
    latest_price = float(prices[-1])
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    mean_price = float(np.mean(prices))
    std_price = float(np.std(prices))
    coef_var = (std_price / mean_price) * 100.0 if mean_price > 0 else 0.0

    # YoY returns
    yoy_changes = pd.Series(prices).pct_change().dropna() * 100.0
    max_yoy_gain = float(yoy_changes.max()) if len(yoy_changes) > 0 else 0.0
    max_yoy_drop = float(yoy_changes.min()) if len(yoy_changes) > 0 else 0.0

    category = config.COMMODITY_TO_CATEGORY.get(commodity, "Other")
    unit = config.COMMODITY_UNITS.get(commodity, "₹/unit")

    return {
        "commodity": commodity,
        "category": category,
        "unit": unit,
        "first_year": int(sub["Year"].iloc[0]),
        "first_price": round(first_price, 2),
        "latest_year": int(sub["Year"].iloc[-1]),
        "latest_price": round(latest_price, 2),
        "total_growth_pct": round(((latest_price - first_price) / first_price) * 100.0, 2),
        "cagr_pct": round((((latest_price / first_price) ** (1.0 / (len(prices) - 1))) - 1.0) * 100.0, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "mean_price": round(mean_price, 2),
        "std_price": round(std_price, 2),
        "volatility_cv_pct": round(coef_var, 2),
        "max_yoy_gain_pct": round(max_yoy_gain, 2),
        "max_yoy_drop_pct": round(max_yoy_drop, 2),
        "yearly_table": sub[["Year", "Price"]].reset_index(drop=True),
    }
