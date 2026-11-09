"""
Agrimetaverse Intelligence - Data Cleaning Module
=================================================
Handles duplicate removal, invalid price detection, commodity-isolated
forward filling, and data quality auditing.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def clean_commodity_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw long-format commodity dataset:
    - Removes duplicate (Commodity, Year) pairs.
    - Validates price sanity (Price > 0).
    - Performs forward fill isolated by commodity (never cross-filling between commodities).
    - Audits missingness and fill operations.
    """
    initial_rows = len(df)
    cleaning_audit = {
        "initial_rows": initial_rows,
        "duplicates_removed": 0,
        "invalid_prices_found": 0,
        "values_forward_filled": 0,
        "commodities_count": df["Commodity"].nunique(),
        "years_present": sorted(df["Year"].unique().tolist()),
    }

    cleaned = df.copy()

    # 1. Deduplication (use Date if available, else Year)
    time_col = "Date" if "Date" in cleaned.columns else "Year"
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["Commodity", time_col], keep="first")
    cleaning_audit["duplicates_removed"] = before_dedup - len(cleaned)

    # 2. Invalid or Negative Prices Check
    invalid_mask = (cleaned["Price"].isna()) | (cleaned["Price"] <= 0)
    cleaning_audit["invalid_prices_found"] = int(invalid_mask.sum())
    cleaned.loc[cleaned["Price"] <= 0, "Price"] = np.nan

    # 3. Commodity-Isolated Forward Filling
    # Sort strictly by Commodity and time_col first
    cleaned = cleaned.sort_values(by=["Commodity", time_col]).reset_index(drop=True)
    
    # Track missing before forward fill
    missing_before = int(cleaned["Price"].isna().sum())

    # Forward fill ONLY within each commodity group
    cleaned["Price"] = cleaned.groupby("Commodity")["Price"].transform(lambda s: s.ffill())
    
    # Backward fill ONLY if earliest observation is missing for a commodity
    cleaned["Price"] = cleaned.groupby("Commodity")["Price"].transform(lambda s: s.bfill())

    missing_after = int(cleaned["Price"].isna().sum())
    cleaning_audit["values_forward_filled"] = missing_before - missing_after
    cleaning_audit["remaining_missing_prices"] = missing_after
    cleaning_audit["final_rows"] = len(cleaned)

    return cleaned, cleaning_audit


def clean_external_features(df_ext: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans external feature DataFrame:
    - Deduplicates by Date/Year.
    - Imputes missing external indicators using forward fill (time-series continuity).
    - Returns audit of filled external variables.
    """
    audit = {
        "initial_rows": len(df_ext),
        "duplicates_removed": 0,
        "filled_by_column": {},
    }
    cleaned = df_ext.copy()
    time_col = "Date" if "Date" in cleaned.columns else "Year"
    
    # Deduplicate by time_col
    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=[time_col], keep="first").sort_values(time_col).reset_index(drop=True)
    audit["duplicates_removed"] = before_dedup - len(cleaned)

    # Time-series forward fill then backfill for external indicators
    feature_cols = [c for c in cleaned.columns if c != "Year"]
    for col in feature_cols:
        null_count = int(cleaned[col].isna().sum())
        if null_count > 0:
            cleaned[col] = cleaned[col].ffill().bfill()
            audit["filled_by_column"][col] = null_count

    return cleaned, audit


def generate_data_quality_report(
    df_prices: pd.DataFrame, df_ext: pd.DataFrame, cleaning_audit: Dict[str, Any]
) -> pd.DataFrame:
    """
    Creates a formal tabular data-quality summary suitable for display and reporting.
    """
    commodities = df_prices["Commodity"].unique()
    records = []

    for c in sorted(commodities):
        c_df = df_prices[df_prices["Commodity"] == c]
        min_p = float(c_df["Price"].min())
        max_p = float(c_df["Price"].max())
        mean_p = float(c_df["Price"].mean())
        std_p = float(c_df["Price"].std()) if len(c_df) > 1 else 0.0
        missing = int(c_df["Price"].isna().sum())

        records.append({
            "Commodity": c,
            "Category": c_df["Category"].iloc[0] if "Category" in c_df.columns else "Other",
            "Observations": len(c_df),
            "Min_Price (₹)": round(min_p, 2),
            "Max_Price (₹)": round(max_p, 2),
            "Mean_Price (₹)": round(mean_p, 2),
            "Std_Dev": round(std_p, 2),
            "Missing_Count": missing,
            "Status": "Verified Healthy" if missing == 0 and min_p > 0 else "Needs Attention",
        })

    return pd.DataFrame(records)
