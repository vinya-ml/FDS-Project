"""
Agrimetaverse Intelligence - Data Loader Module
===============================================
Inspects, loads, converts, and validates historical agricultural commodity
price data and external econometric/climatic drivers.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def inspect_raw_dataset(file_path: Path = config.RAW_DATA_FILE) -> Dict[str, Any]:
    """
    Inspects the raw CSV file without modifying it.
    Detects dimensions, columns, data types, missing values, and structure format.
    """
    if not os.path.exists(file_path):
        return {
            "exists": False,
            "error": f"Dataset file not found at: {file_path}",
            "is_wide": False,
            "columns": [],
            "row_count": 0,
            "col_count": 0,
            "missing_values": 0,
        }

    df_raw = pd.read_csv(file_path)
    cols = df_raw.columns.tolist()
    
    # Check if first column represents commodity and other columns are years
    first_col = cols[0]
    year_cols = [c for c in cols[1:] if str(c).isdigit()]
    is_wide = len(year_cols) >= 5

    return {
        "exists": True,
        "is_wide": is_wide,
        "first_col": first_col,
        "columns": cols,
        "year_columns": year_cols,
        "row_count": len(df_raw),
        "col_count": len(cols),
        "rows": len(df_raw),
        "cols": len(cols),
        "missing_values": int(df_raw.isna().sum().sum()),
        "head_sample": df_raw.head(3).to_dict(orient="records"),
    }


def load_raw_price_data(file_path: Path = None) -> pd.DataFrame:
    """
    Loads price data. If processed file exists, loads it directly.
    Otherwise parses raw dataset (wide format or agriculture_prices_2states_cleaned.csv)
    into standard long time-series format: [Commodity, Category, Date/Year, Price, Unit].
    """
    if file_path is None or str(file_path) == str(config.RAW_DATA_FILE):
        if config.PROCESSED_DATA_FILE.exists():
            df_proc = pd.read_csv(config.PROCESSED_DATA_FILE)
            if "Price" in df_proc.columns and "Commodity" in df_proc.columns:
                return df_proc
        file_path = config.RAW_DATA_FILE

    if not os.path.exists(file_path):
        if config.PROCESSED_DATA_FILE.exists():
            return pd.read_csv(config.PROCESSED_DATA_FILE)
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please place the dataset in data/raw/."
        )

    df_raw = pd.read_csv(file_path)

    # Detect if agriculture_prices_2states_cleaned.csv
    if "Modal_Price" in df_raw.columns and "Commodity" in df_raw.columns and "date" in df_raw.columns:
        from preprocessing.prepare_2states_dataset import process_2states_dataset
        df_cleaned, _ = process_2states_dataset(raw_path=file_path)
        return df_cleaned

    # Detect if already processed long format
    if "Price" in df_raw.columns and "Commodity" in df_raw.columns:
        return df_raw

    # Detect if wide format
    cols = df_raw.columns.tolist()
    first_col = cols[0]
    year_cols = [c for c in cols if str(c).isdigit()]
    if year_cols:
        id_col = first_col
        df_long = pd.melt(
            df_raw,
            id_vars=[id_col],
            value_vars=year_cols,
            var_name="Year",
            value_name="Price",
        )
        df_long = df_long.rename(columns={id_col: "Commodity"})
    else:
        df_long = df_raw.copy()

    # Convert Year to integer and Price to numeric
    if "Year" in df_long.columns:
        df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype(int)
    df_long["Price"] = pd.to_numeric(df_long["Price"], errors="coerce")

    # Clean and canonicalize commodity names
    df_long["Commodity"] = df_long["Commodity"].astype(str).str.strip()
    df_long["Commodity"] = df_long["Commodity"].map(
        lambda name: config.CANONICAL_COMMODITY_MAPPING.get(name, name)
    )

    # Add category and unit
    df_long["Category"] = df_long["Commodity"].map(
        lambda c: config.COMMODITY_TO_CATEGORY.get(c, "Other")
    )
    df_long["Unit"] = df_long["Commodity"].map(
        lambda c: config.COMMODITY_UNITS.get(c, config.DEFAULT_UNIT)
    )

    return df_long


def load_external_features(file_path: Path = config.EXTERNAL_FEATURES_FILE) -> pd.DataFrame:
    """
    Loads external macroeconomic, agro-climatic, and shock features.
    Validates presence of the 13 research paper external factors.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["Date", "Year"] + config.EXTERNAL_FEATURE_COLUMNS)

    df_ext = pd.read_csv(file_path)
    if "Year" in df_ext.columns:
        df_ext["Year"] = pd.to_numeric(df_ext["Year"], errors="coerce").fillna(2024).astype(int)
    
    clean_cols = {col: col.strip() for col in df_ext.columns}
    df_ext = df_ext.rename(columns=clean_cols)
    return df_ext


def get_feature_availability_report(df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a clear transparent report identifying which of the 13 external
    features are present, their units, domain, data source status, and missingness.
    """
    report_rows = []
    total_years = len(getattr(config, "ALL_HISTORICAL_YEARS", [])) or len(df_ext)

    for feature in config.EXTERNAL_FEATURE_COLUMNS:
        meta = config.EXTERNAL_FEATURE_METADATA.get(feature, {})
        is_present = feature in df_ext.columns
        if is_present:
            non_null_count = int(df_ext[feature].notna().sum())
            min_val = float(df_ext[feature].min()) if non_null_count > 0 else np.nan
            max_val = float(df_ext[feature].max()) if non_null_count > 0 else np.nan
            status = "Supplied (Empirical / Reference Index)"
        else:
            non_null_count = 0
            min_val = np.nan
            max_val = np.nan
            status = "Missing (Requires User Upload)"

        report_rows.append({
            "Feature_Code": feature,
            "Label": meta.get("label", feature),
            "Domain": meta.get("domain", "General"),
            "Unit": meta.get("unit", "-"),
            "Present": is_present,
            "Coverage_Years": f"{non_null_count}/{total_years}",
            "Min_Value": min_val,
            "Max_Value": max_val,
            "Data_Status": status,
        })

    return pd.DataFrame(report_rows)


def merge_prices_and_features(
    df_prices: pd.DataFrame, df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges long-format commodity prices with external features aligned on Year.
    Prevents lookahead bias by strictly joining on Year.
    """
    if df_features.empty or "Year" not in df_features.columns:
        return df_prices.copy()

    merged = pd.merge(df_prices, df_features, on="Year", how="left")
    return merged
