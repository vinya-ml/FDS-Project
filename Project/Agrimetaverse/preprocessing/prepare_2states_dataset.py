"""
Agrimetaverse Intelligence - 2-State Mandi Dataset Ingestion & Preprocessing
=============================================================================
Processes agriculture_prices_2states_cleaned.csv covering:
- 2 States: Maharashtra & Karnataka
- 5 Essential Staples: Onion, Potato, Wheat, Tomato, Rice
- 66,200+ raw transaction records across 2023-2025
- Generates continuous daily time series for all 5 commodities (overall & state-specific)
- Extracts 13 agro-climatic, weather, and shock drivers
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "agriculture_prices_2states_cleaned.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "cleaned_commodity_prices.csv"
EXTERNAL_FILE = BASE_DIR / "data" / "external" / "enhanced_features.csv"

COMMODITY_CATEGORIES = {
    "Onion": "Vegetables",
    "Potato": "Vegetables",
    "Tomato": "Vegetables",
    "Wheat": "Grains & Cereals",
    "Rice": "Grains & Cereals",
}


def process_2states_dataset(
    raw_path: Path = RAW_FILE,
    processed_out: Path = PROCESSED_FILE,
    external_out: Path = EXTERNAL_FILE,
):
    print(f"Reading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1. Process 13 External Agro-Climatic Features
    print("Extracting 13 external agro-climatic features...")
    ext = df.groupby("date").agg({
        "rainfall_mm": "mean",
        "temp_mean": "mean",
        "temp_min": "min",
        "temp_max": "max",
        "humidity": "mean",
        "wind_speed": "mean",
        "solar_radiation": "mean",
        "drought_indicator": "mean",
        "flood_indicator": "mean",
        "rainfall_mm_7d_avg": "mean",
        "rainfall_mm_30d_avg": "mean",
        "temp_mean_7d_avg": "mean",
        "temp_mean_30d_avg": "mean",
    }).reset_index()

    ext = ext.rename(columns={
        "date": "Date",
        "rainfall_mm": "Rainfall_mm",
        "temp_mean": "Avg_Temperature_C",
        "temp_min": "Temp_Min",
        "temp_max": "Temp_Max",
        "humidity": "Humidity_%",
        "wind_speed": "Wind_Speed",
        "solar_radiation": "Solar_Radiation",
        "drought_indicator": "Drought_Index",
        "flood_indicator": "Flood_Index",
        "rainfall_mm_7d_avg": "Rainfall_7d_Avg",
        "rainfall_mm_30d_avg": "Rainfall_30d_Avg",
        "temp_mean_7d_avg": "Temp_7d_Avg",
        "temp_mean_30d_avg": "Temp_30d_Avg",
    })

    # Forward & backward fill missing external factors
    ext = ext.ffill().bfill()
    ext["Year"] = ext["Date"].dt.year
    ext.to_csv(external_out, index=False)
    print(f"Saved {len(ext)} external feature records to: {external_out}")

    # 2. Process Commodity Time-Series
    print("Constructing continuous daily price series per commodity & state...")
    cleaned_records = []

    # A) Overall Commodity Series (averaged across states)
    for comm in sorted(df["Commodity"].unique()):
        sub = df[df["Commodity"] == comm].groupby("date")["Modal_Price"].mean()
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        sub_reindexed = sub.reindex(full_idx).ffill().bfill()

        cat = COMMODITY_CATEGORIES.get(comm, "Other")
        for dt, p in zip(full_idx, sub_reindexed.values):
            cleaned_records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Year": dt.year,
                "Commodity": comm,
                "State": "All",
                "Category": cat,
                "Price": round(float(p), 2),
                "Unit": "₹/Quintal",
            })

    # B) State-Specific Commodity Series (Maharashtra & Karnataka)
    for (state, comm), group in df.groupby(["STATE", "Commodity"]):
        sub = group.groupby("date")["Modal_Price"].mean()
        full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        sub_reindexed = sub.reindex(full_idx).ffill().bfill()

        cat = COMMODITY_CATEGORIES.get(comm, "Other")
        series_name = f"{comm} ({state})"
        for dt, p in zip(full_idx, sub_reindexed.values):
            cleaned_records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Year": dt.year,
                "Commodity": series_name,
                "State": state,
                "Category": cat,
                "Price": round(float(p), 2),
                "Unit": "₹/Quintal",
            })

    df_cleaned = pd.DataFrame(cleaned_records)
    df_cleaned.to_csv(processed_out, index=False)
    print(f"Saved {len(df_cleaned)} cleaned commodity records ({df_cleaned['Commodity'].nunique()} unique series) to: {processed_out}")
    return df_cleaned, ext


if __name__ == "__main__":
    process_2states_dataset()
