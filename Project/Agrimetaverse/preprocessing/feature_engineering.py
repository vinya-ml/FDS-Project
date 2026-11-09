"""
Agrimetaverse Intelligence - Feature Engineering Module
======================================================
Derives lagged price indicators, growth rates, rolling past aggregates,
and macro-climate interaction ratios without lookahead bias.
"""

from typing import List
import pandas as pd
import numpy as np


def engineer_commodity_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-series lag features, percentage changes, and rolling statistics
    per commodity strictly using past observations.
    """
    time_col = "Date" if "Date" in df_prices.columns else "Year"
    df = df_prices.copy()
    df = df.sort_values(by=["Commodity", time_col]).reset_index(drop=True)

    # 1. Previous price (Lag 1)
    df["Price_Lag1"] = df.groupby("Commodity")["Price"].shift(1)

    # 2. Lag 2 price
    df["Price_Lag2"] = df.groupby("Commodity")["Price"].shift(2)

    # 3. Price percentage change
    df["Price_Pct_Change"] = df.groupby("Commodity")["Price"].pct_change(1) * 100.0

    # 4. Rolling Mean of past prices (closed='left' to prevent lookahead)
    roll_win = 7 if time_col == "Date" else 2
    df["Price_Rolling_Mean"] = (
        df.groupby("Commodity")["Price"]
        .transform(lambda s: s.shift(1).rolling(window=roll_win, min_periods=1).mean())
    )

    # 5. Volatility / Rolling Std
    df["Price_Rolling_Std"] = (
        df.groupby("Commodity")["Price"]
        .transform(lambda s: s.shift(1).rolling(window=roll_win, min_periods=1).std())
    ).fillna(0.0)

    return df


def engineer_external_interaction_features(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs interaction ratios between external factors and commodity prices:
    - Rain_to_Temp_Ratio: Agro-climatic favorability proxy.
    - Weather_Stress_Index: Drought indicator combined with thermal index.
    """
    df = df_combined.copy()

    # Weather ratio
    if "Rainfall_mm" in df.columns and "Avg_Temperature_C" in df.columns:
        df["Rain_to_Temp_Ratio"] = df["Rainfall_mm"] / (df["Avg_Temperature_C"] + 1e-5)

    if "Drought_Index" in df.columns and "Avg_Temperature_C" in df.columns:
        df["Thermal_Moisture_Stress"] = df["Drought_Index"] * df["Avg_Temperature_C"]

    # Energy-transport ratio
    if "Fuel_Price_Rs/L" in df.columns and "Transport_Cost_Index" in df.columns:
        df["Fuel_to_Transport_Ratio"] = df["Fuel_Price_Rs/L"] / (df["Transport_Cost_Index"] + 1e-5)

    return df


def create_modeling_dataset(
    df_prices: pd.DataFrame, df_features: pd.DataFrame
) -> pd.DataFrame:
    """
    End-to-end feature pipeline producing the fully engineered modeling dataframe.
    """
    time_col = "Date" if ("Date" in df_prices.columns and "Date" in df_features.columns) else "Year"
    merged = pd.merge(df_prices, df_features, on=time_col, how="left")
    
    # Commodity features
    with_ts_features = engineer_commodity_features(merged)
    
    # Interaction features
    final_df = engineer_external_interaction_features(with_ts_features)
    return final_df
