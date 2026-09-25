import numpy as np
import pandas as pd

# 1. Load the raw dataset
input_file = "master_aggriculture_dataset.csv"
print("Loading raw dataset...")
df = pd.read_csv(input_file)
print(f"Initial shape: {df.shape}")

# -------------------------------------------------------------
# Stage 1: Standardize String Formatting
# -------------------------------------------------------------
string_cols = ["STATE", "District", "Market Name", "Commodity", "Variety", "Grade"]
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# -------------------------------------------------------------
# Stage 2: Filter Out Andhra Pradesh (Preserving All 23 Columns)
# -------------------------------------------------------------
# Dropping Andhra Pradesh rows resolves the 100% missing values in:
# temp_min, temp_max, humidity, wind_speed, solar_radiation
initial_count = len(df)
df = df[df["STATE"] != "Andhra Pradesh"].copy()
print(f"Filtered out {initial_count - len(df):,} rows belonging to Andhra Pradesh.")

# -------------------------------------------------------------
# Stage 3: Parse Dates Safely
# -------------------------------------------------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

# -------------------------------------------------------------
# Stage 4: Drop Duplicates
# -------------------------------------------------------------
initial_dedup_len = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_dedup_len - len(df):,} exact duplicate rows.")

# -------------------------------------------------------------
# Stage 5: Enforce Price Logic Bounds
# -------------------------------------------------------------
price_mask = (
    (df["Modal_Price"] > 0)
    & (df["Min_Price"] > 0)
    & (df["Max_Price"] > 0)
    & (df["Min_Price"] <= df["Max_Price"])
)
df = df[price_mask].copy()

# -------------------------------------------------------------
# Stage 6: Time-Series Imputation Across All Numeric Features
# -------------------------------------------------------------
# Sort chronologically per market group
df = df.sort_values(by=["Market Name", "Commodity", "date"]).reset_index(drop=True)

# Include all meteorological and price numeric columns
numeric_features = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "humidity",
    "wind_speed",
    "solar_radiation",
    "rainfall_mm",
    "rainfall_mm_30d_sum",
    "price_lag_7d",
    "price_lag_30d",
    "price_7d_avg",
]
numeric_features = [col for col in numeric_features if col in df.columns]

# A. Forward-fill then backward-fill short trading gaps within each specific mandi
df[numeric_features] = df.groupby(["Market Name", "Commodity"])[
    numeric_features
].transform(lambda group: group.ffill().bfill())

# B. Impute extreme binary flags with 0 if missing
for flag in ["flood_indicator", "drought_indicator"]:
    if flag in df.columns:
        df[flag] = df[flag].fillna(0).astype(int)

# C. Fill any remaining sporadic gaps with the commodity median baseline
for col in numeric_features:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df.groupby("Commodity")[col].transform("median"))

# Drop any unfillable edge rows
df = df.dropna().reset_index(drop=True)

print(f"\nFinal cleaned dataset shape: {df.shape}")
print(f"Remaining null values across dataset:\n{df.isna().sum()}")

# -------------------------------------------------------------
# Stage 7: Export to cleaned_mandi.csv
# -------------------------------------------------------------
output_file = "cleaned_mandi2.csv"
print(f"\nExporting all {len(df):,} rows with all {df.shape[1]} columns to {output_file}...")
df.to_csv(output_file, index=False)
print(f"Saved complete dataset to {output_file}!")
