import numpy as np
import pandas as pd

# 1. Load the raw dataset
input_file = "master_aggriculture_dataset.csv"
print("Loading raw dataset...")
df = pd.read_csv(input_file)
print(f"Initial shape: {df.shape}")

# -------------------------------------------------------------
# Stage 1: Select Only Usable Columns
# -------------------------------------------------------------
# Dropping: temp_min, temp_max, humidity, wind_speed, solar_radiation
required_columns = [
    "STATE",
    "District",
    "Market Name",
    "Commodity",
    "Variety",
    "Grade",
    "date",
    "Min_Price",
    "Max_Price",
    "Modal_Price",
    "temp_mean",
    "rainfall_mm",
    "rainfall_mm_30d_sum",
    "flood_indicator",
    "drought_indicator",
    "price_lag_7d",
    "price_lag_30d",
    "price_7d_avg",
]

# Keep only matching existing columns to prevent KeyErrors
existing_cols = [col for col in required_columns if col in df.columns]
df = df[existing_cols].copy()

# -------------------------------------------------------------
# Stage 2: Standardize String Formatting & Parse Dates
# -------------------------------------------------------------
string_cols = ["STATE", "District", "Market Name", "Commodity", "Variety", "Grade"]
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Parse date column safely
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

# -------------------------------------------------------------
# Stage 3: Drop Duplicates
# -------------------------------------------------------------
initial_len = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_len - len(df)} exact duplicate rows.")

# -------------------------------------------------------------
# Stage 4: Enforce Price Logic Bounds
# -------------------------------------------------------------
# Keep only valid non-negative prices and logical spreads (Min <= Max)
price_mask = (
    (df["Modal_Price"] > 0)
    & (df["Min_Price"] > 0)
    & (df["Max_Price"] > 0)
    & (df["Min_Price"] <= df["Max_Price"])
)
df = df[price_mask].copy()

# -------------------------------------------------------------
# Stage 5: Handle Missing Values per Mandi Group
# -------------------------------------------------------------
# Sort chronologically within each market group for valid time-series filling
df = df.sort_values(by=["Market Name", "Commodity", "date"]).reset_index(drop=True)

# Continuous numeric columns to impute
numeric_features = [
    "temp_mean",
    "rainfall_mm",
    "rainfall_mm_30d_sum",
    "price_lag_7d",
    "price_lag_30d",
    "price_7d_avg",
]
numeric_features = [col for col in numeric_features if col in df.columns]

# A. Forward-fill then backward-fill short trading gaps within each specific market
df[numeric_features] = df.groupby(["Market Name", "Commodity"])[
    numeric_features
].transform(lambda group: group.ffill().bfill())

# B. Impute extreme binary indicators with 0 if missing
for flag in ["flood_indicator", "drought_indicator"]:
    if flag in df.columns:
        df[flag] = df[flag].fillna(0).astype(int)

# C. Fill any remaining gaps with commodity median baseline
for col in numeric_features:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df.groupby("Commodity")[col].transform("median"))

# Drop any remaining unfillable edge rows to guarantee complete rows
df = df.dropna().reset_index(drop=True)

print(f"\nFinal legitimate rows count: {len(df):,}")
print(f"Remaining null values across dataset:\n{df.isna().sum()}")

# -------------------------------------------------------------
# Stage 6: Export ALL Legitimate Rows to cleaned_mandi.csv
# -------------------------------------------------------------
output_file = "cleaned_mandi.csv"
print(f"\nExporting all {len(df):,} clean rows to {output_file}...")
df.to_csv(output_file, index=False)
print(f"Done! All clean rows saved to {output_file}.")