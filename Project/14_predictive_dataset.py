import pandas as pd

df = pd.read_csv("predictive_mandi.csv", parse_dates=["date"])

cols = [
    "STATE", "district", "Market Name", "Commodity", "date",
    "market_commodity", "future_volatility_30d",
    "temp_anomaly", "rain_anomaly",
    "extreme_temp", "extreme_rain",
    "flood_indicator", "drought_indicator",
    "rain_anomaly_7d", "rain_anomaly_14d",
    "rain_anomaly_30d", "price_lag_7d", "price_lag_30d"
]

df = df[cols].dropna(subset=["future_volatility_30d"])

df.to_csv("predictive_final.csv", index=False)

print("Final predictive dataset:", df.shape)
print("\nCommodities:")
print(df["Commodity"].value_counts())
print("\nMissing values:")
print(df.isnull().sum())