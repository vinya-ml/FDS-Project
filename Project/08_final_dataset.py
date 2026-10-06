import pandas as pd

df = pd.read_csv("analysis_mandi.csv", parse_dates=["date"])

cols = [
    "STATE", "district", "Market Name", "Commodity", "date",
    "market_commodity", "Modal_Price", "log_return", "volatility_30d",
    "temp_anomaly", "rain_anomaly", "extreme_temp", "extreme_rain",
    "temp_anomaly_7d", "rain_anomaly_7d", "rain_anomaly_14d",
    "rain_anomaly_30d", "flood_indicator", "drought_indicator",
    "price_lag_7d", "price_lag_30d", "price_7d_avg"
]

df = df[cols].dropna(subset=["volatility_30d"])

df.to_csv("final_mandi.csv", index=False)

print("Final dataset:", df.shape)
print("\nCommodities:\n", df["Commodity"].value_counts())