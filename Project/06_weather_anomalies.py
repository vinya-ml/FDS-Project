import pandas as pd

df = pd.read_csv("volatility_mandi.csv", parse_dates=["date"])

df["month"] = df["date"].dt.month

temp_avg = df.groupby(["STATE", "month"])["temp_mean"].transform("mean")
rain_avg = df.groupby(["STATE", "month"])["rainfall_mm"].transform("mean")

df["temp_anomaly"] = df["temp_mean"] - temp_avg
df["rain_anomaly"] = df["rainfall_mm"] - rain_avg

df.to_csv("weather_mandi.csv", index=False)

print(df[["temp_anomaly", "rain_anomaly"]].describe())