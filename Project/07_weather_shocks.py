import pandas as pd

df = pd.read_csv("weather_mandi.csv", parse_dates=["date"])
df = df.sort_values(["market_commodity", "date"])

t95 = df["temp_anomaly"].quantile(.95)
r95 = df["rain_anomaly"].quantile(.95)

df["extreme_temp"] = (df["temp_anomaly"] > t95).astype(int)
df["extreme_rain"] = (df["rain_anomaly"] > r95).astype(int)

g = df.groupby("market_commodity")

df["temp_anomaly_7d"] = g["temp_anomaly"].transform(
    lambda x: x.rolling(7, min_periods=5).mean()
)

df["rain_anomaly_7d"] = g["rain_anomaly"].transform(
    lambda x: x.rolling(7, min_periods=5).sum()
)

df["rain_anomaly_14d"] = g["rain_anomaly"].transform(
    lambda x: x.rolling(14, min_periods=10).sum()
)

df["rain_anomaly_30d"] = g["rain_anomaly"].transform(
    lambda x: x.rolling(30, min_periods=20).sum()
)

df.to_csv("analysis_mandi.csv", index=False)

print("Extreme temperature:", df["extreme_temp"].sum())
print("Extreme rainfall:", df["extreme_rain"].sum())