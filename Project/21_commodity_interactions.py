import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("predictive_final.csv", parse_dates=["date"])
df = df.dropna(subset=["future_volatility_30d"])

df["onion"] = (df["Commodity"] == "Onion").astype(int)
df["potato"] = (df["Commodity"] == "Potato").astype(int)

for x in ["rain_anomaly", "temp_anomaly", "flood_indicator", "drought_indicator"]:
    df[x+"_onion"] = df[x] * df["onion"]
    df[x+"_potato"] = df[x] * df["potato"]

cols = [
    "rain_anomaly", "temp_anomaly", "flood_indicator", "drought_indicator",
    "rain_anomaly_onion", "rain_anomaly_potato",
    "temp_anomaly_onion", "temp_anomaly_potato",
    "flood_indicator_onion", "flood_indicator_potato",
    "drought_indicator_onion", "drought_indicator_potato",
    "price_lag_7d", "price_lag_30d"
]

df = df.set_index(["market_commodity", "date"])

result = PanelOLS(
    df["future_volatility_30d"],
    df[cols],
    entity_effects=True,
    time_effects=True
).fit(
    cov_type="clustered",
    cluster_entity=True,
    low_memory=True
)

print(result.summary)

with open("commodity_interaction_results.txt", "w") as f:
    f.write(str(result.summary))

print("\nResults saved to commodity_interaction_results.txt")