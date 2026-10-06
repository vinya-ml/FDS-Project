import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("predictive_final.csv", parse_dates=["date"])
df = df.set_index(["market_commodity", "date"])

Xcols = [
    "temp_anomaly", "rain_anomaly",
    "extreme_temp", "extreme_rain",
    "flood_indicator", "drought_indicator",
    "price_lag_7d", "price_lag_30d"
]

model = PanelOLS(
    df["future_volatility_30d"],
    df[Xcols],
    entity_effects=True,
    time_effects=True
)

result = model.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(result.summary)

with open("predictive_results.txt", "w") as f:
    f.write(str(result.summary))