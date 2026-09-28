import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("final_mandi.csv", parse_dates=["date"])

df = df.set_index(["market_commodity", "date"])

y = df["volatility_30d"]

X = df[[
    "temp_anomaly",
    "rain_anomaly",
    "flood_indicator",
    "drought_indicator",
    "price_lag_7d",
    "price_lag_30d"
]]

X = X.astype(float)

model = PanelOLS(
    y, X,
    entity_effects=True,
    time_effects=True
)

result = model.fit(
    cov_type="clustered",
    cluster_entity=True
)

print(result.summary)

with open("baseline_results.txt", "w") as f:
    f.write(str(result.summary))