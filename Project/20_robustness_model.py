import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("alternative_volatility.csv", parse_dates=["date"])
df = df.dropna(subset=["future_abs_return_30d"])
df = df.set_index(["market_commodity", "date"])

X = df[[
    "temp_anomaly", "rain_anomaly",
    "flood_indicator", "drought_indicator",
    "price_lag_7d", "price_lag_30d"
]]

r = PanelOLS(
    df["future_abs_return_30d"], X,
    entity_effects=True, time_effects=True
).fit(
    cov_type="clustered",
    cluster_entity=True
)

print(r.summary)

with open("robustness_results.txt", "w") as f:
    f.write(str(r.summary))