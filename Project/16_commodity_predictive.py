import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("predictive_final.csv", parse_dates=["date"])

X = [
    "temp_anomaly", "rain_anomaly",
    "flood_indicator", "drought_indicator",
    "price_lag_7d", "price_lag_30d"
]

for c in ["Tomato", "Onion", "Potato"]:
    d = df[df["Commodity"] == c].set_index(
        ["market_commodity", "date"]
    )

    model = PanelOLS(
        d["future_volatility_30d"], d[X],
        entity_effects=True, time_effects=True
    )

    result = model.fit(
        cov_type="clustered",
        cluster_entity=True
    )

    print("\n==========", c, "==========")
    print(result.summary)