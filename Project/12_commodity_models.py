import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("final_mandi.csv", parse_dates=["date"])

Xcols = [
    "temp_anomaly", "rain_anomaly",
    "flood_indicator", "drought_indicator",
    "price_lag_7d", "price_lag_30d"
]

with open("commodity_results.txt", "w") as f:
    for c in ["Tomato", "Onion", "Potato"]:
        d = df[df["Commodity"] == c].set_index(
            ["market_commodity", "date"]
        )

        model = PanelOLS(
            d["volatility_30d"], d[Xcols],
            entity_effects=True, time_effects=True
        )

        result = model.fit(
            cov_type="clustered", cluster_entity=True
        )

        print("\n==========", c, "==========")
        print(result.summary)

        f.write(f"\n========== {c} ==========\n")
        f.write(str(result.summary))