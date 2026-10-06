import pandas as pd
from linearmodels.panel import PanelOLS

df = pd.read_csv("predictive_final.csv", parse_dates=["date"])

for w in ["rain_anomaly_7d", "rain_anomaly_14d", "rain_anomaly_30d"]:
    d = df.dropna(subset=[w]).set_index(
        ["market_commodity", "date"]
    )

    X = d[[w, "temp_anomaly", "flood_indicator","drought_indicator", "price_lag_7d", "price_lag_30d"]]

    r = PanelOLS(
        d["future_volatility_30d"], X,
        entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)

    print(f"\n========== {w} ==========")
    print(r.summary)