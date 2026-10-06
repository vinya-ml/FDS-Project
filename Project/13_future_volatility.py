import pandas as pd

df = pd.read_csv("analysis_mandi.csv", parse_dates=["date"])
df = df.sort_values(["market_commodity", "date"])

def future_vol(g):
    g = g.set_index("date")
    r = g["log_return"].sort_index()
    v = r.iloc[::-1].rolling("30D", min_periods=20).std().iloc[::-1]
    return v.reset_index(drop=True)

df["future_volatility_30d"] = (
    df.groupby("market_commodity", group_keys=False).apply(future_vol, include_groups=False).reset_index(drop=True)
)

df.to_csv("predictive_mandi.csv", index=False)

print("Valid:", df["future_volatility_30d"].notna().sum())
print("Missing:", df["future_volatility_30d"].isna().sum())