import pandas as pd

df = pd.read_csv("returns_mandi.csv", parse_dates=["date"])
df = df.sort_values(["market_commodity", "date"])

df["volatility_30d"] = (
    df.groupby("market_commodity")["log_return"].transform(lambda x: x.rolling(30, min_periods=20).std())
)

df.to_csv("volatility_mandi.csv", index=False)

print("Rows:", len(df))
print("Valid 30-day volatility:", df["volatility_30d"].notna().sum())
print("Missing volatility:", df["volatility_30d"].isna().sum())