import pandas as pd

df = pd.read_csv("analysis_mandi.csv", parse_dates=["date"])
df = df.sort_values(["market_commodity", "date"])

df["abs_return"] = df["log_return"].abs()

def future_abs(g):
    g = g.set_index("date")["abs_return"]
    return g.iloc[::-1].rolling("30D", min_periods=20).mean().iloc[::-1]

df["future_abs_return_30d"] = (
    df.groupby("market_commodity", group_keys=False).apply(future_abs, include_groups=False).reset_index(drop=True)
)

df.to_csv("alternative_volatility.csv", index=False)

print("Valid:", df["future_abs_return_30d"].notna().sum())
print("Missing:", df["future_abs_return_30d"].isna().sum())