import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_mandi.csv", parse_dates=["date"])
df = df.sort_values(["market_commodity", "date"])

gap = df.groupby("market_commodity")["date"].diff().dt.days

df["log_return"] = np.where(
    gap <= 3,
    np.log(df["Modal_Price"]) -
    np.log(df.groupby("market_commodity")["Modal_Price"].shift(1)),
    np.nan
)

df.to_csv("returns_mandi.csv", index=False)

print("Rows:", len(df))
print("Valid returns:", df["log_return"].notna().sum())
print("Missing returns:", df["log_return"].isna().sum())