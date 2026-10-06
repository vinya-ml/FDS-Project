import pandas as pd

df = pd.read_csv("cleaned_mandi.csv", parse_dates=["date"])

gap = df.groupby("market_commodity")["date"].diff().dt.days

print("Gaps:", (gap > 1).sum())
print("Largest gap:", gap.max(), "days")
print("\nLargest gaps:")
print(gap[gap > 1].value_counts().head(10))