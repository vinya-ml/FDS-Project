import pandas as pd

import os

data_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(data_dir, "master_aggriculture_dataset.csv")
if not os.path.exists(path):
    path = os.path.join(data_dir, "master_aggriculture_dataset.csv.gz")

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDate range:")

df["date"] = pd.to_datetime(df["date"])
print(df["date"].min(), "to", df["date"].max())

print("\nStates:", df["STATE"].nunique())
print("Markets:", df["Market Name"].nunique())
print("Commodities:", df["Commodity"].nunique())

print("\nTarget commodities:")
print(df[df["Commodity"].isin(["Tomato", "Onion", "Potato"])]
      ["Commodity"].value_counts())