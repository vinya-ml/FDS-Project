import pandas as pd

import os

data_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(data_dir, "master_aggriculture_dataset.csv")
if not os.path.exists(path):
    path = os.path.join(data_dir, "master_aggriculture_dataset.csv.gz")

df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])

df = df.drop_duplicates()
df = df[(df["Modal_Price"] > 0) & (df["Min_Price"] <= df["Max_Price"])]

df = df.drop(columns=[
    "temp_min", "temp_max", "humidity",
    "wind_speed", "solar_radiation"
])

df = df[df["Commodity"].isin(["Tomato", "Onion", "Potato"])]

df = df.sort_values(["STATE", "district", "Market Name", "Commodity", "date"])

df["market_commodity"] = (
    df["STATE"] + "_" + df["district"] + "_" +
    df["Market Name"] + "_" + df["Commodity"]
)

df.to_csv("cleaned_mandi.csv", index=False)

print("Cleaned data:", df.shape)
print("\nCommodities:\n", df["Commodity"].value_counts())
print("\nMissing values:\n", df.isnull().sum())