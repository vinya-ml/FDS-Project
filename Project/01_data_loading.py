import pandas as pd

path = r"C:\Vinya\Sem - 5\Foundations Of Data Science\Project\master_aggriculture_dataset.csv"

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