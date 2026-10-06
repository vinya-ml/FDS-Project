import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)

# Find predictive dataset automatically
files = glob.glob("*.csv")
print("CSV files found:", files)

df = None

for f in files:
    try:
        x = pd.read_csv(f, nrows=5)
        if "future_volatility_30d" in x.columns:
            df = pd.read_csv(f, parse_dates=["date"])
            print("Using:", f)
            break
    except:
        pass

if df is None:
    raise FileNotFoundError("No CSV containing future_volatility_30d was found.")

# 1. Commodity volatility
x = df.groupby("Commodity")["future_volatility_30d"].mean()

plt.figure(figsize=(7,5))
x.plot(kind="bar")
plt.title("Average Future Volatility by Commodity")
plt.ylabel("Future Volatility")
plt.tight_layout()
plt.savefig(f"{PLOTS}/commodity_volatility.png", dpi=300)
plt.close()

# 2. Weather anomalies
for col in ["rain_anomaly", "temp_anomaly"]:
    plt.figure(figsize=(7,5))
    plt.scatter(df[col], df["future_volatility_30d"], s=2, alpha=.2)
    plt.xlabel(col)
    plt.ylabel("Future Volatility")
    plt.title(f"{col} vs Future Volatility")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{col}_vs_volatility.png", dpi=300)
    plt.close()

# 3. Volatility over time
d = df.groupby("date")["future_volatility_30d"].mean()

plt.figure(figsize=(10,5))
plt.plot(d)
plt.xlabel("Date")
plt.ylabel("Future Volatility")
plt.title("Future Volatility Over Time")
plt.tight_layout()
plt.savefig(f"{PLOTS}/volatility_over_time.png", dpi=300)
plt.close()

# 4. Commodity-wise volatility
for c in df["Commodity"].unique():
    d = df[df["Commodity"] == c].groupby("date")["future_volatility_30d"].mean()

    plt.figure(figsize=(10,5))
    plt.plot(d)
    plt.xlabel("Date")
    plt.ylabel("Future Volatility")
    plt.title(f"{c} Future Volatility Over Time")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/{c.lower()}_volatility.png", dpi=300)
    plt.close()

# 5. Rainfall windows
cols = ["rain_anomaly_7d", "rain_anomaly_14d", "rain_anomaly_30d"]
x = df[cols].mean()

plt.figure(figsize=(7,5))
x.plot(kind="bar")
plt.ylabel("Mean Rainfall Anomaly")
plt.title("Rainfall Anomalies Across Time Windows")
plt.tight_layout()
plt.savefig(f"{PLOTS}/rainfall_windows.png", dpi=300)
plt.close()

# 6. Extreme weather
cols = ["extreme_temp", "extreme_rain", "flood_indicator", "drought_indicator"]
x = df[cols].sum()

plt.figure(figsize=(8,5))
x.plot(kind="bar")
plt.ylabel("Number of Observations")
plt.title("Extreme Weather Events")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{PLOTS}/extreme_weather.png", dpi=300)
plt.close()

print("\nAll graphs saved successfully!")
print("Graphs folder:", os.path.abspath(PLOTS))