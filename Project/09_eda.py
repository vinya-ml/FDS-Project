import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("final_mandi.csv", parse_dates=["date"])
os.makedirs("plots", exist_ok=True)

# Price trend
df.groupby(["date", "Commodity"])["Modal_Price"].mean().unstack().plot()
plt.title("Average Mandi Price")
plt.ylabel("Price (INR/Quintal)")
plt.tight_layout()
plt.savefig("plots/01_price_trend.png", dpi=300)
plt.show()
plt.close()

# Volatility trend
df.groupby(["date", "Commodity"])["volatility_30d"].mean().unstack().plot()
plt.title("30-Day Price Volatility")
plt.ylabel("Volatility")
plt.tight_layout()
plt.savefig("plots/02_volatility_trend.png", dpi=300)
plt.show()
plt.close()

# Weather anomalies
df.groupby("date")[["temp_anomaly", "rain_anomaly"]].mean().plot()
plt.title("Average Weather Anomalies")
plt.ylabel("Anomaly")
plt.tight_layout()
plt.savefig("plots/03_weather_anomalies.png", dpi=300)
plt.show()
plt.close()

print("Graphs saved in the 'plots' folder.")