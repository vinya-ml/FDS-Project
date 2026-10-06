import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("final_mandi.csv")

cols = [
    "volatility_30d", "temp_anomaly", "rain_anomaly",
    "extreme_temp", "extreme_rain", "flood_indicator",
    "drought_indicator", "rain_anomaly_7d",
    "rain_anomaly_14d", "rain_anomaly_30d"
]

corr = df[cols].corr()

os.makedirs("plots", exist_ok=True)

print(corr["volatility_30d"].sort_values(ascending=False))

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Weather and Price Volatility Correlation")
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png", dpi=300)
plt.show()
plt.close()