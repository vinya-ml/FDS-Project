# Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather

An empirical time-series and econometric study examining how localized extreme meteorological anomalies (unseasonal precipitation, thermal stress, drought, and floods) drive 30-day commodity price volatility across wholesale agricultural markets in India.

---

## 1. Project Overview

* **Core Research Question:** How strongly do localized extreme weather anomalies predict 30-day commodity price volatility across wholesale agricultural markets?
* **Novelty:** Replaces standard stock and commodity price forecasting with climate resilience and agricultural supply chain analysis.
* **Geographic Coverage:** 28 Indian States (Market/Mandi level).
* **Temporal Horizon:** June 2023 – June 2025 (~737,000 observations).
* **Target Focus:** Perishable crops (e.g., Tomato, Onion) compared against durable staple benchmarks (e.g., Potato).

---

## 2. Dataset Access & External Source

Due to GitHub's file storage ceilings (>100 MB), the raw transactional log (`master_aggriculture_dataset.csv`, ~180 MB) is hosted externally.

* **Primary Dataset Source:** [Kaggle: Indian Crop Market Prices & Weather Data (2023–2025)](https://www.kaggle.com/datasets/jignalgajjar/indian-crop-market-prices-and-weather-data)
* **Pre-processing Script Output:** Running the data cleaning pipeline generates `cleaned_mandi.csv`, which retains all valid observations across all 28 states.

To set up the project locally:
1. Download the raw CSV from the Kaggle link above.
2. Place the file in the project root directory and name it `master_aggriculture_dataset.csv`.
3. Run the cleaning script to produce `cleaned_mandi.csv`:
   ```bash
   python cleaning_script.py