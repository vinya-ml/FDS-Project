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

* **Primary Dataset Source:** [https://www.kaggle.com/datasets/jignalgajjar/indian-crop-market-prices-and-weather-data](https://www.kaggle.com/datasets/jignalgajjar/daily-crop-mandi-price-with-weather-data2023-25)
* **Pre-processing Script Output:** Running the data cleaning pipeline generates `cleaned_mandi.csv`, which retains all valid observations across all 28 states.

To set up the project locally:
1. Download the raw CSV from the Kaggle link above.
2. Place the file in the project root directory and name it `master_aggriculture_dataset.csv`.
3. Run the cleaning script to produce `cleaned_mandi.csv`:
   ```bash
   python cleaning_script.py

---

## 3. Usability Matrix & Selected Features

| Column Name | Raw Data Type | Project Status | Usage / Description |
| :--- | :--- | :--- | :--- |
| `STATE` | String | **Retained** | Regional grouping and spatial fixed-effects modeling. |
| `District` | String | **Retained** | District-level spatial aggregation. |
| `Market Name` | String | **Retained** | Primary time-series entity for continuity tracking. |
| `Commodity` | String | **Retained** | Crop filtering (e.g., Tomato, Onion, Potato). |
| `Variety` | String | **Retained** | Quality control across varieties. |
| `Grade` | String | **Retained** | Quality control (e.g., `FAQ`). |
| `date` | Datetime / String | **Retained** | Master temporal index for rolling window computation. |
| `Min_Price` | Numeric (INR/Quintal)| **Retained** | Intraday spread calculation. |
| `Max_Price` | Numeric (INR/Quintal)| **Retained** | Intraday spread calculation. |
| `Modal_Price` | Numeric (INR/Quintal)| **Retained** | Baseline price for continuous log returns and 30-day volatility. |
| `temp_mean` | Numeric (°C) | **Retained** | Mean temperature for thermal anomaly computation. |
| `rainfall_mm` | Numeric (mm) | **Retained** | Daily localized precipitation shocks. |
| `rainfall_mm_30d_sum` | Numeric (mm) | **Retained** | Cumulative 30-day trailing rainfall volume. |
| `flood_indicator` | Binary (0/1) | **Retained** | Indicator for flood shock regimes. |
| `drought_indicator`| Binary (0/1) | **Retained** | Indicator for drought/heat stress regimes. |
| `price_lag_7d` | Numeric (INR/Quintal)| **Retained** | Autoregressive 7-day price momentum control. |
| `price_lag_30d` | Numeric (INR/Quintal)| **Retained** | Autoregressive 30-day baseline control. |
| `price_7d_avg` | Numeric (INR/Quintal)| **Retained** | Smoothed moving price average. |
| `temp_min` | Numeric | **Dropped** | Excluded due to systematic missing values for Andhra Pradesh. |
| `temp_max` | Numeric | **Dropped** | Excluded due to systematic missing values for Andhra Pradesh. |
| `humidity` | Numeric | **Dropped** | Excluded due to systematic missing values for Andhra Pradesh. |
| `wind_speed` | Numeric | **Dropped** | Excluded due to missing values across select regions. |
| `solar_radiation`| Numeric | **Dropped** | Excluded due to missing values across select regions. |

---

## 4. Preprocessing Pipeline
1. **Column Pruning:** Drop the 5 incomplete climate columns to retain 100% of observations across all 28 states without synthetic imputation bias.
2. **Deduplication:** Remove exact duplicate transactions.
3. **Bounds Enforcement:** Discard records where `Modal_Price <= 0` or `Min_Price > Max_Price`.
4. **Time-Series Imputation:** Group records by market and commodity, forward-filling (`ffill`) and backward-filling (`bfill`) short market closure gaps.
