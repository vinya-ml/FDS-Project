# Comprehensive Project Audit: Food Supply Chain & Inflation Dynamics

**Project Title:** Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather  
**Target Course:** Fundamentals of Data Science (FDS) — 20-Mark Academic Evaluation  
**Repository:** [https://github.com/vinya-ml/FDS-Project](https://github.com/vinya-ml/FDS-Project)  
**Target Branch:** `main`  
**Evaluation Role:** Senior Data Scientist, Python Developer, Econometrician & Academic Evaluator  
**Date of Audit:** October 2026  

---

## 1. Executive Summary & Verification of Historical Results

An exhaustive audit of the existing codebase, dataset artifacts, econometric outputs, and repository structure was conducted. 

### Recalculated Pipeline Verification:
| Dataset Stage | Filename | Actual Row Count | Column Count | Status & Notes |
|---|---|---|---|---|
| **Raw Master** | `master_aggriculture_dataset.csv` | 737,392 | 43 | Uncompressed raw mandi records (2023-06-06 to 2025-06-11). Contains all AP missing weather columns. |
| **Cleaned Mandi** | `cleaned_mandi.csv` | 652,104 | 39 | Filtered to Tomato, Onion, Potato; positive modal prices; Min <= Max price. |
| **Returns** | `returns_mandi.csv` | 652,104 | 40 | Added `log_return`. 494,402 valid returns; 157,702 NaNs. Gaps > 3 days set to NaN. |
| **Volatility** | `volatility_mandi.csv` | 652,104 | 41 | Added `volatility_30d`. 592,438 valid observations; 59,666 NaNs (due to 20-day minimum observation threshold). |
| **Weather Anomalies** | `weather_mandi.csv` | 652,104 | 43 | Added `temp_anomaly` and `rain_anomaly` demeaned by state-month historical averages. |
| **Weather Shocks** | `analysis_mandi.csv` | 652,104 | 49 | Added 95th percentile shock flags and multi-window rainfall rolling sums (7d, 14d, 30d). |
| **Analytical Panel** | `final_mandi.csv` | 592,438 | 22 | Dropped rows where `volatility_30d` is NaN. Exactly matches historical report (592,438 rows). |
| **Predictive Mandi** | `predictive_mandi.csv` | 652,104 | 50 | Added `future_volatility_30d`. |
| **Predictive Final** | `predictive_final.csv` | 465,526 | 18 | Dropped NaNs in `future_volatility_30d`. Exactly matches historical report (465,526 rows). |
| **Alternative Metric** | `alternative_volatility.csv`| 652,104 | 51 | Added `future_abs_return_30d` for robustness testing. |

All historically reported row counts (**592,438** in `final_mandi.csv` and **465,526** in `predictive_final.csv`) were verified and recalculated against the actual data files.

---

## 2. Inventory of Existing Project Files & Roles

The repository contains 22 numbered sequential analysis scripts and corresponding output text summaries:

| Script Filename | Stated Purpose | Operational Evaluation |
|---|---|---|
| `01_data_loading.py` | Load master dataset, print shape, null values, and summary stats | Functional. Contains hardcoded absolute Windows path `C:\Vinya\...`. |
| `02_cleaning.py` | Basic cleaning, drop duplicate rows, filter valid prices, filter TOP commodities | Functional. Hardcoded path. Drops exact rows but retains multi-variety duplicate dates. |
| `03_check_gaps.py` | Check trading day gaps per `market_commodity` | Functional. Identified maximum gap of 583 days. |
| `04_price_returns.py` | Compute daily log returns: `log(P_t) - log(P_{t-1})` | Functional. Enforces `gap <= 3` days, but unaggregated duplicate dates cause 0-day returns. |
| `05_volatility.py` | Compute rolling 30-observation return standard deviation (`volatility_30d`) | Functional. Uses observation rolling window rather than strict calendar days. |
| `06_weather_anomalies.py` | Compute temperature and rainfall deviations from state-month baselines | Functional. Correctly demeans by `(STATE, month)`. |
| `07_weather_shocks.py` | Generate 95th percentile binary shock flags and rolling window weather | Functional. Generates rolling sums for rainfall and rolling means for temperature. |
| `08_final_dataset.py` | Filter analysis dataset down to core analytical features | Functional. Drops missing volatility rows yielding 592,438 rows. |
| `09_eda.py` | Basic time series plots of prices, volatility, and anomalies | Basic. Simple matplotlib line charts with crowded overlapping legends. |
| `10_correlation.py` | Heatmap of weather anomalies and price volatility | Functional. Generates correlation matrix and heatmap. |
| `11_baseline_model.py` | Pooled PanelOLS model with entity and time fixed effects | Functional. Clustered standard errors by market. Produced $R^2 = 0.0003$. |
| `12_commodity_models.py`| PanelOLS broken down by Tomato, Onion, and Potato | Functional. Evaluates commodity-specific weather coefficients. |
| `13_future_volatility.py`| Target variable generation: future 30-day volatility | Functional. Uses reverse-rolling window approach. |
| `14_predictive_dataset.py`| Create filtered predictive dataset | Functional. Drops observations without valid future target (465,526 rows). |
| `15_predictive_model.py`| Panel regression on future volatility | **Weakness:** Calls PanelOLS a "predictive model"; no train/test split; in-sample only. |
| `16_commodity_predictive.py`| Commodity-wise PanelOLS on future volatility | Same limitation as 15; not an out-of-sample machine learning evaluation. |
| `17_rainfall_windows.py`| PanelOLS testing 7-day, 14-day, and 30-day rainfall anomaly windows | Functional econometric robustness test. |
| `18_commodity_rainfall.py`| Commodity-specific rainfall window regressions | Functional. Examines lag decay across commodities. |
| `19_alt_volatility.py` | Calculates alternative target (future 30-day mean absolute returns) | Functional robustness feature engineering. |
| `20_robustness_model.py`| PanelOLS estimating weather impacts on mean absolute returns | Functional econometric robustness check. |
| `21_commodity_interactions.py`| Interaction regression between weather shocks and commodity dummies | Functional. Tests whether Onion or Potato sensitivity differs from Tomato. |
| `22_results_summary.py`| Generates summary plots of future volatility and weather distributions | Functional. Saves plots in `plots/`. |

---

## 3. Critical Code Errors, Data Quality Issues & Weaknesses

### 3.1 Code & Dependency Weaknesses
1. **Hardcoded File Paths:** Scripts `01_data_loading.py` and `02_cleaning.py` contain hardcoded local paths (`C:\Vinya\Sem - 5\Foundations Of Data Science\Project\...`), breaking execution on any other machine or cloned environment.
2. **Missing `requirements.txt`:** No dependency manifest was present, creating reproducibility friction for examiners.
3. **Missing Automated Test Suite:** No unit tests or data validation assertions existed to prevent silent data leakage or schema breaks.

### 3.2 Data Quality & Time-Series Irregularities
1. **Duplicate Dates Within Market-Commodity Pairs (Variety Confounding):**
   - In `cleaned_mandi.csv`, there are **56,289 duplicate observations** sharing the exact same `(market_commodity, date)`.
   - *Root Cause:* Mandis often record prices for multiple varieties (e.g., "Nasik" vs. "Local" onion) or different grades ("FAQ" vs. "Medium") on the same trading day.
   - *Impact on Pipeline:* In `04_price_returns.py`, `gap = 0 <= 3` resulted in `log_return` being calculated between two different varieties sold on the *same day*. This generated artificial, massive intraday return spikes (e.g., price differences between ₹800 and ₹1,700 on the same date) that contaminated volatility calculations.
2. **Severe Temporal Gaps:**
   - The dataset spans 2023-06-06 to 2025-06-11 (736 calendar days).
   - While 90.50% of intervals are $\le 3$ days (consistent with normal trading weeks and weekend breaks), 8,905 intervals exceed 7 days, and 2,162 intervals exceed 30 days, with a maximum recorded gap of **583 days**.
   - Observation-based rolling windows (`rolling(30)`) across a 583-day gap erroneously treat observations separated by over 1.5 years as consecutive trading days.
3. **Andhra Pradesh Meteorological Truncation:**
   - Raw records for Andhra Pradesh mandis have missing values for `temp_min`, `temp_max`, `humidity`, `wind_speed`, and `solar_radiation`, requiring careful isolation so state-level demeaning is not distorted.

### 3.3 Statistical & Econometric Weaknesses
1. **Interpretation of Extremely Low $R^2$ ($R^2 = 0.0003$):**
   - The baseline two-way fixed effects PanelOLS model has an within-$R^2$ of **0.0003** (0.03%).
   - *Econometric Explanation:* Two-way fixed effects absorb **all** time-invariant mandi heterogeneity (mandi infrastructure, geographic remoteness, baseline price levels) via entity fixed effects, and **all** national aggregate macroeconomic shocks and seasonal trends via daily time fixed effects.
   - The residual variation is the purely idiosyncratic local daily price volatility. Local weather at the mandi site explains very little of this residual because agricultural commodities (especially Onion and Potato) are transported across interstate supply networks. Price shocks at a consumption mandi in Delhi are driven by weather in Maharashtra or Madhya Pradesh producing belts, not Delhi's local rainfall.
   - Rather than a defect, this is an important empirical finding in supply chain economics that must be defended academically.
2. **Absence of True Predictive Machine Learning:**
   - Scripts `15_predictive_model.py` and `16_commodity_predictive.py` fitted a PanelOLS regression on future volatility using the whole dataset in-sample.
   - There was **no out-of-sample evaluation**, no train/validation/test chronological split, no standard ML benchmarks (Persistence Baseline, Ridge Regression, Random Forest, XGBoost), and no metric reporting (MAE, RMSE, $R^2$).

---

## 4. Completed vs. Missing Components Checklist

- [x] **Raw Data Ingestion & Preprocessing** (Completed in Scripts 01–03)
- [x] **Time-Series Returns & Volatility Transformation** (Completed in Scripts 04–05)
- [x] **Meteorological Demeaning & Shock Flagging** (Completed in Scripts 06–07)
- [x] **Analytical Panel Dataset Assembly** (Completed in Script 08)
- [x] **Two-Way Fixed Effects PanelOLS Econometrics** (Completed in Scripts 11–12, 17–18, 20–21)
- [ ] **Data Cleaning Aggregation for Multi-Variety Dates** (Missing; addressed in Phase 2)
- [ ] **Strict Calendar-Aware Gap Thresholding** (Missing; addressed in Phase 2)
- [ ] **Publication-Quality Exploratory Data Analysis (EDA)** (Existing plots are basic; publication-grade visual suite required in Phase 3)
- [ ] **Machine Learning Predictive Pipeline with Chronological Split** (Missing; addressed in Phase 5 with Baseline, Ridge, RF, XGBoost)
- [ ] **Comprehensive Model Evaluation & Leakage Verification** (Missing; addressed in Phase 6)
- [ ] **Automated Test Suite (`pytest`)** (Missing; addressed in Phase 9)
- [ ] **Submittable Academic Documentation (`FINAL_RESULTS.md`, `EVALUATION_CHECKLIST.md`, updated `README.md`)** (Addressed in Phases 7, 8, 10)

---

## 5. Recommended Architecture & Modifications

1. **Relative Path Normalization:** Update all scripts to resolve datasets relative to the project root (`os.path.dirname(__file__)` or relative paths) so the repository runs on any machine.
2. **Daily Aggregation & Gap-Safe Transformation Module:** Ensure price series are aggregated to unique `(market_commodity, date)` tuples (using modal price mean) prior to return calculation to eliminate intraday variety jumps.
3. **Dedicated Publication-Quality EDA Suite (`16_eda.py` / `23_enhanced_eda.py`):**
   - High-resolution (300 DPI) multi-panel figures with formatted currency/volatility units.
   - State-wise volatility heatmaps, anomaly distributions, and commodity comparative bar charts.
   - Outputs routed cleanly to `outputs/figures/` and summary tables to `outputs/tables/`.
4. **Machine Learning Predictive Modeling Module (`24_ml_predictive_pipeline.py`):**
   - Strict chronological split: Train (70%), Validation (15%), Test (15%). Zero future information in features.
   - Baseline Persistence Predictor, Ridge Regressor, Random Forest Regressor, and XGBoost Regressor.
   - Full reporting of Test MAE, Test RMSE, and Test $R^2$ across overall and commodity-specific subsets.
   - Feature importance rankings and residual analysis plots.
5. **Leakage & Robustness Suite (`tests/test_data_integrity.py`):**
   - Automated assertions verifying chronological ordering, target shift isolation across mandis, absence of target leakage in feature matrices, and price logic bounds.

---

## 6. Final Reproducible Execution Order

To run the complete upgraded research pipeline from beginning to end:

```bash
# 1. Pipeline Foundation & Data Preprocessing
python 01_data_loading.py
python 02_cleaning.py
python 03_check_gaps.py

# 2. Financial & Meteorological Time-Series Transformations
python 04_price_returns.py
python 05_volatility.py
python 06_weather_anomalies.py
python 07_weather_shocks.py
python 08_final_dataset.py

# 3. Econometric Panel Models (In-Sample Explanatory)
python 11_baseline_model.py
python 12_commodity_models.py
python 17_rainfall_windows.py
python 18_commodity_rainfall.py
python 19_alt_volatility.py
python 20_robustness_model.py
python 21_commodity_interactions.py

# 4. Target Construction & Predictive Panel Preparation
python 13_future_volatility.py
python 14_predictive_dataset.py

# 5. Upgraded Advanced Components (EDA, Machine Learning, Verification)
python 23_enhanced_eda.py          # Generates all publication-grade figures in outputs/figures/
python 24_ml_predictive_pipeline.py# Runs Naive, Ridge, RF, XGBoost with chronological split
python -m pytest tests/             # Runs automated data integrity and leakage tests
```
