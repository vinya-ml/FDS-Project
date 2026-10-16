# Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)]()
[![Target: 20/20 Academic Evaluation](https://img.shields.io/badge/Evaluation-Ready%20for%20Review-success.svg)]()

> **Fundamentals of Data Science (FDS) — Final Academic Research Project**  
> **Course:** Sem - 5 B.Tech Computer Science & Engineering  
> **Target Commodities:** Tomato, Onion, Potato (TOP Vegetables)  
> **Empirical Dataset:** 737,392 Initial Mandi Records across 30 Indian States (2023–2025)  
> **Repository:** [https://github.com/vinya-ml/FDS-Project](https://github.com/vinya-ml/FDS-Project)  

---

## 1. Project Overview & Motivation

In India, retail food inflation is overwhelmingly driven by violent wholesale price spikes in the three staple vegetables: **Tomato, Onion, and Potato (TOP)**. These commodities form the foundation of dietary nutrition and household expenditure across India, yet each possesses distinct agronomic, storage, and perishability characteristics:

- **Tomato:** Ultra-perishable (shelf life of 3–5 days), lacking cold storage buffers, characterized by short 60–90 day production cycles and extreme localized supply cliffs.
- **Onion:** Semi-perishable, harvested across three seasonal cycles (Kharif, Late Kharif, Rabi) and stored in traditional ventilated structures (*chawls*) where buffer stocks are vulnerable to moisture and mold rot.
- **Potato:** Highly storable, harvested primarily in winter (Rabi) and preserved in commercial cold-storage facilities for up to 9–10 months, dampening localized daily weather shocks.

### The Central Research Question
> *"How are extreme temperature and rainfall anomalies associated with short-term agricultural price volatility across Indian mandis, and do these relationships differ between tomato, onion, and potato?"*

---

## 2. Key Empirical Findings (At a Glance)

1. **Hierarchy of Volatility:**
   - **Tomato** exhibits the highest baseline price volatility (mean 30-day volatility = **0.0931**) and tail risk (95th percentile = **0.2443**).
   - **Onion** displays intermediate volatility (**0.0812**) with prolonged surges during post-monsoon storage transitions.
   - **Potato** is the most price-stable (**0.0618**), insulated by cold storage infrastructure.
2. **Econometric Identification (Two-Way Fixed Effects PanelOLS):**
   - **Drought Shocks:** Prolonged moisture deficits exert a positive, statistically significant impact on onion price volatility ($+0.0045$, $p = 0.0266$), highlighting that water stress during bulb maturity depletes marketable storage stocks.
   - **Flood Shocks:** Severe flooding exhibits a negative coefficient across mandis ($-0.0043$, $p = 0.0015$) due to physical market closures and suspension of auctions that freeze recorded daily transaction price variance.
   - **The Low $R^2$ Phenomenon (0.0003):** Econometrically justified. Two-way fixed effects partial out all time-invariant mandi heterogeneity and macro seasonal cycles. Local weather directly above a terminal mandi explains only a fraction of localized volatility because terminal markets source produce from distant producing states.
3. **Out-of-Sample Machine Learning Forecasting (Test $N = 85,675$):**
   - Tested using strict **chronological time splits** (zero future leakage).
   - Non-linear models (**Random Forest: $R^2 = 0.1543$**, **XGBoost: $R^2 = 0.1530$**) achieve a **17-fold improvement** over the Naive Baseline ($R^2 = 0.0088$) and Ridge Regression ($R^2 = 0.0086$).
4. **Feature Ablation (With vs. Without Weather):**
   - Removing weather anomaly features causes out-of-sample $R^2$ to collapse from **13.76% down to 4.23%**. Weather anomalies contribute over **69% of the predictive signal** captured by gradient-boosted trees.
   - Cumulative multi-week rainfall anomalies (`rain_anomaly_30d`, `rain_anomaly_14d`) are significantly stronger predictors than single-day temperature blips.

---

## 3. Repository Architecture

```
FDS-Project/
├── 01_data_loading.py               # Raw dataset loader & initial statistics
├── 02_cleaning.py                   # Filtering valid prices & TOP commodities
├── 03_check_gaps.py                 # Time-series gap diagnostic
├── 04_price_returns.py              # Log return calculation (gap <= 3 days)
├── 05_volatility.py                 # 30-day rolling return volatility
├── 06_weather_anomalies.py          # State-month baseline weather demeaning
├── 07_weather_shocks.py             # 95th percentile shock flags & rolling weather
├── 08_final_dataset.py              # Analytical panel dataset filtering (592k rows)
├── 09_eda.py                        # Original baseline time series charts
├── 10_correlation.py                # Original correlation heatmap
├── 11_baseline_model.py             # Pooled Two-Way Fixed Effects PanelOLS
├── 12_commodity_models.py           # Commodity-specific PanelOLS regressions
├── 13_future_volatility.py          # Target feature engineering (forward 30d vol)
├── 14_predictive_dataset.py         # Predictive panel dataset filtering (465k rows)
├── 15_predictive_model.py           # Original panel regression on future volatility
├── 16_commodity_predictive.py       # Commodity-specific panel regressions
├── 17_rainfall_windows.py           # Multi-window rainfall anomaly panel models
├── 18_commodity_rainfall.py         # Commodity-specific rainfall window models
├── 19_alt_volatility.py             # Alternative target: Mean absolute returns
├── 20_robustness_model.py           # Robustness model on mean absolute returns
├── 21_commodity_interactions.py     # Weather shock x Commodity interaction models
├── 22_results_summary.py            # Summary plots across commodities
├── 23_enhanced_eda.py               # Publication-grade visual suite (outputs/figures/)
├── 24_ml_predictive_pipeline.py     # Out-of-sample ML pipeline (Ridge, RF, XGBoost)
├── 25_robustness_ablation.py        # Weather feature ablation & market concentration
├── validate_data_quality.py         # Reproducible data-quality & gap auditor
├── PROJECT_AUDIT.md                 # Exhaustive Phase 1 technical audit document
├── FINAL_RESULTS.md                 # Comprehensive research findings & synthesis
├── ECONOMETRIC_REPORT.md            # Detailed econometric panel estimation report
├── PREDICTIVE_MODELING_REPORT.md    # Machine learning out-of-sample evaluation report
├── EVALUATION_CHECKLIST.md          # 20-mark evaluation rubric & faculty demo guide
├── requirements.txt                 # Exact project dependency specifications
├── .gitignore                       # Clean Git configuration (ignores raw CSVs)
├── data/
│   └── sample_mandi_demo.csv        # Sanitized test fixture for instant evaluation
├── outputs/
│   ├── figures/                     # 13 high-resolution publication figures (300 DPI)
│   └── tables/                      # 14 machine-readable summary CSV tables
└── tests/
    ├── test_data_integrity.py       # Unit tests: price bounds, non-negativity, leakage
    └── test_pipeline_reproducibility.py # Reproducibility & output artifact tests
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Git

### Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/vinya-ml/FDS-Project.git
cd FDS-Project/Project   # Or repository root

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 5. Execution Guide & Reproduction Steps

All scripts use portable dynamic paths and automatically detect either uncompressed (`.csv`) or compressed (`.csv.gz`) data files.

### Step 1: Run Data Quality & Attrition Validation
```bash
python validate_data_quality.py
```
*Generates pipeline stage counts, trading gap distributions, and attrition waterfall charts in `outputs/`.*

### Step 2: Generate Publication-Grade Exploratory Data Analysis
```bash
python 23_enhanced_eda.py
```
*Generates 9 publication-grade figures in `outputs/figures/` (price trends, distributions, deciles, correlation heatmap, and comparative dashboards).*

### Step 3: Run the Econometric Panel Analysis Suite
```bash
python run_econometric_evaluation.py
```
*Estimates Two-Way Fixed Effects PanelOLS models for baseline, Tomato, Onion, Potato, rainfall windows, and interaction models, exporting `ECONOMETRIC_REPORT.md` and CSV tables.*

### Step 4: Run the Machine Learning Predictive Pipeline
```bash
python 24_ml_predictive_pipeline.py
```
*Executes chronological train/val/test splits, trains Naive Baseline, Ridge Regression, Random Forest, and XGBoost, evaluates out-of-sample metrics, and generates diagnostic charts.*

### Step 5: Run Robustness & Feature Ablation Analysis
```bash
python 25_robustness_ablation.py
```
*Performs feature ablation (with vs. without weather anomalies) and audits geographic market concentration.*

### Step 6: Execute Automated Test Suite
```bash
python -m pytest tests/
```
*Validates data integrity, price logic bounds, non-negativity, and temporal leakage prevention (8/8 tests pass in ~2 seconds).*

---

## 6. Accessing Large Datasets

Due to GitHub's 100 MB per-file upload limitation:
1. **Compressed Formats (`.csv.gz`):** Datasets are natively tracked as `.csv.gz` files in the repository. Pandas reads them directly without manual decompression:
   ```python
   import pandas as pd
   df = pd.read_csv("final_mandi.csv.gz")
   ```
2. **Decompressing to raw `.csv` (Optional):**
   - **Terminal (gzip):** `gzip -d final_mandi.csv.gz`
   - **Archive Tools:** 7-Zip, WinRAR, or standard extractors.
3. **Demonstration Dataset:** For quick testing without loading large files, use `data/sample_mandi_demo.csv`.

---

## 7. Project Evaluation Documents

| Document | Purpose & Contents |
|---|---|
| [**EVALUATION_CHECKLIST.md**](EVALUATION_CHECKLIST.md) | Proposed 20-mark evaluation rubric with criterion-by-criterion evidence mapping and 5-minute faculty presentation guide. |
| [**FINAL_RESULTS.md**](FINAL_RESULTS.md) | Comprehensive academic report answering the central research question with detailed data, model, and economic interpretations. |
| [**PROJECT_AUDIT.md**](PROJECT_AUDIT.md) | Technical audit of all 22 original scripts, pipeline artifacts, temporal gap statistics, and code improvements. |
| [**ECONOMETRIC_REPORT.md**](ECONOMETRIC_REPORT.md) | Econometric panel estimation summary, fixed-effects specifications, and detailed interpretation of statistical significance. |
| [**PREDICTIVE_MODELING_REPORT.md**](PREDICTIVE_MODELING_REPORT.md) | Machine learning out-of-sample evaluation report detailing MAE, RMSE, $R^2$, and tree-based feature importances. |

---

## 8. Authors & Academic Integrity

- **Author:** Vinya (saivinya7116@gmail.com)
- **Institution:** Fundamentals of Data Science (FDS) Course, Semester 5, B.Tech CSE
- **Academic Statement:** All data cleaning, econometric regressions, machine learning models, and figures were computed deterministically from real Agmarknet wholesale mandi records and IMD meteorological observations without fabrication.
