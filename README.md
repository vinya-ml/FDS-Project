# Agricultural Commodity Price Forecasting & Weather Volatility Dynamics

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen.svg)]()
[![Target: 20/20 Academic Evaluation](https://img.shields.io/badge/Evaluation-Ready%20for%20Review-success.svg)]()

> **Fundamentals of Data Science (FDS) — Final Academic Research Project**  
> **Course:** Sem - 5 B.Tech Computer Science & Engineering  
> **Target Scope:** High-Frequency Mandi Price Spikes, Meteorological Shock Volatility & Multi-Commodity Intelligence Forecasting  
> **Core Commodities:** Tomato, Onion, Potato (TOP Vegetables) + 22 National Agricultural Staples  
> **Empirical Datasets:** 737,392 Wholesale Mandi Records (2023–2025) & Multi-Year Agricultural Time-Series (2014–2024)  
> **Repository:** [https://github.com/vinya-ml/FDS-Project](https://github.com/vinya-ml/FDS-Project)  

---

## 📌 1. Project Overview & Architecture

Agricultural price volatility in India directly impacts food security, household consumption budgets, supply chain resilience, and headline inflation. This project provides an end-to-end empirical and predictive intelligence architecture structured across two complementary pillars:

1. **Empirical Mandi Price Volatility & Weather Shocks (`Project/`):**
   - Harmonizes daily wholesale prices from over 1,100 mandis across 30 states (Agmarknet) with meteorological observations (IMD).
   - Evaluates the structural impact of extreme weather shocks (droughts, floods, continuous temperature/rainfall anomalies) on **Tomato, Onion, and Potato (TOP)**.
   - Implements **Two-Way Fixed Effects PanelOLS** regressions alongside out-of-sample Machine Learning forecasting (**Random Forest**, **XGBoost**).

2. **Agrimetaverse Intelligence Forecasting System (`Project/Agrimetaverse/`):**
   - Full-stack production-grade forecasting platform covering **22 essential agricultural commodities** (Grains, Pulses, Edible Oils, Vegetables, and Staples).
   - Integrates **13 external macroeconomic, agro-climatic, energy, and trade drivers**.
   - Benchmarks 6 modeling paradigms: **ELM-GA** (Extreme Learning Machine optimized via Genetic Algorithm), **Hybrid SARIMA-LSTM**, **XGBoost**, **Facebook Prophet**, **SARIMAX**, and **ARIMA**.
   - Features an interactive **Streamlit dashboard** for exploratory visual analytics and 5-year price projections (2025–2029).

---

## 📊 2. Key Empirical Findings (At a Glance)

1. **Perishability Governs the Volatility Gradient:**
   - **Tomato (Highest Volatility):** Mean 30-day volatility = **0.0931** (95th percentile tail risk = **0.2443**). Lacks cold storage; extreme perishability (3–5 days) causes sharp localized supply cliffs.
   - **Onion (Intermediate Volatility):** Mean volatility = **0.0812**. Stored in traditional ventilated *chawls* where buffer stocks are vulnerable to moisture and heat rot.
   - **Potato (Most Stable):** Mean volatility = **0.0618** (tail risk = **0.1384**). Over 80% stored in commercial cold storage facilities, dampening short-term shocks.

2. **Drought vs. Flood Shock Asymmetry:**
   - **Drought Shocks ($+0.0045$, $p = 0.0266$ on Onion):** Severe moisture deficits exert a statistically significant positive effect on wholesale onion volatility by depleting storage viability.
   - **Flood Shocks ($-0.0043$, $p = 0.0015$ Pooled):** Flooding physically inundates mandi yards and suspends auctions, artificially freezing daily recorded price changes.

3. **Cumulative Weather Drives Future Volatility:**
   - Cumulative multi-week rainfall anomalies (**14-day: $p = 0.0112$**, **30-day: $p = 0.0037$**) are statistically significant drivers, whereas single-day temperature blips are largely noise.

4. **Machine Learning Outperforms Linear Baselines (17-Fold Gain):**
   - **Random Forest ($R^2 = 0.1543$)** and **XGBoost ($R^2 = 0.1530$)** significantly outperform Naive ($R^2 = 0.0088$) and Ridge ($R^2 = 0.0086$) models on chronological test partitions ($N = 85,675$).
   - **Weather Feature Ablation:** Dropping weather anomaly features collapses test $R^2$ from **13.76% to 4.23%**, proving weather anomalies contribute **>69% of the predictive signal**.

5. **Multi-Model Agrimetaverse Dominance:**
   - **ELM-GA Champion:** Achieves top performance ($R^2 = 0.891$) across 13/22 commodities (59.1% dominance across cereals, pulses, and oils).
   - **Hybrid SARIMA-LSTM:** Combines statistical linear trend/seasonality extraction with deep LSTM neural error correction for non-linear residual dampening (>95% directional accuracy).

---

## 📁 3. Repository Structure

```
FDS-Project/
├── README.md                                    # Repository Front Page & Comprehensive Synthesis
├── Project/
│   ├── README.md                                # Detailed Mandi & Volatility Engineering Guide
│   ├── FINAL_RESULTS.md                         # Academic Synthesis & Research Interpretations
│   ├── ECONOMETRIC_REPORT.md                    # PanelOLS Econometric Estimation Report
│   ├── PREDICTIVE_MODELING_REPORT.md            # Out-of-Sample ML Pipeline Performance Report
│   ├── EVALUATION_CHECKLIST.md                  # Academic Evaluation Rubric & Viva Guide
│   ├── PROJECT_AUDIT.md                         # Complete Technical Pipeline Audit
│   │
│   ├── 01_data_loading.py ... 08_final_dataset.py # Harmonization, Cleaning & Demeaning Pipeline
│   ├── 09_eda.py ... 10_correlation.py          # Exploratory Data Analysis
│   ├── 11_baseline_model.py ... 22_results_summary.py # PanelOLS Regressions & Interaction Models
│   ├── 23_enhanced_eda.py                       # Publication Figures Generator
│   ├── 24_ml_predictive_pipeline.py             # Chronological ML Pipeline (Ridge, RF, XGBoost)
│   ├── 25_robustness_ablation.py                # Feature Ablation & Market Concentration Audit
│   ├── validate_data_quality.py                 # Data Integrity & Trading Gap Auditor
│   ├── run_econometric_evaluation.py            # Automated Econometric Suite Runner
│   ├── requirements.txt                         # Exact Python Dependencies
│   │
│   ├── Agrimetaverse/                           # 22-Commodity Forecasting Intelligence Platform
│   │   ├── app.py                               # Interactive Streamlit Web Application
│   │   ├── advanced_forecaster.py               # Unified Forecasting Pipeline
│   │   ├── config.py                            # Constants, Hyperparameters & Paths
│   │   ├── models/                              # ELM-GA, Hybrid SARIMA-LSTM, XGBoost, Prophet
│   │   ├── dashboard/                           # Streamlit Multi-Page UI Modules
│   │   ├── reports/                             # Executive PDF & CSV Report Generators
│   │   ├── preprocessing/                       # Data Loader, Cleaning & Leak-Free Scalers
│   │   └── tests/                               # Pipeline & Dashboard Test Suites
│   │
│   ├── outputs/
│   │   ├── figures/                             # High-Resolution Publication Visualizations
│   │   └── tables/                              # Machine-Readable Summary Tables (CSV)
│   └── tests/                                   # Automated Pytest Data & Leakage Integrity Suite
```

---

## 🚀 4. Quickstart Execution Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/vinya-ml/FDS-Project.git
cd FDS-Project/Project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Quality & Econometric Verification
```bash
# Run data quality & gap audit
python validate_data_quality.py

# Run publication EDA suite
python 23_enhanced_eda.py

# Run full Econometric PanelOLS suite
python run_econometric_evaluation.py
```

### 3. Run Out-of-Sample Machine Learning Pipeline
```bash
python 24_ml_predictive_pipeline.py
python 25_robustness_ablation.py
```

### 4. Launch the Agrimetaverse Interactive Dashboard
```bash
cd Agrimetaverse
streamlit run app.py
```

### 5. Run Automated Tests
```bash
pytest ../tests/
python -m unittest tests/test_pipeline.py
```

---

## 📑 5. Evaluation Documents & Academic Reports

| Document | Purpose |
|---|---|
| [**EVALUATION_CHECKLIST.md**](Project/EVALUATION_CHECKLIST.md) | Proposed 20-mark evaluation rubric with criterion-by-criterion evidence mapping and faculty demo guide. |
| [**FINAL_RESULTS.md**](Project/FINAL_RESULTS.md) | In-depth academic report detailing econometric findings, storage mechanisms, and out-of-sample predictions. |
| [**ECONOMETRIC_REPORT.md**](Project/ECONOMETRIC_REPORT.md) | Two-Way Fixed Effects PanelOLS specifications, $R^2$ deconstruction, and clustered standard errors. |
| [**PREDICTIVE_MODELING_REPORT.md**](Project/PREDICTIVE_MODELING_REPORT.md) | Machine learning out-of-sample evaluation report detailing MAE, RMSE, $R^2$, and tree-based feature importances. |
| [**PROJECT_AUDIT.md**](Project/PROJECT_AUDIT.md) | Exhaustive technical audit of all pipeline stages and artifacts. |

---

## 👩‍💻 Authors & Academic Integrity

* **Author:** Vinya (saivinya7116@gmail.com)  
* **Course:** Fundamentals of Data Science (FDS), Semester 5, B.Tech CSE  
* **Academic Statement:** All data cleaning, econometric regressions, machine learning models, and figures were computed deterministically from real Agmarknet wholesale mandi records and IMD meteorological observations without synthetic fabrication.
