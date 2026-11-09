# 🌾 Agricultural Commodity Price Forecasting & Weather Volatility Dynamics
## Agrimetaverse Intelligence — Advanced Forecasting & Decision-Support System

Implementation of the research paper:  
**"Agrimetaverse Intelligence: Advanced Agricultural Commodity Price Forecasting System"**

---

## 📌 1. Project Overview & Objective

Agricultural price volatility in India directly affects smallholder farmers, food security, supply chain operators, and inflation dynamics. This project implements a full-stack, production-grade price forecasting and decision-support intelligence platform covering **22 essential agricultural commodities** across India from **2014 to 2024** (242 core time-series records).

The system integrates **13 external macroeconomic, agro-climatic, energy, and trade drivers**, comparing three foundational time-series paradigms:
1. **ARIMA (Auto-Regressive Integrated Moving Average)**: Univariate autoregressive benchmark tuned via Akaike Information Criterion (AIC).
2. **SARIMAX (Seasonal ARIMA with eXogenous Regressors)**: Multivariate framework integrating standardized external drivers and seasonal dynamics.
3. **Facebook Prophet**: Generalized additive model decomposing nonlinear growth trends and external regressor effects.

---

## 📊 2. Dataset Architecture

- **Primary Source**: `data/raw/DatasetSIH1647.csv`
- **Total Historical Period**: 2014–2024 (11 years)
- **Observations**: 22 commodities × 11 years = 242 annual price records
- **Format Conversion**: Automated detection and reshaping from raw Wide Format (Commodities × Years) into canonical Long Time-Series format (`Commodity`, `Category`, `Year`, `Price`, `Unit`).
- **Commodity Coverage (22 Items Across 5 Sectors)**:
  - **Grains**: Rice, Wheat, Atta (Wheat)
  - **Pulses**: Gram Dal, Tur/Arhar Dal, Urad Dal, Moong Dal, Masoor Dal
  - **Edible Oils**: Groundnut Oil (Packed), Mustard Oil (Packed), Vanaspati (Packed), Soya Oil (Packed), Sunflower Oil (Packed), Palm Oil (Packed)
  - **Vegetables**: Potato, Onion, Tomato
  - **Other Staples**: Sugar, Gur, Milk, Tea Loose, Salt Pack (Iodised)

---

## 🌐 3. The 13 External Regressors

Located in `data/external/enhanced_features.csv`, the platform models the following empirical drivers:
1. **Rainfall (mm)** (`Rainfall_mm`): Annual monsoon precipitation.
2. **Average Temperature (°C)** (`Avg_Temperature_C`): Mean surface ambient temperature.
3. **Drought Index** (`Drought_Index`): Normalized moisture stress factor (0–1).
4. **GDP Growth Rate (%)** (`GDP_Growth_%`): Real annual gross domestic product growth.
5. **Inflation Rate (%)** (`Inflation_%`): Consumer Price Index (CPI) annual headline inflation.
6. **USD/INR Exchange Rate** (`USD_INR_Rate`): Foreign exchange currency rate.
7. **Crude Oil Price ($/barrel)** (`Crude_Oil_Price`): Global Brent crude benchmark.
8. **Fertilizer Price Index** (`Fertilizer_Price_Index`): Agricultural input cost index.
9. **Fuel Price (₹/L)** (`Fuel_Price_Rs/L`): Domestic retail transport fuel price.
10. **Minimum Support Price Index** (`Minimum_Support_Price_Index`): Government procurement floor policy index.
11. **Export Volume Index** (`Export_Volume_Index`): Agricultural trade export volume index.
12. **Storage Capacity Index** (`Storage_Capacity_Index`): Food Corporation of India & state warehousing capacity.
13. **Transportation Cost Index** (`Transport_Cost_Index`): Inland logistics and freight rate index.

---

## 🛡️ 4. Data Preprocessing & Leakage Prevention

1. **Commodity-Isolated Forward Filling**:
   - Missing annual prices are strictly imputed forward within the historical series of that specific commodity. Cross-commodity leakage is completely blocked.
2. **Leak-Free Normalization**:
   - Min-Max normalization ($X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$) fits parameters **strictly on the 2014–2021 training partition**.
   - The fitted training scale is used to transform validation (2022–2023) and test (2024) periods, preventing lookahead bias.
3. **Feature Engineering**:
   - Lag-1 (`Price_Lag1`), Lag-2 (`Price_Lag2`), YoY percentage change (`Price_Pct_Change`), and 2-year rolling past mean.

---

## 🔬 5. Chronological Partitioning & Empirical Benchmarks

### Chronological Splits
- **Training Period**: 2014–2021 (8 years)
- **Validation Period**: 2022–2023 (2 years)
- **Out-of-Sample Testing Period**: 2024 (1 year)

### Published IEEE Paper Table I: Overall Model Performance
| Model Paradigm | Published R² | Published RMSE | Published MAPE (%) | Directional Accuracy (%) | Rank | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMAX** | **0.756** | **12.45** | **8.92%** | **78.4%** | 🥇 Best Model | p = 0.012 |
| **Prophet** | 0.723 | 14.21 | 9.67% | 75.2% | 🥈 Runner-up | p = 0.012 |
| **ARIMA** | 0.689 | 16.78 | 11.34% | 72.1% | 🥉 Baseline | p = 0.012 |

### Published IEEE Paper Table II: Category Performance Metrics Across All Models
| Commodity Category | SARIMAX R² | Prophet R² | ARIMA R² | Optimal Model |
| :--- | :---: | :---: | :---: | :---: |
| **Cereals** (Rice, Wheat, Atta) | **0.812** | 0.756 | 0.712 | **SARIMAX** |
| **Pulses** (Gram, Tur, Urad, Moong, Masoor) | **0.745** | 0.701 | 0.678 | **SARIMAX** |
| **Edible Oils** (Groundnut, Mustard, Soya, etc.) | **0.798** | 0.723 | 0.695 | **SARIMAX** |
| **Vegetables** (Potato, Onion, Tomato) | 0.712 | **0.756** | 0.682 | **Prophet** |
| **Spices** | **0.735** | 0.728 | 0.671 | **SARIMAX** |
| **Other Staples** (Sugar, Milk, Tea, Gur, Salt) | **0.751** | 0.723 | 0.689 | **SARIMAX** |

### Statistical Significance
Rigorous statistical ANOVA testing confirms that differences between the models are statistically significant at **p = 0.012** ($p < 0.05$).

---

## 🚀 6. Installation & Execution

### Prerequisites
- Python 3.11+
- Virtual environment with dependencies

### Quickstart Execution Command
To launch the interactive multi-page dashboard:
```bash
# Using the dedicated virtual environment:
c:\Vinya\venv\Scripts\streamlit.exe run app.py
```
Or with activated virtual environment:
```bash
streamlit run app.py
```

### Running Test Suite
```bash
c:\Vinya\venv\Scripts\python.exe -m unittest tests/test_pipeline.py
```

---

## 📁 7. Project Structure

```
Agrimetaverse/
│
├── app.py                             # Main Streamlit Dashboard Application
├── requirements.txt                   # Dependency specifications
├── README.md                          # Comprehensive documentation
├── config.py                          # Constants, paths, categories, and published targets
│
├── data/
│   ├── raw/
│   │   └── DatasetSIH1647.csv         # Raw 22-commodity wide dataset
│   ├── processed/
│   │   └── cleaned_commodity_prices.csv # Cleaned long-format time-series data
│   └── external/
│       ├── enhanced_features.csv      # 13 external macroeconomic/weather factors
│       └── paper_forecasts_2025_2029.csv # Paper reference reproduction forecasts
│
├── preprocessing/
│   ├── data_loader.py                 # Reshaping, loading & schema verification
│   ├── data_cleaning.py               # Duplicate removal & commodity-isolated ffill
│   ├── normalization.py               # Leak-free training Min-Max scaler
│   └── feature_engineering.py         # Lags, YoY returns & interaction ratios
│
├── models/
│   ├── arima_model.py                 # Grid-searched AIC ARIMA
│   ├── sarimax_model.py               # Standardized exogenous SARIMAX
│   ├── prophet_model.py               # Trend & regressor Prophet forecaster
│   ├── model_training.py              # Chronological train/val/test orchestrator
│   └── model_evaluation.py            # R², RMSE, MAE, MAPE, DA & paper comparator
│
├── forecasting/
│   ├── future_forecast.py             # 1-yr to 5-yr multi-step projections & scenarios
│   └── forecast_visualization.py      # Plotly interactive charts & 95% CI bands
│
├── analysis/
│   ├── eda.py                         # Trends, category indexing & boxplots
│   ├── external_factor_analysis.py    # Correlation matrix & regression scatter plots
│   └── commodity_analysis.py          # Single-commodity diagnostics
│
├── dashboard/
│   ├── home.py                        # KPI cards, status & quick snapshot
│   ├── dataset_explorer.py            # Raw/cleaned data view & quality report
│   ├── eda_page.py                    # Exploratory charts & growth rankings
│   ├── forecasting_page.py            # Interactive forecasting studio
│   ├── model_comparison.py            # Side-by-side metric comparison
│   ├── external_factors.py            # 13 external variables analysis
│   └── reports.py                     # ReportLab PDF generation & CSV exports
│
├── reports/
│   └── Agrimetaverse_Executive_Report.pdf # Generated PDF summary
├── outputs/
│   ├── model_evaluation_metrics.csv   # Saved commodity evaluation table
│   └── overall_metrics.json           # Average model metrics
└── tests/
    └── test_pipeline.py               # Unit & integration test suite
```
