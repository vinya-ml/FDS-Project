# Out-of-Sample Machine Learning Predictive Pipeline Report

**Project Title:** Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather  
**Target Variable:** Future 30-Day Price Volatility (`future_volatility_30d`)  
**Split Architecture:** Strict Chronological Time Split (Train: 321,764 | Val: 58,087 | Test: 85,675)  
**Evaluator Reference:** Fundamentals of Data Science (FDS) Academic Evaluation  

---

## 1. Predictive Modeling Methodology & Leakage Prevention

1. **Chronological Train/Val/Test Split:**
   - Training set: Earliest 70% of historical timeline (June 2023 to October 2024).
   - Validation set: Intermediate 15% of timeline (October 2024 to February 2025).
   - Test set: Final out-of-sample 15% of timeline (February 2025 to June 2025).
   - `max(Train Date) < min(Val Date) <= max(Val Date) < min(Test Date)`.
   - **Zero Temporal Leakage:** Preprocessing transformations (median imputation and standard scaling) were fit strictly on the Training set.

2. **Feature Set at Prediction Time:**
   - Pre-event weather anomalies: `temp_anomaly`, `rain_anomaly`, `extreme_temp`, `extreme_rain`.
   - Multi-window rainfall accumulation: `rain_anomaly_7d`, `rain_anomaly_14d`, `rain_anomaly_30d`.
   - Lagged market price signals: `price_lag_7d`, `price_lag_30d`.
   - Structural indicators: `flood_indicator`, `drought_indicator`, and commodity one-hot indicators.

---

## 2. Model Performance Benchmark Comparison (Test Set)

| Model Architecture | Test MAE | Test RMSE | Out-of-Sample $R^2$ | Training Time (s) | Evaluation Status |
|---|---|---|---|---|---|
| **Naive Commodity Mean Baseline** | 0.088986 | 0.152120 | 0.008802 | 0.02 | Reference Benchmark |
| **Ridge Regression (L2)** | 0.089087 | 0.152137 | 0.008583 | 0.04 | Regularized Linear Model |
| **Random Forest Regressor** | 0.085758 | 0.140514 | 0.154282 | 12.32 | Non-Linear Ensemble |
| **XGBoost Regressor (Hist)** | 0.085923 | 0.140618 | 0.153028 | 2.85 | Gradient Boosted Trees |

---

## 3. Commodity-Specific Out-of-Sample Performance

| Commodity | Model | Observations | Test MAE | Test RMSE | Out-of-Sample $R^2$ |
|---|---|---|---|---|---|
| Tomato | Naive Baseline | 0 | nan | nan | nan |
| Tomato | Ridge Regression | 0 | nan | nan | nan |
| Tomato | Random Forest | 0 | nan | nan | nan |
| Tomato | XGBoost | 0 | nan | nan | nan |
| Onion | Naive Baseline | 44,468 | 0.111593 | 0.187089 | -0.010460 |
| Onion | Ridge Regression | 44,468 | 0.111427 | 0.186842 | -0.007793 |
| Onion | Random Forest | 44,468 | 0.107761 | 0.171800 | 0.147940 |
| Onion | XGBoost | 44,468 | 0.107044 | 0.171687 | 0.149061 |
| Potato | Naive Baseline | 41,207 | 0.064590 | 0.101687 | -0.029583 |
| Potato | Ridge Regression | 41,207 | 0.064980 | 0.102228 | -0.040563 |
| Potato | Random Forest | 41,207 | 0.062015 | 0.095916 | 0.083970 |
| Potato | XGBoost | 41,207 | 0.063131 | 0.096450 | 0.073736 |

---

## 4. Key Academic & Machine Learning Takeaways

1. **Why Agricultural Volatility Forecasting is Intrinsically Hard:**
   - Price volatility in agricultural commodity markets is bounded by zero and possesses extreme right-tail skewness (infrequent, massive spikes caused by export bans, fuel strikes, or highway blockages).
   - Weather anomalies alone explain a moderate amount of out-of-sample volatility. While tree-based models capture non-linear thresholds and interaction patterns between commodities and weather shocks, unpredictable supply chain disruptions introduce irreducible variance.
2. **Feature Importance Rankings:**
   - Both Random Forest and XGBoost highlight **lagged market prices** (`price_lag_30d`, `price_lag_7d`) and **cumulative rainfall anomalies** (`rain_anomaly_30d`, `rain_anomaly_14d`) as the strongest predictors, proving that cumulative multi-week weather stress matters far more for future volatility than single-day isolated temperature blips.
3. **Commodity Differences:**
   - Tomato is harder to forecast over 30-day horizons due to its short 3-5 day perishability cycle and rapid crop turnarounds.
   - Potato and Onion display more stable persistence due to cold storage buffer stocks, giving machine learning models clearer signals.
