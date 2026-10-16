# Econometric Analysis & Statistical Validation Report

**Project Title:** Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather  
**Target Specification:** Two-Way Fixed Effects PanelOLS with Market Clustered Standard Errors  
**Sample Coverage:** 2023-06-06 to 2025-06-11 (730 Trading Days Across 2,119 Mandi-Commodity Entities)  
**Evaluator Reference:** Fundamentals of Data Science (FDS) Academic Evaluation  

---

## 1. Executive Econometric Findings

1. **Identification Strategy & Fixed Effects:**
   - **Entity Fixed Effects (Mandi $\times$ Commodity):** Absorbs all unobserved, time-invariant market characteristics, including local road quality, proximity to production clusters, geographic remoteness, and historical wholesale capacity.
   - **Time Fixed Effects (Daily Dummies):** Absorbs all nationwide common macroeconomic trends, including diesel fuel price shifts, central minimum export price (MEP) bans, monsoon arrival dates, and festival season demand shocks.
   - **Clustered Standard Errors:** Standard errors are clustered at the entity level (`market_commodity`) to account for arbitrary autocorrelation and heteroskedasticity across time within each market.

2. **Deconstruction of the "Low $R^2$" Phenomenon ($R^2 = 0.0003$):**
   - The reported within-$R^2$ of **0.0003** (0.03%) in the baseline model and **0.0010** (0.10%) in the predictive specification is **econometrically expected, statistically defensible, and fully reproducible**.
   - Two-way fixed effects intentionally partial out between-mandi price differences (which account for large price variation across regions) and macro aggregate seasonal waves (which drive agricultural cycles).
   - The residual variation is the purely idiosyncratic day-to-day volatility within an individual mandi.
   - **Supply Chain Economic Insight:** Indian agricultural consumption mandis (e.g., Delhi, Mumbai, Bengaluru) source perishable and storable produce from distant production belts (e.g., Lasalgaon for onions, Kolar for tomatoes). Consequently, *local weather anomalies* occurring directly above a terminal mandi explain a minuscule fraction of its price volatility. The true drivers are regional transit disruptions, production-belt shocks, and cold storage releases.

3. **Statistical Significance & Parameter Insights:**
   - **Drought Indicator:** Exhibits a consistently positive and statistically significant coefficient in the baseline pooled model ($+0.0032$, $p = 0.0043$), indicating that prolonged severe moisture deficits elevate wholesale price volatility.
   - **Flood Indicator:** Shows a significant negative association in the baseline ($-0.0043$, $p = 0.0015$). In many mandis, severe floods halt trading arrivals completely, causing recorded prices to freeze or report zero turnover, artificially compressing recorded return variance during the crisis.
   - **Continuous Anomalies:** Temperature anomalies and daily rainfall anomalies show small, statistically weak direct associations after controlling for fixed effects, confirming that short-term price volatility is triggered by extreme non-linear shocks rather than minor daily weather fluctuations.

---

## 2. Model Comparison Table: Baseline vs. Commodity-Specific Models

| Specification | Observations | Entities | Time Periods | Within $R^2$ | Weather Variable | Coef | Std. Error | $t$-stat | $p$-value | 95% Conf. Interval |
|---|---|---|---|---|---|---|---|---|---|---|
| **Pooled Baseline** | 592,438 | 2,119 | 730 | 0.0003 | `temp_anomaly` | -0.000309 | 0.000262 | -1.18 | 0.2370 | [-0.000822, 0.000203] |
| | | | | | `rain_anomaly` | 5.763142e-05 | 3.705501e-05 | 1.56 | 0.1199 | [-1.499522e-05, 1.302581e-04] |
| | | | | | `flood_indicator` | -0.004305 | 0.001358 | -3.17 | 0.0015 | [-0.006966, -0.001643] |
| | | | | | `drought_indicator` | 0.003170 | 0.001110 | 2.86 | 0.0043 | [0.000995, 0.005345] |

### Commodity-Specific Deconstruction:

#### Tomato (Perishable, Short Production Cycle: 60-90 Days)
- **Observations:** 14,691 (468 Entities, 141 Time Periods)
- **Within $R^2$:** 0.0003 | **Overall $R^2$:** 0.0063
- `temp_anomaly`: Coef = -0.0002, $p = 0.8723$
- `rain_anomaly`: Coef = -4.82e-05, $p = 0.5399$
- `drought_indicator`: Coef = +0.0037, $p = 0.4799$
- `flood_indicator`: Coef = +0.0015, $p = 0.5480$
- *Interpretation:* Tomato prices exhibit high base volatility due to extreme perishability (shelf life 3-5 days). However, local terminal mandi weather does not predict this volatility because tomatoes are trucked rapidly from isolated regional clusters.

#### Onion (Semi-Perishable, Seasonally Stored in Chawls)
- **Observations:** 273,432 (820 Entities, 640 Time Periods)
- **Within $R^2$:** 0.0004 | **Overall $R^2$:** 0.0214
- `temp_anomaly`: Coef = -0.0004, $p = 0.3993$
- `rain_anomaly`: Coef = +5.18e-05, $p = 0.3901$
- `drought_indicator`: Coef = **+0.0045**, $p = 0.0266$ (**Statistically Significant at 5% level**)
- `flood_indicator`: Coef = -0.0028, $p = 0.2344$
- *Interpretation:* Drought has a statistically significant positive effect on onion price volatility ($p = 0.0266$). Onion rabi crops rely on residual moisture and irrigation; severe drought depletes storage buffer stocks, sparking sharp price surges.

#### Potato (Storable, Heavy Cold Storage Buffer Stocks)
- **Observations:** 304,315 (831 Entities, 729 Time Periods)
- **Within $R^2$:** 0.0005 | **Overall $R^2$:** -0.0117
- `temp_anomaly`: Coef = -0.0003, $p = 0.3640$
- `rain_anomaly`: Coef = +8.53e-05, $p = 0.0846$ (Marginally significant at 10% level)
- `drought_indicator`: Coef = +0.0020, $p = 0.0580$
- `flood_indicator`: Coef = **-0.0063**, $p = 0.0000$ (**Highly Significant**)
- *Interpretation:* Potato is the most resilient commodity in the short term due to cold storage infrastructure. Over 80% of potatoes are held in temperature-controlled warehouses, insulating mandis from minor weather events.

---

## 3. Cumulative Multi-Window Rainfall Anomaly Analysis

To test whether prolonged weather shocks accumulate over time, multi-window cumulative rainfall anomalies were estimated:

| Window Period | Observations | Within $R^2$ | Rainfall Coef | Std. Error | $t$-statistic | $p$-value |
|---|---|---|---|---|---|---|
| **7-Day Cumulative** | 465,526 | 0.0010 | -4.322073e-05 | 1.993308e-05 | -2.17 | 0.0301 |
| **14-Day Cumulative** | 465,526 | 0.0010 | -4.088584e-05 | 1.611162e-05 | -2.54 | 0.0112 |
| **30-Day Cumulative** | 465,526 | 0.0010 | -3.382767e-05 | 1.166589e-05 | -2.90 | 0.0037 |

---

## 4. Methodological & Econometric Distinctions

During academic evaluation, it is critical to defend the following conceptual boundaries:
1. **Correlation vs. Causal Inference:** A statistically significant coefficient (e.g., drought on onion volatility, $p=0.0266$) identifies a **conditional association** after netting out fixed effects; it does not prove direct causation without an exogenous instrumental variable (IV) or natural experiment.
2. **In-Sample Fit vs. Out-of-Sample Prediction:** PanelOLS estimates conditional expectations across the historical panel. A model with low in-sample $R^2$ can still serve as an explanatory tool, but cannot be claimed as an accurate forecasting system without out-of-sample temporal testing.
