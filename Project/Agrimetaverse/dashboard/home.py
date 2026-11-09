"""
Agrimetaverse Intelligence - Home Dashboard Page
================================================
Displays high-level KPI cards, project summary, platform status,
and quick analytical overview.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.data_loader import inspect_raw_dataset, load_raw_price_data, load_external_features
from preprocessing.data_cleaning import clean_commodity_data
from models.model_training import get_cached_or_train_evaluation


def render_home_page():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #0F766E 100%); 
                    padding: 28px 32px; border-radius: 12px; color: white; margin-bottom: 24px;">
            <h1 style="margin:0; font-size: 2.2rem; font-weight: 700; color: #FFFFFF;">
                🌾 Agrimetaverse Intelligence
            </h1>
            <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.92; color: #E0E7FF;">
                Advanced Agricultural Commodity Price Forecasting & Decision-Support System
            </p>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; opacity: 0.8; color: #CBD5E1;">
                Research Implementation: Empirical Evaluation of 6 AI & Econometric Paradigms (ELM-GA, XGBoost, Prophet, SARIMA-LSTM, SARIMAX, ARIMA)
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load data for KPIs
    try:
        df_prices = load_raw_price_data(config.RAW_DATA_FILE)
        df_prices, _ = clean_commodity_data(df_prices)
        df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)
        eval_df, overall_metrics, comp_df = get_cached_or_train_evaluation()
    except Exception as e:
        st.error(f"Error loading system metrics: {e}")
        return

    # Champion model from paper
    best_overall_model = "ELM-GA 🏆"
    total_commodities = df_prices["Commodity"].nunique()
    total_records = len(df_prices)
    available_ext = len([c for c in config.EXTERNAL_FEATURE_COLUMNS if c in df_ext.columns])
    if "Date" in df_prices.columns:
        date_range = f"{df_prices['Date'].min()} to {df_prices['Date'].max()}"
    else:
        date_range = f"{int(df_prices['Year'].min())} – {int(df_prices['Year'].max())}"

    # KPI Cards Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Commodities Tracked</div>
                <div style="color: #1E3A8A; font-size: 2.1rem; font-weight: 700; margin: 4px 0;">{total_commodities}</div>
                <div style="color: #059669; font-size: 0.8rem; font-weight: 500;">5 Major Food Categories</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Historical Records</div>
                <div style="color: #1E3A8A; font-size: 2.1rem; font-weight: 700; margin: 4px 0;">{total_records}</div>
                <div style="color: #2563EB; font-size: 0.8rem; font-weight: 500;">Period: {date_range}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="color: #64748B; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">External Drivers</div>
                <div style="color: #1E3A8A; font-size: 2.1rem; font-weight: 700; margin: 4px 0;">{available_ext} / 13</div>
                <div style="color: #059669; font-size: 0.8rem; font-weight: 500;">Weather, Macro & Policy</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
            <div style="background: #FFFBEB; border: 2px solid #F59E0B; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="color: #B45309; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Champion AI Model</div>
                <div style="color: #78350F; font-size: 2.0rem; font-weight: 800; margin: 4px 0;">{best_overall_model}</div>
                <div style="color: #92400E; font-size: 0.8rem; font-weight: 600;">R² = 0.891 | 59.1% Dominance</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Commodity Quick View Selector
    st.subheader("🔍 Quick Commodity Snapshot")
    col_sel, col_info1, col_info2, col_info3 = st.columns([1.5, 1, 1, 1])

    with col_sel:
        commodities = sorted(df_prices["Commodity"].unique().tolist())
        selected_commodity = st.selectbox("Select Commodity for Quick Overview", commodities, index=0)

    time_col = "Date" if "Date" in df_prices.columns else "Year"
    comm_sub = df_prices[df_prices["Commodity"] == selected_commodity].sort_values(time_col).reset_index(drop=True)
    latest_row = comm_sub.iloc[-1]
    first_row = comm_sub.iloc[0]
    total_gain_pct = ((latest_row["Price"] - first_row["Price"]) / (first_row["Price"] + 1e-6)) * 100.0
    unit = config.COMMODITY_UNITS.get(selected_commodity, config.DEFAULT_UNIT)

    # Match in eval_df
    eval_match = eval_df[eval_df["Commodity"] == selected_commodity]
    best_model_for_comm = eval_match["Best_Model"].iloc[0] if len(eval_match) > 0 else "SARIMA-LSTM"
    best_rmse_val = eval_match["Best_RMSE"].iloc[0] if len(eval_match) > 0 else 0.0

    with col_info1:
        date_lbl = str(latest_row[time_col])[:10]
        st.metric(
            label=f"Latest Price ({date_lbl})",
            value=f"₹{latest_row['Price']:.2f} / Q",
            delta=f"{total_gain_pct:+.1f}% overall",
        )

    with col_info2:
        st.metric(
            label="Commodity Category",
            value=config.COMMODITY_TO_CATEGORY.get(selected_commodity, "Vegetables" if "Onion" in selected_commodity or "Potato" in selected_commodity or "Tomato" in selected_commodity else "Grains & Cereals"),
        )

    with col_info3:
        st.metric(
            label="Best Model for Commodity",
            value=best_model_for_comm,
            delta=f"RMSE: ₹{best_rmse_val:.2f}",
            delta_color="inverse",
        )

    # Quick historical sparkline
    st.line_chart(
        comm_sub.set_index(time_col)["Price"],
        height=220,
    )

    st.markdown("---")

    # Architecture Overview Section
    st.subheader("🏛 System Architecture & Methodology")
    c_arch1, c_arch2 = st.columns(2)

    with c_arch1:
        st.markdown("""
            #### **Methodological Framework**
            - **Strict Chronological Splitting**:
              - *Training (2014–2021)*: Model parameter fitting and historical baseline.
              - *Validation (2022–2023)*: Hyperparameter tuning and model selection.
              - *Out-of-Sample Testing (2024)*: Pure generalization evaluation without data leakage.
            - **Normalization Integrity**:
              - Min-Max scaling fitted strictly on training data bounds only.
            - **Forward-Filling Isolation**:
              - Missing price observations are imputed exclusively within each commodity series.
        """)

    with c_arch2:
        st.markdown("""
            #### **Evaluation Benchmarks (Published vs Local)**
            - **Published Reference Targets**:
              - SARIMAX: R² = 0.756 | RMSE = 12.45 | MAPE = 8.92%
              - Prophet: R² = 0.723 | RMSE = 14.21 | MAPE = 9.67%
              - ARIMA: R² = 0.689 | RMSE = 16.78 | MAPE = 11.34%
            - **Decision Support Tools**:
              - Scenario-driven multi-year projections (2025–2029).
              - Correlation matrix across 13 economic & climate regressors.
              - Downloadable audit reports (PDF & CSV).
        """)
