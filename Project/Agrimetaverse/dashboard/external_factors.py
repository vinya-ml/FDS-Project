"""
Agrimetaverse Intelligence - External Factor Analysis Page
==========================================================
Interactive exploration of the 13 econometric, climatic, and supply chain drivers
influencing agricultural commodity markets in India.
Features transparent scientific guidance distinguishing correlation from causation.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.data_loader import load_raw_price_data, load_external_features
from preprocessing.data_cleaning import clean_commodity_data, clean_external_features
from analysis.external_factor_analysis import (
    compute_correlation_matrix,
    plot_correlation_heatmap,
    plot_factor_scatter_regression,
    CAUSATION_DISCLAIMER,
)


def render_external_factors_page():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #1E3A8A; margin:0;">🌐 External Factor & Macro-Climatic Analysis</h2>
            <p style="color: #64748B; margin-top: 4px;">
                Examine empirical relationships between 22 commodity prices and 13 macroeconomic, climatic, trade, and energy indicators.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Scientific Caution Alert
    st.info(f"💡 {CAUSATION_DISCLAIMER}")

    # Load data
    df_prices = load_raw_price_data(config.RAW_DATA_FILE)
    df_prices, _ = clean_commodity_data(df_prices)
    df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)
    df_ext, _ = clean_external_features(df_ext)

    # 1. Cross-Correlation Heatmap
    st.markdown("### 1. Cross-Correlation Matrix: Commodities vs 13 External Drivers")
    c_heat1, c_heat2 = st.columns([1, 3])

    with c_heat1:
        cat_filter = st.selectbox(
            "Filter Commodity Group",
            ["All"] + sorted(list(config.COMMODITY_CATEGORIES.keys())),
            index=0,
        )
        if cat_filter != "All":
            comms_to_show = df_prices[df_prices["Category"] == cat_filter]["Commodity"].unique().tolist()
        else:
            comms_to_show = None

    corr_df = compute_correlation_matrix(df_prices, df_ext, commodity_subset=comms_to_show)
    fig_heat = plot_correlation_heatmap(corr_df)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # 2. Scatter & Regression Fit
    st.markdown("### 2. Pairwise Scatter Plot & Linear Trendline")
    c_sc1, c_sc2 = st.columns(2)

    with c_sc1:
        all_comms = sorted(df_prices["Commodity"].unique().tolist())
        sel_comm = st.selectbox("Select Commodity", all_comms, index=0)

    with c_sc2:
        factor_options = config.EXTERNAL_FEATURE_COLUMNS
        sel_factor = st.selectbox(
            "Select External Factor",
            factor_options,
            format_func=lambda f: f"{config.EXTERNAL_FEATURE_METADATA.get(f, {}).get('label', f)} ({config.EXTERNAL_FEATURE_METADATA.get(f, {}).get('unit', '')})",
            index=0,
        )

    fig_scatter = plot_factor_scatter_regression(df_prices, df_ext, sel_comm, sel_factor)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # 3. External Factor Definitions & Domains
    st.markdown("### 3. Catalog of the 13 External Variables")
    cat_records = []
    for f in config.EXTERNAL_FEATURE_COLUMNS:
        meta = config.EXTERNAL_FEATURE_METADATA.get(f, {})
        cat_records.append({
            "Feature Variable": f,
            "Description": meta.get("label", f),
            "Domain": meta.get("domain", "-"),
            "Unit of Measurement": meta.get("unit", "-"),
            "Role in Forecasting": "Used as Exogenous Regressor in SARIMAX & Prophet",
        })

    st.dataframe(pd.DataFrame(cat_records), use_container_width=True)
