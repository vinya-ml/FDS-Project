"""
Agrimetaverse Intelligence - Exploratory Data Analysis (EDA) Page
================================================================
Interactive visual exploration of historical trends, category indexing,
growth rankings, volatility, and distribution box plots.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.data_loader import load_raw_price_data
from preprocessing.data_cleaning import clean_commodity_data
from analysis.eda import (
    get_growth_summary,
    plot_commodity_trends,
    plot_category_indexed_comparison,
    plot_price_distribution_boxplot,
)


def render_eda_page():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #1E3A8A; margin:0;">📊 Exploratory Data Analysis (EDA)</h2>
            <p style="color: #64748B; margin-top: 4px;">
                Analyze multi-year historical trajectories, category indexed growth, and commodity price distributions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    df_prices = load_raw_price_data(config.RAW_DATA_FILE)
    df_prices, _ = clean_commodity_data(df_prices)

    # 1. Price Trends Section
    st.markdown("### 1. Historical Price Trajectories (2014–2024)")
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        cat_filter = st.selectbox(
            "Filter Category",
            ["All"] + sorted(list(config.COMMODITY_CATEGORIES.keys())),
            index=0,
        )
    with col_f2:
        available_comms = sorted(
            df_prices["Commodity"].unique().tolist()
            if cat_filter == "All"
            else df_prices[df_prices["Category"] == cat_filter]["Commodity"].unique().tolist()
        )
        default_comms = available_comms[:4]
        sel_comms = st.multiselect(
            "Select Specific Commodities to Compare",
            options=available_comms,
            default=default_comms,
        )

    fig_trends = plot_commodity_trends(
        df_prices, selected_commodities=sel_comms, category_filter=cat_filter
    )
    st.plotly_chart(fig_trends, use_container_width=True)

    st.markdown("---")

    # 2. Category Comparisons (Base Year 2014 = 100)
    st.markdown("### 2. Category Price Indices (Base Year 2014 = 100)")
    st.markdown(
        "By normalizing all commodities to an index of 100 in 2014, we can observe "
        "which agricultural sectors experienced the highest structural inflation."
    )
    fig_cat = plot_category_indexed_comparison(df_prices)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # 3. Growth Rankings & CAGR
    st.markdown("### 3. Decade Price Growth & CAGR Rankings (2014 vs 2024)")
    df_growth = get_growth_summary(df_prices)

    col_g1, col_g2 = st.columns([1.5, 1])
    with col_g1:
        st.dataframe(df_growth, use_container_width=True, height=400)
    with col_g2:
        top_growth = df_growth.head(3)
        bot_growth = df_growth.tail(3)
        avg_growth = df_growth["Total_Growth_%"].mean()

        top_html = "".join([f"• {r['Commodity']}: {r['Total_Growth_%']:+.1f}% (₹{r['Price_Initial']:.1f} → ₹{r['Price_Latest']:.1f})<br>" for _, r in top_growth.iterrows()])
        bot_html = "".join([f"• {r['Commodity']}: {r['Total_Growth_%']:+.1f}% (₹{r['Price_Initial']:.1f} → ₹{r['Price_Latest']:.1f})<br>" for _, r in bot_growth.iterrows()])

        st.markdown(f"""
            <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                <div style="color: #065F46; font-size: 0.85rem; font-weight: 600;">🚀 TOP HIGHEST GROWTH COMMODITIES</div>
                <div style="color: #047857; font-size: 0.95rem; font-weight: 600; margin-top: 4px;">
                    {top_html}
                </div>
            </div>
            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
                <div style="color: #991B1B; font-size: 0.85rem; font-weight: 600;">📉 LOWEST / STABLE GROWTH COMMODITIES</div>
                <div style="color: #B91C1C; font-size: 0.95rem; font-weight: 600; margin-top: 4px;">
                    {bot_html}
                </div>
            </div>
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 12px;">
                <div style="color: #1E40AF; font-size: 0.85rem; font-weight: 600;">📊 PORTFOLIO AVERAGE GROWTH</div>
                <div style="color: #1D4ED8; font-size: 1.3rem; font-weight: 700;">{avg_growth:+.1f}%</div>
                <div style="color: #475569; font-size: 0.8rem;">Average price change across all {len(df_growth)} tracked commodity series.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Box Plots Distribution
    st.markdown("### 4. Price Dispersion & Distribution Box Plots")
    fig_box = plot_price_distribution_boxplot(df_prices)
    st.plotly_chart(fig_box, use_container_width=True)
