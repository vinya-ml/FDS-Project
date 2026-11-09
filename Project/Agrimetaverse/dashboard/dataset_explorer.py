"""
Agrimetaverse Intelligence - Dataset Explorer Page
==================================================
Provides interactive inspection of raw and cleaned datasets,
missing value audits, data quality reports, and external feature availability.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.data_loader import (
    inspect_raw_dataset,
    load_raw_price_data,
    load_external_features,
    get_feature_availability_report,
)
from preprocessing.data_cleaning import (
    clean_commodity_data,
    clean_external_features,
    generate_data_quality_report,
)


def render_dataset_explorer_page():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #1E3A8A; margin:0;">📁 Dataset Explorer & Quality Audit</h2>
            <p style="color: #64748B; margin-top: 4px;">
                Inspect historical commodity prices, data-cleaning logs, missing value forward fills, and external feature schema.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Dataset Overview Tabs
    tab_cleaned, tab_raw, tab_quality, tab_external = st.tabs([
        "🧹 Cleaned Dataset",
        "📄 Raw Dataset (Wide)",
        "📊 Data Quality Report",
        "🌐 External Feature Availability",
    ])

    # Load data
    raw_inspect = inspect_raw_dataset(config.RAW_DATA_FILE)
    df_raw_price = pd.read_csv(config.RAW_DATA_FILE)
    df_long = load_raw_price_data(config.RAW_DATA_FILE)
    df_cleaned, cleaning_audit = clean_commodity_data(df_long)
    df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)
    df_ext_clean, ext_audit = clean_external_features(df_ext)

    with tab_cleaned:
        st.markdown("### Cleaned Long-Format Time-Series Data")
        time_span = f"{df_cleaned['Date'].min()} to {df_cleaned['Date'].max()} (Daily)" if "Date" in df_cleaned.columns else f"{int(df_cleaned['Year'].min())} – {int(df_cleaned['Year'].max())}"
        st.markdown(
            f"**Total Records**: {len(df_cleaned):,} | **Commodity Series**: {df_cleaned['Commodity'].nunique()} "
            f"| **Horizon**: {time_span}"
        )

        # Filters
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            cat_list = ["All"] + sorted(list(config.COMMODITY_CATEGORIES.keys()))
            sel_cat = st.selectbox("Filter by Category", cat_list, index=0)
        with c_filter2:
            comm_options = ["All"] + sorted(df_cleaned["Commodity"].unique().tolist())
            if sel_cat != "All":
                comm_options = ["All"] + sorted(
                    df_cleaned[df_cleaned["Category"] == sel_cat]["Commodity"].unique().tolist()
                )
            sel_comm = st.selectbox("Filter by Commodity", comm_options, index=0)

        view_df = df_cleaned.copy()
        if sel_cat != "All":
            view_df = view_df[view_df["Category"] == sel_cat]
        if sel_comm != "All":
            view_df = view_df[view_df["Commodity"] == sel_comm]

        st.dataframe(view_df, use_container_width=True, height=350)

        # Download Cleaned CSV
        csv_data = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_commodity_prices.csv",
            mime="text/csv",
        )

    with tab_raw:
        st.markdown(f"### Raw Dataset Inspection ({config.RAW_DATA_FILE.name})")
        n_rows = raw_inspect.get("rows", raw_inspect.get("row_count", len(df_raw_price)))
        n_cols = raw_inspect.get("cols", raw_inspect.get("col_count", len(df_raw_price.columns)))
        st.info(
            f"Structure: {n_rows:,} rows x {n_cols} columns. "
            f"Contains transactional records across mandis with price and climate attributes."
        )
        st.dataframe(df_raw_price.head(500), use_container_width=True)

        raw_csv = df_raw_price.head(1000).to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Raw Sample ({config.RAW_DATA_FILE.name})",
            data=raw_csv,
            file_name=config.RAW_DATA_FILE.name,
            mime="text/csv",
        )

    with tab_quality:
        st.markdown("### Formal Data Quality & Audit Summary")
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        col_q1.metric("Duplicates Removed", cleaning_audit["duplicates_removed"])
        col_q2.metric("Invalid Prices Detected", cleaning_audit["invalid_prices_found"])
        col_q3.metric("Values Forward Filled", cleaning_audit["values_forward_filled"])
        col_q4.metric("Healthy Series", f"{df_cleaned['Commodity'].nunique()} Series")

        st.markdown("""
            > **Methodological Assurance**: Missing values were strictly forward-filled within each commodity 
            > using previous annual prices. Missing values were **never** cross-filled across different commodities.
        """)

        quality_report = generate_data_quality_report(df_cleaned, df_ext, cleaning_audit)
        st.dataframe(quality_report, use_container_width=True)

    with tab_external:
        st.markdown("### 13 External Regressor Indicators Schema & Availability")
        st.markdown(
            "Evaluation of macroeconomic, agro-climatic, and trade indicators specified in the paper."
        )
        ext_report = get_feature_availability_report(df_ext)
        st.dataframe(ext_report, use_container_width=True)

        st.markdown("#### Preview External Indicator Values (2014–2024)")
        st.dataframe(df_ext, use_container_width=True)

        # Allow user to upload custom external dataset
        st.markdown("---")
        st.markdown("#### 📤 Upload Custom External Features Dataset")
        uploaded_file = st.file_uploader("Upload External Drivers CSV (must contain 'Year' column)", type=["csv"])
        if uploaded_file is not None:
            try:
                custom_ext = pd.read_csv(uploaded_file)
                if "Year" in custom_ext.columns:
                    st.success(f"Custom external dataset uploaded successfully with {len(custom_ext)} years and {len(custom_ext.columns)} columns!")
                    st.dataframe(custom_ext.head(3))
                else:
                    st.error("Uploaded CSV must contain a 'Year' column.")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
