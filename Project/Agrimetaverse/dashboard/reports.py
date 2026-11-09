"""
Agrimetaverse Intelligence - Reports & Export Page
==================================================
Generates downloadable executive reports in PDF format (via ReportLab)
and CSV extracts of cleaned time series, evaluation metrics, and future forecasts.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing.data_loader import load_raw_price_data, load_external_features
from preprocessing.data_cleaning import clean_commodity_data
from models.model_training import get_cached_or_train_evaluation
from reports.report_generator import generate_pdf_report, export_forecasts_csv


def render_reports_page():
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #1E3A8A; margin:0;">📄 Decision Support Reports & Data Export</h2>
            <p style="color: #64748B; margin-top: 4px;">
                Download comprehensive executive reports (PDF) and structured datasets (CSV) for analysts, farmers, and policymakers.
            </p>
        </div>
    """, unsafe_allow_html=True)

    try:
        eval_df, overall_metrics, comp_df = get_cached_or_train_evaluation()
        df_prices = load_raw_price_data(config.RAW_DATA_FILE)
        df_prices, _ = clean_commodity_data(df_prices)
    except Exception as e:
        st.error(f"Error preparing reports: {e}")
        return

    # PDF Report Generation Card
    st.markdown("### 1. Executive Intelligence PDF Report")
    st.markdown("""
        The generated PDF report includes:
        - Executive Summary of the research methodology
        - Chronological split specification (70% train, 15% val, 15% test)
        - Side-by-side benchmark comparison against published research results
        - Commodity-wise performance breakdown and best model recommendations
        - Policy and decision-support guidance for agricultural market participants
    """)

    pdf_path = config.REPORTS_DIR / "Agrimetaverse_Executive_Report.pdf"
    
    # Generate on demand
    if st.button("🔄 Generate Latest Executive PDF Report"):
        with st.spinner("Compiling PDF report with ReportLab..."):
            try:
                generate_pdf_report(eval_df, overall_metrics, comp_df, output_pdf_path=pdf_path)
                st.success("PDF Report generated successfully!")
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")

    # If PDF exists, show download
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📥 Download Executive Intelligence Report (PDF)",
            data=pdf_bytes,
            file_name="Agrimetaverse_Executive_Report.pdf",
            mime="application/pdf",
        )
    else:
        # Pre-generate silently so download is immediately available
        try:
            generate_pdf_report(eval_df, overall_metrics, comp_df, output_pdf_path=pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Executive Intelligence Report (PDF)",
                data=pdf_bytes,
                file_name="Agrimetaverse_Executive_Report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.info("Click the button above to generate the PDF report.")

    st.markdown("---")

    # CSV Data Exports
    st.markdown("### 2. Structured CSV Data Exports")
    c_csv1, c_csv2, c_csv3 = st.columns(3)

    with c_csv1:
        st.markdown("#### Model Evaluation Metrics")
        st.caption(f"Holdout test metrics across all {len(eval_df)} series.")
        csv_eval = eval_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv_eval,
            file_name="model_evaluation_metrics.csv",
            mime="text/csv",
        )

    with c_csv2:
        st.markdown("#### Cleaned Commodity Prices")
        st.caption("Standardized mandi daily price time series.")
        csv_prices = df_prices.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Prices (CSV)",
            data=csv_prices,
            file_name="cleaned_commodity_prices.csv",
            mime="text/csv",
        )

    with c_csv3:
        st.markdown("#### 13 External Regressors")
        st.caption("Weather, macroeconomic, and trade indicators.")
        df_ext = load_external_features(config.EXTERNAL_FEATURES_FILE)
        csv_ext = df_ext.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download External Factors (CSV)",
            data=csv_ext,
            file_name="enhanced_features.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # Data Preview
    st.markdown("### 3. Preview: Model Evaluation Dataset")
    st.dataframe(eval_df, use_container_width=True, height=300)
