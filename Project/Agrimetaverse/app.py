"""
Agrimetaverse Intelligence - Main Application
=============================================
Advanced Agricultural Commodity Price Forecasting & Decision-Support System
Main multi-page Streamlit application orchestrating:
    - Home Dashboard
    - Dataset Explorer & Quality Audit
    - Exploratory Data Analysis (EDA)
    - Future Price Forecasting Studio
    - Model Evaluation & Benchmark Comparison
    - External Factor & Climate Drivers Analysis
    - Executive Reports & CSV Exports
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure Agrimetaverse root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from dashboard.home import render_home_page
from dashboard.dataset_explorer import render_dataset_explorer_page
from dashboard.eda_page import render_eda_page
from dashboard.forecasting_page import render_forecasting_page
from dashboard.model_comparison import render_model_comparison_page
from dashboard.external_factors import render_external_factors_page
from dashboard.reports import render_reports_page

# Configure Page
st.set_page_config(
    page_title="Agrimetaverse Intelligence | Agricultural Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #E2E8F0;
        font-size: 0.95rem;
        padding: 6px 0;
    }

    /* Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stDownloadButton>button {
        border-radius: 8px;
        font-weight: 500;
        background-color: #1E3A8A;
        color: white;
    }

    /* Hide full screen icon for cleaner UI */
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    # Sidebar Branding
    st.sidebar.markdown("""
        <div style="padding: 12px 0 20px 0; border-bottom: 1px solid #334155;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🌾</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">Agrimetaverse</h2>
                    <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Intelligence v2.4</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 0.8rem; color: #CBD5E1;">
                Agricultural Commodity Price Forecasting System (India)
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    pages = {
        "🏠 Home Dashboard": render_home_page,
        "📁 Dataset Explorer": render_dataset_explorer_page,
        "📊 Exploratory Data Analysis (EDA)": render_eda_page,
        "🔮 Price Forecasting Studio": render_forecasting_page,
        "⚖️ Model Evaluation & Comparison": render_model_comparison_page,
        "🌐 External Factor Analysis": render_external_factors_page,
        "📄 Reports & Data Export": render_reports_page,
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        index=0,
    )

    # Sidebar System Info
    st.sidebar.markdown("<br/><br/>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="background: #1E293B; border-radius: 8px; padding: 12px; font-size: 0.78rem; color: #94A3B8;">
            <div style="color: #E2E8F0; font-weight: 600; margin-bottom: 4px;">SYSTEM STATUS</div>
            <div>• Horizon: <b>June 2023 – June 2025</b></div>
            <div>• Commodities: <b>15 Series (MH & KA)</b></div>
            <div>• Models: <b>6 Architectures (ARIMA, SARIMAX, Prophet, XGBoost, ELM-GA, Hybrid)</b></div>
            <div>• External Factors: <b>13 Agro-Climatic Features</b></div>
            <div style="margin-top: 6px; color: #10B981;">● Online & Calibrated</div>
        </div>
    """, unsafe_allow_html=True)

    # Render selected page
    render_func = pages[selected_page_name]
    render_func()

    # Footer
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #94A3B8; font-size: 0.8rem; padding: 20px 0; border-top: 1px solid #E2E8F0;">
            <b>Agrimetaverse Intelligence</b> — Agricultural Price Forecasting & Decision-Support System<br/>
            Based on the research paper: <i>"Agrimetaverse Intelligence: Advanced Agricultural Commodity Price Forecasting System"</i>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
