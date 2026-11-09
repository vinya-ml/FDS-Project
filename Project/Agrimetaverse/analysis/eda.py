"""
Agrimetaverse Intelligence - Exploratory Data Analysis (EDA) Module
==================================================================
Comprehensive visual and statistical exploratory analysis of commodity price
trajectories, category aggregations, volatility, and historical distributions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_growth_summary(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price growth across all tracked commodities:
    - Initial Price
    - Latest Price
    - Absolute Growth (₹)
    - Total Growth (%)
    - Compound Annual Growth Rate (CAGR %)
    """
    records = []
    commodities = sorted(df_prices["Commodity"].unique().tolist())
    has_date = "Date" in df_prices.columns

    for c in commodities:
        sub = df_prices[df_prices["Commodity"] == c]
        if has_date:
            sub = sub.sort_values("Date").reset_index(drop=True)
            p_init = float(sub["Price"].iloc[0])
            p_latest = float(sub["Price"].iloc[-1])
            # Time delta in years
            d_init = pd.to_datetime(sub["Date"].iloc[0])
            d_latest = pd.to_datetime(sub["Date"].iloc[-1])
            n_years = max(0.5, (d_latest - d_init).days / 365.25)
        else:
            sub = sub.sort_values("Year").reset_index(drop=True)
            p_init = float(sub["Price"].iloc[0])
            p_latest = float(sub["Price"].iloc[-1])
            n_years = max(1.0, float(sub["Year"].iloc[-1] - sub["Year"].iloc[0]))

        abs_growth = p_latest - p_init
        pct_growth = (abs_growth / (p_init + 1e-6)) * 100.0
        cagr = (((p_latest / (p_init + 1e-6)) ** (1.0 / n_years)) - 1.0) * 100.0

        cat = sub["Category"].iloc[0] if "Category" in sub.columns else config.COMMODITY_TO_CATEGORY.get(c, "Vegetables")

        records.append({
            "Commodity": c,
            "Category": cat,
            "Price_Initial": round(p_init, 2),
            "Price_Latest": round(p_latest, 2),
            "Absolute_Growth_₹": round(abs_growth, 2),
            "Total_Growth_%": round(pct_growth, 2),
            "CAGR_%": round(cagr, 2),
            "Unit": config.COMMODITY_UNITS.get(c, "₹/Quintal"),
        })

    df_growth = pd.DataFrame(records).sort_values("Total_Growth_%", ascending=False).reset_index(drop=True)
    return df_growth


def plot_commodity_trends(
    df_prices: pd.DataFrame,
    selected_commodities: Optional[List[str]] = None,
    category_filter: Optional[str] = None,
) -> go.Figure:
    """Multi-commodity price trajectory line chart with rich hover details."""
    df = df_prices.copy()
    if category_filter and category_filter != "All":
        df = df[df["Category"] == category_filter]
    if selected_commodities:
        df = df[df["Commodity"].isin(selected_commodities)]

    time_col = "Date" if "Date" in df.columns else "Year"

    fig = px.line(
        df,
        x=time_col,
        y="Price",
        color="Commodity",
        markers=True if time_col == "Year" else False,
        title="Commodity Price Trajectories",
        labels={"Price": f"Price ({config.DEFAULT_UNIT})", time_col: "Date" if time_col == "Date" else "Year"},
        template="plotly_white",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=480,
    )
    return fig


def plot_category_indexed_comparison(df_prices: pd.DataFrame) -> go.Figure:
    """
    Plots category average price trajectory indexed to 100 at baseline start.
    Allows fair relative comparison across diverse price scales.
    """
    df = df_prices.copy()
    time_col = "Date" if "Date" in df.columns else "Year"

    # Compute base price per commodity as initial observation
    base_map = {}
    for c, grp in df.groupby("Commodity"):
        grp_sorted = grp.sort_values(time_col)
        base_map[c] = float(grp_sorted["Price"].iloc[0])

    df["Indexed_Price"] = df.apply(
        lambda r: (r["Price"] / (base_map.get(r["Commodity"], r["Price"]) + 1e-6)) * 100.0,
        axis=1,
    )

    cat_avg = (
        df.groupby(["Category", time_col])["Indexed_Price"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        cat_avg,
        x=time_col,
        y="Indexed_Price",
        color="Category",
        markers=True if time_col == "Year" else False,
        title="Category Price Index Trajectories (Indexed Baseline = 100)",
        labels={"Indexed_Price": "Price Index (Baseline = 100)", time_col: time_col},
        template="plotly_white",
    )
    fig.update_layout(
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_price_distribution_boxplot(df_prices: pd.DataFrame) -> go.Figure:
    """Box plots illustrating dispersion and outlier structure per commodity."""
    fig = px.box(
        df_prices,
        x="Commodity",
        y="Price",
        color="Category" if "Category" in df_prices.columns else None,
        title="Price Distributions Across Commodities",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        yaxis_title=f"Price ({config.DEFAULT_UNIT})",
        height=450,
        margin=dict(l=40, r=40, t=60, b=100),
    )
    return fig
