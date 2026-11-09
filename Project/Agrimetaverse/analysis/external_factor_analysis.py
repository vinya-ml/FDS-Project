"""
Agrimetaverse Intelligence - External Factor Analysis Module
============================================================
Analyzes empirical correlations and relationships between agricultural prices
and 13 macroeconomic, climatic, trade, and energy indicators.
Features transparent scientific guidance distinguishing correlation from causation.
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


CAUSATION_DISCLAIMER = (
    "STATISTICAL NOTE: Correlation indicates co-movement over time, not direct causation. "
    "For instance, correlation between weather patterns and domestic commodity prices "
    "reflects agro-climatic seasonal impacts, harvest yield cycles, and regional mandi arrivals. "
    "Similarly, policy factors like Minimum Support Price (MSP) act with multi-season harvest lags."
)


def compute_correlation_matrix(
    df_prices: pd.DataFrame, df_ext: pd.DataFrame, commodity_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes Pearson correlation coefficients between commodity prices and 13 external features.
    """
    time_col = "Date" if "Date" in df_prices.columns and "Date" in df_ext.columns else "Year"
    
    wide_prices = df_prices.pivot_table(index=time_col, columns="Commodity", values="Price", aggfunc="mean")
    if commodity_subset:
        wide_prices = wide_prices[[c for c in commodity_subset if c in wide_prices.columns]]

    ext_dedup = df_ext.drop_duplicates(subset=[time_col]).set_index(time_col)
    feat_cols = [c for c in config.EXTERNAL_FEATURE_COLUMNS if c in ext_dedup.columns]
    ext_subset = ext_dedup[feat_cols]

    # Combine on index
    combined = pd.concat([wide_prices, ext_subset], axis=1, join="inner")
    
    # Compute full correlation matrix
    corr = combined.corr()

    # Slice cross-correlation: rows = Commodities, cols = External Features
    commodities = wide_prices.columns.tolist()
    cross_corr = corr.loc[commodities, feat_cols].fillna(0.0)
    return cross_corr


def plot_correlation_heatmap(cross_corr: pd.DataFrame) -> go.Figure:
    """Renders interactive cross-correlation heatmap."""
    col_labels = [
        config.EXTERNAL_FEATURE_METADATA.get(c, {}).get("label", c)
        for c in cross_corr.columns
    ]

    fig = px.imshow(
        cross_corr.values,
        x=col_labels,
        y=cross_corr.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Commodity Prices vs External Factors (Pearson Correlation Heatmap)",
        labels=dict(x="External Factor", y="Commodity", color="Correlation (r)"),
        aspect="auto",
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        height=520,
        margin=dict(l=40, r=40, t=60, b=120),
    )
    return fig


def plot_factor_scatter_regression(
    df_prices: pd.DataFrame,
    df_ext: pd.DataFrame,
    commodity: str,
    feature: str,
) -> go.Figure:
    """
    Generates scatter plot with trendline showing commodity price vs external indicator.
    """
    join_col = "Date" if "Date" in df_prices.columns and "Date" in df_ext.columns else "Year"
    comm_df = df_prices[df_prices["Commodity"] == commodity][[join_col, "Price"]].dropna()
    feat_df = df_ext[[join_col, feature]].dropna()
    merged = pd.merge(comm_df, feat_df, on=join_col, how="inner")

    feat_meta = config.EXTERNAL_FEATURE_METADATA.get(feature, {})
    feat_label = feat_meta.get("label", feature)
    feat_unit = feat_meta.get("unit", "")
    comm_unit = config.COMMODITY_UNITS.get(commodity, "₹/Quintal")

    if len(merged) < 2:
        fig = go.Figure()
        fig.update_layout(title="Insufficient overlapping observations for scatter plot")
        return fig

    # Fit linear trendline
    try:
        slope, intercept = np.polyfit(merged[feature], merged["Price"], 1)
        x_range = np.linspace(merged[feature].min(), merged[feature].max(), 50)
        y_trend = slope * x_range + intercept
        r_corr = float(merged["Price"].corr(merged[feature]))
    except Exception:
        slope, intercept, r_corr = 0.0, float(merged["Price"].mean()), 0.0
        x_range = np.array([merged[feature].min(), merged[feature].max()])
        y_trend = np.array([intercept, intercept])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged[feature],
        y=merged["Price"],
        mode="markers",
        marker=dict(size=7, color="#2563EB", opacity=0.7, line=dict(width=0.5, color="#1E293B")),
        name="Observations",
        hovertemplate=f"{join_col}: %{{text}}<br>{feat_label}: %{{x:.2f}} {feat_unit}<br>Price: %{{y:.2f}} {comm_unit}<extra></extra>",
        text=merged[join_col].astype(str),
    ))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_trend,
        mode="lines",
        line=dict(color="#DC2626", width=2.5, dash="dash"),
        name=f"Linear Fit (r = {r_corr:.2f})",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"<b>{commodity}</b> Price vs {feat_label} (r = {r_corr:.2f})",
        xaxis_title=f"{feat_label} ({feat_unit})",
        yaxis_title=f"Price ({comm_unit})",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
