"""
Agrimetaverse Intelligence - Forecast Visualization Module
==========================================================
Interactive Plotly visualizers for price trajectories, 95% confidence
interval envelopes, actual vs predicted evaluations, and category trends.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Curated harmonious color palette
PALETTE = {
    "historical": "#1E3A8A",      # Deep Navy
    "actual_2024": "#059669",     # Emerald Green
    "arima": "#D97706",           # Amber
    "sarimax": "#2563EB",         # Royal Blue
    "prophet": "#7C3AED",         # Vibrant Purple
    "best": "#10B981",            # Emerald
    "confidence_fill": "rgba(37, 99, 235, 0.15)",
    "grid": "#E2E8F0",
    "background": "#FFFFFF",
}


def plot_commodity_forecast(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    commodity: str,
    unit: str = "₹/unit",
    model_name: str = "SARIMAX",
) -> go.Figure:
    """
    Creates an interactive plot with historical series, future forecast,
    and shaded 95% confidence interval ribbon.
    """
    fig = go.Figure()

    # 1. Historical Actual Line
    fig.add_trace(go.Scatter(
        x=historical_df["Year"],
        y=historical_df["Price"],
        mode="lines+markers",
        name="Historical Actual",
        line=dict(color=PALETTE["historical"], width=3),
        marker=dict(size=7, color=PALETTE["historical"]),
        hovertemplate="Year: %{x}<br>Actual Price: " + unit + " %{y:.2f}<extra></extra>",
    ))

    # Connect last historical point to first forecast point for continuity
    last_hist_yr = historical_df["Year"].iloc[-1]
    last_hist_pr = historical_df["Price"].iloc[-1]

    plot_forecast_years = [last_hist_yr] + forecast_df["Year"].tolist()
    plot_forecast_prices = [last_hist_pr] + forecast_df["Forecast_Price"].tolist()
    plot_lower = [last_hist_pr] + forecast_df["Lower_Bound_95%"].tolist()
    plot_upper = [last_hist_pr] + forecast_df["Upper_Bound_95%"].tolist()

    model_color = PALETTE.get(model_name.lower(), PALETTE["sarimax"])

    # 2. Confidence Interval Upper Bound
    fig.add_trace(go.Scatter(
        x=plot_forecast_years,
        y=plot_upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # 3. Confidence Interval Lower Bound with Fill
    fig.add_trace(go.Scatter(
        x=plot_forecast_years,
        y=plot_lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=PALETTE["confidence_fill"],
        name="95% Confidence Band",
        hoverinfo="skip",
    ))

    # 4. Forecast Trajectory Line
    fig.add_trace(go.Scatter(
        x=plot_forecast_years,
        y=plot_forecast_prices,
        mode="lines+markers",
        name=f"{model_name} Forecast",
        line=dict(color=model_color, width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond", color=model_color),
        hovertemplate="Year: %{x}<br>Forecast: " + unit + " %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{commodity}</b> Price Forecast ({model_name}) with 95% Confidence Interval",
            x=0.02,
            font=dict(size=18, family="Inter, Arial, sans-serif"),
        ),
        xaxis=dict(
            title="Year",
            tickmode="linear",
            dtick=1,
            gridcolor=PALETTE["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=f"Price ({unit})",
            gridcolor=PALETTE["grid"],
            zeroline=False,
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=480,
    )

    return fig


def plot_model_comparison_bar(
    eval_df: pd.DataFrame, metric: str = "RMSE", category_filter: Optional[str] = None
) -> go.Figure:
    """
    Renders grouped bar chart comparing ARIMA, SARIMAX, and Prophet across commodities.
    """
    df = eval_df.copy()
    if category_filter and category_filter != "All":
        df = df[df["Category"] == category_filter]

    fig = go.Figure()

    col_map = {
        "ARIMA": f"ARIMA_{metric}",
        "SARIMAX": f"SARIMAX_{metric}",
        "Prophet": f"Prophet_{metric}",
    }

    for model, col in col_map.items():
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=df["Commodity"],
                y=df[col],
                name=model,
                marker_color=PALETTE[model.lower()],
            ))

    fig.update_layout(
        title=f"Model Comparison by {metric} Across Commodities",
        barmode="group",
        template="plotly_white",
        xaxis=dict(tickangle=-45),
        yaxis=dict(title=metric),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=40, r=40, t=80, b=80),
    )

    return fig


def plot_residuals(actuals: np.ndarray, preds: np.ndarray, commodity: str) -> go.Figure:
    """Plots residual distributions and actual vs fitted points."""
    residuals = actuals - preds
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=preds,
        y=residuals,
        mode="markers",
        marker=dict(size=10, color=PALETTE["sarimax"], line=dict(width=1, color="#1E293B")),
        name="Residuals",
        hovertemplate="Fitted: ₹%{x:.2f}<br>Residual: ₹%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", line_width=1.5)

    fig.update_layout(
        title=f"Residual Analysis ({commodity})",
        xaxis_title="Fitted Price (₹)",
        yaxis_title="Residual (Actual - Fitted) (₹)",
        template="plotly_white",
        height=380,
    )
    return fig
