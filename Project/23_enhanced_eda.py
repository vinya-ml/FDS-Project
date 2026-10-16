"""
Enhanced Publication-Quality Exploratory Data Analysis (EDA)
Food Supply Chain & Inflation Dynamics: Mandi Price Spikes vs. Extreme Weather

Generates publication-quality charts in outputs/figures/ and numerical summaries in outputs/tables/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

def run_enhanced_eda():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    tables_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 3: GENERATING PUBLICATION-QUALITY EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    data_path = os.path.join(base_dir, "final_mandi.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "final_mandi.csv.gz")

    print(f"Loading analytical dataset from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    commodity_colors = {
        "Tomato": "#D32F2F",   # Red
        "Onion": "#7B1FA2",    # Purple
        "Potato": "#E65100"    # Amber/Brown
    }

    # -------------------------------------------------------------
    # 1. Price Trends Over Time (Aggregated to 7-Day Rolling Mean & Median)
    # -------------------------------------------------------------
    print("\n[1/9] Generating Commodity Price Trends over Time...")
    daily_price = df.groupby(["date", "Commodity"])["Modal_Price"].agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    plt.figure(figsize=(12, 6), dpi=300)
    for c in ["Tomato", "Onion", "Potato"]:
        sub = daily_price[daily_price["Commodity"] == c].sort_values("date")
        # 7-day moving average for smooth readable trend without discarding spikes
        roll_mean = sub.set_index("date")["mean"].rolling(7, min_periods=3).mean()
        roll_q25 = sub.set_index("date")["q25"].rolling(7, min_periods=3).mean()
        roll_q75 = sub.set_index("date")["q75"].rolling(7, min_periods=3).mean()
        
        plt.plot(roll_mean.index, roll_mean.values, label=f"{c} (7d Mean)",
                 color=commodity_colors[c], linewidth=2)
        plt.fill_between(roll_mean.index, roll_q25.values, roll_q75.values,
                         color=commodity_colors[c], alpha=0.15)

    plt.title("National Average Mandi Prices (2023 - 2025)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11, fontweight="bold")
    plt.ylabel("Modal Price (₹ per Quintal)", fontsize=11, fontweight="bold")
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    p1 = os.path.join(figures_dir, "01_commodity_price_trends.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"    Saved: {p1}")

    # -------------------------------------------------------------
    # 2. Commodity Price Distributions
    # -------------------------------------------------------------
    print("\n[2/9] Generating Commodity Price Distributions...")
    plt.figure(figsize=(11, 5), dpi=300)
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df, x="Commodity", y="Modal_Price",
        palette=commodity_colors, showfliers=False, width=0.4
    )
    plt.title("Modal Price Distribution (Excl. Far Outliers)", fontsize=11, fontweight="bold")
    plt.ylabel("Modal Price (₹ / Quintal)", fontsize=10, fontweight="bold")
    plt.xlabel("Commodity", fontsize=10, fontweight="bold")

    plt.subplot(1, 2, 2)
    for c in ["Tomato", "Onion", "Potato"]:
        sub = df[df["Commodity"] == c]["Modal_Price"]
        # Limit to 99th percentile for density visibility
        sub_clip = sub[sub <= sub.quantile(0.99)]
        sns.kdeplot(sub_clip, label=c, color=commodity_colors[c], linewidth=2)
    plt.title("Modal Price Density Comparison (99th Pct Clip)", fontsize=11, fontweight="bold")
    plt.xlabel("Modal Price (₹ / Quintal)", fontsize=10, fontweight="bold")
    plt.ylabel("Density", fontsize=10, fontweight="bold")
    plt.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    p2 = os.path.join(figures_dir, "02_commodity_price_distributions.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"    Saved: {p2}")

    # -------------------------------------------------------------
    # 3. 30-Day Volatility Distributions Across Commodities
    # -------------------------------------------------------------
    print("\n[3/9] Generating 30-Day Volatility Distributions...")
    plt.figure(figsize=(11, 5), dpi=300)
    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df, x="Commodity", y="volatility_30d",
        palette=commodity_colors, showfliers=False, width=0.4
    )
    plt.title("30-Day Price Volatility by Commodity", fontsize=11, fontweight="bold")
    plt.ylabel("Standard Deviation of Log Returns", fontsize=10, fontweight="bold")
    plt.xlabel("Commodity", fontsize=10, fontweight="bold")

    plt.subplot(1, 2, 2)
    for c in ["Tomato", "Onion", "Potato"]:
        sub = df[df["Commodity"] == c]["volatility_30d"].dropna()
        sub_clip = sub[sub <= sub.quantile(0.99)]
        sns.kdeplot(sub_clip, label=c, color=commodity_colors[c], linewidth=2)
    plt.title("Volatility Density Comparison", fontsize=11, fontweight="bold")
    plt.xlabel("30-Day Rolling Volatility", fontsize=10, fontweight="bold")
    plt.ylabel("Density", fontsize=10, fontweight="bold")
    plt.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(figures_dir, "03_volatility_30d_distributions.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"    Saved: {p3}")

    # Volatility Numerical Summary Table
    vol_summary = df.groupby("Commodity")["volatility_30d"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).round(4)
    vol_summary_path = os.path.join(tables_dir, "eda_commodity_volatility_summary.csv")
    vol_summary.to_csv(vol_summary_path)
    print(f"    Saved summary table: {vol_summary_path}")

    # -------------------------------------------------------------
    # 4. Temperature and Rainfall Anomaly Distributions
    # -------------------------------------------------------------
    print("\n[4/9] Generating Weather Anomaly Distributions...")
    plt.figure(figsize=(12, 5), dpi=300)

    plt.subplot(1, 2, 1)
    sns.histplot(df["temp_anomaly"], bins=50, kde=True, color="#E65100", edgecolor="black", alpha=0.6)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(df["temp_anomaly"].quantile(0.95), color="red", linestyle=":", linewidth=1.5,
                label=f"95th Pct Shock ({df['temp_anomaly'].quantile(0.95):.1f}°C)")
    plt.title("Distribution of Temperature Anomalies (°C)", fontsize=11, fontweight="bold")
    plt.xlabel("Temperature Anomaly (°C from State-Month Baseline)", fontsize=10, fontweight="bold")
    plt.ylabel("Frequency", fontsize=10, fontweight="bold")
    plt.legend(frameon=True, fontsize=9)

    plt.subplot(1, 2, 2)
    # Clip extreme rain anomaly for readable histogram
    rain_clip = df["rain_anomaly"].clip(lower=-20, upper=80)
    sns.histplot(rain_clip, bins=50, kde=True, color="#1976D2", edgecolor="black", alpha=0.6)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(df["rain_anomaly"].quantile(0.95), color="red", linestyle=":", linewidth=1.5,
                label=f"95th Pct Shock ({df['rain_anomaly'].quantile(0.95):.1f} mm)")
    plt.title("Distribution of Rainfall Anomalies (mm)", fontsize=11, fontweight="bold")
    plt.xlabel("Rainfall Anomaly (mm from State-Month Baseline)", fontsize=10, fontweight="bold")
    plt.ylabel("Frequency", fontsize=10, fontweight="bold")
    plt.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    p4 = os.path.join(figures_dir, "04_weather_anomaly_distributions.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"    Saved: {p4}")

    # -------------------------------------------------------------
    # 5. Weather Anomalies vs Price Volatility (Binned Hexbin / Deciles)
    # -------------------------------------------------------------
    print("\n[5/9] Generating Weather Anomalies vs Price Volatility Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Bin temperature anomaly into deciles and compute mean volatility
    df["temp_bin"] = pd.qcut(df["temp_anomaly"], q=10, duplicates="drop")
    temp_grouped = df.groupby(["temp_bin", "Commodity"], observed=True)["volatility_30d"].mean().unstack()
    temp_grouped.index = [f"{b.left:.1f} to {b.right:.1f}°C" for b in temp_grouped.index]
    temp_grouped.plot(kind="bar", ax=axes[0], color=[commodity_colors[c] for c in temp_grouped.columns], width=0.8)
    axes[0].set_title("Price Volatility by Temperature Anomaly Deciles", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Mean 30-Day Volatility", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("Temperature Anomaly Deciles", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Commodity", fontsize=8)

    # Bin rainfall anomaly into deciles
    df["rain_bin"] = pd.qcut(df["rain_anomaly"], q=10, duplicates="drop")
    rain_grouped = df.groupby(["rain_bin", "Commodity"], observed=True)["volatility_30d"].mean().unstack()
    rain_grouped.index = [f"{b.left:.1f} to {b.right:.1f}mm" for b in rain_grouped.index]
    rain_grouped.plot(kind="bar", ax=axes[1], color=[commodity_colors[c] for c in rain_grouped.columns], width=0.8)
    axes[1].set_title("Price Volatility by Rainfall Anomaly Deciles", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Mean 30-Day Volatility", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Rainfall Anomaly Deciles", fontsize=9, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Commodity", fontsize=8)

    plt.tight_layout()
    p5 = os.path.join(figures_dir, "05_weather_vs_volatility_deciles.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"    Saved: {p5}")

    # -------------------------------------------------------------
    # 6. Extreme Weather vs Normal Observations Impact
    # -------------------------------------------------------------
    print("\n[6/9] Generating Extreme Weather vs Normal Comparison...")
    weather_shocks = [
        ("Extreme Heat Shock", "extreme_temp"),
        ("Extreme Rain Shock", "extreme_rain"),
        ("Drought Indicator", "drought_indicator"),
        ("Flood Indicator", "flood_indicator")
    ]

    shock_rows = []
    for label, col in weather_shocks:
        if col in df.columns:
            for c in ["Tomato", "Onion", "Potato"]:
                sub = df[df["Commodity"] == c]
                vol_normal = sub[sub[col] == 0]["volatility_30d"].mean()
                vol_shock = sub[sub[col] == 1]["volatility_30d"].mean()
                diff_pct = ((vol_shock - vol_normal) / vol_normal * 100) if vol_normal > 0 else 0
                shock_rows.append({
                    "Shock": label,
                    "Commodity": c,
                    "Normal_Volatility": round(vol_normal, 4),
                    "Shock_Volatility": round(vol_shock, 4),
                    "Pct_Difference": round(diff_pct, 2)
                })

    shock_df = pd.DataFrame(shock_rows)
    shock_table_path = os.path.join(tables_dir, "eda_extreme_weather_summary.csv")
    shock_df.to_csv(shock_table_path, index=False)

    plt.figure(figsize=(11, 5), dpi=300)
    sns.barplot(
        data=shock_df, x="Shock", y="Pct_Difference", hue="Commodity",
        palette=commodity_colors, edgecolor="black", linewidth=0.8
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("% Difference in 30-Day Volatility (Shock vs. Normal)", fontsize=10, fontweight="bold")
    plt.title("Impact of Extreme Weather Shocks on Agricultural Price Volatility", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    p6 = os.path.join(figures_dir, "06_extreme_weather_impact_comparison.png")
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"    Saved: {p6}")

    # -------------------------------------------------------------
    # 7. State-Wise Price Volatility Variation
    # -------------------------------------------------------------
    print("\n[7/9] Generating State-Wise Volatility Variation...")
    top_states = df["STATE"].value_counts().head(10).index.tolist()
    state_df = df[df["STATE"].isin(top_states)]
    state_vol = state_df.groupby(["STATE", "Commodity"])["volatility_30d"].mean().unstack().loc[top_states]
    
    state_table_path = os.path.join(tables_dir, "eda_statewise_summary.csv")
    state_vol.round(4).to_csv(state_table_path)

    plt.figure(figsize=(12, 6), dpi=300)
    state_vol.plot(kind="bar", figsize=(12, 6), color=[commodity_colors[c] for c in state_vol.columns],
                   edgecolor="black", linewidth=0.8)
    plt.title("Top 10 Agricultural States: Average 30-Day Price Volatility", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Mean 30-Day Return Volatility", fontsize=11, fontweight="bold")
    plt.xlabel("State", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Commodity", frameon=True, fontsize=10)
    plt.tight_layout()
    p7 = os.path.join(figures_dir, "07_statewise_volatility_variation.png")
    plt.savefig(p7, dpi=300)
    plt.close()
    print(f"    Saved: {p7}")

    # -------------------------------------------------------------
    # 8. Correlation Analysis for Relevant Numerical Features
    # -------------------------------------------------------------
    print("\n[8/9] Generating Annotated Correlation Heatmap...")
    num_cols = [
        "volatility_30d", "Modal_Price", "log_return",
        "temp_anomaly", "rain_anomaly", "extreme_temp",
        "extreme_rain", "flood_indicator", "drought_indicator",
        "rain_anomaly_7d", "rain_anomaly_14d", "rain_anomaly_30d",
        "price_lag_7d", "price_lag_30d"
    ]
    avail_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[avail_cols].corr()
    corr_path = os.path.join(tables_dir, "eda_correlation_matrix.csv")
    corr_matrix.round(4).to_csv(corr_path)

    plt.figure(figsize=(12, 9), dpi=300)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.6, cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Weather Anomalies & Agricultural Price Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p8 = os.path.join(figures_dir, "08_correlation_matrix_annotated.png")
    plt.savefig(p8, dpi=300)
    plt.close()
    print(f"    Saved: {p8}")

    # -------------------------------------------------------------
    # 9. Clear Summary Comparison Chart for the 3 Commodities
    # -------------------------------------------------------------
    print("\n[9/9] Generating Clear Summary Comparison Chart...")
    summary_metrics = pd.DataFrame({
        "Median Price (₹/Q)": df.groupby("Commodity")["Modal_Price"].median(),
        "Mean Volatility": df.groupby("Commodity")["volatility_30d"].mean(),
        "95th Pct Volatility": df.groupby("Commodity")["volatility_30d"].quantile(0.95),
        "Observations (k)": df.groupby("Commodity")["Modal_Price"].count() / 1000
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

    # Subplot 1: Median Price
    sns.barplot(x=summary_metrics.index, y=summary_metrics["Median Price (₹/Q)"],
                palette=commodity_colors, ax=axes[0, 0], edgecolor="black")
    axes[0, 0].set_title("A. Median Mandi Price (₹ / Quintal)", fontweight="bold")
    axes[0, 0].set_ylabel("Price (₹/Q)")
    for b in axes[0, 0].patches:
        h = b.get_height()
        axes[0, 0].text(b.get_x() + b.get_width()/2, h + 20, f"₹{h:.0f}", ha="center", fontweight="bold", fontsize=9)

    # Subplot 2: Mean Volatility
    sns.barplot(x=summary_metrics.index, y=summary_metrics["Mean Volatility"],
                palette=commodity_colors, ax=axes[0, 1], edgecolor="black")
    axes[0, 1].set_title("B. Average 30-Day Volatility (Std Dev)", fontweight="bold")
    axes[0, 1].set_ylabel("Volatility")
    for b in axes[0, 1].patches:
        h = b.get_height()
        axes[0, 1].text(b.get_x() + b.get_width()/2, h + 0.003, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    # Subplot 3: 95th Percentile Extreme Volatility
    sns.barplot(x=summary_metrics.index, y=summary_metrics["95th Pct Volatility"],
                palette=commodity_colors, ax=axes[1, 0], edgecolor="black")
    axes[1, 0].set_title("C. Tail Volatility Risk (95th Percentile)", fontweight="bold")
    axes[1, 0].set_ylabel("95th Pct Volatility")
    for b in axes[1, 0].patches:
        h = b.get_height()
        axes[1, 0].text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.4f}", ha="center", fontweight="bold", fontsize=9)

    # Subplot 4: Sample Size (Thousands of Records)
    sns.barplot(x=summary_metrics.index, y=summary_metrics["Observations (k)"],
                palette=commodity_colors, ax=axes[1, 1], edgecolor="black")
    axes[1, 1].set_title("D. Sample Size (Thousands of Observations)", fontweight="bold")
    axes[1, 1].set_ylabel("Observations (k)")
    for b in axes[1, 1].patches:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width()/2, h + 5, f"{h:.1f}k", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("Comparative Market & Risk Profile: Tomato vs. Onion vs. Potato", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p9 = os.path.join(figures_dir, "09_commodity_comprehensive_summary.png")
    plt.savefig(p9, dpi=300)
    plt.close()
    print(f"    Saved: {p9}")

    print("\n[SUCCESS] Phase 3 Enhanced EDA generated all 9 figures and 4 summary tables successfully!")

if __name__ == "__main__":
    run_enhanced_eda()
