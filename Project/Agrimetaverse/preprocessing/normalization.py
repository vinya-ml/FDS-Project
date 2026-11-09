"""
Agrimetaverse Intelligence - Normalization Module
=================================================
Implements Min-Max Normalization:
    X_scaled = (X - X_min) / (X_max - X_min)
Strictly fits parameters on training period data (2014-2021) only,
preventing data leakage into validation and testing partitions.
Retains original price values for all rupee-based user outputs.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class CommodityMinMaxScaler:
    """
    Min-Max Scaler fitted on training data with inverse transform support.
    Can operate per commodity or across a feature column.
    """

    def __init__(self, feature_range=(0.0, 1.0), epsilon: float = 1e-8):
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_val_: Optional[float] = None
        self.max_val_: Optional[float] = None
        self.fitted_: bool = False

    def fit(self, values: pd.Series | np.ndarray) -> "CommodityMinMaxScaler":
        clean_vals = np.asarray(values, dtype=float)
        clean_vals = clean_vals[~np.isnan(clean_vals)]
        if len(clean_vals) == 0:
            self.min_val_ = 0.0
            self.max_val_ = 1.0
        else:
            self.min_val_ = float(np.min(clean_vals))
            self.max_val_ = float(np.max(clean_vals))

        # Handle zero variance edge-case
        if abs(self.max_val_ - self.min_val_) < self.epsilon:
            self.max_val_ = self.min_val_ + 1.0

        self.fitted_ = True
        return self

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before transforming data.")
        arr = np.asarray(values, dtype=float)
        scaled = (arr - self.min_val_) / (self.max_val_ - self.min_val_)
        a, b = self.feature_range
        return scaled * (b - a) + a

    def fit_transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse_transform(self, scaled_values: pd.Series | np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before inverse transforming.")
        arr = np.asarray(scaled_values, dtype=float)
        a, b = self.feature_range
        norm = (arr - a) / (b - a)
        return norm * (self.max_val_ - self.min_val_) + self.min_val_


class DatasetNormalizer:
    """
    Orchestrates per-commodity price scaling and external feature scaling.
    Fits strictly on train_years (2014-2021).
    """

    def __init__(self, train_years: list[int] = list(range(2014, 2022))):
        self.train_years = train_years
        self.commodity_scalers: Dict[str, CommodityMinMaxScaler] = {}
        self.feature_scalers: Dict[str, CommodityMinMaxScaler] = {}

    def fit(self, df_prices: pd.DataFrame, df_features: Optional[pd.DataFrame] = None) -> "DatasetNormalizer":
        """Fit normalization parameters strictly on training partition."""
        if "Date" in df_prices.columns:
            for commodity, group in df_prices.groupby("Commodity"):
                group_sorted = group.sort_values("Date")
                n_train = max(2, int(len(group_sorted) * 0.70))
                train_p = group_sorted.iloc[:n_train]
                scaler = CommodityMinMaxScaler()
                scaler.fit(train_p["Price"].values)
                self.commodity_scalers[commodity] = scaler
        else:
            train_prices = df_prices[df_prices["Year"].isin(self.train_years)]
            for commodity, group in train_prices.groupby("Commodity"):
                scaler = CommodityMinMaxScaler()
                scaler.fit(group["Price"].values)
                self.commodity_scalers[commodity] = scaler

        # Fit external features if provided
        if df_features is not None and not df_features.empty:
            if "Date" in df_features.columns:
                n_train = max(2, int(len(df_features) * 0.70))
                train_features = df_features.sort_values("Date").iloc[:n_train]
                feat_cols = [c for c in train_features.columns if c not in ["Date", "Year"]]
            else:
                train_features = df_features[df_features["Year"].isin(self.train_years)]
                feat_cols = [c for c in train_features.columns if c != "Year"]

            for col in feat_cols:
                scaler = CommodityMinMaxScaler()
                scaler.fit(train_features[col].values)
                self.feature_scalers[col] = scaler

        return self

    def transform_prices(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """Applies training-fitted scaling without modifying original 'Price'."""
        df_out = df_prices.copy()
        scaled_prices = []
        for idx, row in df_out.iterrows():
            commodity = row["Commodity"]
            val = row["Price"]
            if commodity in self.commodity_scalers:
                scaler = self.commodity_scalers[commodity]
                scaled = float(scaler.transform([val])[0])
            else:
                scaled = val
            scaled_prices.append(scaled)
        df_out["Price_Scaled"] = scaled_prices
        return df_out

    def inverse_transform_price(self, commodity: str, scaled_price: float | np.ndarray) -> float | np.ndarray:
        """Restores scaled price predictions back to authentic Indian Rupees (₹)."""
        if commodity in self.commodity_scalers:
            return self.commodity_scalers[commodity].inverse_transform(scaled_price)
        return scaled_price

    def get_scaler_parameters(self) -> pd.DataFrame:
        """Returns diagnostic table of fitted Min and Max bounds per commodity."""
        records = []
        for comm, scaler in self.commodity_scalers.items():
            records.append({
                "Commodity": comm,
                "Train_Min_Price (₹)": round(scaler.min_val_, 2),
                "Train_Max_Price (₹)": round(scaler.max_val_, 2),
                "Train_Range (₹)": round(scaler.max_val_ - scaler.min_val_, 2),
            })
        return pd.DataFrame(records)
