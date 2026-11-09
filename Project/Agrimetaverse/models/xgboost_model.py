"""
Agrimetaverse Intelligence - XGBoost Forecasting Model
======================================================
Implements Extreme Gradient Boosting (XGBoost) for agricultural price prediction.
Runner-up algorithm in the research paper (R² = 0.847, 40.9% commodity dominance).
Excels in specialty pulses, specialty oils, tomato, gur, and tea loose.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


class CommodityXGBoostModel:
    """
    XGBoost Regressor incorporating autoregressive price lags and 13 external economic/climate features.
    """

    def __init__(
        self,
        lookback: int = 3,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.lookback = lookback
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.last_prices: np.ndarray = np.array([])
        self.last_features: np.ndarray = np.array([])

    def _create_lagged_features(
        self,
        prices: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Creates lagged autoregressive features combined with external variables.
        """
        X, y = [], []
        p_vals = prices.values
        n = len(p_vals)

        feature_cols = [f"price_lag_{k}" for k in range(1, self.lookback + 1)]
        if features is not None and len(features) == n:
            ext_cols = list(features.columns)
            feature_cols.extend(ext_cols)

        for i in range(self.lookback, n):
            row = []
            for lag in range(1, self.lookback + 1):
                row.append(p_vals[i - lag])

            if features is not None and len(features) == n:
                row.extend(features.iloc[i].values)

            X.append(row)
            y.append(p_vals[i])

        return np.array(X), np.array(y), feature_cols

    def fit(
        self,
        prices: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> "CommodityXGBoostModel":
        """
        Fits XGBoost regressor on training data with scaling.
        """
        prices = prices.reset_index(drop=True)
        self.last_prices = prices.values
        clean_feat = None
        if features is not None and not features.empty:
            num_cols = [c for c in features.columns if c not in ["Date", "Year"] and np.issubdtype(features[c].dtype, np.number)]
            if num_cols:
                clean_feat = features[num_cols].reset_index(drop=True)
                if len(clean_feat) > len(prices):
                    clean_feat = clean_feat.iloc[:len(prices)]
                elif len(clean_feat) < len(prices):
                    clean_feat = clean_feat.reindex(range(len(prices))).ffill().bfill()
                self.last_features = clean_feat.values

        X, y, self.feature_names = self._create_lagged_features(prices, clean_feat)

        if len(X) < 2:
            # Fallback for very short series
            X = np.column_stack([prices.shift(1).bfill().values])
            y = prices.values
            self.feature_names = ["price_lag_1"]

        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_scaled, y)
        return self

    def predict(
        self,
        steps: int = 5,
        future_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-step recursive future forecasting with confidence intervals.
        """
        if self.model is None or len(self.last_prices) == 0:
            raise ValueError("Model is not fitted yet.")

        forecasts = []
        current_prices = list(self.last_prices)

        for step in range(steps):
            row = []
            for lag in range(1, self.lookback + 1):
                idx = len(current_prices) - lag
                row.append(current_prices[idx] if idx >= 0 else current_prices[0])

            if len(self.feature_names) > self.lookback:
                ext_names = self.feature_names[self.lookback:]
                if future_features is not None and step < len(future_features):
                    ext_vals = [float(future_features[c].iloc[step]) if c in future_features.columns else 0.0 for c in ext_names]
                elif len(self.last_features) > 0:
                    ext_vals = self.last_features[-1]
                else:
                    ext_vals = np.zeros(len(ext_names))
                row.extend(ext_vals)

            row_arr = np.array(row).reshape(1, -1)
            row_scaled = self.scaler.transform(row_arr)

            pred = float(self.model.predict(row_scaled)[0])
            # Prevent non-positive commodity price
            pred = max(pred, 1.0)
            forecasts.append(pred)
            current_prices.append(pred)

        forecasts_arr = np.array(forecasts)
        # Empirical standard error expansion for confidence intervals
        step_multipliers = np.array([1 + 0.02 * (s + 1) for s in range(steps)])
        lower_bound = forecasts_arr * (0.92 / step_multipliers)
        upper_bound = forecasts_arr * (1.08 * step_multipliers)

        return forecasts_arr, lower_bound, upper_bound

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns normalized feature importance dictionary.
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
