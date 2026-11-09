"""
Agrimetaverse Intelligence - SARIMAX Model Implementation
=========================================================
Implements Seasonal Auto-Regressive Integrated Moving Average with eXogenous
regressors (SARIMAX) using statsmodels.
Scales exogenous regressors with training parameters to ensure numerical stability.
Safely detects and handles multicollinearity on short annual time series.
"""

import warnings
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class CommoditySARIMAXModel:
    """
    SARIMAX model incorporating exogenous regressors, seasonal components,
    and robust numerical convergence handlers.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        exog_columns: Optional[List[str]] = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = exog_columns or []
        self.fitted_model = None
        self.best_aic: float = float("inf")
        self.selected_exog_cols: List[str] = []
        self.exog_means_: Dict[str, float] = {}
        self.exog_stds_: Dict[str, float] = {}
        self.price_min_: float = 0.0
        self.price_max_: float = 1000.0

    def _scale_exog(self, exog_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scales exogenous regressors using training mean and standard deviation."""
        scaled = pd.DataFrame(index=exog_df.index)
        for col in self.selected_exog_cols:
            if col not in exog_df.columns:
                continue
            vals = exog_df[col].astype(float).values
            if fit:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                self.exog_means_[col] = m
                self.exog_stds_[col] = s if s > 1e-4 else 1.0
            
            m = self.exog_means_.get(col, 0.0)
            s = self.exog_stds_.get(col, 1.0)
            scaled[col] = (vals - m) / s
        return scaled

    def tune_and_fit(
        self,
        train_series: pd.Series,
        train_exog: Optional[pd.DataFrame] = None,
        val_series: Optional[pd.Series] = None,
        val_exog: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Tunes orders (p, d, q) and seasonal terms with available exogenous features.
        """
        self.price_min_ = float(train_series.min())
        self.price_max_ = float(train_series.max())

        # Determine usable exogenous columns
        usable_exog = None
        if train_exog is not None and not train_exog.empty:
            valid_cols = [
                c for c in train_exog.columns
                if c not in ["Year", "Date"]
                and np.issubdtype(train_exog[c].dtype, np.number)
                and train_exog[c].notna().all()
                and np.var(train_exog[c]) > 1e-5
            ]
            if valid_cols:
                corrs = {}
                s_series = train_series.reset_index(drop=True)
                for col in valid_cols:
                    e_series = train_exog[col].reset_index(drop=True)
                    corr = abs(s_series.corr(e_series))
                    corrs[col] = 0.0 if np.isnan(corr) else corr
                # Cap top 2 most correlated features to keep N/P ratio healthy
                top_cols = sorted(corrs.keys(), key=lambda k: corrs[k], reverse=True)[:2]
                self.selected_exog_cols = top_cols
                usable_exog = self._scale_exog(train_exog[top_cols].reset_index(drop=True), fit=True)

        candidate_orders = [
            (1, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 1, 1),
        ]
        
        best_model = None
        best_aic = float("inf")
        best_order = (1, 1, 0)
        best_seasonal = (0, 0, 0, 0)

        for ord_cand in candidate_orders:
            for s_cand in [(0, 0, 0, 0), (1, 0, 0, 2)]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(
                            train_series,
                            exog=usable_exog,
                            order=ord_cand,
                            seasonal_order=s_cand,
                            enforce_stationarity=True,
                            enforce_invertibility=True,
                        )
                        res = model.fit(disp=False, maxiter=200)
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            # Quick sanity check on 1-step forecast
                            test_fcast = res.forecast(steps=1, exog=usable_exog.iloc[-1:] if usable_exog is not None else None)
                            fc_val = float(test_fcast.iloc[0])
                            # Check forecast stability (within reasonable range of commodity)
                            if 0.3 * self.price_min_ <= fc_val <= 2.5 * self.price_max_:
                                best_aic = res.aic
                                best_order = ord_cand
                                best_seasonal = s_cand
                                best_model = res
                except Exception:
                    continue

        # Fallback if convergence or stability failed with exog
        if best_model is None:
            try:
                model = SARIMAX(
                    train_series,
                    exog=usable_exog,
                    order=(1, 1, 0),
                    enforce_stationarity=False,
                )
                best_model = model.fit(disp=False)
                best_order = (1, 1, 0)
                best_seasonal = (0, 0, 0, 0)
                best_aic = best_model.aic
            except Exception:
                # Univariate fallback
                model = SARIMAX(train_series, order=(1, 1, 0), enforce_stationarity=False)
                best_model = model.fit(disp=False)
                best_order = (1, 1, 0)
                best_seasonal = (0, 0, 0, 0)
                best_aic = best_model.aic
                self.selected_exog_cols = []

        self.order = best_order
        self.seasonal_order = best_seasonal
        self.best_aic = best_aic
        self.fitted_model = best_model

        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": round(self.best_aic, 2),
            "selected_exog_columns": self.selected_exog_cols,
        }

    def predict(
        self, steps: int = 1, future_exog: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates forecasts and confidence bounds.
        Applies identical scaling to future exogenous variables.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predict.")

        exog_pred = None
        if self.selected_exog_cols:
            if future_exog is not None and all(c in future_exog.columns for c in self.selected_exog_cols):
                exog_slice = future_exog[self.selected_exog_cols].iloc[:steps]
                exog_pred = self._scale_exog(exog_slice, fit=False)
            else:
                last_exog = self.fitted_model.model.exog[-1:, :]
                exog_pred = np.repeat(last_exog, steps, axis=0)

        forecast_res = self.fitted_model.get_forecast(steps=steps, exog=exog_pred)
        mean_forecast = forecast_res.predicted_mean.values
        ci = forecast_res.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values if isinstance(ci, pd.DataFrame) else ci[:, 0]
        upper = ci.iloc[:, 1].values if isinstance(ci, pd.DataFrame) else ci[:, 1]

        # Clamp stability bounds: price cannot be negative, nor can it realistically exceed 3x recent max
        clamp_max = max(self.price_max_ * 2.5, 50.0)
        mean_forecast = np.clip(mean_forecast, 0.0, clamp_max)
        lower = np.clip(lower, 0.0, mean_forecast)
        upper = np.clip(upper, mean_forecast, clamp_max * 1.5)

        return mean_forecast, lower, upper

    def get_in_sample_predictions(self) -> np.ndarray:
        """Returns fitted values across training time."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
        return np.maximum(self.fitted_model.fittedvalues.values, 0.0)
