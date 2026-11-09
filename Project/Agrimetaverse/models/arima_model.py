"""
Agrimetaverse Intelligence - ARIMA Model Implementation
======================================================
Implements Auto-Regressive Integrated Moving Average (ARIMA) using statsmodels.
Performs exhaustive grid search over (p, d, q) for p in [0..3], d in [0..2], q in [0..3].
Selects best order via Akaike Information Criterion (AIC) with forecast stability checking.
"""

import warnings
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class CommodityARIMAModel:
    """
    ARIMA model with automated AIC grid search and safe convergence handling.
    """

    def __init__(
        self,
        p_range: range = range(0, 4),
        d_range: range = range(0, 3),
        q_range: range = range(0, 4),
    ):
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.best_order: Tuple[int, int, int] = (1, 1, 0)
        self.best_aic: float = float("inf")
        self.fitted_model = None
        self.training_series: Optional[pd.Series] = None
        self.price_min_: float = 0.0
        self.price_max_: float = 1000.0

    def tune_and_fit(
        self,
        train_series: pd.Series,
        val_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Conducts grid search over (p, d, q) space, selecting optimal order via AIC.
        Ensures selected configuration maintains numerical stability on small samples.
        """
        self.training_series = train_series.copy()
        self.price_min_ = float(train_series.min())
        self.price_max_ = float(train_series.max())

        best_aic = float("inf")
        best_order = (1, 1, 0)
        best_model = None

        search_results = []

        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(train_series, order=(p, d, q))
                            res = model.fit()
                            
                            if np.isfinite(res.aic):
                                # Forecast stability check
                                fcast = float(res.forecast(steps=1).iloc[0])
                                if 0.2 * self.price_min_ <= fcast <= 2.5 * self.price_max_:
                                    search_results.append({
                                        "order": (p, d, q),
                                        "aic": res.aic,
                                        "bic": res.bic,
                                    })
                                    if res.aic < best_aic:
                                        best_aic = res.aic
                                        best_order = (p, d, q)
                                        best_model = res
                    except Exception:
                        continue

        # Fallback if all failed
        if best_model is None:
            for fallback_order in [(1, 1, 0), (0, 1, 0), (1, 0, 0)]:
                try:
                    model = ARIMA(train_series, order=fallback_order)
                    best_model = model.fit()
                    best_order = fallback_order
                    best_aic = best_model.aic
                    break
                except Exception:
                    continue

        self.best_order = best_order
        self.best_aic = best_aic
        self.fitted_model = best_model

        return {
            "best_order": self.best_order,
            "best_aic": round(self.best_aic, 2),
            "search_count": len(search_results),
        }

    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecasts future values along with 95% confidence intervals.
        Returns: (point_forecasts, lower_bounds, upper_bounds)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before calling predict.")

        forecast_res = self.fitted_model.get_forecast(steps=steps)
        mean_forecast = forecast_res.predicted_mean.values
        ci = forecast_res.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values if isinstance(ci, pd.DataFrame) else ci[:, 0]
        upper = ci.iloc[:, 1].values if isinstance(ci, pd.DataFrame) else ci[:, 1]

        clamp_max = max(self.price_max_ * 2.5, 50.0)
        mean_forecast = np.clip(mean_forecast, 0.0, clamp_max)
        lower = np.clip(lower, 0.0, mean_forecast)
        upper = np.clip(upper, mean_forecast, clamp_max * 1.5)

        return mean_forecast, lower, upper

    def get_in_sample_predictions(self) -> np.ndarray:
        """Returns fitted values over training range."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
        return np.maximum(self.fitted_model.fittedvalues.values, 0.0)
