"""
Agrimetaverse Intelligence - Prophet Model Implementation
=========================================================
Implements Facebook Prophet forecasting adapted for annual commodity series.
Handles external regressors, uncertainty intervals, and documents
annual data frequency seasonality limitations.
"""

import logging
import warnings
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress cmdstanpy / prophet verbose output
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class CommodityProphetModel:
    """
    Prophet forecaster supporting external regressors and uncertainty intervals.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        interval_width: float = 0.95,
        regressor_cols: Optional[List[str]] = None,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        self.regressor_cols = regressor_cols or []
        self.fitted_model: Optional[Prophet] = None
        self.selected_regressors: List[str] = []
        self.seasonality_note: str = (
            "Annual series (1 obs/year): Intra-year seasonality (monthly/weekly) "
            "is mathematically unidentified and disabled to avoid rank deficiency. "
            "Model operates on trend + exogenous regressors."
        )

    def tune_and_fit(
        self,
        train_df: pd.DataFrame,
        exog_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Fits Prophet on training data.
        Supports either ['Date', 'Price'] or ['Year', 'Price'].
        """
        df_prophet = pd.DataFrame()
        if "Date" in train_df.columns:
            df_prophet["ds"] = pd.to_datetime(train_df["Date"])
            has_subannual = True
        else:
            df_prophet["ds"] = pd.to_datetime(train_df["Year"].astype(str) + "-12-31")
            has_subannual = False

        df_prophet["y"] = train_df["Price"].values

        # Select valid external regressors
        self.selected_regressors = []
        if exog_df is not None and not exog_df.empty:
            time_col = "Date" if "Date" in train_df.columns and "Date" in exog_df.columns else "Year"
            merged = pd.merge(train_df[[time_col]], exog_df, on=time_col, how="left")
            valid_feats = [
                c for c in exog_df.columns
                if c not in ["Date", "Year"]
                and np.issubdtype(exog_df[c].dtype, np.number)
                and merged[c].notna().all()
            ]
            self.selected_regressors = valid_feats[:3]
            for col in self.selected_regressors:
                df_prophet[col] = merged[col].values

        # Initialize Prophet (enable seasonality for daily series)
        model = Prophet(
            yearly_seasonality=has_subannual,
            weekly_seasonality=has_subannual,
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
        )

        for col in self.selected_regressors:
            model.add_regressor(col)

        # Fit
        model.fit(df_prophet)
        self.fitted_model = model

        return {
            "selected_regressors": self.selected_regressors,
            "seasonality_note": "Enabled weekly and yearly seasonality on daily data" if has_subannual else self.seasonality_note,
            "changepoint_prior_scale": self.changepoint_prior_scale,
        }

    def predict(
        self,
        future_years: Optional[List[int]] = None,
        future_exog: Optional[pd.DataFrame] = None,
        future_dates: Optional[List[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts future values along with uncertainty bounds.
        Supports future_dates (daily) or future_years (annual).
        Returns: (yhat, yhat_lower, yhat_upper)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predict.")

        if future_dates is not None:
            ds_series = pd.to_datetime(future_dates)
            future_df = pd.DataFrame({"ds": ds_series})
            time_col = "Date"
        elif future_years is not None:
            ds_series = pd.to_datetime([f"{y}-12-31" for y in future_years])
            future_df = pd.DataFrame({"ds": ds_series})
            time_col = "Year"
        else:
            raise ValueError("Must provide either future_dates or future_years.")

        # Add regressors if required
        for col in self.selected_regressors:
            if future_exog is not None and col in future_exog.columns:
                if time_col in future_exog.columns:
                    val_map = dict(zip(pd.to_datetime(future_exog[time_col]) if time_col == "Date" else future_exog[time_col], future_exog[col]))
                    keys = ds_series if time_col == "Date" else future_years
                    future_df[col] = [val_map.get(k, future_exog[col].iloc[-1]) for k in keys]
                else:
                    future_df[col] = float(future_exog[col].iloc[-1])
            else:
                future_df[col] = 0.0

        forecast = self.fitted_model.predict(future_df)
        yhat = np.maximum(forecast["yhat"].values, 0.0)
        lower = np.maximum(forecast["yhat_lower"].values, 0.0)
        upper = np.maximum(forecast["yhat_upper"].values, yhat)

        return yhat, lower, upper
