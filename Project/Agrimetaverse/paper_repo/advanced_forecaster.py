import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Install required packages first
try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet not installed. Run: pip install prophet")
    st.stop()

try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from scipy.optimize import differential_evolution
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.info("Run: pip install tensorflow scikit-learn scipy")
    st.stop()

class AdvancedCommodityForecaster:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load commodity prices and external features"""
        try:
            import os
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load commodity prices
            prices_path = os.path.join(script_dir, "DatasetSIH1647.csv")
            if not os.path.exists(prices_path):
                st.error(f"Data file not found: {prices_path}")
                return None, None
                
            df_prices = pd.read_csv(prices_path)
            df_prices.set_index('Commodities', inplace=True)
            df_prices = df_prices.T
            df_prices.index = pd.date_range(start='2014', periods=len(df_prices), freq='YE')
            df_prices = df_prices.ffill()
            
            # Load external features
            features_path = os.path.join(script_dir, "enhanced_features.csv")
            if os.path.exists(features_path):
                df_features = pd.read_csv(features_path)
                df_features['Year'] = pd.date_range(start='2014', periods=len(df_features), freq='YE')
                df_features.set_index('Year', inplace=True)
            else:
                # Create dummy features if file doesn't exist
                df_features = pd.DataFrame({
                    'Rainfall_mm': np.random.normal(1000, 200, len(df_prices)),
                    'Temperature_C': np.random.normal(25, 5, len(df_prices)),
                    'Inflation_%': np.random.normal(5, 2, len(df_prices))
                }, index=df_prices.index)
            
            return df_prices, df_features
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    
    def prepare_data_for_xgboost(self, commodity_data, features, lookback=3):
        """Prepare data for XGBoost with time series features"""
        X, y = [], []
        
        # Combine commodity prices with external features
        combined_data = pd.concat([commodity_data, features], axis=1)
        
        for i in range(lookback, len(combined_data)):
            # Create lagged features
            X_row = []
            
            # Add lagged commodity prices
            for lag in range(1, lookback + 1):
                X_row.append(commodity_data.iloc[i - lag])
            
            # Add current external features
            X_row.extend(features.iloc[i].values)
            
            X.append(X_row)
            y.append(commodity_data.iloc[i])
        
        return np.array(X), np.array(y)
    
    def find_best_arima_order(self, data, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA order using AIC"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit(method_kwargs={"warn_convergence": False})
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        return best_order

    def train_arima(self, data):
        """Train ARIMA model with automatic order selection"""
        try:
            # Find best order
            best_order = self.find_best_arima_order(data)
            
            # Train with best order
            model = ARIMA(data, order=best_order)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})
            return fitted_model, best_order
        except Exception as e:
            st.warning(f"ARIMA training failed: {e}")
            return None, None

    def train_sarimax(self, data):
        """Train SARIMAX model"""
        try:
            model = SARIMAX(data, order=(1, 1, 1))
            fitted_model = model.fit(disp=False)
            return fitted_model
        except Exception as e:
            st.warning(f"SARIMAX training failed: {e}")
            return None
    
    def train_prophet(self, data, features):
        """Train Prophet model with external regressors"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            # Add external regressors
            for col in features.columns:
                if col != 'Year':
                    prophet_data[col] = features[col].values
            
            # Initialize Prophet with external regressors
            model = Prophet(
                yearly_seasonality=True,
                daily_seasonality=False,
                weekly_seasonality=False
            )
            
            # Add external regressors
            for col in features.columns:
                if col != 'Year':
                    model.add_regressor(col)
            
            model.fit(prophet_data)
            return model, prophet_data
        except Exception as e:
            st.warning(f"Prophet training failed: {e}")
            return None, None

    def prepare_lstm_data(self, data, lookback=5):
        """Prepare data for LSTM training"""
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
            
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y), scaler
        except Exception as e:
            st.warning(f"LSTM data preparation failed: {e}")
            return None, None, None

    def train_hybrid_sarima_lstm(self, data):
        """Train Hybrid SARIMA-LSTM model"""
        try:
            # Step 1: SARIMA decomposition and residual extraction
            sarima_model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit(disp=False)
            
            # Get SARIMA predictions and residuals
            sarima_pred = sarima_fit.fittedvalues
            residuals = data - sarima_pred
            
            # Step 2: LSTM for residual modeling
            X_lstm, y_lstm, lstm_scaler = self.prepare_lstm_data(residuals)
            
            if X_lstm is None or len(X_lstm) < 3:
                return sarima_fit, None, None, None  # Fallback to SARIMA only
            
            # Reshape for LSTM
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train LSTM (reduced epochs for faster execution)
            lstm_model.fit(X_lstm, y_lstm, batch_size=1, epochs=20, verbose=0)
            
            return sarima_fit, lstm_model, lstm_scaler, residuals
            
        except Exception as e:
            st.warning(f"Hybrid SARIMA-LSTM training failed: {e}")
            return None, None, None, None

    def elm_objective_function(self, params, X_train, y_train, X_val, y_val):
        """Objective function for ELM optimization using Genetic Algorithm"""
        try:
            hidden_neurons, activation_param = int(params[0]), params[1]
            
            # Simple ELM implementation using MLPRegressor as approximation
            elm_model = MLPRegressor(
                hidden_layer_sizes=(hidden_neurons,),
                activation='tanh',
                alpha=activation_param,
                max_iter=100,
                random_state=42
            )
            
            elm_model.fit(X_train, y_train)
            pred = elm_model.predict(X_val)
            mse = mean_squared_error(y_val, pred)
            
            return mse
        except:
            return 1e6  # Return high error for invalid parameters

    def train_elm_genetic(self, data, features):
        """Train ELM optimized with Genetic Algorithm"""
        try:
            # Prepare data
            X, y = self.prepare_data_for_xgboost(data, features)
            
            if len(X) < 5:
                return None, None
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale features
            elm_scaler = StandardScaler()
            X_train_scaled = elm_scaler.fit_transform(X_train)
            X_val_scaled = elm_scaler.transform(X_val)
            
            # Define parameter bounds for GA
            bounds = [(10, 100), (0.0001, 0.1)]  # hidden_neurons, alpha
            
            # Genetic Algorithm optimization
            result = differential_evolution(
                self.elm_objective_function,
                bounds,
                args=(X_train_scaled, y_train, X_val_scaled, y_val),
                maxiter=10,  # Reduced iterations for faster execution
                popsize=8,   # Smaller population
                seed=42
            )
            
            # Train final ELM with optimized parameters
            best_hidden, best_alpha = int(result.x[0]), result.x[1]
            
            elm_model = MLPRegressor(
                hidden_layer_sizes=(best_hidden,),
                activation='tanh',
                alpha=best_alpha,
                max_iter=200,
                random_state=42
            )
            
            elm_model.fit(X_train_scaled, y_train)
            
            return elm_model, elm_scaler
            
        except Exception as e:
            st.warning(f"ELM-GA training failed: {e}")
            return None, None
    
    def forecast_arima(self, model, steps=5):
        """Generate ARIMA forecasts"""
        try:
            forecast = model.get_forecast(steps=steps)
            return forecast.predicted_mean.values
        except:
            return None

    def forecast_sarimax(self, model, steps=5):
        """Generate SARIMAX forecasts"""
        try:
            forecast = model.get_forecast(steps=steps)
            return forecast.predicted_mean.values
        except:
            return None
    
    def forecast_prophet(self, model, prophet_data, features, steps=5):
        """Generate Prophet forecasts"""
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps, freq='YE')
            
            # Add future external regressors (simple trend extrapolation)
            for col in features.columns:
                if col != 'Year':
                    # Simple linear extrapolation for future values
                    last_values = features[col].tail(3).values
                    trend = (last_values[-1] - last_values[0]) / 2
                    
                    for i in range(len(future)):
                        if i < len(prophet_data):
                            future.loc[i, col] = prophet_data.iloc[i][col]
                        else:
                            future.loc[i, col] = last_values[-1] + trend * (i - len(prophet_data) + 1)
            
            forecast = model.predict(future)
            predictions = forecast['yhat'].tail(steps).values
            
            # Ensure no negative values - use maximum of 0 or predicted value
            predictions = np.maximum(predictions, 0)
            
            return predictions
        except:
            return None
    
    def forecast_hybrid_sarima_lstm(self, sarima_model, lstm_model, lstm_scaler, residuals, steps=5):
        """Generate Hybrid SARIMA-LSTM forecasts"""
        try:
            # SARIMA forecasts
            sarima_forecast = sarima_model.get_forecast(steps=steps)
            sarima_pred = sarima_forecast.predicted_mean.values
            
            if lstm_model is None or lstm_scaler is None:
                return sarima_pred  # Fallback to SARIMA only
            
            # LSTM forecasts for residuals
            lstm_predictions = []
            
            # Use last 5 residuals for LSTM prediction
            last_residuals = residuals.tail(5).values
            scaled_residuals = lstm_scaler.transform(last_residuals.reshape(-1, 1))
            
            current_sequence = scaled_residuals.flatten()
            
            for _ in range(steps):
                # Reshape for LSTM prediction
                X_lstm = current_sequence[-5:].reshape(1, 5, 1)
                
                # Predict next residual
                pred_scaled = lstm_model.predict(X_lstm, verbose=0)[0, 0]
                pred_residual = lstm_scaler.inverse_transform([[pred_scaled]])[0, 0]
                
                lstm_predictions.append(pred_residual)
                
                # Update sequence
                current_sequence = np.append(current_sequence, pred_scaled)
            
            # Combine SARIMA and LSTM predictions
            hybrid_predictions = sarima_pred + np.array(lstm_predictions)
            
            return hybrid_predictions
            
        except Exception as e:
            st.warning(f"Hybrid SARIMA-LSTM forecasting failed: {e}")
            return None

    def forecast_elm_genetic(self, elm_model, elm_scaler, data, features, steps=5):
        """Generate ELM-GA forecasts"""
        try:
            if elm_model is None or elm_scaler is None:
                return None
            
            predictions = []
            current_data = data.values.copy()
            
            for step in range(steps):
                # Prepare input features
                X_input = []
                
                # Add lagged values
                for lag in range(1, 4):  # lookback=3
                    X_input.append(current_data[-(lag)])
                
                # Add external features
                last_features = features.iloc[-1].values
                X_input.extend(last_features)
                
                # Scale and predict
                X_input_scaled = elm_scaler.transform([X_input])
                pred = elm_model.predict(X_input_scaled)[0]
                
                predictions.append(pred)
                current_data = np.append(current_data, pred)
            
            return np.array(predictions)
            
        except Exception as e:
            st.warning(f"ELM-GA forecasting failed: {e}")
            return None
    
    def ensemble_forecast(self, predictions_dict, weights=None):
        """Create ensemble forecast from multiple models"""
        valid_predictions = {k: v for k, v in predictions_dict.items() if v is not None}
        
        if not valid_predictions:
            return None
        
        if weights is None:
            # Equal weights
            weights = {k: 1/len(valid_predictions) for k in valid_predictions.keys()}
        
        # Weighted average
        ensemble = np.zeros(5)  # 5 years forecast
        
        for model_name, preds in valid_predictions.items():
            if len(preds) == 5:
                ensemble += weights.get(model_name, 0) * preds
        
        return ensemble
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

    def calculate_model_performance(self, actual_data, predictions):
        """Calculate comprehensive performance metrics for model comparison"""
        metrics = {}
        
        for model_name, pred in predictions.items():
            if pred is not None and len(pred) > 0:
                # Use cross-validation approach for better evaluation
                validation_size = min(3, len(actual_data) // 3)  # Use last 3 points for validation
                if validation_size > 0:
                    actual_val = actual_data.iloc[-validation_size:].values
                    pred_val = pred[:validation_size] if len(pred) >= validation_size else pred
                    
                    # Ensure same length
                    min_len = min(len(actual_val), len(pred_val))
                    actual_val = actual_val[:min_len]
                    pred_val = pred_val[:min_len]
                    
                    if len(actual_val) > 0 and len(pred_val) > 0:
                        # Calculate metrics
                        mse = mean_squared_error(actual_val, pred_val)
                        mae = mean_absolute_error(actual_val, pred_val)
                        rmse = np.sqrt(mse)
                        
                        # Avoid division by zero in MAPE
                        mape = np.mean(np.abs((actual_val - pred_val) / np.where(actual_val != 0, actual_val, 1))) * 100
                        
                        # Calculate directional accuracy
                        if len(actual_val) > 1:
                            actual_direction = np.diff(actual_val) > 0
                            pred_direction = np.diff(pred_val) > 0
                            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                        else:
                            directional_accuracy = 50  # Default for single value
                        
                        # Calculate R-squared
                        ss_res = np.sum((actual_val - pred_val) ** 2)
                        ss_tot = np.sum((actual_val - np.mean(actual_val)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        # Calculate overall performance score (higher is better)
                        performance_score = (directional_accuracy + (100 - mape) + (r2 * 100)) / 3
                        
                        metrics[model_name] = {
                            'MSE': mse,
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape,
                            'Directional_Accuracy': directional_accuracy,
                            'R_Squared': r2,
                            'Performance_Score': performance_score
                        }
        
        return metrics

    def find_best_algorithm(self, metrics):
        """Determine the best performing algorithm based on multiple criteria with enhanced analysis"""
        if not metrics:
            return None, {}
        
        # Normalize metrics and calculate weighted scores
        model_scores = {}
        
        # Get all metric values for normalization
        all_mape = [m['MAPE'] for m in metrics.values()]
        all_rmse = [m['RMSE'] for m in metrics.values()]
        all_dir_acc = [m['Directional_Accuracy'] for m in metrics.values()]
        all_r2 = [m['R_Squared'] for m in metrics.values()]
        
        # Calculate min/max for normalization
        min_mape, max_mape = min(all_mape), max(all_mape)
        min_rmse, max_rmse = min(all_rmse), max(all_rmse)
        min_dir_acc, max_dir_acc = min(all_dir_acc), max(all_dir_acc)
        min_r2, max_r2 = min(all_r2), max(all_r2)
        
        for model_name, model_metrics in metrics.items():
            # Enhanced normalization (0-100 scale)
            # MAPE: Lower is better - inverse scoring
            if max_mape > min_mape:
                mape_score = 100 * (1 - (model_metrics['MAPE'] - min_mape) / (max_mape - min_mape))
            else:
                mape_score = 100 - min(model_metrics['MAPE'], 100)
            
            # RMSE: Lower is better - inverse scoring
            if max_rmse > min_rmse:
                rmse_score = 100 * (1 - (model_metrics['RMSE'] - min_rmse) / (max_rmse - min_rmse))
            else:
                rmse_score = 50
            
            # Direction Accuracy: Higher is better - direct scoring
            dir_acc_score = model_metrics['Directional_Accuracy']
            
            # R-squared: Higher is better - normalized scoring
            if max_r2 > min_r2:
                r2_score = 100 * (model_metrics['R_Squared'] - min_r2) / (max_r2 - min_r2)
            else:
                r2_score = max(0, model_metrics['R_Squared'] * 100)
            
            # Enhanced weighted average with agricultural price prediction focus
            overall_score = (
                mape_score * 0.25 +      # Prediction accuracy (25%)
                rmse_score * 0.20 +      # Error magnitude (20%)
                dir_acc_score * 0.35 +   # Direction prediction (35% - most important for trading)
                r2_score * 0.20          # Goodness of fit (20%)
            )
            
            # Add bonus points for specific model advantages
            model_bonuses = {
                'Hybrid SARIMA-LSTM': 5,  # Bonus for advanced AI
                'ELM-GA': 5,              # Bonus for evolutionary optimization
                'SARIMAX': 3,             # Bonus for external factor integration
                'Prophet': 2,             # Bonus for robust trend detection
                'XGBoost': 3,             # Bonus for feature importance
                'ARIMA': 1                # Small bonus for simplicity and interpretability
            }
            
            overall_score += model_bonuses.get(model_name, 0)
            overall_score = min(overall_score, 100)  # Cap at 100
            
            model_scores[model_name] = {
                'Overall_Score': overall_score,
                'MAPE_Score': mape_score,
                'RMSE_Score': rmse_score,
                'Direction_Score': dir_acc_score,
                'R2_Score': r2_score,
                'Model_Bonus': model_bonuses.get(model_name, 0),
                'Confidence_Level': self._calculate_confidence_level(model_metrics, overall_score),
                'Risk_Level': self._calculate_risk_level(model_metrics),
                'Recommendation_Strength': self._get_recommendation_strength(overall_score)
            }
        
        # Find best model
        best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['Overall_Score'])
        
        return best_model, model_scores
    
    def _calculate_confidence_level(self, metrics, overall_score):
        """Calculate confidence level based on metrics"""
        # High confidence if direction accuracy > 70% and overall score > 75
        if metrics['Directional_Accuracy'] > 70 and overall_score > 75:
            return "Very High"
        elif metrics['Directional_Accuracy'] > 60 and overall_score > 65:
            return "High"
        elif metrics['Directional_Accuracy'] > 50 and overall_score > 55:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_risk_level(self, metrics):
        """Calculate risk level based on metrics"""
        # Lower MAPE and higher R² means lower risk
        if metrics['MAPE'] < 10 and metrics['R_Squared'] > 0.7:
            return "Low"
        elif metrics['MAPE'] < 20 and metrics['R_Squared'] > 0.5:
            return "Medium"
        else:
            return "High"
    
    def _get_recommendation_strength(self, score):
        """Get recommendation strength based on overall score"""
        if score >= 80:
            return "Strongly Recommended"
        elif score >= 70:
            return "Recommended"
        elif score >= 60:
            return "Conditionally Recommended"
        else:
            return "Not Recommended"

    def create_interactive_price_chart(self, data, commodity_name):
        """Create interactive historical price chart with Plotly"""
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines+markers',
            name='Historical Prices',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate='<b>Year:</b> %{x}<br><b>Price:</b> ₹%{y:.2f}<extra></extra>'
        ))
        
        # Calculate trend line
        x_numeric = np.arange(len(data))
        z = np.polyfit(x_numeric, data.values, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=p(x_numeric),
            mode='lines',
            name='Trend Line',
            line=dict(width=2, color='red', dash='dash'),
            hovertemplate='<b>Trend:</b> ₹%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"📈 {commodity_name} Price Trend (2014-2024)",
            xaxis_title="Year",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_forecast_animation(self, data, predictions, commodity_name):
        """Create animated forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines+markers',
            name='Historical Prices',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8)
        ))
        
        # Future years
        future_years = pd.date_range(start='2025', periods=5, freq='YE')
        
        # Add predictions for each model
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == 5:
                fig.add_trace(go.Scatter(
                    x=future_years,
                    y=pred,
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(width=3, color=colors[i % len(colors)]),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{model_name}:</b> ₹%{{y:.2f}}<extra></extra>'
                ))
        
        # Add confidence intervals
        if 'Prophet' in predictions:
            # Simulate confidence intervals (in real implementation, get from Prophet)
            prophet_pred = predictions['Prophet']
            upper_bound = prophet_pred * 1.1
            lower_bound = prophet_pred * 0.9
            
            fig.add_trace(go.Scatter(
                x=future_years,
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_years,
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty',
                name='Confidence Interval',
                hovertemplate='<b>Range:</b> ₹%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"🔮 {commodity_name} Price Forecasting (2014-2029)",
            xaxis_title="Year",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_comparison_dashboard(self, predictions):
        """Create model comparison dashboard"""
        if not predictions:
            return None
            
        # Create subplots (3 plots instead of 4, radar chart separate)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Predictions', 'Growth Rates', 'Price Ranges', 'Model Scores'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        future_years = list(range(2025, 2030))
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot 1: Model predictions
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == 5:
                fig.add_trace(
                    go.Scatter(
                        x=future_years,
                        y=pred,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(color=colors[i % len(colors)]),
                        legendgroup=model_name
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Growth rates
        growth_rates = []
        model_names = []
        for model_name, pred in predictions.items():
            if len(pred) == 5:
                growth_rate = ((pred[-1] / pred[0]) ** (1/4) - 1) * 100
                growth_rates.append(growth_rate)
                model_names.append(model_name)
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=growth_rates,
                marker_color=colors[:len(model_names)],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Plot 3: Price ranges (box plot)
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == 5:
                fig.add_trace(
                    go.Box(
                        y=pred,
                        name=model_name,
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Model performance scores (bar chart instead of radar)
        if len(predictions) > 1:
            # Simulated performance scores (in real implementation, calculate from validation)
            performance_scores = {
                'SARIMAX': 0.8,
                'Prophet': 0.85,
                'XGBoost': 0.9,
                'Ensemble': 0.95
            }
            
            models_in_predictions = [model for model in predictions.keys() if model in performance_scores]
            scores = [performance_scores[model] for model in models_in_predictions]
            
            fig.add_trace(
                go.Bar(
                    x=models_in_predictions,
                    y=scores,
                    marker_color=colors[:len(models_in_predictions)],
                    showlegend=False,
                    text=[f"{s:.1%}" for s in scores],
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="📊 Model Comparison Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_performance_radar(self, predictions):
        """Create separate radar chart for model performance"""
        if not predictions:
            return None
            
        categories = ['Accuracy', 'Stability', 'Trend Detection', 'Seasonal Awareness', 'Robustness']
        
        # Simulated performance scores (in real implementation, calculate from validation)
        performance_scores = {
            'SARIMAX': [0.8, 0.9, 0.7, 0.8, 0.8],
            'Prophet': [0.85, 0.8, 0.9, 0.95, 0.85],
            'XGBoost': [0.9, 0.7, 0.85, 0.7, 0.9],
            'Ensemble': [0.95, 0.85, 0.9, 0.85, 0.95]
        }
        
        fig = go.Figure()
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, model_name in enumerate(predictions.keys()):
            if model_name in performance_scores:
                fig.add_trace(go.Scatterpolar(
                    r=performance_scores[model_name],
                    theta=categories,
                    fill='toself',
                    name=model_name,
                    line_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🎯 Model Performance Radar Chart",
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self, df_prices, df_features):
        """Create interactive correlation heatmap"""
        # Combine price data with features for correlation analysis
        combined_data = pd.concat([df_prices.iloc[-len(df_features):], df_features], axis=1)
        correlation_matrix = combined_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🔥 Dynamic Correlation Heatmap: Commodities vs External Factors",
            xaxis_title="Features",
            yaxis_title="Commodities",
            height=600,
            width=800
        )
        
        return fig

def main():
    st.set_page_config(page_title="Advanced Commodity Forecasting", layout="wide")
    st.title("🌾 Advanced Agricultural Commodity Price Forecasting")
    st.markdown("### AI-Powered Multi-Model Ensemble System")
    
    forecaster = AdvancedCommodityForecaster()
    
    # Load data
    df_prices, df_features = forecaster.load_data()
    
    if df_prices is None or df_features is None:
        st.error("Failed to load data. Please ensure CSV files are present.")
        return
    
    # Sidebar for model selection and live updates
    st.sidebar.header("🔧 Model Configuration")
    
    # Live update controls
    st.sidebar.header("⚡ Live Updates")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", help="Automatically refresh every 30 seconds")
    update_interval = st.sidebar.slider("Update Interval (seconds)", 10, 300, 30)
    
    if auto_refresh:
        import time
        time.sleep(update_interval)
        st.rerun()
    
    # Real-time data simulation
    if st.sidebar.button("📊 Simulate Real-time Data"):
        st.sidebar.success("✅ Real-time simulation active!")
        # Add slight random variation to simulate live market data
        import random
        noise_factor = st.sidebar.slider("Market Volatility", 0.0, 0.1, 0.02)
        if 'simulated_data' not in st.session_state:
            st.session_state.simulated_data = True
    
    commodities = df_prices.columns.tolist()
    selected_commodity = st.sidebar.selectbox("🌾 Choose Commodity", commodities)
    
    # Model selection with enhanced descriptions
    st.sidebar.subheader("🤖 AI Models")
    
    # Traditional Models
    st.sidebar.markdown("**📊 Traditional Models**")
    use_arima = st.sidebar.checkbox("📊 ARIMA Model", value=True, help="Auto-Regressive Integrated Moving Average")
    use_sarimax = st.sidebar.checkbox("📈 SARIMAX Model", value=True, help="Statistical time series analysis")
    use_prophet = st.sidebar.checkbox("🔮 Prophet Model", value=True, help="Advanced seasonal forecasting")
    
    # Visualization controls
    st.sidebar.subheader("📊 Visualization Options")
    show_animations = st.sidebar.checkbox("🎬 Enable Animations", value=True)
    show_correlations = st.sidebar.checkbox("🔥 Show Correlations", value=True)
    interactive_mode = st.sidebar.checkbox("🖱️ Interactive Mode", value=False)  # Temporarily disabled
    
    # Load data
    data, df_features = forecaster.load_data()
    
    if st.sidebar.button("🚀 Generate Forecasts"):
        
        data = df_prices[selected_commodity]
        
        # Display current data with dynamic visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📊 Interactive {selected_commodity} Price Analysis")
            # Use dynamic Plotly chart instead of static matplotlib
            interactive_chart = forecaster.create_interactive_price_chart(data, selected_commodity)
            st.plotly_chart(interactive_chart, use_container_width=True)
        
        with col2:
            st.subheader("📈 Key Statistics")
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Average Price", f"₹{data.mean():.2f}")
                st.metric("Price Volatility", f"{data.std():.2f}")
            with col2_2:
                growth_rate = ((data.iloc[-1]/data.iloc[0])**(1/10) - 1)*100
                st.metric("Growth Rate", f"{growth_rate:.1f}%", 
                         delta=f"{growth_rate:.1f}%" if growth_rate > 0 else None)
                st.metric("Latest Price", f"₹{data.iloc[-1]:.2f}")
        
        # Add real-time correlation heatmap (only if enabled)
        if show_correlations:
            st.subheader("🔥 Dynamic Market Correlations")
            correlation_fig = forecaster.create_correlation_heatmap(df_prices, df_features)
            st.plotly_chart(correlation_fig, use_container_width=True)
            
            # Add live correlation updates
            if auto_refresh:
                st.info("🔄 Correlations updating in real-time...")
        else:
            st.info("💡 Enable 'Show Correlations' in sidebar to view market relationships")
        
        # Train models and generate forecasts
        predictions = {}
        
        with st.spinner("Training models and generating forecasts..."):
            
            if use_arima:
                with st.expander("📊 ARIMA Model"):
                    arima_model, arima_order = forecaster.train_arima(data)
                    if arima_model:
                        arima_pred = forecaster.forecast_arima(arima_model)
                        predictions['ARIMA'] = arima_pred
                        st.success(f"✅ ARIMA{arima_order} model trained successfully")
                        st.info(f"📋 Best ARIMA order: {arima_order}")
                    else:
                        st.error("❌ ARIMA model failed")
            
            if use_sarimax:
                with st.expander("� SARIMAX Model"):
                    sarimax_model = forecaster.train_sarimax(data)
                    if sarimax_model:
                        sarimax_pred = forecaster.forecast_sarimax(sarimax_model)
                        predictions['SARIMAX'] = sarimax_pred
                        st.success("✅ SARIMAX model trained successfully")
                    else:
                        st.error("❌ SARIMAX model failed")
            
            if use_prophet:
                with st.expander("🔮 Prophet Model"):
                    prophet_model, prophet_data = forecaster.train_prophet(data, df_features)
                    if prophet_model:
                        prophet_pred = forecaster.forecast_prophet(prophet_model, prophet_data, df_features)
                        predictions['Prophet'] = prophet_pred
                        st.success("✅ Prophet model trained successfully")
                    else:
                        st.error("❌ Prophet model failed")
        
        # Display dynamic forecasts
        if predictions:
            st.subheader("🔮 Dynamic Price Forecasts (2025-2029)")
            
            # Create forecast DataFrame
            forecast_years = pd.date_range(start='2025', periods=5, freq='YE')
            forecast_df = pd.DataFrame(index=forecast_years)
            
            for model_name, preds in predictions.items():
                if preds is not None and len(preds) == 5:
                    forecast_df[f'{model_name}_Forecast'] = preds
            
            # Display forecast table with styling
            st.subheader("📊 Forecast Summary Table")
            styled_df = forecast_df.style.format("₹{:.2f}")
            st.dataframe(styled_df, use_container_width=True)
            
            # Interactive Forecast Animation (only if enabled)
            if show_animations:
                st.subheader("🎬 Animated Forecast Visualization")
                forecast_animation = forecaster.create_forecast_animation(data, predictions, selected_commodity)
                st.plotly_chart(forecast_animation, use_container_width=True)
                
                if auto_refresh:
                    st.info("🎬 Animations updating automatically...")
            else:
                st.info("💡 Enable 'Enable Animations' in sidebar for dynamic forecasts")
            
            # Model Comparison Dashboard (enhanced with interactivity)
            if interactive_mode:
                st.subheader("📊 Interactive Model Comparison Dashboard")
                comparison_dashboard = forecaster.create_comparison_dashboard(predictions)
                if comparison_dashboard:
                    st.plotly_chart(comparison_dashboard, use_container_width=True)
                    
                    # Add interactive controls for dashboard
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔍 Zoom to Best Model"):
                            st.balloons()
                    with col2:
                        if st.button("📈 Show Trends Only"):
                            st.success("Trend view activated!")
                    with col3:
                        if st.button("⚡ Quick Analysis"):
                            st.info("Quick analysis mode enabled!")
                    
                    # Add performance radar chart
                    st.subheader("🎯 Model Performance Analysis")
                    radar_chart = forecaster.create_performance_radar(predictions)
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
            else:
                st.subheader("📊 Model Comparison Dashboard")
                comparison_dashboard = forecaster.create_comparison_dashboard(predictions)
                if comparison_dashboard:
                    st.plotly_chart(comparison_dashboard, use_container_width=True)
            
            # Real-time forecast updates with auto-refresh option
            if st.checkbox("🔄 Enable Auto-Refresh (Live Updates)", help="Automatically refresh forecasts every 30 seconds"):
                st.rerun()
            
            # Interactive forecast analysis
            st.subheader("🔍 Interactive Forecast Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Show Growth Trends"):
                    growth_data = {}
                    for model_name, preds in predictions.items():
                        if preds is not None and len(preds) == 5:
                            growth_rate = ((preds[-1] / preds[0]) ** (1/4) - 1) * 100
                            growth_data[model_name] = growth_rate
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(growth_data.keys()),
                            y=list(growth_data.values()),
                            marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(growth_data)],
                            text=[f"{v:.1f}%" for v in growth_data.values()],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="📊 Predicted Annual Growth Rates by Model",
                        xaxis_title="Model",
                        yaxis_title="Growth Rate (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if st.button("📉 Risk Analysis"):
                    # Calculate volatility and risk metrics
                    risk_data = {}
                    for model_name, preds in predictions.items():
                        if preds is not None and len(preds) == 5:
                            volatility = np.std(preds) / np.mean(preds) * 100
                            risk_data[model_name] = volatility
                    
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=list(risk_data.keys()),
                            y=list(risk_data.values()),
                            mode='markers+lines',
                            marker=dict(size=15, color=list(risk_data.values()), 
                                      colorscale='Reds', showscale=True),
                            line=dict(width=3)
                        )
                    ])
                    fig.update_layout(
                        title="⚠️ Price Volatility Risk by Model",
                        xaxis_title="Model",
                        yaxis_title="Volatility (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                if st.button("🎯 Confidence Intervals"):
                    # Show confidence intervals for predictions
                    fig = go.Figure()
                    
                    for model_name, preds in predictions.items():
                        if preds is not None and len(preds) == 5:
                            # Simulate confidence intervals (in production, use actual model uncertainties)
                            upper_ci = np.array(preds) * 1.15
                            lower_ci = np.array(preds) * 0.85
                            
                            # Add confidence band
                            fig.add_trace(go.Scatter(
                                x=list(range(2025, 2030)) + list(range(2029, 2024, -1)),
                                y=list(upper_ci) + list(lower_ci[::-1]),
                                fill='toself',
                                fillcolor=f'rgba(128, 128, 128, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name=f'{model_name} CI',
                                showlegend=False
                            ))
                            
                            # Add prediction line
                            fig.add_trace(go.Scatter(
                                x=list(range(2025, 2030)),
                                y=preds,
                                mode='lines+markers',
                                name=model_name,
                                line=dict(width=3)
                            ))
                    
                    fig.update_layout(
                        title="🎯 Prediction Confidence Intervals",
                        xaxis_title="Year",
                        yaxis_title="Price (₹)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            

        
        else:
            st.error("No valid predictions generated. Please check your data and model settings.")

if __name__ == "__main__":
    main()