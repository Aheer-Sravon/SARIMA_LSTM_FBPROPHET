"""
Prophet Forecaster for Time Series Prediction
Using Facebook Prophet for cups_sold forecasting
"""

# Suppress the specific warning
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ProphetForecaster:
    """
    Facebook Prophet forecaster for cups_sold prediction.
    
    Prophet is a procedure for forecasting time series data based on an additive model 
    where non-linear trends are fit with yearly, weekly, and daily seasonality, 
    plus holiday effects.
    
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        model: Fitted Prophet model
    """
    
    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        test_path: Union[str, Path],
        target_col: str = 'cups_sold',
        seasonality_mode: str = 'multiplicative',
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ) -> None:
        """
        Initialize the Prophet forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality (recommended: True)
            daily_seasonality: Whether to include daily seasonality
        """
        self.train_path: Path = Path(train_path)
        self.val_path: Path = Path(val_path)
        self.test_path: Path = Path(test_path)
        self.target_col: str = target_col
        self.seasonality_mode: str = seasonality_mode
        self.yearly_seasonality: bool = yearly_seasonality
        self.weekly_seasonality: bool = weekly_seasonality
        self.daily_seasonality: bool = daily_seasonality
        
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.train_val_data: Optional[pd.DataFrame] = None
        
        self.model: Optional[Prophet] = None
        
        self.load_data()
    
    def load_data(self) -> None:
        """Load and preprocess training, validation, and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
        
        # Combine train and validation for model training
        self.train_val_data = pd.concat([self.train_data, self.val_data]).reset_index(drop=True)
    
    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns).
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with 'ds' (datestamp) and 'y' (value) columns
        """
        prophet_df = pd.DataFrame({
            'ds': df['date'],
            'y': df[self.target_col]
        })
        return prophet_df
    
    def train(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.80,
        verbose: bool = False
    ) -> None:
        """
        Train the Prophet model.
        
        Args:
            growth: 'linear' or 'logistic'
            changepoint_prior_scale: Controls flexibility of trend (higher = more flexible)
            seasonality_prior_scale: Controls flexibility of seasonality (higher = more flexible)
            holidays_prior_scale: Controls flexibility of holidays
            interval_width: Width of uncertainty intervals (0.8 = 80%)
            verbose: Whether to print training progress
        """
        # Initialize model
        self.model = Prophet(
            growth=growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            interval_width=interval_width
        )
        
        # Prepare training data
        train_prophet = self.prepare_prophet_data(self.train_val_data)
        
        # Suppress logs if not verbose
        if not verbose:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
        
        # Fit model
        self.model.fit(train_prophet)
        
        if verbose:
            print("Prophet model trained successfully!")
    
    def predict(self) -> np.ndarray:
        """
        Generate predictions for test set.
        
        Returns:
            Array of predictions for test period
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare test data
        test_prophet = self.prepare_prophet_data(self.test_data)
        
        # Make predictions
        forecast = self.model.predict(test_prophet)
        predictions = forecast['yhat'].values
        
        return predictions
    
    def forecast_future(self, n_steps: int, start_date: str) -> pd.DataFrame:
        """
        Forecast future values.
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
        
        Returns:
            DataFrame with forecasted values and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Create future dataframe
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = self.model.predict(future)
        
        return pd.DataFrame({
            'date': forecast['ds'],
            'predicted_cups_sold': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper']
        })
    
    def evaluate(self, predictions: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            predictions: Pre-computed predictions (if None, will compute them)
        
        Returns:
            Dictionary containing MAE, RMSE, and MAPE metrics
        """
        if predictions is None:
            predictions = self.predict()
        
        actual: np.ndarray = self.test_data[self.target_col].values
        
        mae: float = mean_absolute_error(actual, predictions)
        rmse: float = np.sqrt(mean_squared_error(actual, predictions))
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape: float = np.mean(
                np.abs((actual[non_zero_mask] - predictions[non_zero_mask]) / actual[non_zero_mask])
            ) * 100
        else:
            mape = float('nan')  # Or 0.0 if all zeros (adjust as needed)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def fit_and_evaluate(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            verbose=verbose
        )
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics
    
    def plot_components(self) -> None:
        """Plot the forecast components (trend, seasonality, etc.)."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Create future dataframe for plotting
        future = self.model.make_future_dataframe(periods=len(self.test_data))
        forecast = self.model.predict(future)
        
        # Plot components
        fig = self.model.plot_components(forecast)
        return fig
    
    def plot_forecast(self) -> None:
        """Plot the forecast with actual data."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Prepare all data
        all_data = pd.concat([self.train_val_data, self.test_data]).reset_index(drop=True)
        all_prophet = self.prepare_prophet_data(all_data)
        
        # Make forecast
        forecast = self.model.predict(all_prophet)
        
        # Plot
        fig = self.model.plot(forecast)
        return fig
