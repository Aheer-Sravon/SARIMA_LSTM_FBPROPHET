from pathlib import Path

# Suppress the specific warning
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")

from pathlib import Path
from typing import Optional, Tuple, Union, Dict
import matplotlib.pyplot as plt

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
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
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
        self.residuals: Optional[np.ndarray] = None  # For diagnostics
        
        self.load_data()
    
    def load_data(self) -> None:
        """Load and preprocess data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse dates
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Combine train and val for fitting
        self.train_val_data = pd.concat([self.train_data, self.val_data])
    
    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet (ds, y columns)."""
        prophet_df = df.reset_index().rename(columns={'date': 'ds', self.target_col: 'y'})
        return prophet_df[['ds', 'y']]
    
    def train(
        self,
        growth: str = 'linear',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        verbose: bool = False
    ) -> None:
        """Train the Prophet model on combined train+val data."""
        import logging
        logging.getLogger('prophet').setLevel(logging.INFO if verbose else logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.INFO if verbose else logging.WARNING)  # Also for backend

        self.model = Prophet(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        
        train_prophet = self.prepare_prophet_data(self.train_val_data)
        self.model.fit(train_prophet)  # No verbose here
        
        # Compute residuals for diagnostics
        forecast = self.model.predict(train_prophet)
        actual = train_prophet['y'].values
        self.residuals = actual - forecast['yhat'].values
    
    def predict(self) -> np.ndarray:
        """Generate predictions on test set."""
        if self.model is None:
            raise ValueError("Model must be trained before predicting")
        
        test_prophet = self.prepare_prophet_data(self.test_data)
        forecast = self.model.predict(test_prophet)
        return forecast['yhat'].values
    
    def forecast_future(self, n_steps: int, start_date: str) -> pd.DataFrame:
        """
        Forecast future values.
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
        
        Returns:
            DataFrame with forecasted values, including uncertainty bounds
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        future = pd.DataFrame({'ds': future_dates})
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
        
        actual = self.test_data[self.target_col].values
        
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
    
    def plot_components(self, save_path: Optional[str] = None) -> None:
        """Plot the forecast components (trend, seasonality, etc.)."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Create future dataframe for plotting
        future = self.model.make_future_dataframe(periods=len(self.test_data))
        forecast = self.model.predict(future)
        
        # Plot components
        fig = self.model.plot_components(forecast)
        if save_path:
            plt.savefig(save_path)
            print(f"Components plot saved to {save_path}")
        plt.show()
    
    def plot_forecast(self, save_path: Optional[str] = None) -> None:
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
        if save_path:
            plt.savefig(save_path)
            print(f"Forecast plot saved to {save_path}")
        plt.show()
    
    def plot_diagnostics(self, save_path: Optional[str] = None):
        """Plot model diagnostics (residuals histogram)."""
        if self.residuals is None:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        plt.figure(figsize=(12, 6))
        plt.hist(self.residuals, bins=30, color='blue', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Histogram (Prophet)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        
        plt.show()

    def plot_actual_vs_predicted(self, predictions: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values
        
        pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        }).to_csv("./weather_fbprophet_actual_vs_predicted.csv")
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title('Actual vs Predicted Cups Sold (Prophet)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        plt.show()

    def plot_future_forecast(self, n_steps: int = 30, start_date: Optional[str] = None, save_path: Optional[str] = None):
        """Plot future forecast with uncertainty."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.Timedelta(days=1))
        forecast_df = self.forecast_future(n_steps, start_date)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_cups_sold'], label='Forecast', color='green')
        plt.fill_between(forecast_df['date'], forecast_df['lower_bound'], forecast_df['upper_bound'], color='green', alpha=0.2, label='Uncertainty')
        plt.xlabel('Date')
        plt.ylabel('Predicted Cups Sold')
        plt.title('Future Forecast (Prophet)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        plt.show()

# Instantiate and run
model = ProphetForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'test.csv'
)

# Train the model (equivalent to optimize + train in SARIMAX)
model.train(
    growth='linear',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    verbose=True
)

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables needed for Prophet)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_future_forecast.png'
)
