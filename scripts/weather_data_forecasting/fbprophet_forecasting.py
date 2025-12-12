from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress the specific warning
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")

class ProphetForecaster:
    """
    Facebook Prophet forecaster for cups_sold prediction.
    
    Prophet is a procedure for forecasting time series data based on an additive model 
    where non-linear trends are fit with yearly, weekly, and daily seasonality, 
    plus holiday effects.
    
    Attributes:
        data_path: Path to the full dataset CSV file
        model: Fitted Prophet model
    """
    
    def __init__(
        self,
        data_path,
        target_col = 'cups_sold',
        test_size = 7,  # Last 7 days for testing
        seasonality_mode = 'additive',
        yearly_seasonality = True,
        weekly_seasonality = True,
        daily_seasonality = False
    ) -> None:
        """
        Initialize the Prophet forecaster.
        
        Args:
            data_path: Path to the full dataset CSV file
            target_col: Name of the target column to forecast
            test_size: Number of last days to use for testing
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality (recommended: True)
            daily_seasonality: Whether to include daily seasonality
        """
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.test_size = test_size
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.full_data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.residuals = None  # For diagnostics
        
        self.load_and_split_data()
    
    def load_and_split_data(self):
        """Load the full dataset and split into train/test."""
        self.full_data = pd.read_csv(self.data_path)
        
        # Parse dates and sort by date
        self.full_data['date'] = pd.to_datetime(self.full_data['date'])
        self.full_data = self.full_data.sort_values('date').reset_index(drop=True)
        
        # Split data: all except last 7 days for training, last 7 days for testing
        train_size = len(self.full_data) - self.test_size
        
        # Train data (all except last 7 days)
        self.train_data = self.full_data.iloc[:train_size].copy()
        self.train_data.set_index('date', inplace=True)
        
        # Test data (last 7 days)
        self.test_data = self.full_data.iloc[train_size:].copy()
        self.test_data.set_index('date', inplace=True)
        
        print(f"Dataset loaded: {len(self.full_data)} total rows")
        print(f"Train data: {len(self.train_data)} rows (all except last {self.test_size} days)")
        print(f"Test data: {len(self.test_data)} rows (last {self.test_size} days)")
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet (ds, y columns)."""
        prophet_df = df.reset_index().rename(columns={'date': 'ds', self.target_col: 'y'})
        return prophet_df[['ds', 'y']]
    
    def train(
        self,
        growth = 'linear',
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10.0,
        verbose = False
    ) -> None:
        """Train the Prophet model on training data."""
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
        
        train_prophet = self.prepare_prophet_data(self.train_data)
        self.model.fit(train_prophet)  # No verbose here
        
        # Compute residuals for diagnostics
        forecast = self.model.predict(train_prophet)
        actual = train_prophet['y'].values
        self.residuals = actual - forecast['yhat'].values
        
        print(f"Model trained on {len(self.train_data)} days of data")
    
    def predict(self):
        """Generate predictions on test set (last 7 days)."""
        if self.model is None:
            raise ValueError("Model must be trained before predicting")
        
        test_prophet = self.prepare_prophet_data(self.test_data)
        forecast = self.model.predict(test_prophet)
        return forecast['yhat'].values
    
    def forecast_future(self, n_steps, start_date):
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
    
    def evaluate(self, predictions = None):
        """
        Evaluate model performance on test set (last 7 days).
        
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
            mape = float('nan')
        
        print(f"Evaluation on last {self.test_size} days:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def fit_and_evaluate(
        self,
        growth: str = 'linear',
        changepoint_prior_scale = 0.05,
        seasonality_prior_scale = 10.0,
        verbose: bool = False
    ):
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
    
    def plot_components(self, save_path = None):
        """Plot the forecast components (trend, seasonality, etc.)."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Create future dataframe for plotting
        future = self.model.make_future_dataframe(periods=self.test_size)
        forecast = self.model.predict(future)
        
        # Plot components
        fig = self.model.plot_components(forecast)
        if save_path:
            plt.savefig(save_path)
            print(f"Components plot saved to {save_path}")
        plt.show()
    
    def plot_forecast(self, save_path = None):
        """Plot the forecast with actual data."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Prepare all data
        all_data = pd.concat([self.train_data, self.test_data]).reset_index(drop=True)
        all_prophet = self.prepare_prophet_data(all_data)
        
        # Make forecast
        forecast = self.model.predict(all_prophet)
        
        # Plot
        fig = self.model.plot(forecast)
        if save_path:
            plt.savefig(save_path)
            print(f"Forecast plot saved to {save_path}")
        plt.show()
    
    def plot_diagnostics(self, save_path = None):
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

    def plot_actual_vs_predicted(self, predictions = None, save_path = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values
        
        # Save predictions to CSV
        pred_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        })
        pred_df.to_csv("../log/weather_fbprophet_actual_vs_predicted.csv", index=False)
        print(f"Predictions saved to ../log/weather_fbprophet_actual_vs_predicted.csv")
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue', marker='o')
        plt.plot(test_dates, predictions, label='Predicted', color='orange', marker='s')
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title(f'Actual vs Predicted Cups Sold (Prophet) - Last {self.test_size} Days')
        plt.legend()
        plt.grid(True)
        
        # Add value labels for the last 7 days
        for i, (date, act, pred) in enumerate(zip(test_dates, actual, predictions)):
            plt.text(date, act, f'{act:.0f}', ha='center', va='bottom')
            plt.text(date, pred, f'{pred:.0f}', ha='center', va='top')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        plt.show()

    def plot_future_forecast(self, n_steps, start_date = None, save_path = None):
        """Plot future forecast with uncertainty."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.Timedelta(days=1))
        forecast_df = self.forecast_future(n_steps, start_date)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_cups_sold'], label='Forecast', color='green', marker='o')
        plt.fill_between(forecast_df['date'], forecast_df['lower_bound'], forecast_df['upper_bound'], color='green', alpha=0.2, label='Uncertainty')
        plt.xlabel('Date')
        plt.ylabel('Predicted Cups Sold')
        plt.title(f'Future {n_steps}-Day Forecast (Prophet)')
        plt.legend()
        plt.grid(True)
        
        # Add value labels
        for _, row in forecast_df.iterrows():
            plt.text(row['date'], row['predicted_cups_sold'], f'{row["predicted_cups_sold"]:.0f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        plt.show()

# Instantiate and run
model = ProphetForecaster(
    data_path=Path(__file__).parent.parent.parent / 'data' / 'intermediate' / 'merged_daily_weather_all.csv',  # Update to your single dataset
    target_col='cups_sold',
    test_size=7
)

# Train the model
model.train(
    growth='linear',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    verbose=True
)

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_diagnostics_plot.png')

predictions = model.predict()

# Evaluate on the last 7 days
metrics = model.evaluate(predictions)
print("\nFinal Metrics:", metrics)

# Plot actual vs predicted for the last 7 days
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_actual_vs_predicted.png'
)

# Future forecast example
future_preds = model.forecast_future(
    n_steps=5,
    start_date=str(model.test_data.index[-1] + pd.Timedelta(days=1))  # Start from day after last test date
)
print("\nFuture Forecast:")
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date=str(model.test_data.index[-1] + pd.Timedelta(days=1)),
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_future_forecast.png'
)

# Optional: Show train/test split info
print(f"\nTrain dates: {model.train_data.index[0]} to {model.train_data.index[-1]}")
print(f"Test dates: {model.test_data.index[0]} to {model.test_data.index[-1]}")
