import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
from tqdm import tqdm

import matplotlib.pyplot as plt

# Suppress convergence and value warnings by category
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Suppress specific UserWarnings by message (using regex to match exactly)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")

class SARIMAForecaster:
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) forecaster for time series prediction.
    
    Attributes:
        data_path: Path to the full dataset CSV file
        best_order: Optimal (p, d, q) parameters
        best_s_order: Optimal seasonal (P, D, Q, s) parameters
        model: Fitted SARIMAX model
    """
    
    def __init__(
        self, 
        data_path,
        target_col = 'cups_sold',
        test_size = 7  # Last 7 days for testing
    ):
        """
        Initialize the SARIMA forecaster.
        
        Args:
            data_path: Path to the full dataset CSV file
            target_col: Name of the target column to forecast
            test_size: Number of last days to use for testing
        """
        self.data_path: Path = Path(data_path)
        self.target_col: str = target_col
        self.test_size = test_size
        
        self.full_data = None
        self.train_data = None
        self.test_data = None
        self.target = None
        
        self.best_order = None
        self.best_s_order = None
        self.model = None
        self.res = None
        
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
        self.train_data = self.train_data.asfreq('D').ffill()  # Set daily frequency and forward fill
        
        # Test data (last 7 days)
        self.test_data = self.full_data.iloc[train_size:].copy()
        self.test_data.set_index('date', inplace=True)
        self.test_data = self.test_data.asfreq('D').ffill()
        
        # Set target series for training
        self.target = self.train_data[self.target_col]
        
        print(f"Dataset loaded: {len(self.full_data)} total rows")
        print(f"Train data: {len(self.train_data)} rows (all except last {self.test_size} days)")
        print(f"Test data: {len(self.test_data)} rows (last {self.test_size} days)")

    def optimize(
        self,
        p_range = range(0, 6),  # Smaller default for efficiency
        q_range = range(0, 6),
        P_range = range(0, 6),
        Q_range = range(0, 6),
        d = 0,
        D = 0,
        s = 7,
        verbose = True
    ) -> None:
        """
        Optimize SARIMA parameters using grid search and AIC.
        
        Args:
            p_range: Range for AR (AutoRegressive) parameter
            q_range: Range for MA (Moving Average) parameter
            P_range: Range for seasonal AR parameter
            Q_range: Range for seasonal MA parameter
            d: Degree of differencing (default: 1)
            D: Degree of seasonal differencing (default: 0)
            s: Seasonal period (default: 7 for weekly data)
            verbose: Whether to show progress bar
        """
        parameters = list(product(p_range, q_range, P_range, Q_range))
        results = []
        
        iterator = tqdm(parameters, desc="Optimizing SARIMA") if verbose else parameters
        
        for param in iterator:
            try:
                model = SARIMAX(
                    self.target,
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], s),
                    simple_differencing=False
                )
                res = model.fit(disp=False, maxiter=500)  # Better convergence
                aic: float = res.aic
                results.append([param, aic])
            except Exception:
                continue
        
        if results:
            result_df: pd.DataFrame = pd.DataFrame(results, columns=['params', 'aic'])
            result_df = result_df.sort_values('aic').reset_index(drop=True)
            best_param = result_df['params'].iloc[0]
            
            self.best_order = (best_param[0], d, best_param[1])
            self.best_s_order = (best_param[2], D, best_param[3], s)
            
            if verbose:
                print(f"\nBest order: {self.best_order}")
                print(f"Best seasonal order: {self.best_s_order}")
                print(f"Best AIC: {result_df['aic'].iloc[0]:.2f}")
                print("\nTop 5 parameter combinations:")
                print(result_df.head())

    def fit(self):
        """Fit the SARIMA model with optimized parameters."""
        if self.best_order is None or self.best_s_order is None:
            self.optimize(verbose=False)
        
        if self.best_order and self.best_s_order:
            self.model = SARIMAX(
                self.target,
                order=self.best_order,
                seasonal_order=self.best_s_order,
                simple_differencing=False
            )
            self.res = self.model.fit(disp=False, maxiter=500)
            print(f"Model fitted on {len(self.train_data)} days of data")

    def predict(self):
        """
        Generate predictions for test set (last 7 days).
        
        Returns:
            List of predictions for test period
        """
        if self.best_order is None or self.best_s_order is None:
            raise ValueError("Model must be optimized before making predictions")
        
        if self.res is None:
            self.fit()
        
        # In-sample prediction for the last 7 days
        predictions = self.res.get_forecast(steps=self.test_size)
        
        return predictions.predicted_mean.values

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
        
        actual: np.ndarray = self.test_data[self.target_col].values
        predictions_array = np.array(predictions)
        
        mae: float = mean_absolute_error(actual, predictions_array)
        rmse: float = np.sqrt(mean_squared_error(actual, predictions_array))

        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((actual[non_zero_mask] - predictions_array[non_zero_mask]) / 
                actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0  # If all actual values are zero, set MAPE to 0 (or nan if preferred)
        
        print(f"Evaluation on last {self.test_size} days:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def fit_and_evaluate(self):
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.optimize(verbose=False)
        self.fit()
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics

    def get_summary(self):
        """Get summary of the fitted model."""
        if self.res is None:
            return "Model not yet trained"
        return str(self.res.summary())

    def forecast_future(self, n_steps, start_date):
        """
        Forecast future values.
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
        
        Returns:
            DataFrame with forecasted values
        """
        if self.res is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps)
        predictions = forecast.predicted_mean.values
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_cups_sold': predictions
        })

    def plot_diagnostics(self, save_path = None):
        """Plot model diagnostics (residuals, etc.)."""
        if self.res is None:
            raise ValueError("Model must be trained before plotting diagnostics")
        self.res.plot_diagnostics(figsize=(12, 8))
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        plt.show()

    def plot_actual_vs_predicted(self, predictions = None, save_path = None):
        """Plot actual vs predicted values on the test set (last 7 days)."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col]

        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual.values,
            'Predicted': predictions
        })
        result_df.to_csv("../log/weather_sarima_actual_vs_predicted.csv", index=False)
        print("Predictions saved to ../log/weather_sarima_actual_vs_predicted.csv")
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue', marker='o')
        plt.plot(test_dates, predictions, label='Predicted', color='orange', marker='s')
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title(f'Actual vs Predicted Cups Sold (SARIMA) - Last {self.test_size} Days')
        plt.legend()
        plt.grid(True)
        
        # Add value labels for the last 7 days
        for i, (date, act, pred) in enumerate(zip(test_dates, actual.values, predictions)):
            plt.text(date, act, f'{act:.0f}', ha='center', va='bottom')
            plt.text(date, pred, f'{pred:.0f}', ha='center', va='top')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        plt.show()

    def plot_future_forecast(self, n_steps = 30, start_date = None, save_path = None):
        """Plot future forecast starting from the end of the test data or a specified date."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.Timedelta(days=1))
        forecast_df = self.forecast_future(n_steps, start_date)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_cups_sold'], label='Forecast', color='green', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Predicted Cups Sold')
        plt.title(f'Future {n_steps}-Day Forecast (SARIMA)')
        plt.grid(True)
        
        # Add value labels
        for _, row in forecast_df.iterrows():
            plt.text(row['date'], row['predicted_cups_sold'], f'{row["predicted_cups_sold"]:.0f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        plt.show()

# Instantiate and run
model = SARIMAForecaster(
    data_path=Path(__file__).parent.parent.parent / 'data' / 'intermediate' / 'merged_daily_weather_all.csv',
    target_col='cups_sold',
    test_size=7
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 6), verbose=True)

model.fit()

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_diagnostics_plot.png')

predictions = model.predict()

# Evaluate on the last 7 days
metrics = model.evaluate(predictions)
print("\nFinal Metrics:", metrics)

# Plot actual vs predicted for the last 7 days
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_actual_vs_predicted.png'
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
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_future_forecast.png'
)

# Optional: Show train/test split info
print(f"\nTrain dates: {model.train_data.index[0]} to {model.train_data.index[-1]}")
print(f"Test dates: {model.test_data.index[0]} to {model.test_data.index[-1]}")
