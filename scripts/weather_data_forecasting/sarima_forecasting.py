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
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        best_order: Optimal (p, d, q) parameters
        best_s_order: Optimal seasonal (P, D, Q, s) parameters
        model: Fitted SARIMAX model
    """
    
    def __init__(
        self, 
        train_path, 
        val_path, 
        test_path,
        target_col = 'cups_sold'
    ):
        """
        Initialize the SARIMA forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
        """
        self.train_path: Path = Path(train_path)
        self.val_path: Path = Path(val_path)
        self.test_path: Path = Path(test_path)
        self.target_col: str = target_col
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_val = None
        self.full = None
        self.target = None
        
        self.best_order = None
        self.best_s_order = None
        self.model = None
        self.res = None
        
        self.load_data()

    def load_data(self):
        """Load and preprocess training, validation, and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column and set as index
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.asfreq('D')  # Explicitly set daily frequency to avoid warnings
            df = df.ffill()  # Handle any NaNs if gaps exist
        
        # Combine train and validation for model training
        self.train_val = pd.concat([self.train_data, self.val_data]).sort_index()
        
        # Combine all data for sequential prediction
        self.full = pd.concat([self.train_val, self.test_data]).sort_index()
        
        # Set target series
        self.target = self.train_val[self.target_col]

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
            self.optimize()
        
        if self.best_order and self.best_s_order:
            self.model = SARIMAX(
                self.target,
                order=self.best_order,
                seasonal_order=self.best_s_order,
                simple_differencing=False
            )
            self.res = self.model.fit(disp=False, maxiter=500)

    def predict(self):
        """
        Generate predictions for test set using in-sample prediction.
        Uses single model fit (fast, like original script).
        
        Returns:
            List of predictions for test period
        """
        if self.best_order is None or self.best_s_order is None:
            raise ValueError("Model must be optimized before making predictions")
        
        if self.res is None:
            self.fit()
        
        train_len = len(self.train_val)
        horizon = len(self.test_data)
        
        # In-sample prediction (like original script)
        predictions = self.res.get_prediction(
            start=train_len,
            end=train_len + horizon - 1
        )
        
        return predictions.predicted_mean.values

    def evaluate(self, predictions = None):
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
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col]

        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual.values,
            'Predicted': predictions
        })
        result_df.to_csv("../log/weather_sarima_actual_vs_predicted.csv")
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title('Actual vs Predicted Cups Sold (SARIMA)')
        plt.legend()
        plt.grid(True)
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
        plt.plot(forecast_df['date'], forecast_df['predicted_cups_sold'], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Predicted Cups Sold')
        plt.title('Future Forecast (SARIMA)')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        plt.show()

# Instantiate and run
model = SARIMAForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'test.csv'
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 6), verbose=True)

model.fit()

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables for SARIMA)
future_preds = model.forecast_future(
    n_steps=8,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarima_future_forecast.png'
)
