import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pathlib import Path
import pandas as pd
from itertools import product
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

# Add at the beginning of your script
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class SARIMAForecaster:
    """
    SARIMA (Seasonal AutoRegressive Integrated Moving Average) forecaster for time series prediction.
    
    Supports both daily and weekly frequencies.
    
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        target_col: Name of the target column to forecast
        frequency: Time series frequency ('D' for daily, 'W-FRI' for weekly)
        best_order: Optimal (p, d, q) parameters
        best_s_order: Optimal seasonal (P, D, Q, s) parameters
        model: Fitted SARIMAX model
    """
    
    def __init__(
        self, 
        train_path,
        test_path,
        target_col='weekly_sales',
        frequency='W-FRI'
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.frequency = frequency
        
        self.train_data = None
        self.test_data = None  # Removed val_data
        self.train_val = None
        self.full = None
        self.target = None
        
        self.best_order = None
        self.best_s_order = None
        self.model = None
        self.res = None
        
        self.load_data()

    def load_data(self):
        """Load and preprocess training and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column and set as index
        for df_name in ['train_data', 'test_data']:  # Removed 'val_data'
            df = getattr(self, df_name)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Set frequency explicitly to avoid the warning
            df.index = pd.DatetimeIndex(df.index, freq=self.frequency)
            
            # Only fill forward if it's daily data and you expect no gaps
            if self.frequency == 'D':
                df = df.asfreq('D').ffill()
                setattr(self, df_name, df)
        
        # Use train data directly (no validation)
        self.train_val = self.train_data
        
        # Combine all data for sequential prediction
        self.full = pd.concat([self.train_val, self.test_data]).sort_index()
        
        # Set target series
        self.target = self.train_val[self.target_col]

    def optimize(
        self,
        p_range: range = range(0, 3),
        q_range: range = range(0, 3),
        P_range: range = range(0, 3),
        Q_range: range = range(0, 3),
        d: int = 1,
        D: int = 0,
        s: int = 7,
        verbose: bool = True
    ):
        """
        Optimize SARIMA parameters using grid search and AIC.
        
        Args:
            p_range: Range for AR (AutoRegressive) parameter
            q_range: Range for MA (Moving Average) parameter
            P_range: Range for seasonal AR parameter
            Q_range: Range for seasonal MA parameter
            d: Degree of differencing (default: 1)
            D: Degree of seasonal differencing (default: 0)
            s: Seasonal period (default: 7 for weekly patterns, 52 for yearly in weekly data)
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
                res = model.fit(disp=False, maxiter=500, method='nm')
                aic = res.aic
                results.append([param, aic])
            except Exception:
                continue
        
        if results:
            result_df = pd.DataFrame(results, columns=['params', 'aic'])
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
            self.res = self.model.fit(disp=False, maxiter=500, method='nm')

    def predict(self):
        """
        Generate predictions for test set using recursive forecasting.
        
        Returns:
            List of predictions for test period
        """
        if self.best_order is None or self.best_s_order is None:
            raise ValueError("Model must be optimized before making predictions")
        
        full_target = self.full[self.target_col]
        train_len = len(self.train_val)
        horizon = len(self.test_data)
        predictions = []
        
        for i in range(train_len, train_len + horizon):
            try:
                model = SARIMAX(
                    full_target[:i],
                    order=self.best_order,
                    seasonal_order=self.best_s_order,
                    simple_differencing=False
                )
                res = model.fit(disp=False, maxiter=500, method='nm')
                pred = res.get_forecast(steps=1).predicted_mean.iloc[0]
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Prediction failed at step {i}: {e}")
                predictions.append(predictions[-1] if predictions else 0.0)
        
        return predictions

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
        
        mae = mean_absolute_error(actual, predictions_array)
        rmse = np.sqrt(mean_squared_error(actual, predictions_array))

        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape: float = np.mean(
                np.abs((actual[non_zero_mask] - predictions_array[non_zero_mask]) / 
                actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0
        
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

    def get_summary(self) -> str:
        """Get summary of the fitted model."""
        if self.res is None:
            return "Model not yet trained"
        return str(self.res.summary())

    def forecast_future(self, n_steps: int, start_date: str) -> pd.DataFrame:
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
        
        # Generate future dates based on frequency
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq=self.frequency)
        
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps)
        predictions = forecast.predicted_mean.values
        
        # Determine column name based on frequency
        pred_col = 'predicted_weekly_sales' if self.frequency.startswith('W') else 'predicted_cups_sold'
        
        return pd.DataFrame({
            'date': future_dates,
            pred_col: predictions
        })

    def plot_diagnostics(self, save_path = None):
        """Plot model diagnostics (residuals, etc.)."""
        if self.res is None:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        self.res.plot_diagnostics(figsize=(12, 8))
        
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        
        # Handle display based on frequency
        if self.frequency.startswith('W'):
            plt.pause(10)
            plt.close()
        else:
            plt.show()

    def plot_actual_vs_predicted(self, predictions = None, fig_save_path = None, csv_save_path = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col]
        
        # Save CSV for daily data
        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual.values,
            'Predicted': predictions
        })
        result_df.to_csv(csv_save_path)
        
        # Determine labels based on frequency
        title = 'Actual vs Predicted Weekly Sales (SARIMA)'
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if fig_save_path:
            plt.savefig(fig_save_path)
            print(f"Actual vs Predicted plot saved to {fig_save_path}")
        
        # Handle display based on frequency
        if self.frequency.startswith('W'):
            plt.pause(10)
            plt.close()
        else:
            plt.show()

    def plot_future_forecast(self, n_steps: int = 30, start_date = None, save_path = None):
        """Plot future forecast starting from the end of the test data or a specified date."""
        if start_date is None:
            if self.frequency.startswith('W'):
                start_date = str(self.test_data.index[-1] + pd.offsets.Week(weekday=4))  # Next Friday
            else:
                start_date = str(self.test_data.index[-1] + pd.Timedelta(days=1))
        
        forecast_df = self.forecast_future(n_steps, start_date)
        
        # Determine labels and column names based on frequency
        if self.frequency.startswith('W'):
            pred_col = 'predicted_weekly_sales'
            ylabel = 'Predicted Weekly Sales'
        else:
            pred_col = 'predicted_cups_sold'
            ylabel = 'Predicted Cups Sold'
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df[pred_col], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.title('Future Forecast (SARIMA)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        # Handle display based on frequency
        if self.frequency.startswith('W'):
            plt.pause(10)
            plt.close()
        else:
            plt.show()

# Read the full dataframe
df = pd.read_csv(
    Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart' / 'walmart.csv'
)

# Process for the first 5 stores
stores = [1, 3]
base_data_path = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart'
base_fig_path = Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots'

for store in stores:

    print(f"For store: {store}")

    df_store = df[df['Store'] == store].copy()
    df_store['Date'] = pd.to_datetime(df_store['Date'], format='%d-%m-%Y')
    df_store = df_store.sort_values('Date')
    df_store.rename(columns={'Date': 'date', 'Weekly_Sales': 'weekly_sales'}, inplace=True)
    
    # Split the data chronologically - LAST 50 points for testing
    n = len(df_store)
    test_size = 7  # Fixed 50 test points
    train_size = n - test_size

    train = df_store.iloc[:train_size]
    test = df_store.iloc[train_size:]

    # Skip validation split entirely
    # Save splits to CSV files for the store
    train_path = base_data_path / f'train_{store}.csv'
    test_path = base_data_path / f'test_{store}.csv'

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # Instantiate and run for this store
    model = SARIMAForecaster(
        train_path=train_path,
        test_path=test_path,
        target_col='weekly_sales'
    )
    
    # Optimize with smaller ranges if needed
    model.optimize(
        p_range=range(0, 4),
        q_range=range(0, 3),
        P_range=range(0, 3),
        Q_range=range(0, 2),
        d=1,
        D=0,
        s=2,
        verbose=True
    )
    
    model.fit()
    
    # Plot diagnostics after fitting
    diag_path = base_fig_path / f'sarima_diagnostics_plot_{store}.png'
    model.plot_diagnostics(save_path=diag_path)
    
    predictions = model.predict()
    
    metrics = model.evaluate(predictions)
    print(f"\nStore {store} metrics: {metrics}")
    
    # Plot actual vs predicted
    avp_path = base_fig_path / f'sarima_actual_vs_predicted_{store}.png'
    model.plot_actual_vs_predicted(predictions, fig_save_path=avp_path, csv_save_path=f"../log/walmart_sarima_actual_vs_predicted_{store}.csv")
    
    # Future forecast example (no exogenous variables for SARIMA)
    future_preds = model.forecast_future(
        n_steps=5,
        start_date='2012-11-02'
    )
    print(f"\nStore {store} future predictions:\n{future_preds}")
    
    # Plot future forecast
    ff_path = base_fig_path / f'sarima_future_forecast_{store}.png'
    model.plot_future_forecast(
        n_steps=5,
        start_date='2012-11-02',
        save_path=ff_path
    )
