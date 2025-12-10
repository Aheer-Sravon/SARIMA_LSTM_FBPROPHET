import sys
import warnings
from pathlib import Path
import pandas as pd
from typing import Optional, List
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import SARIMAXForecaster
sys.path.pop()

from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive.*")

class WeeklySARIMAXForecaster(SARIMAXForecaster):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        target_col='weekly_sales',
        exog_cols=['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    ):
        super().__init__(train_path, val_path, test_path, target_col, exog_cols)

    def load_data(self) -> None:
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column and set as index
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.asfreq('W-FRI')  # Set weekly frequency
        
        # Combine train and validation for model training
        self.train_val = pd.concat([self.train_data, self.val_data]).sort_index()
        
        # Combine all data for sequential prediction
        self.full = pd.concat([self.train_val, self.test_data]).sort_index()
        
        # Set target series and exogenous variables
        self.target = self.train_val[self.target_col]
        self.train_exog = self.train_val[self.exog_cols]
        self.test_exog = self.test_data[self.exog_cols]
        
        # Compatibility with parent class attributes
        self.train_val_data = self.train_val
        self.full_data = self.full
        self.exog = self.train_exog
        self.full_exog = self.full[self.exog_cols]

    def forecast_future(self, n_steps: int, start_date: str, exog_future: pd.DataFrame) -> pd.DataFrame:
        if self.res is None:
            raise ValueError("Model must be trained before forecasting")
        
        if len(exog_future) != n_steps:
            raise ValueError(f"exog_future must have {n_steps} rows")
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps, exog=exog_future)
        predictions = forecast.predicted_mean.values
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_weekly_sales': predictions
        })

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """Plot model diagnostics."""
        if self.res is None:
            raise ValueError("Model must be fitted first")
        try:
            self.res.plot_diagnostics(figsize=(12, 8))
        except Exception as e:
            print(f"Diagnostics plotting failed: {str(e)}")
            print("Residual variance:", np.var(self.res.resid))
            # Optionally plot basic residual histogram
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(self.res.resid)
            plt.title('Residuals')
            plt.subplot(2, 2, 2)
            from statsmodels.graphics.gofplots import qqplot
            qqplot(self.res.resid, line='s')
            plt.title('Normal Q-Q')
            # Skip KDE and correlogram if failed

        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        plt.pause(10)
        plt.close()

    def plot_actual_vs_predicted(self, predictions: Optional[List[float]] = None, save_path: Optional[str] = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col]
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.title('Actual vs Predicted Weekly Sales (SARIMAX)')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        plt.pause(10)
        plt.close()

    def plot_future_forecast(self, n_steps: int = 30, exog_future: Optional[pd.DataFrame] = None, start_date: Optional[str] = None, save_path: Optional[str] = None):
        """Plot future forecast starting from the end of the test data or a specified date."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.offsets.Week(weekday=4))  # Next Friday
        if exog_future is None or len(exog_future) != n_steps:
            raise ValueError(f"Provide exog_future with {n_steps} rows and columns: {self.exog_cols}")
        forecast_df = self.forecast_future(n_steps, start_date, exog_future)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_weekly_sales'], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Predicted Weekly Sales')
        plt.title('Future Forecast (SARIMAX)')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        plt.pause(10)
        plt.close()

# Read the full dataframe
df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart' / 'walmart.csv')

# Process for the first 5 stores
stores = [1, 2, 3, 4, 5]
base_data_path = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart'
base_fig_path = Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots'

for store in stores:
    print(f"For store: {store}")

    df_store = df[df['Store'] == store].copy()
    df_store['Date'] = pd.to_datetime(df_store['Date'], format='%d-%m-%Y')
    df_store = df_store.sort_values('Date')
    df_store.rename(columns={'Date': 'date', 'Weekly_Sales': 'weekly_sales'}, inplace=True)
    
    # Split the data chronologically
    n = len(df_store)
    train_size = int(0.7 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    
    train = df_store.iloc[:train_size]
    val = df_store.iloc[train_size:train_size + val_size]
    test = df_store.iloc[train_size + val_size:]
    
    # Save splits to CSV files for the store
    train_path = base_data_path / f'train_{store}.csv'
    val_path = base_data_path / f'validation_{store}.csv'
    test_path = base_data_path / f'test_{store}.csv'
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    # Instantiate and run for this store
    model = WeeklySARIMAXForecaster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        target_col='weekly_sales'
    )
    
    # Optimize with smaller ranges if needed
    model.optimize(
        p_range=range(0, 4),
        q_range=range(0, 4),
        P_range=range(0, 2),
        Q_range=range(0, 2),
        d=1,
        D=1,
        s=3,
        verbose=True
    )
    
    model.train()
    
    # Plot diagnostics after fitting
    diag_path = base_fig_path / f'sarimax_diagnostics_plot_{store}.png'
    model.plot_diagnostics(save_path=diag_path)
    
    predictions = model.predict()
    
    metrics = model.evaluate(predictions)
    print(f"\nStore {store} metrics: {metrics}")
    
    # Plot actual vs predicted
    avp_path = base_fig_path / f'sarimax_actual_vs_predicted_{store}.png'
    model.plot_actual_vs_predicted(predictions, save_path=avp_path)
    
    # Future forecast example
    # Use last exog values repeated for future
    exog_last = model.test_data[model.exog_cols].iloc[-1]
    exog_future = pd.DataFrame([exog_last] * 5, columns=model.exog_cols)
    
    future_preds = model.forecast_future(
        n_steps=5,
        start_date='2012-11-02',
        exog_future=exog_future
    )
    print(f"\nStore {store} future predictions:\n{future_preds}")
    
    # Plot future forecast
    ff_path = base_fig_path / f'sarimax_future_forecast_{store}.png'
    model.plot_future_forecast(
        n_steps=5,
        exog_future=exog_future,
        start_date='2012-11-02',
        save_path=ff_path
    )
