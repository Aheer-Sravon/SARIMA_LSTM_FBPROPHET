import sys
import warnings
from pathlib import Path
import pandas as pd
from typing import Optional, List

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import SARIMAForecaster
sys.path.pop()

# Suppress the specific warning
warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters for seasonal ARMA.*")

class WeeklySARIMAForecaster(SARIMAForecaster):
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
        
        # Set target series
        self.target = self.train_val[self.target_col]

    def forecast_future(self, n_steps: int, start_date: str) -> pd.DataFrame:
        if self.res is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps)
        predictions = forecast.predicted_mean.values
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_weekly_sales': predictions
        })

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
        plt.title('Actual vs Predicted Weekly Sales (SARIMA)')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        plt.show()

    def plot_future_forecast(self, n_steps: int = 30, start_date: Optional[str] = None, save_path: Optional[str] = None):
        """Plot future forecast starting from the end of the test data or a specified date."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.offsets.Week(weekday=4))  # Next Friday
        forecast_df = self.forecast_future(n_steps, start_date)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_weekly_sales'], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Predicted Weekly Sales')
        plt.title('Future Forecast (SARIMA)')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        plt.show()

# Instantiate and run
model = WeeklySARIMAForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart' / 'test.csv',
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
    s=7,
    verbose=True
)

model.fit()

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots' / 'sarima_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots' / 'sarima_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables for SARIMA)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2012-11-02'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2012-11-02',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots' / 'sarima_future_forecast.png'
)
