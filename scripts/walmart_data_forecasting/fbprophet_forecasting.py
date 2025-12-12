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
    Facebook Prophet forecaster for weekly sales prediction.
    
    Attributes:
        train_path: Path to training data CSV file
        test_path: Path to test data CSV file
        model: Fitted Prophet model
    """
    
    def __init__(
        self,
        train_path,
        test_path,
        target_col = 'weekly_sales',
        seasonality_mode = 'additive',
        yearly_seasonality = True,
        weekly_seasonality = True,
        daily_seasonality = False
    ) -> None:
        """
        Initialize the Prophet forecaster.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality (recommended: True)
            daily_seasonality: Whether to include daily seasonality
        """
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.train_data = None
        self.test_data = None
        self.full_data = None
        self.model = None
        self.residuals = None
        
        self.load_data()
    
    def load_data(self):
        """Load training and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse dates and set as index
        for df_name in ['train_data', 'test_data']:
            df = getattr(self, df_name)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Combine for full data reference
        self.full_data = pd.concat([self.train_data, self.test_data]).sort_index()
        
        print(f"Train data: {len(self.train_data)} weeks")
        print(f"Test data: {len(self.test_data)} weeks")
    
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
        logging.getLogger('cmdstanpy').setLevel(logging.INFO if verbose else logging.WARNING)
        
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
        self.model.fit(train_prophet)
        
        # Compute residuals for diagnostics
        forecast = self.model.predict(train_prophet)
        actual = train_prophet['y'].values
        self.residuals = actual - forecast['yhat'].values
        
        print(f"Model trained on {len(self.train_data)} weeks of data")
    
    def predict(self):
        """Generate predictions on test set."""
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
            DataFrame with forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        
        return pd.DataFrame({
            'date': forecast['ds'],
            'predicted_weekly_sales': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper']
        })
    
    def evaluate(self, predictions=None):
        """
        Evaluate model performance on test set.
        
        Returns:
            Dictionary containing MAE, RMSE, and MAPE metrics
        """
        if predictions is None:
            predictions = self.predict()
        
        actual = self.test_data[self.target_col].values
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((actual[non_zero_mask] - predictions[non_zero_mask]) / actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def fit_and_evaluate(
        self,
        growth='linear',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        verbose=False
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
    
    def plot_components(self, save_path=None):
        """Plot the forecast components (trend, seasonality, etc.)."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Create future dataframe for plotting
        future = self.model.make_future_dataframe(periods=len(self.test_data), freq='W')
        forecast = self.model.predict(future)
        
        # Plot components
        fig = self.model.plot_components(forecast)
        if save_path:
            plt.savefig(save_path)
            print(f"Components plot saved to {save_path}")
        plt.show()
    
    def plot_forecast(self, save_path=None):
        """Plot the forecast with actual data."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting")
        
        # Prepare all data
        all_prophet = self.prepare_prophet_data(self.full_data)
        
        # Make forecast
        forecast = self.model.predict(all_prophet)
        
        # Plot
        fig = self.model.plot(forecast)
        if save_path:
            plt.savefig(save_path)
            print(f"Forecast plot saved to {save_path}")
        plt.show()
    
    def plot_diagnostics(self, save_path=None):
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
    
    def plot_actual_vs_predicted(self, predictions=None, fig_save_path=None, csv_save_path=None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values
        
        # Create log directory if it doesn't exist
        Path("../log").mkdir(exist_ok=True)
        
        # Save predictions to CSV
        pred_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        })
        pred_df.to_csv(csv_save_path)
        print(f"Predictions saved to {csv_save_path}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue', marker='o')
        plt.plot(test_dates, predictions, label='Predicted', color='orange', marker='s')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.title(f'Actual vs Predicted Weekly Sales (Prophet) - {len(self.test_data)} Weeks')
        plt.legend()
        plt.grid(True)
        
        if fig_save_path:
            plt.savefig(fig_save_path)
            print(f"Actual vs Predicted plot saved to {fig_save_path}")
        
        plt.show()
    
    def plot_future_forecast(self, n_steps, start_date=None, save_path=None):
        """Plot future forecast with uncertainty."""
        if start_date is None:
            start_date = str(self.test_data.index[-1] + pd.offsets.Week(weekday=4))  # Next Friday
        
        forecast_df = self.forecast_future(n_steps, start_date)
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_weekly_sales'], 
                label='Forecast', color='green', marker='o')
        plt.fill_between(forecast_df['date'], forecast_df['lower_bound'], 
                        forecast_df['upper_bound'], color='green', alpha=0.2, 
                        label='Uncertainty')
        plt.xlabel('Date')
        plt.ylabel('Predicted Weekly Sales')
        plt.title(f'Future {n_steps}-Week Forecast (Prophet)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
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
    print(f"\n{'='*50}")
    print(f"For store: {store}")
    print('='*50)

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

    # Save splits to CSV files for the store
    train_path = base_data_path / f'train_{store}.csv'
    test_path = base_data_path / f'test_{store}.csv'

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    # Instantiate and run for this store
    model = ProphetForecaster(
        train_path=train_path,
        test_path=test_path,
        target_col='weekly_sales'
    )
    
    # Train the model
    model.train(
        growth='linear',
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=0.1,
        verbose=True
    )
    
    # Plot diagnostics after fitting
    diag_path = base_fig_path / f'fbprophet_diagnostics_plot_{store}.png'
    model.plot_diagnostics(save_path=diag_path)
    
    predictions = model.predict()
    
    # Evaluate on the test set
    metrics = model.evaluate(predictions)
    print(f"\nStore {store} metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Plot actual vs predicted
    avp_path = base_fig_path / f'fbprophet_actual_vs_predicted_{store}.png'
    model.plot_actual_vs_predicted(predictions, fig_save_path=avp_path, csv_save_path=f"../log/walmart_fbprophet_actual_vs_predicted_{store}.csv")
    
    # Future forecast example
    future_preds = model.forecast_future(
        n_steps=5,
        start_date='2012-11-02'
    )
    print(f"\nStore {store} future predictions:")
    print(future_preds)
    
    # Plot future forecast
    ff_path = base_fig_path / f'fbprophet_future_forecast_{store}.png'
    model.plot_future_forecast(
        n_steps=5,
        start_date='2012-11-02',
        save_path=ff_path
    )
    
    # Show train/test split info
    print(f"\nTrain dates: {train['date'].iloc[0].date()} to {train['date'].iloc[-1].date()}")
    print(f"Test dates: {test['date'].iloc[0].date()} to {test['date'].iloc[-1].date()}")

print("\n" + "="*50)
print("Prophet forecasting completed for all stores!")
print("="*50)
