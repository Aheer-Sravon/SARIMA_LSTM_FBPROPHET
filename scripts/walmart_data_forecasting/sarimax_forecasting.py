import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
plt.ion()

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")
warnings.filterwarnings("ignore", message="Non-stationary starting seasonal autoregressive.*")


warnings.filterwarnings("ignore")


class SARIMAXForecaster:
    """
    Unified SARIMAX forecaster for time series prediction.
    
    Supports both daily and weekly frequencies with exogenous variables.
    """
    
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        target_col='cups_sold',
        frequency='D',
        exog_cols=None
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.frequency = frequency
        
        # Set default exogenous variables
        if exog_cols is None:
            if frequency == 'D':
                self.exog_cols = ['weekday_num']
            else:  # Weekly
                self.exog_cols = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        else:
            self.exog_cols = exog_cols
        
        # Data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_val = None
        self.full_data = None
        self.target = None
        self.exog = None
        self.full_exog = None
        
        # Model attributes
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
            
            if self.frequency == 'D':
                df = df.asfreq('D').ffill()
            else:  # Weekly
                df = df.asfreq('W-FRI')
        
        # Combine train and validation
        self.train_val = pd.concat([self.train_data, self.val_data]).sort_index()
        
        # Combine all data
        self.full_data = pd.concat([self.train_val, self.test_data]).sort_index()
        
        # Set target and exogenous
        self.target = self.train_val[self.target_col]
        
        # Check available exogenous columns
        available_exog = [col for col in self.exog_cols if col in self.train_val.columns]
        if available_exog:
            self.exog = self.train_val[available_exog]
            self.full_exog = self.full_data[available_exog]
            self.exog_cols = available_exog
        else:
            self.exog = None
            self.full_exog = None
    
    def optimize(
        self,
        p_range=range(0, 3),
        q_range=range(0, 3),
        P_range=range(0, 2),
        Q_range=range(0, 2),
        d=None,
        D=None,
        s=None,
        verbose=True
    ):
        """Optimize SARIMAX parameters using grid search and AIC."""
        # Set defaults based on frequency
        if d is None:
            d = 1 if self.frequency == 'W-FRI' else 0
        
        if D is None:
            D = 1 if self.frequency == 'W-FRI' else 0
        
        if s is None:
            s = 52 if self.frequency == 'W-FRI' else 7
        
        parameters = list(product(p_range, q_range, P_range, Q_range))
        results = []
        
        iterator = tqdm(parameters, desc="Optimizing SARIMAX") if verbose else parameters
        
        for param in iterator:
            try:
                model = SARIMAX(
                    self.target,
                    exog=self.exog,
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], s),
                    simple_differencing=False,
                    enforce_invertibility=True,
                    enforce_stationarity=True
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
        else:
            print("No parameters converged. Using defaults.")
            self.best_order = (1, d, 1)
            self.best_s_order = (0, D, 0, s)
    
    def fit(self):
        """Fit the SARIMAX model with optimized parameters."""
        if self.best_order is None or self.best_s_order is None:
            self.optimize(verbose=False)
        
        if self.best_order and self.best_s_order:
            self.model = SARIMAX(
                self.target,
                exog=self.exog,
                order=self.best_order,
                seasonal_order=self.best_s_order,
                simple_differencing=False,
                enforce_invertibility=True,
                enforce_stationarity=True
            )
            self.res = self.model.fit(disp=False, maxiter=500, method='nm')
    
    def train(self):
        """Alias for fit method for compatibility."""
        self.fit()
    
    def predict(self, window=1):
        """Generate predictions for test set using recursive forecasting."""
        if self.best_order is None or self.best_s_order is None:
            raise ValueError("Model must be optimized before making predictions")
        
        full_target = self.full_data[self.target_col]
        train_len = len(self.train_val)
        horizon = len(self.test_data)
        total_len = train_len + horizon
        
        predictions = []
        
        for i in range(train_len, total_len, window):
            try:
                model = SARIMAX(
                    full_target[:i],
                    exog=self.full_exog[:i] if self.full_exog is not None else None,
                    order=self.best_order,
                    seasonal_order=self.best_s_order,
                    simple_differencing=False
                )
                res = model.fit(disp=False, maxiter=500, method='nm')
                
                forecast_steps = min(window, total_len - i)
                if self.full_exog is not None:
                    oos_exog = self.full_exog[i:i+forecast_steps]
                    oos_pred = res.get_forecast(steps=forecast_steps, exog=oos_exog).predicted_mean.values
                else:
                    oos_pred = res.get_forecast(steps=forecast_steps).predicted_mean.values
                
                predictions.extend(oos_pred)
            except Exception as e:
                print(f"Warning: Prediction failed at step {i}: {e}")
                last_pred = predictions[-1] if predictions else 0.0
                predictions.extend([last_pred] * min(window, total_len - i))
        
        return predictions[:horizon]
    
    def forecast_future(self, n_steps, start_date, exog_future):
        """Forecast future values with exogenous variables."""
        if self.res is None:
            raise ValueError("Model must be trained before forecasting")
        
        if exog_future is not None and len(exog_future) != n_steps:
            raise ValueError(f"exog_future must have {n_steps} rows")
        
        # Generate future dates
        if self.frequency == 'D':
            future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        else:
            future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps, exog=exog_future)
        predictions = forecast.predicted_mean.values
        
        # Determine column name
        if self.target_col == 'weekly_sales':
            pred_col = 'predicted_weekly_sales'
        else:
            pred_col = 'predicted_cups_sold'
        
        return pd.DataFrame({
            'date': future_dates,
            pred_col: predictions
        })
    
    def evaluate(self, predictions=None):
        """Evaluate model performance on test set."""
        if predictions is None:
            predictions = self.predict()
        
        actual = self.test_data[self.target_col].values
        predictions_array = np.array(predictions)
        
        mae = mean_absolute_error(actual, predictions_array)
        rmse = np.sqrt(mean_squared_error(actual, predictions_array))
        
        # Calculate MAPE
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(
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
    
    def fit_and_evaluate(self, verbose=True):
        """Fit the model and evaluate on test set."""
        self.optimize(verbose=verbose)
        self.fit()
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics
    
    def get_summary(self):
        """Get summary of the fitted model."""
        if self.res is None:
            return "Model not yet trained"
        return str(self.res.summary())
    
    def plot_diagnostics(self, save_path=None):
        """Plot model diagnostics."""
        if self.res is None:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        self.res.plot_diagnostics(figsize=(12, 8))
        
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        
        if self.frequency == 'W-FRI':
            plt.pause(10)
            plt.close()
        else:
            plt.show()
    
    def plot_actual_vs_predicted(self, predictions=None, save_path=None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col]
        
        # Create results directory if it doesn't exist
        Path("../log").mkdir(exist_ok=True)
        
        # Save to CSV
        if self.frequency == 'D':
            csv_path = "../log/weather_sarimax_actual_vs_predicted.csv"
        else:
            csv_path = "../log/walmart_sarimax_actual_vs_predicted.csv"
        
        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual.values,
            'Predicted': predictions
        })
        result_df.to_csv(csv_path, index=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        
        if self.target_col == 'weekly_sales':
            plt.ylabel('Weekly Sales')
            title = 'Actual vs Predicted Weekly Sales (SARIMAX)'
        else:
            plt.ylabel('Cups Sold')
            title = 'Actual vs Predicted Cups Sold (SARIMAX)'
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        if self.frequency == 'W-FRI':
            plt.pause(10)
            plt.close()
        else:
            plt.show()
    
    def plot_future_forecast(self, n_steps=30, exog_future=None, start_date=None, save_path=None):
        """Plot future forecast."""
        if start_date is None:
            if self.frequency == 'D':
                start_date = str(self.test_data.index[-1] + pd.Timedelta(days=1))
            else:
                start_date = str(self.test_data.index[-1] + pd.offsets.Week(weekday=4))
        
        if exog_future is None:
            raise ValueError(f"Provide exog_future with {n_steps} rows")
        
        forecast_df = self.forecast_future(n_steps, start_date, exog_future)
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df.iloc[:, 1], label='Forecast', color='green')
        plt.xlabel('Date')
        
        if self.target_col == 'weekly_sales':
            plt.ylabel('Predicted Weekly Sales')
        else:
            plt.ylabel('Predicted Cups Sold')
        
        plt.title('Future Forecast (SARIMAX)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        if self.frequency == 'W-FRI':
            plt.pause(10)
            plt.close()
        else:
            plt.show()

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
    model = SARIMAXForecaster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        target_col='weekly_sales',
        frequency='W-FRI'
    )
    
    # Optimize with smaller ranges if needed
    model.optimize(
        p_range=range(0, 4),
        q_range=range(0, 4),
        P_range=range(0, 2),
        Q_range=range(0, 2),
        d=1,
        D=0,
        s=7,
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
