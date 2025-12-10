import sys
from pathlib import Path
import pandas as pd
from typing import Optional, List
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch

import matplotlib.pyplot as plt
plt.ion()

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import LSTMForecaster
sys.path.pop()

# Suppress warnings if needed
import warnings
warnings.filterwarnings("ignore")

NUM_STORES = 5

class WeeklyLSTMForecaster(LSTMForecaster):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        target_col='weekly_sales',
        exog_cols=['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],
        seq_length=7
    ):
        self.exog_cols = exog_cols
        self.seq_length = seq_length
        super().__init__(train_path, val_path, test_path, target_col, seq_length)

    def load_data(self) -> None:
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column and set as index
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.asfreq('W-FRI')  # Set weekly frequency
        
        # Combine train and validation for scaling
        self.train_val_data = pd.concat([self.train_data, self.val_data]).sort_index()
        
        # Full data for reference
        self.full_data = pd.concat([self.train_val_data, self.test_data]).sort_index()

        # Scale target
        train_val_y = self.train_val_data[[self.target_col]].values
        self.scaler_y.fit(train_val_y)

        # Prepare sequences with exog_cols
        self._prepare_sequences()

    def _prepare_sequences(self) -> None:
        """Prepare input sequences with exogenous variables."""
        full_data = pd.concat([self.train_val_data, self.test_data])
        
        # Scale target
        y_scaled = self.scaler_y.transform(full_data[[self.target_col]])
        
        # Scale exog (use separate scaler for exog)
        self.scaler_exog = MinMaxScaler()
        exog_scaled = self.scaler_exog.fit_transform(full_data[self.exog_cols])
        
        # Combine scaled target and exog
        features = np.hstack([y_scaled, exog_scaled])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length, 0])  # Predict next target
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split back
        train_end = len(self.train_data)
        val_end = train_end + len(self.val_data)
        
        self.X_train = X[:train_end - self.seq_length]
        self.y_train = y[:train_end - self.seq_length]
        
        self.X_val = X[train_end - self.seq_length:val_end - self.seq_length]
        self.y_val = y[train_end - self.seq_length:val_end - self.seq_length]
        
        self.X_test = X[val_end - self.seq_length:]
        self.y_test = y[val_end - self.seq_length:]

    def forecast_future(self, n_steps: int, start_date: str, exog_future: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        if len(exog_future) != n_steps:
            raise ValueError(f"exog_future must have {n_steps} rows")
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        
        # Scale future exog
        exog_future_scaled = self.scaler_exog.transform(exog_future[self.exog_cols])
        
        # Get last window from full data
        last_y_scaled = self.scaler_y.transform(
            self.full_data[self.target_col].values[-self.seq_length:].reshape(-1, 1)
        )
        last_exog_scaled = self.scaler_exog.transform(
            self.full_data[self.exog_cols].iloc[-self.seq_length:]
        )
        current_window = np.hstack([last_y_scaled, last_exog_scaled])
        
        predictions = []
        self.model.eval()
        for i in range(n_steps):
            X_input = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model(X_input).cpu().numpy()[0, 0]
            
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update window
            new_step = np.concatenate([[pred_scaled], exog_future_scaled[i]])
            current_window = np.vstack([current_window[1:], new_step])
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_weekly_sales': predictions
        })

    def plot_actual_vs_predicted(self, predictions: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.title('Actual vs Predicted Weekly Sales (LSTM)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        plt.show()

    def plot_future_forecast(self, n_steps: int, exog_future: pd.DataFrame, start_date: str = '2012-11-02', save_path: Optional[str] = None):
        """Plot future forecast."""
        forecast_df = self.forecast_future(n_steps, start_date, exog_future)
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_weekly_sales'], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Predicted Weekly Sales')
        plt.title('Future Forecast (LSTM)')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")

        plt.show()

if __name__ == '__main__':
    # Define paths
    base_data_path = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart'
    base_fig_path = Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots'
    
    base_data_path.mkdir(parents=True, exist_ok=True)
    base_fig_path.mkdir(parents=True, exist_ok=True)
    
    # Load and merge data
    df = pd.read_csv(base_data_path / 'walmart.csv')
    
    for store in range(1, NUM_STORES + 1):
        print(f"For store: {store}")
        df_store = df[df['Store'] == store].copy()
        df_store['Date'] = pd.to_datetime(df_store['Date'], format='%d-%m-%Y')
        df_store = df_store.sort_values('Date')
        df_store.rename(columns={'Date': 'date', 'Weekly_Sales': 'weekly_sales'}, inplace=True)
        
        # Aggregate to store-level to handle department duplicates
        agg_dict = {'weekly_sales': 'sum'}
        for col in ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
            if col in df_store.columns:
                agg_dict[col] = 'first'
        for md in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
            if md in df_store.columns:
                agg_dict[md] = 'first'
        df_store = df_store.groupby('date').agg(agg_dict).reset_index()
        
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
        model = WeeklyLSTMForecaster(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            target_col='weekly_sales',
            exog_cols=['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        )
        
        # Train
        model.train(
            epochs=100,
            batch_size=32,
            learning_rate=0.0005,
            patience=15,
            verbose=True
        )
        
        # Plot diagnostics after fitting
        diag_path = base_fig_path / f'lstm_diagnostics_plot_{store}.png'
        model.plot_diagnostics(save_path=diag_path)
        
        predictions = model.predict()
        
        metrics = model.evaluate(predictions)
        print(f"\nStore {store} metrics: {metrics}")
        
        # Plot actual vs predicted
        avp_path = base_fig_path / f'lstm_actual_vs_predicted_{store}.png'
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
        ff_path = base_fig_path / f'lstm_future_forecast_{store}.png'
        model.plot_future_forecast(
            n_steps=5,
            exog_future=exog_future,
            start_date='2012-11-02',
            save_path=ff_path
        )
