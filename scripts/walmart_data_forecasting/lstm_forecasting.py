from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import warnings
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
plt.ion()

# Suppress warnings if needed
warnings.filterwarnings("ignore")

NUM_STORES = 5

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    Architecture based on the TensorFlow notebook:
    - 3 LSTM layers (128 -> 64 -> 128 units)
    - Batch normalization after first two LSTM layers
    - Dropout for regularization
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 128], dropout_rates=[0.3, 0.2, 0.1]):
        super(LSTMModel, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Third LSTM layer
        self.lstm3 = nn.LSTM(input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        # Output layer
        self.fc = nn.Linear(hidden_sizes[2], 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden, seq)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # back to (batch, seq, hidden)
        lstm_out = self.dropout1(lstm_out)
        
        # Second LSTM layer
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.bn2(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.dropout2(lstm_out)
        
        # Third LSTM layer
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = self.dropout3(lstm_out)
        
        # Take last timestep for prediction
        last_timestep = lstm_out[:, -1, :]
        out = self.fc(last_timestep)
        
        return out


class LSTMForecaster:
    """
    Unified LSTM forecaster for time series prediction.
    
    Supports both daily and weekly frequencies with or without exogenous variables.
    For daily data: uses weekday as exogenous feature
    For weekly data: uses provided exogenous variables or none
    
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        target_col: Name of the target column to forecast
        frequency: Time series frequency ('D' for daily, 'W-FRI' for weekly)
        seq_length: Length of input sequences
        exog_cols: List of exogenous variable column names
    """
    
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        target_col='cups_sold',
        frequency='D',
        seq_length=7,
        exog_cols=None
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.frequency = frequency
        self.seq_length = seq_length
        
        # Set default exogenous variables if not provided
        if exog_cols is None:
            if frequency == 'D':
                self.exog_cols = []  # Will add weekday automatically
            else:  # Weekly
                self.exog_cols = []  # Empty by default, user must specify if needed
        else:
            self.exog_cols = exog_cols
        
        # Data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_val_data = None
        self.full_data = None
        
        # Scaling attributes
        self.scaler_y = MinMaxScaler()
        self.scaler_exog = None
        
        # Model attributes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Data sequences
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.load_data()
    
    def load_data(self):
        """Load and preprocess data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse dates and set frequency
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Set frequency
            if self.frequency == 'D':
                df = df.asfreq('D')
            else:  # Weekly
                df = df.asfreq('W-FRI')
        
        # Combine train and val for scaling
        self.train_val_data = pd.concat([self.train_data, self.val_data]).sort_index()
        
        # Full data for reference
        self.full_data = pd.concat([self.train_val_data, self.test_data]).sort_index()
        
        # Prepare sequences
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Prepare input sequences with features."""
        # Get full data for sequence creation
        full_data = pd.concat([self.train_val_data, self.test_data]).sort_index()
        
        # Scale target
        y_scaled = self.scaler_y.fit_transform(full_data[[self.target_col]])
        
        # Prepare features
        features_list = [y_scaled]
        
        # Add weekday feature for daily data
        if self.frequency == 'D':
            full_data['weekday_norm'] = full_data.index.weekday / 6.0
            features_list.append(full_data[['weekday_norm']].values)
        
        # Add exogenous variables if specified
        if self.exog_cols:
            # Check which columns are available
            available_exog = [col for col in self.exog_cols if col in full_data.columns]
            if available_exog:
                self.scaler_exog = MinMaxScaler()
                exog_scaled = self.scaler_exog.fit_transform(full_data[available_exog])
                features_list.append(exog_scaled)
                self.exog_cols = available_exog  # Update with only available columns
            else:
                self.exog_cols = []
        
        # Combine all features
        features = np.hstack(features_list)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length, 0])  # Predict next value
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split back to train/val/test
        train_end = len(self.train_data)
        val_end = train_end + len(self.val_data)
        
        self.X_train = X[:train_end - self.seq_length]
        self.y_train = y[:train_end - self.seq_length]
        
        self.X_val = X[train_end - self.seq_length:val_end - self.seq_length]
        self.y_val = y[train_end - self.seq_length:val_end - self.seq_length]
        
        self.X_test = X[val_end - self.seq_length:]
        self.y_test = y[val_end - self.seq_length:]
    
    def train(
        self,
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        patience=15,
        verbose=True
    ):
        """Train the LSTM model with early stopping."""
        input_size = self.X_train.shape[2]  # Number of features
        
        self.model = LSTMModel(input_size=input_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.train_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    def fit(self, *args, **kwargs):
        """Alias for train method."""
        self.train(*args, **kwargs)
    
    def predict(self):
        """Generate predictions on test set."""
        if self.model is None:
            raise ValueError("Model must be trained before predicting")
        
        test_dataset = TimeSeriesDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        return self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    def forecast_future(self, n_steps, start_date, exog_future=None):
        """
        Forecast future values using recursive prediction.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Generate future dates
        if self.frequency == 'D':
            future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        else:
            future_dates = pd.date_range(start=start_date, periods=n_steps, freq='W-FRI')
        
        # Prepare exogenous future data
        future_exog = None
        if self.exog_cols and exog_future is not None:
            if len(exog_future) != n_steps:
                raise ValueError(f"exog_future must have {n_steps} rows")
            future_exog = self.scaler_exog.transform(exog_future[self.exog_cols])
        
        # Get last window from full data
        last_y_scaled = self.scaler_y.transform(
            self.full_data[self.target_col].values[-self.seq_length:].reshape(-1, 1)
        )
        
        # Build current window
        features_list = [last_y_scaled]
        
        # Add exogenous variables if available
        if self.exog_cols and self.scaler_exog is not None:
            last_exog_scaled = self.scaler_exog.transform(
                self.full_data[self.exog_cols].iloc[-self.seq_length:]
            )
            features_list.append(last_exog_scaled)
        
        current_window = np.hstack(features_list)
        
        # Generate predictions
        predictions = []
        for i in range(n_steps):
            # Prepare input
            X_input = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_scaled = self.model(X_input).cpu().numpy()[0, 0]
            
            # Inverse transform
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Prepare next step features - FIXED
            next_features_list = []
            
            # 1. Add scaled prediction (reshape to ensure 2D)
            next_features_list.append(np.array([[pred_scaled]]))
            
            # 2. Add exogenous variables for next step if available
            if future_exog is not None:
                # future_exog[i] is 1D, reshape to 2D
                next_features_list.append(future_exog[i].reshape(1, -1))
            
            # FIX: Concatenate properly
            if len(next_features_list) == 1:
                new_step = next_features_list[0]
            else:
                # All features are already 2D, concatenate along axis=1
                new_step = np.concatenate(next_features_list, axis=1)
            
            # Update window
            current_window = np.vstack([current_window[1:], new_step])
        
        # Determine prediction column name
        pred_col = 'predicted_weekly_sales'
        
        return pd.DataFrame({
            'date': future_dates,
            pred_col: predictions
        })
    
    def evaluate(self, predictions=None):
        """
        Evaluate model performance on test set.
        
        Args:
            predictions: Pre-computed predictions
        
        Returns:
            Dictionary containing MAE, RMSE, and MAPE metrics
        """
        if predictions is None:
            predictions = self.predict()
        
        actual = self.scaler_y.inverse_transform(self.y_test).flatten()
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        # Calculate MAPE
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
    
    def fit_and_evaluate(self, epochs=100, batch_size=32, learning_rate=0.0005, patience=15, verbose=True):
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(epochs, batch_size, learning_rate, patience, verbose=verbose)
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics
    
    def get_summary(self):
        """Get summary of the model architecture."""
        if self.model is None:
            return "Model not yet trained"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = "LSTM Model Summary:\n"
        summary += f"  Input size: {self.X_train.shape[2]}\n"
        summary += f"  Sequence length: {self.seq_length}\n"
        summary += f"  Total parameters: {total_params:,}\n"
        summary += f"  Trainable parameters: {trainable_params:,}\n"
        summary += f"  Device: {self.device}\n"
        
        return summary
    
    def plot_diagnostics(self, save_path=None):
        """Plot training and validation loss history."""
        if not self.train_losses:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss (LSTM)')
        plt.legend()
        plt.grid(True)
        
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
        actual = self.test_data[self.target_col].values
        
        # Save results to CSV
        Path("../log").mkdir(exist_ok=True)
        
        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        })
        result_df.to_csv("../log/walmart_lstm_actual_vs_predicted.csv", index=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        
        if self.target_col == 'weekly_sales':
            plt.ylabel('Weekly Sales')
            title = 'Actual vs Predicted Weekly Sales (LSTM)'
        else:
            plt.ylabel('Cups Sold')
            title = 'Actual vs Predicted Cups Sold (LSTM)'
        
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
        
        if self.exog_cols and exog_future is None:
            print(f"Warning: exog_future not provided but model uses exogenous variables: {self.exog_cols}")
        
        forecast_df = self.forecast_future(n_steps, start_date, exog_future)
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df.iloc[:, 1], label='Forecast', color='green')
        plt.xlabel('Date')
        
        if self.target_col == 'weekly_sales':
            plt.ylabel('Predicted Weekly Sales')
        else:
            plt.ylabel('Predicted Cups Sold')
        
        plt.title('Future Forecast (LSTM)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        if self.frequency == 'W-FRI':
            plt.pause(10)
            plt.close()
        else:
            plt.show()

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
    model = LSTMForecaster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        frequency='W-FRI',
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
