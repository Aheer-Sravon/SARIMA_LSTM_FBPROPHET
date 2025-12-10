from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
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
    
    def __init__(
        self, 
        input_size, 
        hidden_sizes = [128, 64, 128],
        dropout_rates = [0.3, 0.2, 0.1]
    ):
        super(LSTMModel, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            batch_first=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            batch_first=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Third LSTM layer
        self.lstm3 = nn.LSTM(
            input_size=hidden_sizes[1],
            hidden_size=hidden_sizes[2],
            batch_first=True
        )
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
    LSTM forecaster for cups_sold prediction using PyTorch.
    
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        seq_length: Length of input sequences
        model: LSTM model
    """
    
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        target_col = 'cups_sold',
        seq_length = 7  # Weekly window
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            seq_length: Length of input sequences (default: 7 for weekly)
        """
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.seq_length = seq_length
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_val_data = None
        
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses = []  # Ensure initialized here
        self.val_losses = []  # Ensure initialized here
        
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
        
        # Parse dates
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Combine train and val for scaling
        self.train_val_data = pd.concat([self.train_data, self.val_data])
        
        # Scale target
        train_val_y = self.train_val_data[[self.target_col]].values
        self.scaler_y.fit(train_val_y)
        
        # Prepare sequences with weekday_num
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Prepare input sequences with weekday feature."""
        full_data = pd.concat([self.train_val_data, self.test_data])
        full_data['weekday_norm'] = full_data.index.weekday / 6.0  # Normalize 0-6 to 0-1
        
        # Scale target
        y_scaled = self.scaler_y.transform(full_data[[self.target_col]])
        
        # Combine scaled target and weekday
        features = np.hstack([y_scaled, full_data[['weekday_norm']].values])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length, 0])  # Predict next cups_sold
        
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
    
    def train(
        self,
        epochs = 100,
        batch_size = 32,
        learning_rate = 0.0005,
        patience = 15,
        verbose = True
    ):
        """Train the LSTM model with early stopping."""
        input_size = self.X_train.shape[2]  # features
        
        self.model = LSTMModel(input_size=input_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
    
    def forecast_future(self, n_steps, start_date):
        """
        Forecast future values using recursive prediction.
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
        
        Returns:
            DataFrame with forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Get last window from full data (train_val + test)
        full_data = pd.concat([self.train_val_data, self.test_data])
        last_window_scaled = self.scaler_y.transform(
            full_data[self.target_col].values[-self.seq_length:].reshape(-1, 1)
        )
        last_weekdays = full_data.index[-self.seq_length:].weekday.values / 6.0
        
        current_window = np.hstack([last_window_scaled, last_weekdays.reshape(-1, 1)])
        
        # Generate future dates and weekdays
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        future_weekdays = future_dates.weekday.values / 6.0
        
        predictions = []
        for weekday in future_weekdays:
            # Prepare input
            X_input = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_scaled = self.model(X_input).cpu().numpy()[0, 0]
            
            # Inverse transform
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Update window
            weekday_norm = weekday / 6.0
            new_step = np.array([[pred_scaled, weekday_norm]])
            current_window = np.vstack([current_window[1:], new_step])
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_cups_sold': predictions
        })
    
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
        
        actual = self.scaler_y.inverse_transform(self.y_test).flatten()
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((actual[non_zero_mask] - predictions[non_zero_mask]) / actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0  # Or float('nan') if preferred; 0.0 assumes all zeros mean perfect relative match
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def fit_and_evaluate(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        patience: int = 15,
        verbose: bool = True
    ):
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(epochs, batch_size, learning_rate, patience, verbose=verbose)
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics

    def plot_diagnostics(self, save_path = None):
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
        
        plt.show()

    def plot_actual_vs_predicted(self, predictions = None, save_path = None):
        """Plot actual vs predicted values on the test set."""
        if predictions is None:
            predictions = self.predict()
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values

        pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        }).to_csv("../log/weather_lstm_actual_vs_predicted.csv")
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title('Actual vs Predicted Cups Sold (LSTM)')
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
        plt.title('Future Forecast (LSTM)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        plt.show()

# Instantiate and run
model = LSTMForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'weather' / 'test.csv'
)

# Train the model (equivalent to optimize + train in SARIMAX)
model.train(epochs=100, batch_size=32, learning_rate=0.0005, patience=15, verbose=True)

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'lstm_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'lstm_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables needed for LSTM, uses recursive with weekdays)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'lstm_future_forecast.png'
)
