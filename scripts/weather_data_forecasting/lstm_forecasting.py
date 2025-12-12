from pathlib import Path
import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

SEED = 250

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
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
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.bn3(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.dropout3(lstm_out)
        
        last_timestep = lstm_out[:, -1, :]
        out = self.fc(last_timestep)
        
        return out

class LSTMForecaster:
    """
    LSTM forecaster for cups_sold prediction using PyTorch.
    
    Attributes:
        data_path: Path to data CSV file
        target_col: Name of the target column to forecast
        seq_length: Length of input sequences
        test_size: Number of days to use for testing (default: 7)
    """
    
    def __init__(
        self,
        data_path,
        target_col = 'cups_sold',
        seq_length = 7,  # Weekly window
        test_size = 7    # Last 7 days for testing
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            data_path: Path to data CSV file
            target_col: Name of the target column to forecast
            seq_length: Length of input sequences (default: 7 for weekly)
            test_size: Number of days to use for testing (default: 7)
        """
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.seq_length = seq_length
        self.test_size = test_size
        
        self.data = None
        self.train_data = None
        self.test_data = None
        
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses = []
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.load_data()
    
    def load_data(self):
        """Load and preprocess data."""
        # Load single dataset
        self.data = pd.read_csv(self.data_path)
        
        # Parse dates
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        
        # Split data: all except last 7 days for training, last 7 days for testing
        self.train_data = self.data.iloc[:-self.test_size]
        self.test_data = self.data.iloc[-self.test_size:]
        
        print(f"Training data: {len(self.train_data)} days")
        print(f"Testing data: {len(self.test_data)} days")
        
        # Scale target using only training data
        train_y = self.train_data[[self.target_col]].values
        self.scaler_y.fit(train_y)
        
        # Prepare sequences
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Prepare input sequences with weekday feature."""
        # Create weekday_norm feature
        if 'weekday_num' in self.data.columns:
            self.data['weekday_norm'] = self.data['weekday_num'] / 6.0
        else:
            self.data['weekday_norm'] = self.data.index.weekday / 6.0
        
        # Scale the target values for entire dataset
        y_scaled = self.scaler_y.transform(self.data[[self.target_col]])
        
        # Combine scaled target and weekday
        features = np.hstack([y_scaled, self.data[['weekday_norm']].values])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split sequences into train and test
        # Training: all sequences that don't use test data in their window
        # Test: sequences that predict the last 7 days
        train_end = len(self.train_data) - self.seq_length
        test_start = len(self.data) - self.test_size - self.seq_length
        
        self.X_train = X[:train_end]
        self.y_train = y[:train_end]
        
        self.X_test = X[test_start:]
        self.y_test = y[test_start:]
        
        print(f"Training sequences: {len(self.X_train)}")
        print(f"Testing sequences: {len(self.X_test)}")
    
    def train(
        self,
        epochs = 100,
        batch_size = 16,
        learning_rate = 0.0005,
        verbose = True
    ):
        """Train the LSTM model on all training data (no validation)."""
        input_size = self.X_train.shape[2]
        
        self.model = LSTMModel(input_size=input_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.train_losses = []
        
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
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    def predict(self):
        """Generate predictions on test set (last 7 days)."""
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
    
    def evaluate(self, predictions=None):
        """
        Evaluate model performance on test set (last 7 days only).
        
        Args:
            predictions: Pre-computed predictions (if None, will compute them)
        
        Returns:
            Dictionary containing MAE, RMSE, and MAPE metrics
        """
        if predictions is None:
            predictions = self.predict()
        
        # Get actual values for the test period
        actual = self.scaler_y.inverse_transform(self.y_test).flatten()
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(
                np.abs((actual[non_zero_mask] - predictions[non_zero_mask]) / actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0
        
        print(f"\nEvaluation on last {self.test_size} days:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
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
        verbose: bool = True
    ):
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(epochs, batch_size, learning_rate, verbose=verbose)
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics

    def plot_diagnostics(self, save_path=None):
        """Plot training loss history."""
        if not self.train_losses:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss (LSTM)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        
        plt.show()

    def plot_actual_vs_predicted(self, predictions=None, save_path=None):
        """Plot actual vs predicted values on the test set (last 7 days)."""
        if predictions is None:
            predictions = self.predict()
        
        # Get actual values for the test period
        actual = self.scaler_y.inverse_transform(self.y_test).flatten()
        test_dates = self.test_data.index
        
        # Create DataFrame for logging
        result_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        })
        
        # Save to CSV
        output_path = Path(__file__).parent / 'log' / 'weather_lstm_actual_vs_predicted.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', marker='o', color='blue', linewidth=2)
        plt.plot(test_dates, predictions, label='Predicted', marker='s', color='orange', linestyle='--', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Cups Sold')
        plt.title(f'Actual vs Predicted Cups Sold (LSTM) - Last {self.test_size} Days')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels
        for i, (date, actual_val, pred_val) in enumerate(zip(test_dates, actual, predictions)):
            plt.annotate(f'{actual_val:.0f}', (date, actual_val), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{pred_val:.0f}', (date, pred_val), textcoords="offset points", xytext=(0,-15), ha='center', color='orange')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Actual vs Predicted plot saved to {save_path}")
        
        plt.show()
        
        # Print individual predictions
        print("\nDaily Predictions (Last 7 Days):")
        print("=" * 50)
        for i, (date, actual_val, pred_val) in enumerate(zip(test_dates, actual, predictions)):
            error = actual_val - pred_val
            error_percent = (error / actual_val * 100) if actual_val != 0 else 0
            print(f"{date.date()}:")
            print(f"  Actual: {actual_val:.1f}")
            print(f"  Predicted: {pred_val:.1f}")
            print(f"  Error: {error:+.1f} ({error_percent:+.1f}%)")
            print("-" * 30)

# Instantiate and run
model = LSTMForecaster(
    data_path=Path(__file__).parent.parent.parent / 'data' / 'intermediate' / 'merged_daily_weather_all.csv',
    test_size=7  # Evaluate on last 7 days only
)

# Train the model (no validation, no early stopping)
print("Training LSTM model on all data except last 7 days...")
model.train(epochs=100, batch_size=16, learning_rate=0.0005, verbose=True)

# Make predictions on last 7 days
print("\nMaking predictions on last 7 days...")
predictions = model.predict()

# Evaluate on last 7 days
print("\nEvaluating model performance...")
metrics = model.evaluate(predictions)

# Plot diagnostics
print("\nGenerating plots...")
model.plot_diagnostics(
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'lstm_diagnostics_plot.png'
)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'lstm_actual_vs_predicted.png'
)

# Print final summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"Training: {len(model.train_data)} days")
print(f"Testing: {len(model.test_data)} days")
print("\nMetrics on last 7 days:")
for metric_name, metric_value in metrics.items():
    if metric_name == 'MAPE':
        print(f"{metric_name}: {metric_value:.2f}%")
    else:
        print(f"{metric_name}: {metric_value:.2f}")
print("=" * 60)
