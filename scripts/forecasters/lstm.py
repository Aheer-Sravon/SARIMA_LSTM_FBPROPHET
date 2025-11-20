"""
LSTM Forecaster for Time Series Prediction using PyTorch
Adapted from TensorFlow implementation in notebooks to PyTorch
"""

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        input_size: int, 
        hidden_sizes: List[int] = [128, 64, 128],
        dropout_rates: List[float] = [0.3, 0.2, 0.1]
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        # Apply batch norm to last timestep
        last_output = lstm_out[:, -1, :]
        last_output = self.bn1(last_output)
        # Reshape for next LSTM
        lstm_out = lstm_out[:, :-1, :]
        lstm_out = torch.cat([lstm_out, last_output.unsqueeze(1)], dim=1)
        lstm_out = self.dropout1(lstm_out)
        
        # Second LSTM layer
        lstm_out, _ = self.lstm2(lstm_out)
        last_output = lstm_out[:, -1, :]
        last_output = self.bn2(last_output)
        lstm_out = lstm_out[:, :-1, :]
        lstm_out = torch.cat([lstm_out, last_output.unsqueeze(1)], dim=1)
        lstm_out = self.dropout2(lstm_out)
        
        # Third LSTM layer
        lstm_out, _ = self.lstm3(lstm_out)
        last_time_step = lstm_out[:, -1, :]
        last_time_step = self.dropout3(last_time_step)
        
        # Output
        out = self.fc(last_time_step)
        
        return out


class LSTMForecaster:
    """
    LSTM forecaster for cups_sold prediction using PyTorch.
    
    This implementation follows the architecture from the notebook:
    - Window size: 7 days
    - Features: scaled cups_sold + normalized weekday
    - Split: 80% train, 10% val, 10% test
    - Learning rate: 0.0005
    - Callbacks: EarlyStopping, ReduceLROnPlateau
    """
    
    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        test_path: Union[str, Path],
        target_col: str = 'cups_sold',
        window_size: int = 7,
        hidden_sizes: List[int] = [128, 64, 128],
        dropout_rates: List[float] = [0.3, 0.2, 0.1],
        seed: int = 250,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the LSTM forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            window_size: Number of time steps to look back
            hidden_sizes: List of hidden units for each LSTM layer
            dropout_rates: Dropout rates for each LSTM layer
            seed: Random seed for reproducibility
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.train_path: Path = Path(train_path)
        self.val_path: Path = Path(val_path)
        self.test_path: Path = Path(test_path)
        self.target_col: str = target_col
        self.window_size: int = window_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.dropout_rates: List[float] = dropout_rates
        self.seed: int = seed
        
        self.device: str = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.full_data: Optional[pd.DataFrame] = None
        
        self.scaler_y: MinMaxScaler = MinMaxScaler()
        
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        self.model: Optional[LSTMModel] = None
        self.best_model_state: Optional[Dict[str, Any]] = None
        
        self.load_data()
        self.prepare_data()
        self.build_model()
    
    def load_data(self) -> None:
        """Load and preprocess data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse dates
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
        
        # Combine all data
        self.full_data = pd.concat([self.train_data, self.val_data, self.test_data]).sort_values('date').reset_index(drop=True)
    
    def prepare_data(self) -> None:
        """Prepare sequences for LSTM."""
        # Normalize weekday (0-6 to 0-1)
        self.full_data['weekday_norm'] = self.full_data['weekday_num'] / 6.0
        
        # Scale target
        cups_sold = self.full_data[self.target_col].values.reshape(-1, 1)
        self.scaler_y.fit(cups_sold)
        cups_sold_scaled = self.scaler_y.transform(cups_sold)
        
        # Create features: [cups_sold_scaled, weekday_norm]
        features = np.column_stack([cups_sold_scaled, self.full_data['weekday_norm'].values])
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            y.append(cups_sold_scaled[i+self.window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split indices
        train_end = len(self.train_data) - self.window_size
        val_end = train_end + len(self.val_data)
        
        self.X_train, self.y_train = X[:train_end], y[:train_end]
        self.X_val, self.y_val = X[train_end:val_end], y[train_end:val_end]
        self.X_test, self.y_test = X[val_end:], y[val_end:]
    
    def build_model(self) -> None:
        """Build the LSTM model."""
        input_size = 2  # cups_sold_scaled + weekday_norm
        self.model = LSTMModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rates=self.dropout_rates
        ).to(self.device)
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        patience: int = 15,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """Train the LSTM model with early stopping and LR scheduler."""
        # Datasets and loaders
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Scheduler and early stopping
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=patience // 2
        )
        best_val_loss = float('inf')
        patience_counter = 0
        history = []
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            
            train_loss /= len(train_loader.dataset)
            
            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    val_loss += criterion(outputs, y_batch).item() * len(X_batch)
            
            val_loss /= len(val_loader.dataset)
            
            history.append({'train_loss': train_loss, 'val_loss': val_loss})
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input data (if None, uses test data)
        
        Returns:
            Predictions in original scale
        """
        if X is None:
            X = self.X_test
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def forecast_future(self, n_steps: int, start_date: str) -> pd.DataFrame:
        """
        Forecast future values recursively.
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
        
        Returns:
            DataFrame with forecasted values
        """
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
        future_weekdays = future_dates.weekday.values
        
        # Get last window from full data
        last_data = self.full_data.tail(self.window_size).copy()
        last_data['weekday_norm'] = last_data['weekday_num'] / 6.0
        cups_sold_scaled = self.scaler_y.transform(last_data[[self.target_col]].values)
        last_window = np.concatenate([cups_sold_scaled, last_data[['weekday_norm']].values], axis=1)
        
        predictions = []
        current_window = last_window.copy()
        
        self.model.eval()
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
    
    def evaluate(self, predictions: Optional[np.ndarray] = None) -> Dict[str, float]:
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
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(epochs, batch_size, learning_rate, patience, verbose=verbose)
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics
