"""
CNN-LSTM Forecaster for Time Series Prediction using PyTorch
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


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for time series forecasting.
    Architecture based on the TensorFlow notebook:
    - LSTM layer (128 units) with BatchNorm and Dropout
    - Conv1D layer (128 filters, kernel=3) with BatchNorm and Dropout
    - LSTM layer (128 units) with Dropout
    - Dense output layer
    """
    
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 128,
        cnn_filters: int = 128,
        cnn_kernel_size: int = 3,
        dropout_rates: List[float] = [0.3, 0.2, 0.1]
    ):
        super(CNNLSTMModel, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # Conv1D layer (causal padding)
        self.conv1 = nn.Conv1d(
            in_channels=lstm_hidden_size,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size - 1  # Causal padding
        )
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)  # (batch, seq, hidden)
        
        # Apply batch norm to the sequence
        lstm_out_transposed = lstm_out.transpose(1, 2)  # (batch, hidden, seq)
        lstm_out_normalized = self.bn1(lstm_out_transposed)
        lstm_out = lstm_out_normalized.transpose(1, 2)  # (batch, seq, hidden)
        lstm_out = self.dropout1(lstm_out)
        
        # Conv1D layer
        # Conv1D expects (batch, channels, length)
        conv_input = lstm_out.transpose(1, 2)  # (batch, hidden, seq)
        conv_out = self.conv1(conv_input)  # (batch, filters, seq+padding)
        
        # Remove future padding (causal)
        conv_out = conv_out[:, :, :-self.conv1.padding[0]]  # (batch, filters, seq)
        conv_out = self.relu(conv_out)
        conv_out = self.bn2(conv_out)
        conv_out = self.dropout2(conv_out)
        
        # Back to (batch, seq, features) for LSTM
        conv_out = conv_out.transpose(1, 2)  # (batch, seq, filters)
        
        # Second LSTM layer
        lstm_out, _ = self.lstm2(conv_out)  # (batch, seq, hidden)
        last_time_step = lstm_out[:, -1, :]  # (batch, hidden)
        last_time_step = self.dropout3(last_time_step)
        
        # Output
        out = self.fc(last_time_step)  # (batch, 1)
        
        return out


class CNNLSTMForecaster:
    """
    CNN-LSTM forecaster for cups_sold prediction using PyTorch.
    
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
        lstm_hidden_size: int = 128,
        cnn_filters: int = 128,
        cnn_kernel_size: int = 3,
        dropout_rates: List[float] = [0.3, 0.2, 0.1],
        seed: int = 150,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the CNN-LSTM forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            window_size: Number of time steps to look back
            lstm_hidden_size: Number of hidden units in LSTM layers
            cnn_filters: Number of filters in CNN layer
            cnn_kernel_size: Kernel size for CNN
            dropout_rates: List of dropout rates for each layer
            seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.window_size = window_size
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.dropout_rates = dropout_rates
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.full_data: Optional[pd.DataFrame] = None
        
        # Scalers
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Model
        self.model: Optional[CNNLSTMModel] = None
        self.best_model_state: Optional[dict] = None
        
        # Data arrays
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        self.load_data()
        self.prepare_data()
    
    def load_data(self) -> None:
        """Load and preprocess training, validation, and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # Parse date column
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Combine all data
        self.full_data = pd.concat([self.train_data, self.val_data, self.test_data])
    
    def prepare_data(self) -> None:
        """Prepare data for training: create windows and scale."""
        # Normalize weekday (0-6 scaled to 0-1)
        self.full_data['weekday_norm'] = self.full_data['weekday_num'] / 6.0
        
        # Scale target
        cups_sold_scaled = self.scaler_y.fit_transform(self.full_data[[self.target_col]])
        
        # Combine features: scaled cups_sold and normalized weekday
        X = np.concatenate([cups_sold_scaled, self.full_data[['weekday_norm']].values], axis=1)
        y = cups_sold_scaled
        
        # Create windowed dataset
        X_data, y_data = self._create_dataset(X, y, self.window_size)
        
        # Split into train, val, test (80%, 10%, 10%)
        train_size = int(len(X_data) * 0.8)
        val_size = int(len(X_data) * 0.1)
        
        self.X_train = X_data[:train_size]
        self.y_train = y_data[:train_size]
        
        self.X_val = X_data[train_size:train_size + val_size]
        self.y_val = y_data[train_size:train_size + val_size]
        
        self.X_test = X_data[train_size + val_size:]
        self.y_test = y_data[train_size + val_size:]
    
    def _create_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window dataset."""
        X_data, y_data = [], []
        for i in range(len(X) - window_size):
            X_data.append(X[i:i + window_size])
            y_data.append(y[i + window_size])
        return np.array(X_data), np.array(y_data)
    
    def build_model(self) -> None:
        """Build the CNN-LSTM model."""
        input_size = self.X_train.shape[2]  # Number of features
        self.model = CNNLSTMModel(
            input_size=input_size,
            lstm_hidden_size=self.lstm_hidden_size,
            cnn_filters=self.cnn_filters,
            cnn_kernel_size=self.cnn_kernel_size,
            dropout_rates=self.dropout_rates
        ).to(self.device)
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0005,
        patience: int = 10,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the CNN-LSTM model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            patience: Early stopping patience
            reduce_lr_patience: Patience for learning rate reduction
            reduce_lr_factor: Factor to reduce learning rate
            min_lr: Minimum learning rate
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            self.build_model()
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        val_dataset = TimeSeriesDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=reduce_lr_factor,
            patience=reduce_lr_patience, min_lr=min_lr
        )
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
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
            history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
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
        cups_sold_scaled = self.scaler_y.transform(last_data[[self.target_col]])
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
        patience: int = 10,
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
