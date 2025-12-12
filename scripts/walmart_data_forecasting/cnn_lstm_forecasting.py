import warnings
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
plt.ion()

# Suppress warnings if needed
warnings.filterwarnings("ignore")

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for time series forecasting.
    """
    
    def __init__(self, input_size, lstm_hidden_size=128, cnn_filters=128, 
                 cnn_kernel_size=3, dropout_rates=[0.3, 0.2, 0.1]):
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
            padding=cnn_kernel_size - 1
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
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.dropout1(lstm_out)
        
        # Conv1D layer
        conv_input = lstm_out.permute(0, 2, 1)
        conv_out = self.conv1(conv_input)
        conv_out = self.relu(conv_out)
        conv_out = conv_out[:, :, :- (self.conv1.kernel_size[0] - 1)]
        conv_out = self.bn2(conv_out)
        conv_out = self.dropout2(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        
        # Second LSTM layer
        lstm_out, _ = self.lstm2(conv_out)
        lstm_out = self.dropout3(lstm_out)
        
        # Take last timestep
        last_timestep = lstm_out[:, -1, :]
        out = self.fc(last_timestep)
        
        return out

class CNNLSTMForecaster:
    """
    CNN-LSTM forecaster for cups_sold prediction using PyTorch.
    
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        seq_length: Length of input sequences
        model: CNN-LSTM model
    """
    
    def __init__(self, train_path, test_path, 
                 target_col='weekly_sales', seq_length=7, exog_cols=None):
        """
        Initialize the CNN-LSTM forecaster.
        
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            seq_length: Length of input sequences (default: 7 for weekly)
            exog_cols: List of exogenous feature columns (optional)
        """
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.target_col = target_col
        self.seq_length = seq_length
        self.exog_cols = exog_cols if exog_cols is not None else []
        
        self.train_data = None
        self.test_data = None
        self.train_val_data = None
        
        self.scaler_y = MinMaxScaler()
        self.scaler_exog = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses = []
        self.val_losses = []
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.load_data()
        self.prepare_data()
    
    def load_data(self):
        """Load and preprocess data."""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)  # Changed: removed val_data
        
        # Parse dates
        for df in [self.train_data, self.test_data]:  # Removed val_data
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif df.index.name != 'date':
                df.index = pd.RangeIndex(start=0, stop=len(df))
        
        # No validation, use train data directly
        self.train_val_data = self.train_data  # Changed
    
    def prepare_data(self):
        """Prepare input sequences with weekday feature and exogenous features."""
        # Combine all data for sequence creation
        full_data = pd.concat([self.train_data, self.test_data])
        
        # Add weekday feature if we have datetime index
        if isinstance(full_data.index, pd.DatetimeIndex):
            full_data['weekday_norm'] = full_data.index.weekday / 6.0
        else:
            # Create a dummy weekday feature if no datetime index
            full_data['weekday_norm'] = np.tile(np.arange(7) / 6.0, 
                                                len(full_data) // 7 + 1)[:len(full_data)]
        
        # Scale target
        train_y = self.train_data[[self.target_col]].values
        self.scaler_y.fit(train_y)
        y_scaled = self.scaler_y.transform(full_data[[self.target_col]])
        
        # Combine scaled target and weekday
        features_list = [y_scaled, full_data[['weekday_norm']].values]
        
        # Add scaled exogenous features if provided
        if self.exog_cols:
            self.scaler_exog = {}
            for col in self.exog_cols:
                if col in full_data.columns:
                    scaler = MinMaxScaler()
                    train_exog = self.train_data[[col]].values
                    scaler.fit(train_exog)
                    self.scaler_exog[col] = scaler
                    exog_scaled = scaler.transform(full_data[[col]].values)
                    features_list.append(exog_scaled)
        
        # Combine all features
        features = np.hstack(features_list)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(y_scaled[i+self.seq_length, 0])  # Predict next target value
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split to train/test only
        train_end = len(self.train_data)
        
        self.X_train = X[:train_end - self.seq_length]
        self.y_train = y[:train_end - self.seq_length]
        
        self.X_test = X[train_end - self.seq_length:]
        self.y_test = y[train_end - self.seq_length:]
        
        # Create validation split from training data (80/20)
        val_size = int(0.2 * len(self.X_train))
        self.X_val = self.X_train[:val_size]
        self.y_val = self.y_train[:val_size]
        self.X_train = self.X_train[val_size:]
        self.y_train = self.y_train[val_size:]
    
    def train(self, epochs=100, batch_size=32, learning_rate=0.0005, 
              patience=10, verbose=True):
        """Train the CNN-LSTM model with early stopping."""
        input_size = self.X_train.shape[2]  # features
        
        self.model = CNNLSTMModel(input_size=input_size).to(self.device)
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
                torch.save(self.model.state_dict(), 'best_cnnlstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cnnlstm_model.pth'))
    
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
        
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
            exog_future: DataFrame with future exogenous values (optional)
        
        Returns:
            DataFrame with forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Get all data for the last window
        full_data = pd.concat([self.train_val_data, self.test_data])
        
        # Handle start_date: if not a datetime, create one
        try:
            start_date_dt = pd.to_datetime(start_date)
        except:
            # If start_date is not a valid date, use the day after last data point
            if hasattr(full_data.index, '__len__') and len(full_data.index) > 0:
                last_date = full_data.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    start_date_dt = last_date + pd.Timedelta(days=1)
                else:
                    start_date_dt = pd.Timestamp.now()
            else:
                start_date_dt = pd.Timestamp.now()
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date_dt, periods=n_steps, freq='D')
        future_weekdays = future_dates.weekday.values / 6.0
        
        # Get last window from scaled data
        y_scaled = self.scaler_y.transform(full_data[[self.target_col]])
        last_window_scaled = y_scaled[-self.seq_length:]
        
        # Get last weekdays from data or create dummy
        if isinstance(full_data.index, pd.DatetimeIndex):
            last_weekdays = full_data.index[-self.seq_length:].weekday.values / 6.0
        else:
            # Create dummy weekdays if no datetime index
            last_weekdays = np.tile(np.arange(7) / 6.0, 
                                    self.seq_length // 7 + 1)[:self.seq_length]
        
        # Start with base features (target + weekday)
        current_window = np.hstack([last_window_scaled, last_weekdays.reshape(-1, 1)])
        
        # Add last exogenous features if available
        if self.exog_cols:
            for col in self.exog_cols:
                if col in full_data.columns and col in self.scaler_exog:
                    exog_values = full_data[col].values[-self.seq_length:]
                    exog_scaled = self.scaler_exog[col].transform(exog_values.reshape(-1, 1))
                    current_window = np.hstack([current_window, exog_scaled])
        
        predictions = []
        self.model.eval()
        
        for i in range(n_steps):
            # Prepare input
            X_input = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred_scaled = self.model(X_input).cpu().numpy()[0, 0]
            
            # Inverse transform
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Create new step with predicted target + weekday
            new_step = [[pred_scaled, future_weekdays[i]]]
            
            # Add exogenous values for this step if available
            if self.exog_cols:
                for col in self.exog_cols:
                    if col in self.scaler_exog:
                        if exog_future is not None and i < len(exog_future) and col in exog_future.columns:
                            exog_val = exog_future.iloc[i][col]
                            exog_scaled = self.scaler_exog[col].transform([[exog_val]])[0, 0]
                        else:
                            # If no future exog, repeat the last one from training data
                            exog_val = full_data[col].values[-1]
                            exog_scaled = self.scaler_exog[col].transform([[exog_val]])[0, 0]
                        new_step[0].append(exog_scaled)
            
            new_step = np.array(new_step)
            current_window = np.vstack([current_window[1:], new_step])
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_weekly_sales': predictions  # Use correct column name
        })
    
    def evaluate(self, predictions=None):
        """
        Evaluate model performance on test set.
        
        Args:
            predictions: Pre-computed predictions (if None, will compute them)
        
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
    
    def fit_and_evaluate(self, epochs=100, batch_size=32, learning_rate=0.0005,
                         patience=10, verbose=True):
        """
        Fit the model and evaluate on test set.
        
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.train(epochs, batch_size, learning_rate, patience, verbose=verbose)
        predictions = self.predict()
        metrics = self.evaluate(predictions)
        
        return predictions, metrics
    
    def plot_diagnostics(self, save_path=None):
        """Plot training and validation loss history."""
        if not self.train_losses:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss (CNN-LSTM)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Diagnostics plot saved to {save_path}")
        
        plt.show()
        plt.pause(5)
        plt.close()
    
    def plot_actual_vs_predicted(self, predictions=None, fig_save_path=None, csv_save_path=None):
        """Plot actual vs predicted values on the test set.
        
        Args:
            predictions: Pre-computed predictions (if None, will compute them)
            fig_save_path: Path to save the figure (optional)
            csv_save_path: Path to save the CSV data (optional)
        """
        if predictions is None:
            predictions = self.predict()
        
        test_dates = self.test_data.index
        actual = self.test_data[self.target_col].values
        
        # Save to CSV if csv_save_path is provided
        pd.DataFrame({
            'Date': test_dates,
            'Actual': actual,
            'Predicted': predictions
        }).to_csv(csv_save_path, index=False)
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', color='blue')
        plt.plot(test_dates, predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales' if self.target_col == 'weekly_sales' else 'Cups Sold')
        plt.title('Actual vs Predicted (CNN-LSTM)')
        plt.legend()
        plt.grid(True)
        
        # Save figure if fig_save_path is provided
        if fig_save_path:
            plt.savefig(fig_save_path)
            print(f"Actual vs Predicted plot saved to {fig_save_path}")
        
        plt.show()
        plt.pause(5)
        plt.close()
    
    def plot_future_forecast(self, n_steps=30, start_date=None, save_path=None):
        """Plot future forecast starting from the end of the test data or a specified date."""
        if start_date is None:
            if hasattr(self.test_data.index, '__len__') and len(self.test_data.index) > 0:
                last_date = self.test_data.index[-1]
                if isinstance(last_date, pd.Timestamp):
                    start_date = str(last_date + pd.Timedelta(days=1))
                else:
                    start_date = str(pd.Timestamp.now())
            else:
                start_date = str(pd.Timestamp.now())
        
        forecast_df = self.forecast_future(n_steps, start_date)
        
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_weekly_sales'], label='Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Predicted Weekly Sales')
        plt.title('Future Forecast (CNN-LSTM)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Future Forecast plot saved to {save_path}")
        
        plt.show()
        plt.pause(5)
        plt.close()

# Define paths
base_data_path = Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'walmart'
base_fig_path = Path(__file__).parent.parent.parent / 'figures' / 'walmart_forecast_plots'

base_data_path.mkdir(parents=True, exist_ok=True)
base_fig_path.mkdir(parents=True, exist_ok=True)

# Load and merge data
df = pd.read_csv(base_data_path / 'walmart.csv')

stores = [1, 3]

for store in stores:
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
    
    # Split the data chronologically - LAST 50 points for testing
    n = len(df_store)
    test_size = 7  # Fixed 50 test points
    train_size = n - test_size

    train = df_store.iloc[:train_size]
    test = df_store.iloc[train_size:]

     # Save splits to CSV files for the store (NO VALIDATION)
    train_path = base_data_path / f'train_{store}.csv'
    test_path = base_data_path / f'test_{store}.csv'

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    # Instantiate and run for this store
    model = CNNLSTMForecaster(
        train_path=train_path,
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
    diag_path = base_fig_path / f'cnn_lstm_diagnostics_plot_{store}.png'
    model.plot_diagnostics(save_path=diag_path)
    
    predictions = model.predict()
    
    metrics = model.evaluate(predictions)
    print(f"\nStore {store} metrics: {metrics}")
    
    # Plot actual vs predicted
    avp_path = base_fig_path / f'cnn_lstm_actual_vs_predicted_{store}.png'
    model.plot_actual_vs_predicted(
        predictions,
        fig_save_path=avp_path,
        csv_save_path=f"../log/walmart_cnn_lstm_actual_vs_predicted_{store}.csv"
    )
    
    # Future forecast example
    # Use last exog values repeated for future
    if hasattr(model, 'exog_cols') and model.exog_cols:
        exog_last = model.test_data[model.exog_cols].iloc[-1]
        exog_future = pd.DataFrame([exog_last] * 5, columns=model.exog_cols)
        
        future_preds = model.forecast_future(
            n_steps=5,
            start_date='2012-11-02',
            exog_future=exog_future
        )
    else:
        future_preds = model.forecast_future(
            n_steps=5,
            start_date='2012-11-02'
        )
    print(f"\nStore {store} future predictions:\n{future_preds}")

    # Plot future forecast
    ff_path = base_fig_path / f'cnn_lstm_future_forecast_{store}.png'
    model.plot_future_forecast(
        n_steps=5,
        start_date='2012-11-02',
        save_path=ff_path
    )
