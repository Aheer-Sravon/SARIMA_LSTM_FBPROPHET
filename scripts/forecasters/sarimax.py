import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

# Suppress convergence and value warnings by category
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Suppress specific UserWarnings by message (using regex to match exactly)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found.*")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.*")

from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
from tqdm import tqdm

class SARIMAXForecaster:
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
    forecaster for time series prediction.
   
    Based on notebook implementation:
    - Uses weekday_num as exogenous variable
    - Recursive forecasting for test predictions
    - Grid search optimization for (p, q, P, Q) parameters
    - Fixed d=0, D=0, s=7 (weekly seasonality)
   
    Attributes:
        train_path: Path to training data CSV file
        val_path: Path to validation data CSV file
        test_path: Path to test data CSV file
        best_order: Optimal (p, d, q) parameters
        best_s_order: Optimal seasonal (P, D, Q, s) parameters
        model: Fitted SARIMAX model
        exog_cols: List of exogenous variable column names
    """
   
    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        test_path: Union[str, Path],
        target_col: str = 'cups_sold',
        exog_cols: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the SARIMAX forecaster.
       
        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file
            test_path: Path to test CSV file
            target_col: Name of the target column to forecast
            exog_cols: List of exogenous variable column names (default: ['weekday_num'])
        """
        self.train_path: Path = Path(train_path)
        self.val_path: Path = Path(val_path)
        self.test_path: Path = Path(test_path)
        self.target_col: str = target_col
        self.exog_cols: List[str] = exog_cols if exog_cols is not None else ['weekday_num']
       
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.train_val_data: Optional[pd.DataFrame] = None
        self.full_data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.exog: Optional[pd.DataFrame] = None
        self.full_exog: Optional[pd.DataFrame] = None
       
        self.best_order: Optional[Tuple[int, int, int]] = None
        self.best_s_order: Optional[Tuple[int, int, int, int]] = None
        self.model: Optional[SARIMAX] = None
        self.res: Optional[Any] = None
       
        self.load_data()
   
    def load_data(self) -> None:
        """Load and preprocess training, validation, and test data."""
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
       
        # Parse date column and set as index with frequency
        for df in [self.train_data, self.val_data, self.test_data]:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.asfreq('D').ffill()  # Set daily freq and handle gaps
       
        # Combine train and validation for model training
        self.train_val_data = pd.concat([self.train_data, self.val_data]).sort_index()
       
        # Combine all data for sequential prediction
        self.full_data = pd.concat([self.train_val_data, self.test_data]).sort_index()
       
        # Set target and exogenous variables
        self.target = self.train_val_data[self.target_col]
        self.exog = self.train_val_data[self.exog_cols]
        self.full_exog = self.full_data[self.exog_cols]
   
    def optimize(
        self,
        p_range: range = range(0, 3),  # Smaller default for efficiency
        q_range: range = range(0, 3),
        P_range: range = range(0, 3),
        Q_range: range = range(0, 3),
        d: int = 0,
        D: int = 0,
        s: int = 7,
        verbose: bool = True
    ) -> None:
        """
        Optimize SARIMAX parameters using grid search and AIC.
        Based on notebook: tests all combinations of (p, q, P, Q) with fixed d=0, D=0, s=7
       
        Args:
            p_range: Range for AR (AutoRegressive) parameter
            q_range: Range for MA (Moving Average) parameter
            P_range: Range for seasonal AR parameter
            Q_range: Range for seasonal MA parameter
            d: Degree of differencing (default: 0 as per notebook)
            D: Degree of seasonal differencing (default: 0 as per notebook)
            s: Seasonal period (default: 7 for weekly data)
            verbose: Whether to show progress bar
        """
        parameters = list(product(p_range, q_range, P_range, Q_range))
        results: List[List[Union[Tuple[int, int, int, int], float]]] = []
       
        iterator = tqdm(parameters, desc="Optimizing SARIMAX") if verbose else parameters
       
        for param in iterator:
            try:
                model = SARIMAX(
                    self.target,
                    exog=self.exog,
                    order=(param[0], d, param[1]),
                    seasonal_order=(param[2], D, param[3], s),
                    simple_differencing=False
                )
                res = model.fit(disp=False, maxiter=500, method='nm')  # Better convergence
                aic: float = res.aic
                results.append([param, aic])
            except Exception:
                continue
       
        if results:
            result_df: pd.DataFrame = pd.DataFrame(results, columns=['params', 'aic'])
            result_df = result_df.sort_values('aic').reset_index(drop=True)
            best_param: Tuple[int, int, int, int] = result_df['params'].iloc[0]
           
            self.best_order = (best_param[0], d, best_param[1])
            self.best_s_order = (best_param[2], D, best_param[3], s)
           
            if verbose:
                print(f"\nBest order: {self.best_order}")
                print(f"Best seasonal order: {self.best_s_order}")
                print(f"Best AIC: {result_df['aic'].iloc[0]:.2f}")
                print(f"\nTop 5 parameter combinations:")
                print(result_df.head())
   
    def train(self) -> None:
        """Train the SARIMAX model with optimized parameters."""
        if self.best_order is None or self.best_s_order is None:
            self.optimize()
       
        if self.best_order and self.best_s_order:
            self.model = SARIMAX(
                self.target,
                exog=self.exog,
                order=self.best_order,
                seasonal_order=self.best_s_order,
                simple_differencing=False
            )
            self.res = self.model.fit(disp=False, maxiter=500, method='nm')  # Better convergence
   
    def predict(self, window: int = 1) -> List[float]:
        """
        Generate predictions for test set using recursive forecasting.
        Based on notebook's recursive_forecast function.
       
        Args:
            window: Step size for recursive forecasting (default: 1)
       
        Returns:
            List of predictions for test period
        """
        if self.best_order is None or self.best_s_order is None:
            raise ValueError("Model must be optimized before making predictions")
       
        full_target: pd.Series = self.full_data[self.target_col]
        train_len: int = len(self.train_val_data)
        horizon: int = len(self.test_data)
        total_len: int = train_len + horizon
       
        predictions: List[float] = []
       
        # Recursive forecasting
        for i in range(train_len, total_len, window):
            try:
                model = SARIMAX(
                    full_target[:i],
                    exog=self.full_exog[:i],
                    order=self.best_order,
                    seasonal_order=self.best_s_order,
                    simple_differencing=False
                )
                res = model.fit(disp=False, maxiter=500, method='nm')
               
                # Corrected: Use get_forecast for out-of-sample predictions
                oos_exog = self.full_exog[i:i+window]
                oos_pred = res.get_forecast(steps=len(oos_exog), exog=oos_exog).predicted_mean.values
                predictions.extend(oos_pred)
               
            except Exception as e:
                print(f"Warning: Prediction failed at step {i}: {e}")
                # Append repeated value for window size
                last_pred = predictions[-1] if predictions else 0.0
                predictions.extend([last_pred] * min(window, total_len - i))
       
        # Trim to exact horizon length
        return predictions[:horizon]
   
    def forecast_future(self, n_steps: int, start_date: str, exog_future: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast future values with exogenous variables.
       
        Args:
            n_steps: Number of steps to forecast
            start_date: Start date for forecast (format: 'YYYY-MM-DD')
            exog_future: DataFrame with exogenous variables for future periods (must have n_steps rows)
       
        Returns:
            DataFrame with forecasted values
        """
        if self.res is None:
            raise ValueError("Model must be trained before forecasting")
       
        # Validate exog_future
        if len(exog_future) != n_steps:
            raise ValueError(f"exog_future must have {n_steps} rows")
       
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=n_steps, freq='D')
       
        # Make predictions
        forecast = self.res.get_forecast(steps=n_steps, exog=exog_future)
        predictions = forecast.predicted_mean.values
       
        return pd.DataFrame({
            'date': future_dates,
            'predicted_cups_sold': predictions
        })
   
    def evaluate(self, predictions: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
       
        Args:
            predictions: Pre-computed predictions (if None, will compute them)
       
        Returns:
            Dictionary containing MAE, RMSE, and MAPE metrics
        """
        if predictions is None:
            predictions = self.predict()
       
        actual: np.ndarray = self.test_data[self.target_col].values
        predictions_array = np.array(predictions)
       
        mae: float = mean_absolute_error(actual, predictions_array)
        rmse: float = np.sqrt(mean_squared_error(actual, predictions_array))

        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape: float = np.mean(
                np.abs((actual[non_zero_mask] - predictions_array[non_zero_mask]) / 
                actual[non_zero_mask])
            ) * 100
        else:
            mape = 0.0  # If all actual values are zero, MAPE is undefined
       
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
   
    def fit_and_evaluate(self, verbose: bool = True) -> Tuple[List[float], Dict[str, float]]:
        """
        Fit the model and evaluate on test set.
       
        Returns:
            Tuple of (predictions, metrics dictionary)
        """
        self.optimize(verbose=verbose)
        self.train()
        predictions = self.predict()
        metrics = self.evaluate(predictions)
       
        return predictions, metrics
   
    def get_summary(self) -> str:
        """Get summary of the fitted model."""
        if self.res is None:
            return "Model not yet trained"
        return str(self.res.summary())
