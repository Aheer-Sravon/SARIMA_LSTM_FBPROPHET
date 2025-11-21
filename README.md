# Time Series Forecasting for Cups Sold

This repository contains Python scripts and Jupyter notebooks for forecasting daily "cups_sold" using various time series models. The models include SARIMA/SARIMAX, LSTM, CNN-LSTM, and Facebook Prophet. The code is designed for univariate time series prediction, incorporating features like weekday normalization for deep learning models and seasonal parameters for statistical models.

The project demonstrates model training, hyperparameter optimization (where applicable), recursive forecasting on a test set, evaluation metrics (MAE, RMSE, MAPE), and future predictions. It is based on CSV data in the `data/` directory, with preprocessed splits (train.csv, validation.csv, test.csv) in `data/preprocessed/`.

## Features
- **Models Implemented**:
  - SARIMA/SARIMAX: Statistical autoregressive models with seasonal components.
  - LSTM: Recurrent neural network for sequence prediction.
  - CNN-LSTM: Hybrid convolutional-recurrent model for capturing spatial and temporal patterns.
  - Prophet: Additive model for forecasting with seasonality and trends.
- **Forecasting Capabilities**: Recursive one-step-ahead predictions on test data and multi-step future forecasts.
- **Evaluation**: Computes MAE, RMSE, and MAPE; handles low-value data sensitivities.
- **Notebooks**: Exploratory analysis and model prototypes (e.g., SARIMA and SARIMAX.ipynb, LSTM_final.ipynb, CNN-LSTM_final.ipynb, Summary Notebook_final.ipynb for comparisons).
- **Scripts**: Modular forecaster classes in `scripts/forecasters/` (e.g., sarima.py) and executable wrappers in `scripts/` (e.g., sarima_forecasting.py).

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not provided, install the following packages manually:
   ```
   pip install numpy pandas torch scikit-learn statsmodels prophet tqdm matplotlib
   ```
   - Note: PyTorch may require platform-specific installation (e.g., for GPU support: see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).
   - Prophet requires additional setup on some systems (e.g., `pip install pystan` first on older versions).

## Usage
### Data Preparation
- The `data/` directory contains three subfolders:
  - `preprocessed/`: Contains processed data ready for modeling (train.csv, validation.csv, test.csv).
  - `intermediate/`: For any intermediate data processing outputs (if applicable).
  - `raw/`: Original raw data files.
- Ensure 'date' is in 'YYYY-MM-DD' format and set as index where required in the CSVs.

### Running Scripts
Each model has a forecasting script in the `scripts/` directory. Run them from the root or `scripts/` directory (adjust paths if needed). The forecaster classes are imported from the `forecasters` module.

- **SARIMA**:
  ```
  python scripts/sarima_forecasting.py
  ```
  - Optimizes parameters via grid search, fits the model, predicts on test set, evaluates, and forecasts future steps.

- **LSTM**:
  ```
  python scripts/lstm_forecasting.py
  ```
  - Trains with early stopping and learning rate reduction.

- **CNN-LSTM**:
  ```
  python scripts/cnn_lstm_forecasting.py
  ```
  - Similar to LSTM but with convolutional layers.

- **Prophet**:
  ```
  python scripts/fbprophet_forecasting.py
  ```
  - Trains with configurable seasonality and changepoint priors.

Example output includes metrics (e.g., {'MAE': 3.37, 'RMSE': 4.61, 'MAPE': 50.48}) and future predictions as a DataFrame.

### Notebooks
- **SARIMA and SARIMAX.ipynb**: Explores SARIMA/SARIMAX fitting, diagnostics, and forecasting.
- **LSTM_final.ipynb**: LSTM model training and evaluation.
- **CNN-LSTM_final.ipynb**: CNN-LSTM hybrid implementation.
- **Summary Notebook_final.ipynb**: Compares all models with visualizations (e.g., bar plots of metrics).

Open notebooks with Jupyter:
```
jupyter notebook notebooks/
```

## Project Structure
```
├── data/
│   ├── preprocessed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── intermediate/
│   │   └── (intermediate files)
│   └── raw/
│       └── (raw data files)
├── notebooks/
│   ├── SARIMA and SARIMAX.ipynb
│   ├── LSTM_final.ipynb
│   ├── CNN-LSTM_final.ipynb
│   └── Summary Notebook_final.ipynb
├── scripts/
│   ├── forecasters/
│   │   ├── sarima.py                # SARIMA forecaster class
│   │   ├── lstm.py                  # LSTM forecaster class (PyTorch)
│   │   ├── cnn_lstm.py              # CNN-LSTM forecaster class (PyTorch)
│   │   └── fbprophet.py             # Prophet forecaster class
│   ├── sarima_forecasting.py        # Executable for SARIMA
│   ├── lstm_forecasting.py          # Executable for LSTM
│   ├── cnn_lstm_forecasting.py      # Executable for CNN-LSTM
│   └── fbprophet_forecasting.py     # Executable for Prophet
├── README.md
└── requirements.txt
```

## Notes
- **Performance Considerations**: MAPE may appear high (40-60%) due to low-scale data (e.g., single-digit sales days). Focus on MAE/RMSE for better insight.
- **Customization**: Adjust hyperparameters in scripts (e.g., epochs for LSTM, p_range for SARIMA).
- **Limitations**: Models are univariate; extend with exogenous variables (e.g., weather) for better accuracy.
- **Environment**: Tested on Python 3.12+. Deep learning models use PyTorch (ported from TensorFlow in notebooks).

## Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a pull request with improvements (e.g., new models, bug fixes).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
