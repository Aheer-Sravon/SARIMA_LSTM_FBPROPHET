# sarima_forecasting.py
from forecasters import SARIMAForecaster
import pandas as pd

# Instantiate and run
model = SARIMAForecaster(
    train_path='../data/preprocessed/train.csv',
    val_path='../data/preprocessed/validation.csv',
    test_path='../data/preprocessed/test.csv'
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 3), verbose=True)

model.fit()

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Future forecast example (no exogenous variables for SARIMA)
future_preds = model.forecast_future(n_steps=3, start_date='2025-02-08')
print(future_preds)
