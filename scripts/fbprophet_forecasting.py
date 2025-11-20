# prophet_forecasting.py
from forecasters import ProphetForecaster
import pandas as pd

# Instantiate and run
model = ProphetForecaster(
    train_path='../data/preprocessed/train.csv',
    val_path='../data/preprocessed/validation.csv',
    test_path='../data/preprocessed/test.csv'
)

# Train with parameters (no optimize needed for Prophet)
model.train(
    growth='linear',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    verbose=True
)

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Future forecast example (no exogenous variables for Prophet)
future_preds = model.forecast_future(n_steps=3, start_date='2025-02-08')
print(future_preds)

