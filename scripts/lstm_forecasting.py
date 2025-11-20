# lstm_forecasting.py
from forecasters import LSTMForecaster
import pandas as pd

# Instantiate and run
model = LSTMForecaster(
    train_path='../data/preprocessed/train.csv',
    val_path='../data/preprocessed/validation.csv',
    test_path='../data/preprocessed/test.csv'
)

# Train the model (equivalent to optimize + train in SARIMAX)
model.train(epochs=100, batch_size=32, learning_rate=0.0005, patience=15, verbose=True)

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Future forecast example (no exogenous variables needed for LSTM, uses recursive with weekdays)
future_preds = model.forecast_future(n_steps=3, start_date='2025-02-08')
print(future_preds)
