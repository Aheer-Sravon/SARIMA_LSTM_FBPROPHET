import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import CNNLSTMForecaster
sys.path.pop()

# Instantiate and run
model = CNNLSTMForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'test.csv'
)

# Train the model (equivalent to optimize + train in SARIMAX)
model.train(epochs=100, batch_size=32, learning_rate=0.0005, patience=15, verbose=True)

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'cnn_lstm_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'cnn_lstm_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables needed for CNN-LSTM, uses recursive with weekdays)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'cnn_lstm_future_forecast.png'
)
