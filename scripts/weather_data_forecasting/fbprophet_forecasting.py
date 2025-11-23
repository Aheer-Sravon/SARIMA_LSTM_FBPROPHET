import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import ProphetForecaster
sys.path.pop()

# Instantiate and run
model = ProphetForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'test.csv'
)

# Train the model (equivalent to optimize + train in SARIMAX)
model.train(
    growth='linear',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    verbose=True
)

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables needed for Prophet)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'prophet_future_forecast.png'
)
