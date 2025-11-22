# sarima_forecasting.py
from forecasters import SARIMAForecaster

# Instantiate and run
model = SARIMAForecaster(
    train_path='../data/preprocessed/train.csv',
    val_path='../data/preprocessed/validation.csv',
    test_path='../data/preprocessed/test.csv'
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 4), verbose=True)

model.fit()

# Plot diagnostics after fitting
model.plot_diagnostics(save_path='../figures/sarima_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path='../figures/sarima_actual_vs_predicted.png'
)

# Future forecast example (no exogenous variables for SARIMA)
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08'
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    save_path='../figures/sarima_future_forecast.png'
)
