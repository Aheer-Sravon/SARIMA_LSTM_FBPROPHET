import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from forecasters import SARIMAXForecaster
sys.path.pop()

import pandas as pd

# Instantiate and run
model = SARIMAXForecaster(
    train_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'train.csv',
    val_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'validation.csv',
    test_path=Path(__file__).parent.parent.parent / 'data' / 'preprocessed' / 'test.csv'
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 4), verbose=True)

model.train()

# Plot diagnostics after fitting
model.plot_diagnostics(save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarimax_diagnostics_plot.png')

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Plot actual vs predicted
model.plot_actual_vs_predicted(
    predictions,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarimax_actual_vs_predicted.png'
)

# Future forecast example (provide exog_future with 'weekday_num' column)
exog_future = pd.DataFrame({'weekday_num': [5, 6, 0, 1, 2]})  # Sat (5), Sun (6), Mon (0), Tue (1), Wed (2) for 2025-02-08 to 12
future_preds = model.forecast_future(
    n_steps=5,
    start_date='2025-02-08',
    exog_future=exog_future
)
print(future_preds)

# Plot future forecast
model.plot_future_forecast(
    n_steps=5,
    start_date='2025-02-08',
    exog_future=exog_future,
    save_path=Path(__file__).parent.parent.parent / 'figures' / 'weather_forecast_plots' / 'sarimax_future_forecast.png'
)
