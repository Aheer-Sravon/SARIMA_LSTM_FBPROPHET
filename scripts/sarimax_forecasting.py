from forecasters import SARIMAXForecaster
import pandas as pd

# Instantiate and run
model = SARIMAXForecaster(
    train_path='../data/preprocessed/train.csv',
    val_path='../data/preprocessed/validation.csv',
    test_path='../data/preprocessed/test.csv'
)

# Optimize with smaller ranges if needed
model.optimize(p_range=range(0, 3), verbose=True)

model.train()

predictions = model.predict()

metrics = model.evaluate(predictions)
print(metrics)

# Future forecast example (provide exog_future with 'weekday_num' column)
exog_future = pd.DataFrame({'weekday_num': [5, 6, 0]})  # Sat (5), Sun (6), Mon (0) for 2025-02-08 to 10
future_preds = model.forecast_future(n_steps=3, start_date='2025-02-08', exog_future=exog_future)
print(future_preds)
