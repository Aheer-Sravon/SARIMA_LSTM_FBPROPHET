"""
Example usage of all forecasting models for cups_sold prediction.

This script demonstrates how to use each forecaster:
1. LSTM Forecaster (PyTorch)
2. CNN-LSTM Forecaster (PyTorch)
3. SARIMA Forecaster
4. SARIMAX Forecaster
5. Prophet Forecaster

Each example includes:
- Model initialization
- Training
- Prediction
- Evaluation
- Future forecasting
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import forecasters
from forecasters import LSTMForecaster, CNNLSTMForecaster, SARIMAForecaster, SARIMAXForecaster, ProphetForecaster


def example_lstm():
    """Example usage of LSTM forecaster."""
    print("=" * 80)
    print("LSTM Forecaster Example (PyTorch)")
    print("=" * 80)
    
    # Initialize forecaster
    forecaster = LSTMForecaster(
        train_path='../data/preprocessed/train.csv',
        val_path='../data/preprocessedvalidation.csv',
        test_path='test.csv',
        target_col='cups_sold',
        window_size=7,
        hidden_sizes=[128, 64, 128],
        dropout_rates=[0.3, 0.2, 0.1],
        seed=250
    )
    
    # Train and evaluate
    print("\nTraining LSTM model...")
    predictions, metrics = forecaster.fit_and_evaluate(
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        patience=15,
        verbose=True
    )
    
    # Print metrics
    print("\n" + "=" * 50)
    print("Test Set Performance:")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50)
    
    # Forecast future 7 days
    print("\nForecasting future 7 days (Feb 01-07, 2025)...")
    future_forecast = forecaster.forecast_future(n_steps=7, start_date='2025-02-01')
    print(future_forecast)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    actual = forecaster.scaler_y.inverse_transform(forecaster.y_test).flatten()
    plt.plot(range(len(actual)), actual, label='Actual', marker='o', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cups Sold', fontsize=12)
    plt.title('LSTM: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'lstm_predictions.png'")
    
    return forecaster, predictions, metrics


def example_cnn_lstm():
    """Example usage of CNN-LSTM forecaster."""
    print("\n" + "=" * 80)
    print("CNN-LSTM Forecaster Example (PyTorch)")
    print("=" * 80)
    
    # Initialize forecaster
    forecaster = CNNLSTMForecaster(
        train_path='train.csv',
        val_path='validation.csv',
        test_path='test.csv',
        target_col='cups_sold',
        window_size=7,
        lstm_hidden_size=128,
        cnn_filters=128,
        cnn_kernel_size=3,
        dropout_rates=[0.3, 0.2, 0.1],
        seed=150
    )
    
    # Train and evaluate
    print("\nTraining CNN-LSTM model...")
    predictions, metrics = forecaster.fit_and_evaluate(
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        patience=10,
        verbose=True
    )
    
    # Print metrics
    print("\n" + "=" * 50)
    print("Test Set Performance:")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50)
    
    # Forecast future 7 days
    print("\nForecasting future 7 days (Feb 01-07, 2025)...")
    future_forecast = forecaster.forecast_future(n_steps=7, start_date='2025-02-01')
    print(future_forecast)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    actual = forecaster.scaler_y.inverse_transform(forecaster.y_test).flatten()
    plt.plot(range(len(actual)), actual, label='Actual', marker='o', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cups Sold', fontsize=12)
    plt.title('CNN-LSTM: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cnn_lstm_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'cnn_lstm_predictions.png'")
    
    return forecaster, predictions, metrics


def example_sarima():
    """Example usage of SARIMA forecaster."""
    print("\n" + "=" * 80)
    print("SARIMA Forecaster Example")
    print("=" * 80)
    
    # Initialize forecaster
    forecaster = SARIMAForecaster(
        train_path='../data/preprocessed/train.csv',
        val_path='../data/preprocessed/validation.csv',
        test_path='../data/preprocessed/test.csv',
        target_col='cups_sold'
    )
    
    # Optimize parameters (this may take a while)
    print("\nOptimizing SARIMA parameters...")
    forecaster.optimize(
        p_range=range(0, 4),
        q_range=range(0, 4),
        P_range=range(0, 4),
        Q_range=range(0, 4),
        d=1,
        D=0,
        s=7,
        verbose=True
    )
    
    # Train and evaluate
    print("\nTraining SARIMA model...")
    forecaster.train()
    predictions = forecaster.predict()
    metrics = forecaster.evaluate(predictions)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("Test Set Performance:")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50)
    
    # Print model summary
    print("\nModel Summary:")
    print(forecaster.get_summary())
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    actual = forecaster.test['cups_sold'].values
    plt.plot(range(len(actual)), actual, label='Actual', marker='o', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cups Sold', fontsize=12)
    plt.title('SARIMA: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarima_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'sarima_predictions.png'")
    
    return forecaster, predictions, metrics


def example_sarimax():
    """Example usage of SARIMAX forecaster."""
    print("\n" + "=" * 80)
    print("SARIMAX Forecaster Example (with exogenous variables)")
    print("=" * 80)
    
    # Initialize forecaster with exogenous variables (weekday_num)
    forecaster = SARIMAXForecaster(
        train_path='train.csv',
        val_path='validation.csv',
        test_path='test.csv',
        target_col='cups_sold',
        exog_cols=['weekday_num']
    )
    
    # Optimize parameters (this may take a while)
    print("\nOptimizing SARIMAX parameters...")
    forecaster.optimize(
        p_range=range(0, 6),
        q_range=range(0, 6),
        P_range=range(0, 6),
        Q_range=range(0, 6),
        d=0,
        D=0,
        s=7,
        verbose=True
    )
    
    # Train and evaluate
    print("\nTraining SARIMAX model...")
    forecaster.train()
    predictions = forecaster.predict()
    metrics = forecaster.evaluate(predictions)
    
    # Print metrics
    print("\n" + "=" * 50)
    print("Test Set Performance:")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50)
    
    # Forecast future with exogenous variables
    print("\nForecasting future 7 days (Feb 01-07, 2025)...")
    future_dates = pd.date_range(start='2025-02-01', periods=7, freq='D')
    future_weekdays = future_dates.weekday.values
    exog_future = pd.DataFrame({'weekday_num': future_weekdays})
    
    future_forecast = forecaster.forecast_future(
        n_steps=7,
        start_date='2025-02-01',
        exog_future=exog_future
    )
    print(future_forecast)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    actual = forecaster.test['cups_sold'].values
    plt.plot(range(len(actual)), actual, label='Actual', marker='o', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cups Sold', fontsize=12)
    plt.title('SARIMAX: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarimax_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'sarimax_predictions.png'")
    
    return forecaster, predictions, metrics


def example_prophet():
    """Example usage of Prophet forecaster."""
    print("\n" + "=" * 80)
    print("Prophet Forecaster Example (Facebook Prophet)")
    print("=" * 80)
    
    # Initialize forecaster
    forecaster = ProphetForecaster(
        train_path='train.csv',
        val_path='validation.csv',
        test_path='test.csv',
        target_col='cups_sold',
        seasonality_mode='multiplicative',
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Train and evaluate
    print("\nTraining Prophet model...")
    predictions, metrics = forecaster.fit_and_evaluate(
        growth='linear',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        verbose=False
    )
    
    # Print metrics
    print("\n" + "=" * 50)
    print("Test Set Performance:")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("=" * 50)
    
    # Forecast future 7 days
    print("\nForecasting future 7 days (Feb 01-07, 2025)...")
    future_forecast = forecaster.forecast_future(n_steps=7, start_date='2025-02-01')
    print(future_forecast[['date', 'predicted_cups_sold']])
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    actual = forecaster.test['cups_sold'].values
    plt.plot(range(len(actual)), actual, label='Actual', marker='o', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linewidth=2)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cups Sold', fontsize=12)
    plt.title('Prophet: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prophet_predictions.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'prophet_predictions.png'")
    
    return forecaster, predictions, metrics


def compare_all_models():
    """Compare all models side by side."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # Run all models
    print("\n[1/5] Running LSTM...")
    _, _, lstm_metrics = example_lstm()
    results['LSTM'] = lstm_metrics
    
    print("\n[2/5] Running CNN-LSTM...")
    _, _, cnn_lstm_metrics = example_cnn_lstm()
    results['CNN-LSTM'] = cnn_lstm_metrics
    
    print("\n[3/5] Running SARIMA...")
    _, _, sarima_metrics = example_sarima()
    results['SARIMA'] = sarima_metrics
    
    print("\n[4/5] Running SARIMAX...")
    _, _, sarimax_metrics = example_sarimax()
    results['SARIMAX'] = sarimax_metrics
    
    print("\n[5/5] Running Prophet...")
    _, _, prophet_metrics = example_prophet()
    results['Prophet'] = prophet_metrics
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    
    # Save to CSV
    comparison_df.to_csv('model_comparison.csv')
    print("\nComparison table saved to 'model_comparison.csv'")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [results[model][metric] for model in results]
        ax.bar(results.keys(), values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to 'model_comparison.png'")
    
    return results


if __name__ == "__main__":
    # You can run individual examples or compare all models
    
    # Option 1: Run individual model
    print("Running individual model examples...")
    # example_lstm()
    # example_cnn_lstm()
    example_sarima()
    # example_sarimax()
    # example_prophet()
    
    # Option 2: Compare all models (this will take longer)
    # compare_all_models()
