echo -e "\nSARIMA for Weather Data\n"
uv run ./sarima_forecasting.py > ../log/weather_sarima.txt

echo -e "SARIMAX for Weather Data\n"
uv run ./sarimax_forecasting.py > ../log/weather_sarimax.txt

echo -e "LSTM for Weather Data\n"
uv run ./lstm_forecasting.py > ../log/weather_lstm.txt

echo -e "CNN-LSTM for Weather Data\n"
uv run ./cnn_lstm_forecasting.py > ../log/weather_cnn_lstm.txt

echo -e "FbProphet for Weather Data\n"
uv run ./fbprophet_forecasting.py > ../log/weather_fbprophet.txt

echo -e "All run completed.\n"
