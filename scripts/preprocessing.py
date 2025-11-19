import pandas as pd

months = [
    'april-24', 'may-24', 'june-24', 'july-24', 'august-24',
    'september-24', 'october-24', 'november-24', 'december-24',
    'january-25', 'february-25'
]

dfs = []
for month in months:
    df = pd.read_csv(f'../data/raw/{month}.csv')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

df_all['datetime'] = pd.to_datetime(df_all['datetime'], format='mixed')

df_all['date'] = df_all['datetime'].dt.date

grouped = df_all.groupby('date')

lower_temp = grouped['temp'].min()
upper_temp = grouped['temp'].max()
average_temp = grouped['temp'].mean()
feels_like = grouped['feelslike'].mean()
dew = grouped['dew'].mean()
humidity = grouped['humidity'].mean()
wind_gust = grouped['windgust'].max()
wind_speed = grouped['windspeed'].mean()
weekday_num = grouped['datetime'].first().dt.weekday

new_df = pd.concat([lower_temp, upper_temp, average_temp, feels_like, dew, humidity, wind_gust, wind_speed, weekday_num], axis=1)
new_df.columns = ['lower temp', 'upper temp', 'average temp', 'feels like', 'dew', 'humidity', 'wind gust', 'wind speed', 'weekday_num']

new_df = new_df.sort_index()
new_df = new_df.reset_index(names='date')

# Load sales data and merge cups_sold
sales_df = pd.read_csv('../data/raw/processed_sales.csv')
sales_df['date'] = pd.to_datetime(sales_df['date']).dt.date
new_df = pd.merge(new_df, sales_df[['date', 'cups_sold']], on='date', how='left')

# Move cups_sold next to date (optional, for nicer column order)
new_df.insert(1, 'cups_sold', new_df.pop('cups_sold'))

# Optional: fill missing cups_sold with 0 (uncomment if needed)
new_df['cups_sold'] = new_df['cups_sold'].fillna(0).astype(int)

new_df.to_csv('../data/intermediate/merged_daily_weather_all.csv', index=False)
