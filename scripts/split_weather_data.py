import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('../data/intermediate/merged_daily_weather_all.csv')

# Ensure the data is sorted by date (assuming it's already sorted, but to be safe)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Calculate the sizes for train, validation, and test sets
n = len(df)
train_size = int(0.7 * n)
val_size = int(0.1 * n)
test_size = n - train_size - val_size  # Ensures the total adds up to n

# Split the data chronologically
train = df.iloc[:train_size]
val = df.iloc[train_size:train_size + val_size]
test = df.iloc[train_size + val_size:]

# Optionally, save the splits to new CSV files
train.to_csv('../data/preprocessed/weather/train.csv', index=False)
val.to_csv('../data/preprocessed/weather/validation.csv', index=False)
test.to_csv('../data/preprocessed/weather/test.csv', index=False)

# Print the shapes to verify
print(f"Total rows: {n}")
print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")
print(f"Test shape: {test.shape}")
