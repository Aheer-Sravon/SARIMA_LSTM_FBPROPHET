import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('../data/raw/walmart.csv')
df.columns = df.columns.str.lower()

# Parse the date
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Aggregate data: sum weekly_sales, mean for other numeric columns
df_agg = df.groupby('date').agg({
    'weekly_sales': 'sum',
    'holiday_flag': 'mean',
    'temperature': 'mean',
    'fuel_price': 'mean',
    'cpi': 'mean',
    'unemployment': 'mean'
}).reset_index()

# Sort by date (though groupby preserves order)
df_agg = df_agg.sort_values('date').reset_index(drop=True)

# Calculate the sizes for train, validation, and test sets
n = len(df_agg)
train_size = int(0.7 * n)
val_size = int(0.1 * n)
test_size = n - train_size - val_size  # Ensures the total adds up to n

# Split the data chronologically
train = df_agg.iloc[:train_size]
val = df_agg.iloc[train_size:train_size + val_size]
test = df_agg.iloc[train_size + val_size:]

# Optionally, save the splits to new CSV files
train.to_csv('../data/preprocessed/walmart/train.csv', index=False)
val.to_csv('../data/preprocessed/walmart/validation.csv', index=False)
test.to_csv('../data/preprocessed/walmart/test.csv', index=False)

# Print the shapes to verify
print(f"Total rows: {n}")
print(f"Train shape: {train.shape}")
print(f"Validation shape: {val.shape}")
print(f"Test shape: {test.shape}")
