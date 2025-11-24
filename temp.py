import pandas as pd

df = pd.read_csv("./data/raw/walmart.csv")
df.columns = df.columns.str.lower()

df['date'] = pd.to_datetime(df['date'], dayfirst=True)

store_dfs = {store: group.reset_index(drop=True) for store, group in df.groupby('store')}

print(store_dfs[1])
print(store_dfs[2])
