import pandas as pd
df = pd.read_csv('sales.csv')
df['revenue'] = df['quantity'] * df['unit_price']
top_rev_cat = df.groupby('category')['revenue'].sum().idxmax()
print(top_rev_cat)
top_region = df.groupby('region')['revenue'].mean().idxmax()
print(top_region)
df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', errors='coerce')
df['month'] = df['order_date'].dt.to_period('M')

prod_monthly = df.groupby(['product', 'month'])['revenue'].sum()
prod_table = prod_monthly.unstack(fill_value=0)

first_month = prod_table.iloc[:, 0]
last_month = prod_table.iloc[:, -1]
declining_prod = prod_table[last_month < first_month]

declining_prod.index.tolist()