from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sales(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['order_date'] = pd.to_datetime(
        df['order_date'], errors='coerce')
    bad = int(df['order_date'].isna().sum())
    print('Unparseable dates:', bad)
    df['region'] = df['region'].str.strip().fillna('Unknown')
    df['discount'] = df['discount'].fillna(0)
    df['revenue'] = (df['quantity'] * df['unit_price']
                     * (1 - df['discount']))
    return df


def inspect(df: pd.DataFrame) -> None:
    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
    nums = ['quantity', 'unit_price', 'discount', 'revenue']
    print(df[nums].describe().round(1))
    print('lines:', len(df))
    print('orders:', df['order_id'].nunique())


def revenue_by_category(df: pd.DataFrame) -> pd.Series:
    return (df.groupby('category')['revenue'].sum()
              .sort_values(ascending=False))


def aov_by_region(df: pd.DataFrame) -> pd.Series:
    orders = df.groupby(['region', 'order_id'])['revenue'].sum()
    return (orders.groupby('region').mean()
                  .sort_values(ascending=False))


def monthly_units(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=['order_date']).copy()
    d['month'] = d['order_date'].dt.to_period('M')
    return (d.groupby(['product', 'month'])['quantity']
             .sum().unstack('month', fill_value=0))


def declining(df: pd.DataFrame) -> pd.Series:
    # drop the last month: it is only half complete
    m = monthly_units(df).iloc[:, :-1]
    recent = m.iloc[:, -3:].mean(axis=1)
    earlier = m.iloc[:, -6:-3].mean(axis=1)
    change = (recent - earlier) / earlier.where(earlier > 0)
    return change.dropna().sort_values()


def top_categories(df: pd.DataFrame, n: int = 3) -> pd.Series:
    t = df.groupby(['region', 'category'])['revenue'].sum()
    g = t.groupby(level='region', group_keys=False)
    return g.nlargest(n)


def main() -> None:
    df = load_sales(Path('sales.csv'))
    inspect(df)
    cat = revenue_by_category(df)
    assert abs(cat.sum() - df['revenue'].sum()) < 1e-6
    print(cat.round(0))
    print(aov_by_region(df).round(0))
    print(declining(df).round(2).head(3))
    print(top_categories(df))


if __name__ == '__main__':
    main()
