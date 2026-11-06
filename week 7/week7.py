from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path('charts')
BLUE, ORANGE = '#0072B2', '#D55E00'


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def line_chart(df: pd.DataFrame) -> str:
    m = df.groupby('month')['revenue_k'].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m.index, m.values, color=BLUE)
    ax.set_xticks(m.index[::3])
    ax.set_ylabel('Revenue (Rs thousand)')
    ax.set_title(f'Revenue peaked in {m.idxmax()} at {m.max():,.0f}k')
    save(fig, '1_revenue_by_month.png')
    return str(m.idxmax())


def bar_chart(df: pd.DataFrame) -> pd.Series:
    d = df.groupby('department')['revenue_k'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d.index, d.values, color=BLUE)
    ax.set_xlabel('Revenue (Rs thousand)')
    ax.set_title(f'{d.index[-1]} earns {d.iloc[-1] / d.sum():.0%} of revenue')
    save(fig, '2_revenue_by_department.png')
    return d


def hist_chart(df: pd.DataFrame) -> float:
    s = df['revenue_k']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=30, color=BLUE)
    ax.axvline(s.median(), color=ORANGE, label='median')
    ax.set_xlabel('Monthly revenue per region and department (k)')
    ax.set_title('Most rows are small; a few are very large')
    ax.legend()
    save(fig, '3_revenue_distribution.png')
    return float(s.skew())


def box_chart(df: pd.DataFrame) -> pd.Series:
    order = df.groupby('region')['revenue_k'].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df.loc[df['region'] == r, 'revenue_k'] for r in order.index]
    ax.boxplot(data)
    ax.set_xticklabels(order.index)
    ax.set_ylabel('Revenue per row (k)')
    ax.set_title('Karachi and Lahore lead; spread grows with size')
    save(fig, '4_revenue_by_region.png')
    return order


def scatter_chart(df: pd.DataFrame) -> float:
    typical = df.groupby(['region', 'department'])['units'].transform('mean')
    y = df['units'] / typical
    r = df['discount_pct'].corr(y)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['discount_pct'], y, s=12, alpha=0.5, color=BLUE)
    ax.set_xlabel('Discount (%)')
    ax.set_ylabel('Units vs typical (1.0 = typical)')
    ax.set_title(f'Deeper discounts, more units (r = {r:.2f})')
    save(fig, '5_discount_vs_units.png')
    return float(r)


def main() -> None:
    df = pd.read_csv('shop_monthly.csv')
    print(df.shape, df['month'].min(), df['month'].max())
    print('peak month:', line_chart(df))
    print(bar_chart(df).round(0).to_string())
    print('skew:', round(hist_chart(df), 2))
    print(box_chart(df).round(1).to_string())
    print('r:', round(scatter_chart(df), 2))


if __name__ == '__main__':
    main()
