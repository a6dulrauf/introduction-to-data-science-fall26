from __future__ import annotations

from pathlib import Path

import pandas as pd

COL = 'monthly_salary_pkr'


def summarize(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    return pd.Series({
        'n': len(s), 'mean': s.mean(), 'median': s.median(),
        'std': s.std(), 'q1': q1, 'q3': q3, 'iqr': q3 - q1,
        'skew': s.skew()})


def iqr_flag(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    gap = 1.5 * (q3 - q1)
    return (s < q1 - gap) | (s > q3 + gap)


def by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)[COL]
    return g.agg(n='size', median='median',
                 mean='mean', std='std').round(0)


def add_one(s: pd.Series, pay: float) -> pd.Series:
    return pd.concat([s, pd.Series([pay])], ignore_index=True)


def main() -> None:
    df = pd.read_csv(Path('salaries.csv'))
    s = df[COL]
    print(df.shape, df.isna().sum().sum())
    print(summarize(s).round(2).to_string())
    flagged = df[iqr_flag(s)]
    print('flagged:', len(flagged))
    print(flagged['level'].value_counts().to_string())
    print(by_group(df, 'level'))
    print(by_group(df, 'department'))
    for pay in (5_000_000,):
        t = add_one(s, pay)
        print(round(t.mean()), t.median(), round(t.std()))


if __name__ == '__main__':
    main()
