from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path('data/raw/members.csv')
OUT = Path('data/processed/members_clean.csv')
EXPORT = pd.Timestamp('2026-10-01')
CITY = {'karachi': 'Karachi', 'khi': 'Karachi',
        'krachi': 'Karachi', 'lahore': 'Lahore',
        'lhr': 'Lahore', 'islamabad': 'Islamabad',
        'isb': 'Islamabad'}
GENDER = {'m': 'Male', 'male': 'Male', 'f': 'Female',
          'female': 'Female', 'other': 'Other'}
LOG: list[tuple[str, str, str]] = []


def log(column: str, problem: str, decision: str) -> None:
    LOG.append((column, problem, decision))


def to_number(s: pd.Series) -> pd.Series:
    t = s.str.lower()
    t = t.str.replace(r'yrs|rs|,|\s', '', regex=True)
    k = t.str.endswith('k').fillna(False)
    num = pd.to_numeric(t.str.rstrip('k'), errors='coerce')
    return num.where(~k, num * 1000)


def in_range(s: pd.Series, low: float, high: float,
             name: str) -> pd.Series:
    bad = s.notna() & ~s.between(low, high)
    log(name, f'{int(bad.sum())} outside {low:g}-{high:g}',
        'set to missing; never guess a value')
    return s.where(~bad)


def clean_phone(s: pd.Series) -> pd.Series:
    d = s.str.replace(r'\D', '', regex=True)
    p92 = d.str.startswith('92', na=False)
    d = d.where(~p92, '0' + d.str[2:])
    short = d.str.len().eq(10) & d.str.startswith('3', na=False)
    d = d.where(~short, '0' + d)
    return d.where(d.str.fullmatch(r'03\d{9}').fillna(False))


def parse_dates(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    dmy = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
    txt = pd.to_datetime(s, format='%b %d, %Y', errors='coerce')
    d = iso.fillna(dmy).fillna(txt)
    return d.where(d <= EXPORT)


def iqr_flag(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    gap = 1.5 * (q3 - q1)
    return (s < q1 - gap) | (s > q3 + gap)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.apply(lambda c: c.str.strip())
    df = df.replace({'': np.nan, 'N/A': np.nan, '?': np.nan})
    df['full_name'] = df['full_name'].str.title()
    df['gender'] = df['gender'].str.lower().map(GENDER)
    df['gender'] = df['gender'].fillna('Unknown')
    city = df['city'].str.lower().map(CITY)
    assert not (df['city'].notna() & city.isna()).any()
    df['city'] = city.fillna('Unknown')
    n = len(df)
    df = df.drop_duplicates()
    log('rows', f'{n - len(df)} duplicate rows', 'dropped')
    df['phone'] = clean_phone(df['phone'])
    df['age'] = in_range(to_number(df['age']), 14, 90, 'age')
    fee = to_number(df['monthly_fee'])
    df['monthly_fee'] = in_range(fee, 1000, 20000, 'fee')
    sess = to_number(df['sessions_per_month'])
    df['sessions_per_month'] = in_range(sess, 0, 31, 'sessions')
    sat = to_number(df['satisfaction'])
    df['satisfaction'] = in_range(sat, 1, 5, 'satisfaction')
    for col in ('join_date', 'last_visit'):
        df[col] = parse_dates(df[col])
    key = df['full_name'] + df['phone'].fillna('?')
    dup = key.duplicated(keep=False)
    df['possible_duplicate'] = dup & df['phone'].notna()
    df['fee_outlier'] = iqr_flag(df['monthly_fee'])
    df['visit_before_join'] = df['last_visit'] < df['join_date']
    return df


def run_tests(df: pd.DataFrame) -> None:
    assert df['member_id'].is_unique
    assert df['age'].dropna().between(14, 90).all()
    assert df['monthly_fee'].dropna().between(1000, 20000).all()
    assert df['city'].isin([*CITY.values(), 'Unknown']).all()
    assert df['join_date'].dropna().le(EXPORT).all()
    assert df['phone'].dropna().str.fullmatch(r'03\d{9}').all()


def main() -> None:
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    df = clean(raw)
    run_tests(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f'Rows: {len(raw)} -> {len(df)}')
    print(df.isna().sum()[lambda s: s > 0].to_string())
    flags = ['possible_duplicate', 'fee_outlier',
             'visit_before_join']
    print(df[flags].sum().to_string())
    for column, problem, decision in LOG:
        print(f'{column}: {problem} -> {decision}')


if __name__ == '__main__':
    main()
