from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(6)
COL = 'monthly_salary_pkr'


def sample_means(s: pd.Series, n: int, reps: int) -> np.ndarray:
    values = s.to_numpy()
    return np.array([RNG.choice(values, n, replace=False).mean()
                     for _ in range(reps)])


def biased_mean(df: pd.DataFrame, n: int) -> float:
    top = df.nlargest(n, 'years_experience')
    return float(top[COL].mean())


def partial_corr(df: pd.DataFrame, x: str, y: str, z: str) -> float:
    def leftover(col: str) -> np.ndarray:
        fit = np.polyfit(df[z], df[col], 1)
        return (df[col] - np.polyval(fit, df[z])).to_numpy()
    return float(np.corrcoef(leftover(x), leftover(y))[0, 1])


def posterior(prior: float, sens: float, fpr: float) -> float:
    hit = prior * sens
    return hit / (hit + (1 - prior) * fpr)


def main() -> None:
    pay = pd.read_csv('salaries.csv')
    s = pay[COL]
    print('population mean:', round(s.mean()))
    for n in (10, 30, 100):
        m = sample_means(s, n, 1000)
        print(n, round(m.mean()), round(m.std()),
              round(s.std() / np.sqrt(n)))
    print('biased sample of 30:', round(biased_mean(pay, 30)))
    beach = pd.read_csv('beach_days.csv')
    r = beach['ice_cream_sales'].corr(beach['rescue_calls'])
    print('ice cream vs rescues:', round(r, 2))
    p = partial_corr(beach, 'ice_cream_sales', 'rescue_calls',
                     'temperature_c')
    print('after removing temperature:', round(p, 2))
    t = pd.read_csv('treatment.csv')
    print(t.groupby('treatment')['recovered'].mean().round(2))
    print(t.groupby(['severity', 'treatment'])['recovered']
          .agg(['mean', 'size']).round(2))
    print(round(posterior(0.01, 0.90, 0.10), 3))


if __name__ == '__main__':
    main()
