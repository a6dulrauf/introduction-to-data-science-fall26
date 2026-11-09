import pandas as pd
import numpy as np

df = pd.read_csv("salaries.csv")
s = df["monthly_salary_pkr"].dropna().to_numpy()

before = {
    "mean": np.mean(s),
    "median": np.median(s),
    "std": np.std(s),
}
print(before)

s2 = np.append(s, 5_000_000)

after = {
    "mean": np.mean(s2),
    "median": np.median(s2),
    "std": np.std(s2),
}
print(after)

for k in before:
    change = after[k] - before[k]
    pct = change / before[k] * 100
    print(f"{k}: {before[k]:,.0f} -> {after[k]:,.0f}  (change {change:,.0f}, {pct:.1f}%)")