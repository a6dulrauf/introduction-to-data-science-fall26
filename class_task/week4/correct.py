import pandas as pd
import numpy as np

df = pd.read_csv("members.csv", dtype=str, keep_default_na=False)
print(len(df), len(df.drop_duplicates()))
print("\n")

age = df["age"].str.strip()
not_plain = age[~age.str.fullmatch(r"\d+")]
print(not_plain.value_counts()) 
print("\n")

nums = pd.to_numeric(age, errors="coerce")
impossible = nums[(nums <= 0) | (nums > 120)]
print(impossible.value_counts())
print("\n")
print((age == "").sum())