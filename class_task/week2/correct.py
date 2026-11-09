from collections import Counter
import csv
from pathlib import Path

path = Path('order.csv')
with path.open(newline='', encoding='utf-8') as f:
  rows = list(csv.DictReader(f))
before_counts = Counter(row['city'] for row in rows)
print(before_counts)

corrected_cities = []
for row in rows:
  corrected_city = row['city'].strip().lower()

  if(corrected_city == 'khi'):
    corrected_city = 'karachi'

  corrected_cities.append(corrected_city)

after_counts = Counter(corrected_cities)
print(after_counts)