from __future__ import annotations

import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

MISSING = {'', 'na', 'n/a', 'null', 'none', 'nan', '?'}
Row = dict[str, str | None]
Data = dict[str, list[str]]


@dataclass
class Column:
    name: str
    kind: str
    missing: int
    unique: int
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    median: float | None = None


def is_missing(value: str | None) -> bool:
    return value is None or value.strip().lower() in MISSING


def infer_kind(values: list[str]) -> str:
    present = [v.strip() for v in values if not is_missing(v)]
    if not present:
        return 'empty'
    for caster, label in ((int, 'int'), (float, 'float')):
        try:
            for v in present:
                caster(v)
            return label
        except ValueError:
            continue
    return 'str'


def profile(name: str, raw: list[str]) -> Column:
    present = [v.strip() for v in raw if not is_missing(v)]
    kind = infer_kind(raw)
    col = Column(name, kind, len(raw) - len(present),
                 len(set(present)))
    if kind in ('int', 'float'):
        nums = [float(v) for v in present]
        col.low, col.high = min(nums), max(nums)
        col.mean = statistics.fmean(nums)
        col.median = statistics.median(nums)
    return col


def read_rows(path: Path) -> tuple[list[str], list[Row]]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('no header row')
        return list(reader.fieldnames), list(reader)


def find_duplicates(data: Data) -> list[tuple[str, str]]:
    names = list(data)
    return [(a, b)
            for i, a in enumerate(names)
            for b in names[i + 1:]
            if data[a] == data[b]]


def report(path: Path) -> None:
    names, rows = read_rows(path)
    bad = sum(1 for r in rows
              if None in r or None in r.values())
    data = {n: [r.get(n) or '' for r in rows] for n in names}
    print(f'File: {path.name}')
    print(f'Rows: {len(rows)}  Columns: {len(names)}')
    print(f'Malformed rows: {bad}')
    for n in names:
        c = profile(n, data[n])
        line = (f'- {c.name}: {c.kind}, missing={c.missing}, '
                f'unique={c.unique}')
        if c.mean is not None:
            line += (f', min={c.low:g}, max={c.high:g}, '
                     f'mean={c.mean:.2f}, median={c.median:g}')
        if c.unique <= 1 and c.kind != 'empty':
            line += '  [constant]'
        print(line)
    for a, b in find_duplicates(data):
        print(f'Duplicate columns: {a} and {b}')


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('Usage: python profiler.py <file.csv>')
        return 2
    path = Path(argv[1])
    try:
        report(path)
    except (OSError, ValueError) as error:
        print(f'Could not profile {path}: {error}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
