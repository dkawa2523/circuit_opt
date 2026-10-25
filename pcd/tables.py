"""Read authored CSV tables without silently changing their column structure."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path


def read_csv_rows(
    path: str | Path,
    required: Iterable[str] = (),
) -> tuple[list[str], list[dict[str, str]]]:
    """Read one rectangular table; domain-specific values stay with its caller."""

    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        if not columns or any(not name.strip() for name in columns):
            raise ValueError(f"CSV has an empty or missing header: {path}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"CSV has duplicate column names: {path}")
        if missing := sorted(set(required) - set(columns)):
            raise ValueError(f"CSV is missing columns {missing}: {path}")
        rows: list[dict[str, str]] = []
        for values in reader:
            if not values:
                continue
            if len(values) != len(columns):
                raise ValueError(f"CSV row {reader.line_num} has {len(values)} cells; expected {len(columns)}: {path}")
            rows.append(dict(zip(columns, values, strict=True)))
    return columns, rows
