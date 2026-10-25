"""Stream a flat evaluation table with columns discovered during a study."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pcd.core.models import to_plain


@contextmanager
def evaluation_table(path: Path) -> Iterator[Callable[[Iterable[Mapping[str, Any]]], None]]:
    """Spool rows on disk, then write one CSV using the union of metric columns."""

    columns: dict[str, None] = {}
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", dir=path.parent) as spool:

        def append(rows: Iterable[Mapping[str, Any]]) -> None:
            for row in rows:
                columns.update(dict.fromkeys(row))
                spool.write(json.dumps(to_plain(row), ensure_ascii=False, allow_nan=False) + "\n")

        yield append
        spool.seek(0)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(json.loads(line) for line in spool)
