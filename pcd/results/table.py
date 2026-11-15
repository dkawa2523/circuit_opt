"""Stream a flat evaluation table with columns discovered during a study."""

from __future__ import annotations

import csv
import json
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from pcd.core.models import to_plain

_IDENTIFIER_COLUMNS = (
    "table_schema",
    "dataset_id",
    "study_id",
    "case_schema",
    "resolved_case_schema",
    "runtime_fingerprint_sha256",
    "solver_fingerprint_sha256",
    "candidate_id",
    "scenario_id",
    "raw_cache_key",
)


def read_evaluation_csv(path: Path) -> pd.DataFrame:
    """Preserve literal identifiers while letting pandas infer metric types."""

    return pd.read_csv(path, converters=dict.fromkeys(_IDENTIFIER_COLUMNS, str))


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
