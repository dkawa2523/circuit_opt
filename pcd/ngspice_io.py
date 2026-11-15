"""Translate ngspice output files into canonical simulation data.

This module decodes the generated ``wrdata`` layout and reads saved AC CSV
observations. Measurements consume the resulting canonical data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation import SimulationResult


def read_frequency_response(
    path: str | Path,
    extra_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a canonical AC CSV or an ngspice complex ``wrdata`` file."""

    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first_line = handle.readline().strip()
    if first_line.split(",", 1)[0].strip() == "frequency_Hz":
        return _read_canonical_frequency_response(path, extra_columns)

    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] < 6:
        raise ValueError(f"AC output needs 6 columns (scale, re, im per vector): {path}")
    expected = 3 * (2 + len(extra_columns or ()))
    if extra_columns is not None and values.shape[1] != expected:
        raise ValueError(f"AC output expected {expected} columns for its probes, found {values.shape[1]}: {path}")
    data = {
        "frequency_Hz": values[:, 0],
        "voltage_re": values[:, 1],
        "voltage_im": values[:, 2],
        "current_re": values[:, 4],
        "current_im": values[:, 5],
    }
    for index, name in enumerate(extra_columns or (), start=2):
        base = 3 * index
        data[f"{name}_re"] = values[:, base + 1]
        data[f"{name}_im"] = values[:, base + 2]
    return pd.DataFrame(data)


def _read_canonical_frequency_response(path: Path, extra_columns: Sequence[str] | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["frequency_Hz", "voltage_re", "voltage_im", "current_re", "current_im"]
    required.extend(f"{name}_{part}" for name in (extra_columns or ()) for part in ("re", "im"))
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"canonical AC output is missing columns {missing}: {path}")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return frame


def read_transient_output(
    path: str | Path, probe_columns: Sequence[str], *, has_source_current: bool = True
) -> SimulationResult:
    """Read scale/value pairs for time, voltage, optional source current and each probe."""

    values = np.loadtxt(path, ndmin=2)
    width = values.shape[1]
    first_probe = 3 if has_source_current else 2
    expected = 2 * (first_probe + len(probe_columns))
    if width != expected:
        raise ValueError(f"transient output expected {expected} columns for its probes, found {width}: {path}")
    vectors = values[:, 1::2]
    return SimulationResult(
        time_s=vectors[:, 0],
        voltage_V=vectors[:, 1],
        current_A=vectors[:, 2] if has_source_current else None,
        probes={name: vectors[:, first_probe + index] for index, name in enumerate(probe_columns)},
    )
