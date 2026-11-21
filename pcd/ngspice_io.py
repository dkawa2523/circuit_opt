"""Translate ngspice output files into canonical simulation data.

This module decodes the generated ``wrdata`` layout. Saved canonical CSVs
are read by the result-record layer.
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
    """Read ngspice's complex ``wrdata`` output."""

    path = Path(path)
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
