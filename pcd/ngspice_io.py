"""Translate ngspice output files into canonical simulation data.

This module knows historical ``wrdata`` layouts and canonical CSV columns. It
does not compute impedance, power, objectives, or engineering acceptance.
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

    values = np.loadtxt(path)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 6:
        raise ValueError(f"AC output needs 6 columns (scale, re, im per vector): {path}")
    data = {
        "frequency_Hz": values[:, 0],
        "voltage_re": values[:, 1],
        "voltage_im": values[:, 2],
        "current_re": values[:, 4],
        "current_im": values[:, 5],
    }
    for index, name in enumerate(extra_columns or (), start=2):
        base = 3 * index
        if base + 2 >= values.shape[1]:
            raise ValueError(f"AC output is missing vector {name!r}: {path}")
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


# Older ngspice releases wrote a different number of columns for the same
# three vectors, so the standard layouts remain pinned by column count.
_TRANSIENT_COLUMNS = {6: (0, 3, 5), 5: (0, 2, 4), 4: (0, 1, 3), 3: (0, 1, 2)}


def read_transient_output(path: str | Path, probe_columns: Sequence[str] | None = None) -> SimulationResult:
    """Read one ngspice transient ``wrdata`` file."""

    values = np.loadtxt(path)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    width = values.shape[1]
    if width == 2:
        return SimulationResult(time_s=values[:, 0], voltage_V=values[:, 1], current_A=None)

    names = tuple(probe_columns or ())
    if names:
        vectors = values[:, 1::2]
        expected = 3 + len(names)
        if vectors.shape[1] < expected:
            raise ValueError(f"expected {expected} vectors for probes {list(names)}, found {vectors.shape[1]}: {path}")
        return SimulationResult(
            time_s=vectors[:, 0],
            voltage_V=vectors[:, 1],
            current_A=vectors[:, 2],
            probes={name: vectors[:, 3 + index] for index, name in enumerate(names)},
        )

    layout = _TRANSIENT_COLUMNS.get(min(width, 6))
    if layout is None:
        raise ValueError(f"cannot parse wrdata output: {path}")
    time_index, voltage_index, current_index = layout
    return SimulationResult(
        time_s=values[:, time_index],
        voltage_V=values[:, voltage_index],
        current_A=values[:, current_index],
    )
