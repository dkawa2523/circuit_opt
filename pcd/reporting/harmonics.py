"""Harmonic-comparison figure from periodic electrical measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .figures import FigureOutput, _headless_pyplot, _save


def render_harmonic_spectrum(
    table: pd.DataFrame,
    out: FigureOutput,
    *,
    title: str | None = None,
) -> Path | None:
    """Render the periodic harmonic table relative to each signal's H1."""

    if table.empty:
        return None
    panels = [group for _, group in table.groupby("signal", sort=False) if np.isfinite(group["relative_dBc"].iloc[0])]
    if not panels:
        return None
    plt = _headless_pyplot()
    figure, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2), squeeze=False)
    for axis, group in zip(axes[0], panels, strict=True):
        _draw_harmonic_panel(axis, group)
    if title:
        figure.suptitle(title, fontsize=12)
    return _save(figure, out)


def _draw_harmonic_panel(axis: Any, table: pd.DataFrame) -> None:
    labels = [f"H{order}" for order in table["harmonic"]]
    levels = table["relative_dBc"].to_numpy(float)
    fundamental, unit = table[["amplitude", "unit"]].iloc[0]
    title = "Voltage harmonics" if unit == "V" else "Current harmonics"
    axis.plot(labels, [max(value, -120.0) for value in levels], "o-", color="tab:blue")
    axis.axhline(-120.0, ls=":", color="grey", lw=0.8)
    for label, level in zip(labels, levels, strict=True):
        if level < -120:
            axis.annotate("< -120", (label, -120), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axis.set_ylim(-125.0, max(5.0, max(levels) + 5.0))
    axis.set_xlabel("harmonic order; display floor -120 dBc (original values retained)")
    axis.set_ylabel("amplitude relative to H1 [dBc]")
    axis.set_title(f"{title}\nH1 = {fundamental:.4g} {unit}")
    axis.grid(alpha=0.25, axis="y")
