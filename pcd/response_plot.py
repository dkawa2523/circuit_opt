"""Looking at what a run produced.

:mod:`pcd.netlist_viz` draws the circuit; this draws its *response*.  Both files
a run writes get a panel:

    waveform.csv -> electrode voltage against time
    ac.csv       -> Smith chart and reflection against frequency

The Smith chart is the reason this exists.  A matching network is designed by
moving the input impedance toward the centre of that chart, and until now the
sweep was written to disk and never looked at -- so a match could be optimized
but not inspected.

Drawing only; every number shown comes from :mod:`pcd.analysis`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import DEFAULT_Z0, at_frequency, input_impedance

#: Constant-resistance and constant-reactance circles drawn as the chart grid,
#: normalized to the reference impedance.
_R_CIRCLES = (0.2, 0.5, 1.0, 2.0, 5.0)
_X_CIRCLES = (0.2, 0.5, 1.0, 2.0, 5.0)


def _headless_pyplot() -> Any:
    """Select matplotlib's file-only backend before any figure is created."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def render_response(
    waveform: pd.DataFrame | None,
    ac: pd.DataFrame | None,
    out: str | Path,
    title: str | None = None,
    z0: float = DEFAULT_Z0,
    marker_hz: float | None = None,
) -> Path:
    """Draw whichever panels the available data supports."""

    plt = _headless_pyplot()
    panels = [p for p in ("waveform", "smith", "reflection") if _has_data(p, waveform, ac)]
    if not panels:
        raise ValueError("nothing to plot: neither a waveform nor a frequency response was given")

    figure, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    impedance = input_impedance(ac, z0) if ac is not None and not ac.empty else None

    for axis, panel in zip(axes, panels, strict=True):
        if panel == "waveform":
            _draw_waveform(axis, waveform)
        elif panel == "smith":
            _draw_smith(axis, impedance, z0, marker_hz)
        else:
            _draw_reflection(axis, impedance, marker_hz)

    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170)
    plt.close(figure)
    return out


def _has_data(panel: str, waveform: pd.DataFrame | None, ac: pd.DataFrame | None) -> bool:
    frame = waveform if panel == "waveform" else ac
    return frame is not None and not frame.empty


def _draw_waveform(axis: Any, waveform: pd.DataFrame | None) -> None:
    if waveform is None:
        return
    time_ns = waveform["time_s"].to_numpy(float) * 1e9
    axis.plot(time_ns, waveform["voltage_V"].to_numpy(float), lw=1.2, label="electrode")
    if "source_voltage_V" in waveform:
        axis.plot(time_ns, waveform["source_voltage_V"].to_numpy(float), lw=0.9, alpha=0.7, label="source")
        axis.legend(fontsize=8)
    axis.set_xlabel("time [ns]")
    axis.set_ylabel("voltage [V]")
    axis.set_title("Transient")
    axis.grid(alpha=0.3)


def _draw_smith(axis: Any, impedance: pd.DataFrame | None, z0: float, marker_hz: float | None) -> None:
    """Reflection coefficient on the unit disc, with the usual impedance grid."""

    _draw_smith_grid(axis)
    if impedance is None:
        return

    axis.plot(
        impedance["reflection_re"].to_numpy(float),
        impedance["reflection_im"].to_numpy(float),
        lw=1.4,
        color="tab:blue",
    )

    if marker_hz is not None:
        row = at_frequency(impedance, marker_hz)
        axis.plot([row["reflection_re"]], [row["reflection_im"]], "o", color="tab:red", ms=7, zorder=5)
        axis.annotate(
            f"{row['frequency_Hz'] / 1e6:.2f} MHz\n{row['resistance_ohm']:.1f}{row['reactance_ohm']:+.1f}j Ω\n"
            f"VSWR {row['vswr']:.2f}",
            (row["reflection_re"], row["reflection_im"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )
    axis.set_title(f"Smith chart (Z0 = {z0:g} Ω)")


def _draw_smith_grid(axis: Any) -> None:
    """Constant-R and constant-X circles, clipped to the unit disc."""

    theta = np.linspace(0, 2 * np.pi, 400)
    axis.plot(np.cos(theta), np.sin(theta), color="0.35", lw=1.0)
    axis.axhline(0.0, color="0.35", lw=0.8)

    for r in _R_CIRCLES:
        centre, radius = r / (1 + r), 1 / (1 + r)
        axis.plot(centre + radius * np.cos(theta), radius * np.sin(theta), color="0.8", lw=0.6)
    for x in _X_CIRCLES:
        # Constant-reactance arcs are circles centred at (1, 1/x); only the part
        # inside the unit disc is meaningful, so the rest is masked away.
        radius = 1 / x
        for sign in (1, -1):
            cx, cy = 1.0, sign * radius
            px, py = cx + radius * np.cos(theta), cy + radius * np.sin(theta)
            inside = px**2 + py**2 <= 1.0
            axis.plot(np.where(inside, px, np.nan), np.where(inside, py, np.nan), color="0.8", lw=0.6)

    axis.set_xlim(-1.08, 1.08)
    axis.set_ylim(-1.08, 1.08)
    axis.set_aspect("equal")
    axis.axis("off")


def _draw_reflection(axis: Any, impedance: pd.DataFrame | None, marker_hz: float | None) -> None:
    if impedance is None:
        return
    freq_mhz = impedance["frequency_Hz"].to_numpy(float) / 1e6
    axis.semilogx(freq_mhz, impedance["reflection_db"].to_numpy(float), lw=1.4, color="tab:blue")
    if marker_hz is not None:
        row = at_frequency(impedance, marker_hz)
        axis.axvline(row["frequency_Hz"] / 1e6, color="tab:red", lw=1.0, ls="--")
    axis.set_xlabel("frequency [MHz]")
    axis.set_ylabel("|Γ| [dB]")
    axis.set_title("Reflection")
    axis.grid(alpha=0.3, which="both")
