"""Standard figures generated only from canonical saved-run data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd.analysis.ac import DEFAULT_Z0, at_frequency, frequency_sweep_metrics, input_impedance
from pcd.probes import ComponentObservation

_R_CIRCLES = (0.2, 0.5, 1.0, 2.0, 5.0)
_X_CIRCLES = (0.2, 0.5, 1.0, 2.0, 5.0)


def _headless_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(figure: Any, out: str | Path) -> Path:
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    _headless_pyplot().close(figure)
    return destination


def render_frequency_response(
    ac: pd.DataFrame,
    out: str | Path,
    *,
    title: str | None = None,
    z0: float = DEFAULT_Z0,
    marker_hz: float | None = None,
    sweep_metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Render impedance, Smith, and return-loss views of one AC result."""

    if ac.empty:
        raise ValueError("frequency response is empty")
    plt = _headless_pyplot()
    impedance = input_impedance(ac, z0)
    sweep = sweep_metrics if sweep_metrics is not None else frequency_sweep_metrics(ac, z0)
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.0))
    _draw_impedance_rectangular(axes[0, 0], impedance, marker_hz)
    _draw_smith(axes[0, 1], impedance, z0, marker_hz)
    _draw_impedance_polar(axes[1, 0], impedance, marker_hz)
    _draw_return_loss(axes[1, 1], impedance, marker_hz, sweep)
    if title:
        figure.suptitle(title, fontsize=12)
    return _save(figure, out)


def _draw_impedance_rectangular(axis: Any, impedance: pd.DataFrame, marker_hz: float | None) -> None:
    frequency_mhz = impedance["frequency_Hz"].to_numpy(float) / 1e6
    marker = "o" if len(impedance) == 1 else None
    _plot_frequency(axis, frequency_mhz, impedance["resistance_ohm"], label="R", lw=1.4, marker=marker)
    _plot_frequency(axis, frequency_mhz, impedance["reactance_ohm"], label="X", lw=1.4, marker=marker)
    _set_frequency_limits(axis, frequency_mhz)
    _mark_frequency(axis, marker_hz)
    axis.set_xlabel("frequency [MHz]")
    axis.set_ylabel("impedance [ohm]")
    axis.set_title("Input impedance")
    axis.grid(alpha=0.3, which="both")
    axis.legend()


def _draw_impedance_polar(axis: Any, impedance: pd.DataFrame, marker_hz: float | None) -> None:
    frequency_mhz = impedance["frequency_Hz"].to_numpy(float) / 1e6
    magnitude = impedance["magnitude_ohm"].to_numpy(float)
    phase = np.degrees(
        np.angle(impedance["resistance_ohm"].to_numpy(float) + 1j * impedance["reactance_ohm"].to_numpy(float))
    )
    phase_axis = axis.twinx()
    marker = "o" if len(impedance) == 1 else None
    _plot_frequency(axis, frequency_mhz, magnitude, color="tab:blue", label="|Z|", lw=1.4, marker=marker)
    _plot_frequency(phase_axis, frequency_mhz, phase, color="tab:orange", label="phase", lw=1.2, marker=marker)
    _set_frequency_limits(axis, frequency_mhz)
    _mark_frequency(axis, marker_hz)
    axis.set_xlabel("frequency [MHz]")
    axis.set_ylabel("|Z| [ohm]", color="tab:blue")
    phase_axis.set_ylabel("phase [deg]", color="tab:orange")
    axis.set_title("Impedance magnitude and phase")
    axis.grid(alpha=0.3, which="both")


def _draw_smith(axis: Any, impedance: pd.DataFrame, z0: float, marker_hz: float | None) -> None:
    _draw_smith_grid(axis)
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
            f"{row['frequency_Hz'] / 1e6:.3g} MHz\n"
            f"{row['resistance_ohm']:.3g}{row['reactance_ohm']:+.3g}j ohm\n"
            f"VSWR {row['vswr']:.3g}",
            (row["reflection_re"], row["reflection_im"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )
    axis.set_title(f"Smith chart (Z0 = {z0:g} ohm)")


def _draw_smith_grid(axis: Any) -> None:
    theta = np.linspace(0, 2 * np.pi, 400)
    axis.plot(np.cos(theta), np.sin(theta), color="0.35", lw=1.0)
    axis.axhline(0.0, color="0.35", lw=0.8)
    for resistance in _R_CIRCLES:
        centre, radius = resistance / (1 + resistance), 1 / (1 + resistance)
        axis.plot(centre + radius * np.cos(theta), radius * np.sin(theta), color="0.8", lw=0.6)
    for reactance in _X_CIRCLES:
        radius = 1 / reactance
        for sign in (1, -1):
            px = 1.0 + radius * np.cos(theta)
            py = sign * radius + radius * np.sin(theta)
            inside = px**2 + py**2 <= 1.0
            axis.plot(np.where(inside, px, np.nan), np.where(inside, py, np.nan), color="0.8", lw=0.6)
    axis.set_xlim(-1.08, 1.08)
    axis.set_ylim(-1.08, 1.08)
    axis.set_aspect("equal")
    axis.axis("off")


def _draw_return_loss(
    axis: Any,
    impedance: pd.DataFrame,
    marker_hz: float | None,
    sweep_metrics: Mapping[str, Any] | None,
) -> None:
    frequency_mhz = impedance["frequency_Hz"].to_numpy(float) / 1e6
    return_loss = -impedance["reflection_db"].to_numpy(float)
    marker = "o" if len(impedance) == 1 else None
    _plot_frequency(axis, frequency_mhz, return_loss, lw=1.4, color="tab:blue", marker=marker)
    _set_frequency_limits(axis, frequency_mhz)
    _mark_frequency(axis, marker_hz)
    _mark_half_power_band(axis, sweep_metrics)
    axis.set_xlabel("frequency [MHz]")
    axis.set_ylabel("return loss [dB]")
    axis.set_title("Input reflection")
    axis.grid(alpha=0.3, which="both")


def _mark_half_power_band(axis: Any, sweep_metrics: Mapping[str, Any] | None) -> None:
    if not sweep_metrics:
        return
    resonance = sweep_metrics.get("half_power_resonance")
    if not isinstance(resonance, Mapping):
        return
    lower = float(resonance["lower_frequency_Hz"]) / 1e6
    upper = float(resonance["upper_frequency_Hz"]) / 1e6
    peak = float(resonance["resonant_frequency_Hz"]) / 1e6
    axis.axvspan(lower, upper, color="tab:green", alpha=0.09, label="power -3 dB band")
    axis.axvline(peak, color="tab:green", lw=1.0, ls="-.", label="power peak")
    axis.legend(fontsize=8)


def _mark_frequency(axis: Any, marker_hz: float | None) -> None:
    if marker_hz is not None:
        axis.axvline(marker_hz / 1e6, color="tab:red", lw=1.0, ls="--")


def _set_frequency_limits(axis: Any, frequency_mhz: np.ndarray) -> None:
    if len(frequency_mhz) == 1:
        centre = float(frequency_mhz[0])
        axis.set_xlim(centre * 0.8, centre * 1.2)


def _plot_frequency(axis: Any, frequency_mhz: np.ndarray, values: Any, **style: Any) -> None:
    if len(frequency_mhz) > 1 and float(frequency_mhz[-1] / frequency_mhz[0]) >= 10.0:
        axis.semilogx(frequency_mhz, values, **style)
    else:
        axis.plot(frequency_mhz, values, **style)


def render_transient_response(
    waveform: pd.DataFrame,
    out: str | Path,
    *,
    title: str | None = None,
    load_current_column: str | None = None,
) -> Path:
    """Render voltage, current, and available instantaneous power traces."""

    if waveform.empty:
        raise ValueError("waveform is empty")
    panels = ["voltage"]
    if _finite_column(waveform, "current_A") or _finite_column(waveform, load_current_column):
        panels.append("current")
    if _has_source_power(waveform) or _has_load_power(waveform, load_current_column):
        panels.append("power")

    plt = _headless_pyplot()
    figure, axes = plt.subplots(len(panels), 1, figsize=(8.4, 3.0 * len(panels)), squeeze=False)
    time_us = waveform["time_s"].to_numpy(float) * 1e6
    for axis, panel in zip(axes[:, 0], panels, strict=True):
        if panel == "voltage":
            _draw_transient_voltage(axis, waveform, time_us)
        elif panel == "current":
            _draw_transient_current(axis, waveform, time_us, load_current_column)
        else:
            _draw_transient_power(axis, waveform, time_us, load_current_column)
        axis.set_xlabel("time [us]")
        axis.grid(alpha=0.3)
    if title:
        figure.suptitle(title, fontsize=12)
    return _save(figure, out)


def _finite_column(frame: pd.DataFrame, column: str | None) -> bool:
    return bool(column and column in frame and np.isfinite(frame[column].to_numpy(float)).any())


def _has_source_power(waveform: pd.DataFrame) -> bool:
    return _finite_column(waveform, "source_voltage_V") and _finite_column(waveform, "current_A")


def _has_load_power(waveform: pd.DataFrame, load_current_column: str | None) -> bool:
    return _finite_column(waveform, "voltage_V") and _finite_column(waveform, load_current_column)


def _draw_transient_voltage(axis: Any, waveform: pd.DataFrame, time_us: np.ndarray) -> None:
    axis.plot(time_us, waveform["voltage_V"].to_numpy(float), lw=1.2, label="load port")
    if _finite_column(waveform, "source_voltage_V"):
        axis.plot(time_us, waveform["source_voltage_V"].to_numpy(float), lw=0.9, alpha=0.75, label="source")
    axis.set_ylabel("voltage [V]")
    axis.set_title("Transient voltage")
    axis.legend()


def _draw_transient_current(
    axis: Any,
    waveform: pd.DataFrame,
    time_us: np.ndarray,
    load_current_column: str | None,
) -> None:
    if _finite_column(waveform, "current_A"):
        axis.plot(time_us, -waveform["current_A"].to_numpy(float), lw=1.1, label="source delivered")
    if _finite_column(waveform, load_current_column):
        axis.plot(time_us, waveform[load_current_column].to_numpy(float), lw=1.1, label="load")
    axis.set_ylabel("current [A]")
    axis.set_title("Transient current")
    axis.legend()


def _draw_transient_power(
    axis: Any,
    waveform: pd.DataFrame,
    time_us: np.ndarray,
    load_current_column: str | None,
) -> None:
    if _has_source_power(waveform):
        source_power = -waveform["source_voltage_V"].to_numpy(float) * waveform["current_A"].to_numpy(float)
        axis.plot(time_us, source_power, lw=1.1, label="source delivered")
    if _has_load_power(waveform, load_current_column):
        load_power = waveform["voltage_V"].to_numpy(float) * waveform[load_current_column].to_numpy(float)
        axis.plot(time_us, load_power, lw=1.1, label="load accepted")
    axis.set_ylabel("instantaneous power [W]")
    axis.set_title("Transient electrical power")
    axis.legend()


def render_operating_point(
    metrics: dict[str, Any],
    components: tuple[ComponentObservation, ...],
    out: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Render available power and component-stress values at one point."""

    powers = _metric_values(
        metrics,
        {
            "source_real_power_W": "source",
            "load_real_power_W": "load",
            "network_loss_W": "network loss",
            "modeled_component_loss_W": "modeled loss",
        },
    )
    voltages = _component_values(metrics, components, "voltage_peak_V")
    currents = _component_values(metrics, components, "current_rms_A")
    panels = [("Electrical power", "power [W]", powers), ("Component peak voltage", "voltage [V]", voltages)]
    panels.append(("Component RMS current", "current [A]", currents))
    panels = [panel for panel in panels if panel[2]]
    if not panels:
        return None

    plt = _headless_pyplot()
    figure, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for axis, (panel_title, ylabel, values) in zip(axes[0], panels, strict=True):
        labels, numbers = zip(*values, strict=True)
        axis.bar(labels, numbers, color="tab:blue", alpha=0.85)
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25, axis="y")
    if title:
        figure.suptitle(title, fontsize=12)
    return _save(figure, out)


def _metric_values(metrics: dict[str, Any], names: dict[str, str]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for name, label in names.items():
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append((label, float(value)))
    return values


def _component_values(
    metrics: dict[str, Any],
    components: tuple[ComponentObservation, ...],
    suffix: str,
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for component in components:
        value = metrics.get(f"component_{component.metric_id}_{suffix}")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append((component.reference, float(value)))
    return values
