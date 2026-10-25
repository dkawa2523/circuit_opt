"""User-selected plots consume saved signal catalogs and shared numeric tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pcd.analysis.traces import ac_trace, harmonic_trace, transfer_trace

from .figures import _headless_pyplot


def save_figure(figure: Any, destination: Path, name: str, formats: list[str]) -> list[str]:
    files = []
    try:
        figure.tight_layout()
        for extension in formats:
            file = f"{name}.{extension}"
            figure.savefig(destination / file, dpi=170)
            files.append(file)
    finally:
        _headless_pyplot().close(figure)
    return files


def _signal(catalog: dict[str, Any], name: str, domain: str) -> dict[str, Any]:
    item = catalog.get("signals", {}).get(name)
    if not item or domain not in item["columns"]:
        raise ValueError(f"signal {name!r} has no saved {domain} data; add a measurement probe and rerun")
    return item


def _range(frame: pd.DataFrame, axis: str, limits: list[float] | None) -> pd.DataFrame:
    if limits is None:
        return frame
    tolerance = max(abs(limits[0]), abs(limits[1])) * 1e-9
    if limits[0] < frame[axis].min() - tolerance or limits[1] > frame[axis].max() + tolerance:
        raise ValueError(f"requested plot range {limits} lies outside saved {axis} data")
    # Retain every solver knot; axes clip display without resampling measurements.
    return frame


def _check_log_waveform(frame: pd.DataFrame, column: str, recipe: dict[str, Any]) -> None:
    visible = frame.loc[frame["time_s"].between(*recipe["time_s"])] if "time_s" in recipe else frame
    for option, values in (("xscale", visible["time_s"]), ("yscale", visible[column])):
        if recipe.get(option) == "log" and (values <= 0).any():
            raise ValueError(f"plot {option}=log requires positive values in the displayed range")


def _coverage(table: pd.DataFrame, kind: str, recipe: dict[str, Any]) -> dict[str, Any]:
    """Describe available plotted observations without changing the saved table."""

    axis = {"waveform": "time_s", "harmonics": "harmonic"}.get(kind, "frequency_Hz")
    limits = recipe.get(axis)
    displayed = table.loc[table[axis].between(*limits)] if limits is not None else table
    columns = {
        "ac": ["magnitude", "phase_deg"],
        "bode": ["gain_dB", "phase_deg"],
        # Zero-amplitude harmonics are known values, displayed at the stated floor.
        "harmonics": ["amplitude"],
    }.get(kind, [str(name) for name in table if name != axis])
    finite = {name: int(np.isfinite(displayed[name].to_numpy(float)).sum()) for name in columns}
    rows = len(displayed)
    status = "complete" if rows and all(count == rows for count in finite.values()) else "partial"
    if not any(finite.values()):
        status = "not_available"
    return {"status": status, "displayed_rows": rows, "finite_values": finite}


def render_recipe(
    recipe: dict[str, Any],
    destination: Path,
    formats: list[str],
    catalog: dict[str, Any],
    waveform: pd.DataFrame,
    ac: pd.DataFrame | None,
    *,
    title: str,
    fundamental_hz: float | None,
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind, name = recipe["type"], recipe["name"]
    domain = "transient" if kind in {"waveform", "harmonics"} else "ac"
    if domain == "transient" and waveform.empty:
        raise ValueError(f"plot {name!r} needs saved transient data")
    if domain == "ac" and (ac is None or ac.empty):
        raise ValueError(f"plot {name!r} needs saved AC data")
    ac = cast(pd.DataFrame, ac)
    figure_title = f"{recipe['title']}\n{title}" if "title" in recipe else title
    if kind == "bode" and "title" not in recipe:
        figure_title += f"\n{recipe['numerator']} / {recipe['denominator']}"
    plt = _headless_pyplot()
    figure = None
    try:
        tables: dict[str, pd.DataFrame] = {}
        if kind == "bode":
            first, second = (_signal(catalog, recipe[key], "ac") for key in ("numerator", "denominator"))
            if first["unit"] != second["unit"]:
                raise ValueError("a dimensionless Bode ratio requires numerator and denominator of the same unit")
            table = transfer_trace(ac, first["columns"]["ac"], second["columns"]["ac"])
            _range(table, "frequency_Hz", recipe.get("frequency_Hz"))
            tables[name] = table
            figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.3), sharex=True)
            marker = "o" if len(table) == 1 else None
            axes[0].plot(table["frequency_Hz"], table["gain_dB"], marker=marker)
            axes[0].set_ylabel("gain [dB]")
            axes[1].plot(table["frequency_Hz"], table["phase_deg"], marker=marker)
            axes[1].set_ylabel("phase [deg]")
            axes[1].set_xlabel("frequency [Hz]")
        else:
            grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
            for signal in recipe["signals"]:
                item = _signal(catalog, signal, domain)
                grouped[item.get("unit") or "unspecified"].append((signal, item))
            rows = len(grouped) * (2 if kind == "ac" else 1)
            figure, axes = plt.subplots(rows, 1, figsize=(8.4, 2.8 * rows), squeeze=False)
            index = 0
            for unit, channels in grouped.items():
                axis = axes[index, 0]
                phase_axis = axes[index + 1, 0] if kind == "ac" else None
                for signal, item in channels:
                    label = item["label"] or signal
                    if kind == "waveform":
                        table = _range(waveform, "time_s", recipe.get("time_s"))
                        column = item["columns"]["transient"][0]
                        _check_log_waveform(table, column, recipe)
                        axis.plot(table["time_s"], table[column], label=label)
                        tables[signal] = table[["time_s", column]]
                        axis.set_xlabel("time [s]")
                        if "time_s" in recipe:
                            axis.set_xlim(recipe["time_s"])
                        axis.set_xscale(recipe.get("xscale", "linear"))
                        axis.set_yscale(recipe.get("yscale", "linear"))
                    elif kind == "ac":
                        phase_axis = axes[index + 1, 0]
                        table = ac_trace(ac, item["columns"]["ac"])
                        _range(table, "frequency_Hz", recipe.get("frequency_Hz"))
                        marker = "o" if len(table) == 1 else None
                        axis.plot(table["frequency_Hz"], table["magnitude"], marker=marker, label=label)
                        phase_axis.plot(table["frequency_Hz"], table["phase_deg"], marker=marker, label=label)
                        for part in (axis, phase_axis):
                            part.set_xlabel("frequency [Hz]")
                        phase_axis.set_ylabel("phase [deg]")
                        tables[signal] = table
                    else:
                        frequency = recipe.get("fundamental_Hz", fundamental_hz)
                        if frequency is None:
                            raise ValueError("harmonics require an explicit periodic fundamental_Hz")
                        selected = waveform
                        if "time_s" in recipe:
                            _range(waveform, "time_s", recipe["time_s"])
                            selected = waveform.loc[waveform["time_s"].between(*recipe["time_s"])]
                        table = harmonic_trace(
                            selected["time_s"].to_numpy(float),
                            selected[item["columns"]["transient"][0]].to_numpy(float),
                            float(frequency),
                            recipe.get("count"),
                            measurement=measurement,
                        )
                        levels = table["relative_dBc"].to_numpy(float)
                        axis.plot(table["harmonic"], np.maximum(levels, -120), "o-", label=label)
                        axis.axhline(-120, color="grey", lw=0.8, ls=":")
                        axis.set_xlabel("harmonic order (display floor: -120 dBc)")
                        previous_top = axis.get_ylim()[1]
                        axis.set_ylim(-125, max(5, previous_top, float(np.nanmax(levels)) + 5))
                        tables[signal] = table
                    axis.set_ylabel(
                        "level [dBc]" if kind == "harmonics" else f"{'magnitude' if kind == 'ac' else 'value'} [{unit}]"
                    )
                axis.legend(fontsize=8)
                if phase_axis is not None:
                    phase_axis.legend(fontsize=8)
                index += 2 if kind == "ac" else 1
        for axis in figure.axes:
            axis.grid(alpha=0.3, which="both")
        if domain == "ac":
            frequency_limits = recipe.get(
                "frequency_Hz", [float(ac["frequency_Hz"].min()), float(ac["frequency_Hz"].max())]
            )
            if frequency_limits[0] == frequency_limits[1]:
                frequency_limits = [frequency_limits[0] * 0.8, frequency_limits[1] * 1.2]
            for axis in figure.axes:
                axis.set_xscale(recipe.get("xscale", "log"))
                axis.set_xlim(frequency_limits)
        data_files = []
        coverage = {}
        for index, (signal, table) in enumerate(tables.items()):
            file = f"{name}_data_{index + 1}.csv"
            table.to_csv(destination / file, index=False)
            coverage[signal] = _coverage(table, kind, recipe)
            data_files.append(
                {
                    "signal": signal,
                    "file": file,
                    "unit": "dimensionless" if kind == "bode" else catalog["signals"][signal]["unit"],
                }
            )
        status = "complete" if all(item["status"] == "complete" for item in coverage.values()) else "partial"
        if all(item["status"] == "not_available" for item in coverage.values()):
            status = "not_available"
        reason = None
        if status != "complete":
            reason = (
                "no finite plotted observations in the displayed range"
                if status == "not_available"
                else "some requested plotted values are unavailable; see coverage"
            )
            figure_title += f"\n{status.upper()}: {reason}"
        figure.suptitle(figure_title, fontsize=10 if reason else 12)
        return {
            "status": status,
            **({"reason": reason} if reason else {}),
            "coverage": coverage,
            "recipe": recipe,
            "figures": save_figure(figure, destination, name, formats),
            "data": data_files,
        }
    finally:
        if figure is not None:
            plt.close(figure)
