"""Harmonic-comparison figure from periodic electrical measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .figures import _headless_pyplot, _save

HarmonicLevels = tuple[list[str], list[float], float]


def render_harmonic_spectrum(
    metrics: Mapping[str, Any],
    out: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Render voltage and current harmonic amplitudes relative to H1."""

    panels = _harmonic_panels(metrics)
    if not panels:
        return None
    plt = _headless_pyplot()
    figure, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2), squeeze=False)
    for axis, (panel_title, unit, values) in zip(axes[0], panels, strict=True):
        _draw_harmonic_panel(axis, panel_title, unit, values)
    if title:
        figure.suptitle(title, fontsize=12)
    return _save(figure, out)


def _harmonic_panels(metrics: Mapping[str, Any]) -> list[tuple[str, str, HarmonicLevels]]:
    panels = []
    voltage = _harmonic_levels(metrics.get("voltage_harmonic_amplitude_V"))
    current = _harmonic_levels(metrics.get("current_harmonic_amplitude_A"))
    if voltage is not None:
        panels.append(("Voltage harmonics", "V", voltage))
    if current is not None:
        panels.append(("Current harmonics", "A", current))
    return panels


def _draw_harmonic_panel(axis: Any, title: str, unit: str, values: HarmonicLevels) -> None:
    labels, levels, fundamental = values
    colors = ["tab:blue", *("tab:orange" for _ in labels[1:])]
    axis.bar(labels, levels, color=colors, alpha=0.85)
    axis.set_ylim(min(-20.0, min(levels) - 10.0), 5.0)
    axis.set_ylabel("amplitude relative to H1 [dBc]")
    axis.set_title(f"{title}\nH1 = {fundamental:.4g} {unit}")
    axis.grid(alpha=0.25, axis="y")


def _harmonic_levels(values: Any) -> HarmonicLevels | None:
    harmonics = _ordered_harmonics(values)
    if not harmonics or harmonics[0][0] != 1 or harmonics[0][1] <= 0:
        return None
    fundamental = harmonics[0][1]
    labels = [f"H{order}" for order, _ in harmonics]
    levels = [20.0 * math.log10(max(amplitude / fundamental, 1e-15)) for _, amplitude in harmonics]
    return labels, levels, fundamental


def _ordered_harmonics(values: Any) -> list[tuple[int, float]]:
    if not isinstance(values, Mapping):
        return []
    harmonics = [_harmonic_item(name, value) for name, value in values.items()]
    return sorted(item for item in harmonics if item is not None)


def _harmonic_item(name: Any, value: Any) -> tuple[int, float] | None:
    text = str(name)
    if not text.startswith("h") or not text[1:].isdigit():
        return None
    amplitude = float(value)
    if not math.isfinite(amplitude) or amplitude < 0:
        return None
    return int(text[1:]), amplitude
