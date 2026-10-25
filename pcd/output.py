"""Small presentation settings; they never change a physical solve."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .artifacts import artifact_path_segment
from .inputs import integer, reject_unknown


def output_settings(raw: Any = None) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("output must be a mapping")
    reject_unknown(raw, {"formats", "schematic", "plots", "data"}, "output")
    formats = raw.get("formats", ["png"])
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(item, str) or item not in {"png", "svg", "pdf"} for item in formats)
    ):
        raise ValueError("output.formats must be a nonempty list of png, svg, pdf")
    if len(set(formats)) != len(formats):
        raise ValueError("output.formats contains duplicates")
    schematic = raw.get("schematic", True)
    if isinstance(schematic, bool):
        schematic = {"enabled": schematic}
    if not isinstance(schematic, Mapping):
        raise ValueError("output.schematic must be a boolean or mapping")
    reject_unknown(schematic, {"enabled", "view"}, "output.schematic")
    if not isinstance(schematic.get("enabled", True), bool):
        raise ValueError("output.schematic.enabled must be boolean")
    if not isinstance(schematic.get("view", "physical"), str) or schematic.get("view", "physical") not in {
        "physical",
        "instrumented",
    }:
        raise ValueError("output.schematic.view must be physical or instrumented")
    data = raw.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("output.data must be a mapping")
    reject_unknown(data, {"waveforms"}, "output.data")
    if not isinstance(data.get("waveforms", "selected"), str) or data.get("waveforms", "selected") not in {
        "all",
        "selected",
    }:
        raise ValueError("output.data.waveforms must be all or selected (export scope; raw evidence is retained)")
    plots = raw.get("plots", "auto")
    if plots != "auto" and not isinstance(plots, list):
        raise ValueError("output.plots must be auto or a list")
    resolved = []
    names: set[str] = set()
    if isinstance(plots, list):
        for index, item in enumerate(plots):
            if not isinstance(item, Mapping):
                raise ValueError("each output plot must be a mapping")
            kind = item.get("type")
            allowed = {"type", "name", "title"}
            allowed |= {
                "waveform": {"signals", "time_s", "xscale", "yscale"},
                "ac": {"signals", "frequency_Hz", "xscale"},
                "bode": {"numerator", "denominator", "frequency_Hz", "xscale"},
                "harmonics": {"signals", "time_s", "fundamental_Hz", "count"},
            }.get(str(kind), set())
            if not isinstance(kind, str) or kind not in {"waveform", "ac", "bode", "harmonics"}:
                raise ValueError("output plot type must be waveform, ac, bode, or harmonics")
            reject_unknown(item, allowed, f"output.plots[{index}]")
            recipe = deepcopy(dict(item))
            if "title" in item and (not isinstance(item["title"], str) or not item["title"].strip()):
                raise ValueError("output plot title must be a nonempty string")
            name = str(item.get("name", f"{kind}_{index + 1}"))
            if name != artifact_path_segment(name, 64) or name in names or name == "circuit":
                raise ValueError("output plot names must be unique safe filename segments (up to 64 characters)")
            names.add(name)
            recipe["name"] = name
            if kind == "bode":
                if not all(
                    isinstance(item.get(key), str) and item[key].strip() for key in ("numerator", "denominator")
                ):
                    raise ValueError("bode requires numerator and denominator signal names")
            else:
                signals = item.get("signals")
                if (
                    not isinstance(signals, list)
                    or not signals
                    or any(not isinstance(v, str) or not v for v in signals)
                ):
                    raise ValueError("output plot signals must be a nonempty list of signal names")
                if len(set(signals)) != len(signals):
                    raise ValueError("output plot signals contain duplicates")
            for key in ("time_s", "frequency_Hz"):
                if key in item:
                    limits = item[key]
                    if not isinstance(limits, list) or len(limits) != 2:
                        raise ValueError(f"plot {key} must contain [start, stop]")
                    low, high = map(float, limits)
                    if not (math.isfinite(low) and math.isfinite(high) and low < high and low >= 0):
                        raise ValueError(f"plot {key} requires finite increasing nonnegative limits")
                    recipe[key] = [low, high]
            for key in ("xscale", "yscale"):
                if key in item and (not isinstance(item[key], str) or item[key] not in {"linear", "log"}):
                    raise ValueError(f"plot {key} must be linear or log")
            if "count" in item:
                recipe["count"] = integer(item["count"], "plot.count", minimum=1)
            if "fundamental_Hz" in item:
                frequency = float(item["fundamental_Hz"])
                if not math.isfinite(frequency) or frequency <= 0:
                    raise ValueError("plot.fundamental_Hz must be positive and finite")
                recipe["fundamental_Hz"] = frequency
            resolved.append(recipe)
    return {
        "formats": list(formats),
        "schematic": {"enabled": schematic.get("enabled", True), "view": schematic.get("view", "physical")},
        "plots": resolved if isinstance(plots, list) else "auto",
        "data": {"waveforms": data.get("waveforms", "selected")},
    }
