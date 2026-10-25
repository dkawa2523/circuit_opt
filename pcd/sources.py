"""Resolve structured electrical sources and their observation vectors once."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .inputs import reject_unknown

if TYPE_CHECKING:
    from .case import Case


def source_specs(case: Case) -> tuple[dict[str, Any], ...]:
    declared, single = case.data.get("sources"), case.data.get("source")
    if declared is not None and single is not None:
        raise ValueError("use either source or sources, not both")
    if declared is None:
        declared = [] if single is None else [single]
    if not isinstance(declared, list):
        raise TypeError("sources must be a list of mappings")
    if any(not isinstance(item, Mapping) for item in declared):
        raise TypeError("source / sources entries must be mappings")
    for index, item in enumerate(declared):
        if "raw" in item:
            reject_unknown(item, {"raw"}, f"source[{index}]")
            continue
        common = {"type", "name", "p", "n", "variables", "ac_magnitude", "ac_phase_deg"}
        kind = str(item.get("type", "sine_voltage"))
        if kind == "current_dc":
            fields = {"current_A", "value_A", "value", "ac_magnitude_A"}
        elif kind in {"dc_voltage", "voltage_dc", "dc"}:
            fields = {"voltage_V", "value_V", "value", "ac_magnitude_V"}
        elif kind in {"pulse", "voltage_pulse"}:
            fields = {"v1_V", "v2_V", "delay_s", "rise_s", "fall_s", "width_s", "period_s", "ac_magnitude_V"}
        else:
            fields = {"dc_V", "amplitude_V", "amplitude", "frequency_Hz", "frequency", "phase_deg", "ac_magnitude_V"}
        reject_unknown(item, common | fields, f"source[{index}]")
        for aliases in (
            {"frequency_Hz", "frequency"},
            {"amplitude_V", "amplitude"},
            {"value", "value_V", "voltage_V"},
            {"value", "value_A", "current_A"},
            {"ac_magnitude", "ac_magnitude_V", "ac_magnitude_A"},
        ):
            if len(aliases & set(item)) > 1:
                raise ValueError(f"source[{index}] must use only one of {sorted(aliases)}")
    return tuple(dict(item) for item in declared)


def source_name(source: Mapping[str, Any], index: int) -> str:
    prefix = "I" if source.get("type") == "current_dc" else "V"
    return str(source.get("name", f"{prefix}src" if index == 0 else f"{prefix}src{index}"))


def imported_sources(case: Case) -> dict[str, dict[str, Any]]:
    """Observe top-level source declarations without reinterpreting circuit topology."""

    from .netlist_import import executable_netlist_lines, flatten_netlist_file

    circuit = case.data.get("circuit") or {}
    if not isinstance(circuit, Mapping) or circuit.get("builder") != "from_netlist":
        return {}
    declared = circuit.get("netlist_file")
    if not isinstance(declared, str):
        return {}
    path = Path(declared)
    path = path if path.is_absolute() else case.base_dir / path
    if not path.is_file():
        return {}
    text, _dependencies = flatten_netlist_file(path)
    sources = {}
    conditional_depth = 0
    replaced = (
        {source_name(item, index).lower() for index, item in enumerate(source_specs(case)) if "raw" not in item}
        if circuit.get("source_policy", "replace_named") == "replace_named"
        else set()
    )
    for line in executable_netlist_lines(text, mode=circuit.get("netlist_mode", "fragment")):
        parts = line.text.split()
        if parts and parts[0].lower() in {".if", ".elseif", ".else", ".endif"}:
            if parts[0].lower() == ".if":
                conditional_depth += 1
            elif parts[0].lower() == ".endif":
                conditional_depth = max(0, conditional_depth - 1)
        if (
            line.top_level
            and len(parts) >= 4
            and parts[0][:1].upper() in {"V", "I"}
            and parts[0].lower() not in replaced
        ):
            uncertain = conditional_depth > 0 or parts[0] in sources
            sources[parts[0]] = {
                "p": parts[1],
                "n": parts[2],
                "type": "imported_ambiguous"
                if uncertain
                else ("imported_voltage" if parts[0][:1].upper() == "V" else "imported_current"),
            }
    return sources


def selected_source(case: Case) -> tuple[str, Mapping[str, Any] | None]:
    measurement = case.data.get("measurement") or {}
    pairs = [(source_name(item, index), item) for index, item in enumerate(source_specs(case)) if "raw" not in item]
    structured = dict(pairs)
    names = [name for name, _source in pairs]
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("structured source names must be nonempty and unique")
    requested = str(measurement.get("current_source", "")).strip()
    imported = imported_sources(case)
    if overlap := sorted({name.lower() for name in names} & {name.lower() for name in imported}):
        raise ValueError(
            f"sources {overlap} occur in both YAML and the preserved netlist; use source_policy: replace_named or remove the duplicate YAML source"
        )
    if requested and structured and requested not in structured and requested not in imported:
        raise ValueError(f"measurement.current_source {requested!r} is not a declared structured source")
    if not requested and len(structured) + len(imported) > 1:
        raise ValueError("multiple sources require measurement.current_source to identify the observed source port")
    name = requested or next(iter(structured), "Vsrc")
    if not structured or name in imported:
        if not requested and imported:
            name = next(iter(imported))
        selected = imported.get(name)
        if selected is not None and selected["type"] == "imported_current":
            raise ValueError(
                "an imported current source needs an explicit zero-volt branch meter as measurement.current_source and its source-port voltage as measurement.source_voltage"
            )
        if selected is not None and selected["type"] == "imported_voltage":
            return name, selected
    return name, structured.get(name)


def source_vectors(case: Case) -> tuple[str, str, str]:
    name, source = selected_source(case)
    measurement = case.data.get("measurement") or {}
    voltage = measurement.get("source_voltage")
    if voltage is not None:
        vector = re.fullmatch(r"v\(\s*([^(),\s]+)\s*(?:,\s*([^(),\s]+)\s*)?\)", str(voltage).strip(), re.IGNORECASE)
        if vector is None:
            raise ValueError("measurement.source_voltage must be a voltage vector v(p) or v(p,n)")
        if source is not None and not str(source.get("type", "")).startswith("imported_"):
            requested = (vector[1].lower(), (vector[2] or "0").lower())
            port = (str(source.get("p", "src")).lower(), str(source.get("n", "0")).lower())
            if requested != port:
                raise ValueError(
                    "measurement.source_voltage must observe the selected source's p,n port with the same polarity"
                )
    if voltage is None:
        if source is None:
            circuit = case.data.get("circuit") or {}
            if "netlist_file" in circuit or any("raw" in item for item in source_specs(case)):
                raise ValueError(
                    "raw/imported sources require measurement.source_voltage to name the observed voltage vector"
                )
            voltage = "v(src)"
        else:
            positive, negative = str(source.get("p", "src")), str(source.get("n", "0"))
            voltage = f"v({positive})" if negative == "0" else f"v({positive},{negative})"
    current = f"i(Vmeasure_{name})" if source is not None and source.get("type") == "current_dc" else f"i({name})"
    return name, str(voltage), current


def ac_source_name(case: Case, requested: str | None) -> str | None:
    names = [source_name(item, index) for index, item in enumerate(source_specs(case)) if "raw" not in item]
    if requested is not None:
        if requested not in names:
            raise ValueError(f"solver.ac.source {requested!r} must name a structured source")
        return requested
    if len(names) > 1:
        raise ValueError(
            "multiple sources require solver.ac.source to select the AC excitation independently of probes"
        )
    return names[0] if names else None
