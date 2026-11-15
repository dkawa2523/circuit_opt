"""Resolve structured electrical sources and their observation vectors once."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .inputs import reject_unknown
from .probes import source_meter_reference

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


_SPICE_ZERO = re.compile(r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:e[+-]?\d+)?[a-z]*", re.IGNORECASE)


def _zero_source(fields: list[str], *, transient_only: bool = False) -> bool:
    """Recognize a zero independent source over the requested analysis domain."""

    values = [field.lower() for field in fields]
    if not values:
        return True
    if values[0] != "ac":
        if values[0] == "dc":
            values = values[1:]
        if not values or _SPICE_ZERO.fullmatch(values[0]) is None:
            return False
        values = values[1:]
    if not values:
        return True
    if values[0] != "ac" or len(values) > 3:
        return False
    return transient_only or (len(values) >= 2 and _SPICE_ZERO.fullmatch(values[1]) is not None)


def imported_sources(case: Case, *, include_nested: bool = False) -> dict[str, dict[str, Any]]:
    """Observe top-level source declarations without reinterpreting circuit topology."""

    deck = _imported_source_deck(case)
    if deck is None:
        return {}
    text, mode = deck
    circuit = case.data.get("circuit") or {}
    replaced = (
        {source_name(item, index).lower() for index, item in enumerate(source_specs(case)) if "raw" not in item}
        if str(circuit.get("source_policy", "replace_named")).strip().lower() == "replace_named"
        else set()
    )
    return {
        name: item
        for name, item in _source_cards(text, mode=mode, include_nested=include_nested).items()
        if name not in replaced
    }


def _imported_source_deck(case: Case) -> tuple[str, str] | None:
    from .netlist_import import flatten_netlist_file

    circuit = case.data.get("circuit") or {}
    if not isinstance(circuit, Mapping) or circuit.get("builder") != "from_netlist":
        return None
    declared = circuit.get("netlist_file")
    if not isinstance(declared, str):
        return None
    path = Path(declared)
    path = path if path.is_absolute() else case.base_dir / path
    if not path.is_file():
        return None
    mode = circuit.get("netlist_mode", "fragment")
    text, _dependencies = flatten_netlist_file(path, mode=mode)
    return text, mode


def _source_cards(text: str, *, mode: str = "fragment", include_nested: bool = False) -> dict[str, dict[str, Any]]:
    from .netlist_import import executable_netlist_lines, spice_card_fields

    sources = {}
    conditional_depth = 0
    for index, line in enumerate(executable_netlist_lines(text, mode=mode)):
        parts = spice_card_fields(line.statement)
        if parts and parts[0].lower() in {".if", ".elseif", ".else", ".endif"}:
            if parts[0].lower() == ".if":
                conditional_depth += 1
            elif parts[0].lower() == ".endif":
                conditional_depth = max(0, conditional_depth - 1)
        if (line.top_level or include_nested) and len(parts) >= 3 and parts[0][:1].upper() in {"V", "I"}:
            name = parts[0]
            key = name.lower() if line.top_level else f"{index}:{name.lower()}"
            prior = sources.get(key)
            uncertain = not line.top_level or conditional_depth > 0 or prior is not None
            is_voltage = name[:1].upper() == "V"
            specification = [field.lower() for field in parts[3:]]
            zero = _zero_source(specification) and (prior is None or prior["zero"])
            sources[key] = {
                "name": name,
                "p": parts[1],
                "n": parts[2],
                "type": "imported_ambiguous"
                if uncertain
                else ("imported_voltage" if is_voltage else "imported_current"),
                "is_meter": is_voltage and zero,
                "zero": zero,
                "transient_zero": _zero_source(specification, transient_only=True)
                and (prior is None or prior["transient_zero"]),
                "ac_excitation": _has_ac_excitation(specification) or (prior is not None and prior["ac_excitation"]),
            }
    return sources


def _has_ac_excitation(specification: list[str]) -> bool:
    for index, field in enumerate(specification):
        if field == "ac":
            # ngspice uses unity when the AC keyword has no magnitude.
            magnitude = specification[index + 1] if index + 1 < len(specification) else "1"
            if _SPICE_ZERO.fullmatch(magnitude) is None:
                return True
    return False


def _circuit_source_cards(case: Case) -> dict[str, dict[str, Any]]:
    cards = imported_sources(case, include_nested=True)
    cards.update(_source_cards(_authored_source_statements(case), include_nested=True))
    return cards


def _authored_source_statements(case: Case) -> str:
    circuit = case.data.get("circuit") or {}
    statements = [str(item["raw"]) for item in source_specs(case) if "raw" in item]
    if isinstance(circuit, Mapping) and circuit.get("builder", "from_yaml") == "from_yaml":
        components = circuit.get("components")
    else:
        components = None
    if isinstance(components, list | tuple):
        statements.extend(
            str(item["raw"])
            if "raw" in item
            else " ".join(str(item.get(key, "")) for key in ("ref", "n1", "n2", "value"))
            for item in components
            if isinstance(item, Mapping)
        )
    return "\n".join(statements)


def transient_power_balance_issue(
    case: Case, observed_source: str | None, *, source_scope: Mapping[str, Any] | None = None
) -> str | None:
    """A selected source observation can close the balance for one identifiable drive."""

    if source_scope is not None:
        return source_scope["transient_unavailable_reason"]
    if observed_source is None:
        return "no source port was observed"
    declared = source_specs(case)
    if any("raw" in source for source in declared):
        return "raw source declarations do not establish complete transient source power"

    cards = _circuit_source_cards(case)
    for index, source in enumerate(declared):
        name = source_name(source, index)
        cards[name.lower()] = {"name": name, "type": "declared", "zero": False}
    if issue := _transient_card_issue(cards, observed_source):
        return issue
    if issue := _unmetered_energy_issue(_authored_source_statements(case)):
        return issue
    if deck := _imported_source_deck(case):
        return _unmetered_energy_issue(deck[0], mode=deck[1])
    return None


def _transient_card_issue(cards: Mapping[str, dict[str, Any]], observed_source: str) -> str | None:
    active = _transient_drives(cards)
    if uncertain := [item["name"] for item in active.values() if item["type"] == "imported_ambiguous"]:
        return f"nested or conditional sources prevent a complete transient source balance: {', '.join(uncertain)}"

    observed = observed_source.lower()
    if observed not in active:
        return "the selected source observation does not directly identify an independent transient drive"
    if others := [item["name"] for key, item in active.items() if key != observed]:
        return f"additional transient sources are outside the observed source port: {', '.join(others)}"
    return None


def _transient_drives(cards: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: item for key, item in cards.items() if not item.get("transient_zero", item["zero"])}


def _unmetered_energy_issue(deck: str, *, mode: str = "fragment") -> str | None:
    from .netlist_import import executable_netlist_lines

    # Ideal controlled and behavioral sources, and XSPICE code models, can
    # exchange power through output terminals without an explicit power input.
    # A positive profiled resistor is still an R card and dissipates V*I.
    names = dict.fromkeys(
        statement.split(maxsplit=1)[0]
        for line in executable_netlist_lines(deck, mode=mode)
        if (statement := line.statement)[:1].upper() in {"A", "B", "E", "F", "G", "H"}
    )
    if names:
        return f"unmetered behavioral or controlled elements {', '.join(names)} leave the power balance incomplete"
    return None


def ac_power_balance_issue(source_scope: Mapping[str, Any] | None) -> str | None:
    """Read AC balance evidence without trusting an older rendered inventory."""

    if source_scope is None:
        return None
    if "ac_power_balance_unavailable_reason" in source_scope:
        return source_scope["ac_power_balance_unavailable_reason"]
    if source_scope.get("basis") == "rendered_ngspice_deck":
        return "saved source inventory does not establish complete AC power balance. Reanalyze the saved netlist."
    return None


def selected_source(case: Case) -> tuple[str | None, Mapping[str, Any] | None]:
    measurement = case.data.get("measurement") or {}
    pairs = [(source_name(item, index), item) for index, item in enumerate(source_specs(case)) if "raw" not in item]
    structured = {name.lower(): (name, source) for name, source in pairs}
    if len(pairs) != len(structured) or any(not name.strip() for name in structured):
        raise ValueError("structured source names must be nonempty and unique")
    requested = str(measurement.get("current_source", "")).strip()
    imported = imported_sources(case)
    if overlap := sorted(structured.keys() & imported.keys()):
        raise ValueError(
            f"sources {overlap} occur in both YAML and the preserved netlist; use source_policy: replace_named or remove the duplicate YAML source"
        )
    key = requested.lower()
    if requested and structured and key not in structured and key not in imported:
        raise ValueError(f"measurement.current_source {requested!r} is not a declared structured source")
    if not requested and len(structured) + len(imported) > 1:
        raise ValueError("multiple sources require measurement.current_source to identify the observed source port")
    if not key:
        key = next(iter(structured or imported), "")
    if key in structured:
        return structured[key]
    if key in imported:
        selected = imported[key]
        name = selected["name"]
        if selected["type"] == "imported_current":
            raise ValueError(
                "an imported current source needs an explicit zero-volt branch meter as measurement.current_source and its source-port voltage as measurement.source_voltage"
            )
        if selected["type"] == "imported_voltage":
            return name, selected
        return name, None
    return requested or None, None


def source_vectors(case: Case) -> tuple[str | None, str | None, str | None]:
    name, source = selected_source(case)
    measurement = case.data.get("measurement") or {}
    voltage = measurement.get("source_voltage")
    if name is None:
        # Preserve the implicit branch used by older raw decks when it exists.
        raw = _circuit_source_cards(case).get("vsrc")
        if raw is not None and raw["type"] == "imported_voltage":
            name = raw["name"]
        elif voltage is None:
            return None, None, None
        else:
            raise ValueError("measurement.source_voltage requires a source or measurement.current_source")
    if voltage is not None:
        requested = _voltage_terminals(str(voltage))
        if requested is None:
            raise ValueError("measurement.source_voltage must be a voltage vector v(p) or v(p,n)")
        explicit_meter = source is not None and source.get("is_meter") and measurement.get("current_source")
        if source is not None and not explicit_meter:
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
    branch = source_meter_reference(name) if source is not None and source.get("type") == "current_dc" else name
    current = f"i({branch})"
    return name, str(voltage), current


def _voltage_terminals(vector: str) -> tuple[str, str] | None:
    if match := re.fullmatch(r"v\(\s*([^(),\s]+)\s*(?:,\s*([^(),\s]+)\s*)?\)", vector.strip(), re.IGNORECASE):
        return match[1].lower(), (match[2] or "0").lower()
    return None


def ac_source_name(case: Case, requested: str | None) -> str | None:
    names = [source_name(item, index) for index, item in enumerate(source_specs(case)) if "raw" not in item]
    if requested is not None:
        if isinstance(requested, str):
            for name in names:
                if name.lower() == requested.lower():
                    return name
        raise ValueError(f"solver.ac.source {requested!r} must name a structured source")
    if len(names) > 1:
        raise ValueError(
            "multiple sources require solver.ac.source to select the AC excitation independently of probes"
        )
    return names[0] if names else None


def ac_port_issue(
    case: Case,
    observed_source: str | None,
    requested_drive: str | None,
    *,
    source_scope: Mapping[str, Any] | None = None,
) -> str | None:
    """Explain why the observed AC result does not identify a driving point."""

    if source_scope is not None:
        return source_scope["ac_unavailable_reason"]
    if observed_source is None:
        return "no source port was observed"
    drive = ac_source_name(case, requested_drive)
    known = any(
        source_name(item, index).lower() == observed_source.lower()
        for index, item in enumerate(source_specs(case))
        if "raw" not in item
    )
    return _ac_card_issue(_circuit_source_cards(case), observed_source, drive, known)


def _ac_card_issue(
    cards: Mapping[str, dict[str, Any]], observed_source: str, drive: str | None = None, known: bool = False
) -> str | None:
    imported_drives = [item for item in cards.values() if item["ac_excitation"]]
    if uncertain := [item["name"] for item in imported_drives if item["type"] == "imported_ambiguous"]:
        return f"nested or conditional AC excitations prevent identifying a driving point: {', '.join(uncertain)}"
    active = ([drive] if drive is not None else []) + [item["name"] for item in imported_drives]
    if len(active) > 1:
        return f"multiple AC excitations are active: {', '.join(active)}"
    drive = active[0] if active else None
    observed = observed_source.lower()
    if drive is None or drive.lower() == observed:
        return None
    if known or (observed in cards and not cards[observed]["is_meter"]):
        return f"observed source {observed_source!r} differs from AC excitation {drive!r}"
    return None


def rendered_source_scope(
    deck: str,
    observed_source: str | None,
    *,
    observed_voltage: str | None = None,
    observed_current: str | None = None,
    mode: str = "deck",
) -> dict[str, Any]:
    """Identify independent drives and bind the recorded V/I to their actual terminals."""
    cards = _source_cards(deck, mode=mode, include_nested=True)
    unresolved = bool(re.search(r"(?im)^\s*\.(?:inc|include|lib)\s", deck))
    controls = len(re.findall(r"(?im)^\s*\.control(?:\s|$)", deck))
    reason = None
    if unresolved or controls > (1 if mode == "deck" else 0):
        reason = (
            "source inventory is incomplete because the rendered deck contains external input or extra control blocks"
        )
    ac_excitations = [item["name"] for item in cards.values() if item["ac_excitation"]]
    ac_issue = _ac_card_issue(cards, observed_source or "")
    transient_port_issue = reason
    transient_source = observed_source or ""
    if observed_source is None:
        ac_issue = "no source port observation is declared"
        transient_port_issue = reason or ac_issue
    elif observed_source and not ac_excitations:
        ac_issue = "the rendered deck contains no independent AC excitation"
    if observed_source and (observed_voltage or observed_current) and reason is None:
        current_vector = observed_current or f"i({observed_source})"
        match = re.fullmatch(r"i\(\s*([^(),\s]+)\s*\)", current_vector.strip(), re.IGNORECASE)
        branch = cards.get(match[1].lower()) if match else None
        if ac_issue is None:
            drive = cards[ac_excitations[0].lower()]
            if not _source_port_matches(deck, branch, drive, observed_voltage or "", mode):
                ac_issue = f"observed current and voltage do not establish the AC source port of {drive['name']!r}"

        matching_drives = [
            drive["name"]
            for drive in _transient_drives(cards).values()
            if _source_port_matches(deck, branch, drive, observed_voltage or "", mode)
        ]
        if len(matching_drives) == 1:
            transient_source = matching_drives[0]
        else:
            transient_port_issue = "observed current and voltage do not establish an independent transient source port"
    energy_issue = _unmetered_energy_issue(deck, mode=mode)
    return {
        "basis": "rendered_ngspice_deck",
        "ac_excitations": ac_excitations,
        "ac_unavailable_reason": reason or ac_issue,
        "ac_power_balance_unavailable_reason": reason or ac_issue or energy_issue,
        "transient_port_unavailable_reason": transient_port_issue,
        "transient_unavailable_reason": transient_port_issue
        or _transient_card_issue(cards, transient_source)
        or energy_issue,
    }


def _source_port_matches(
    deck: str, branch: Mapping[str, Any] | None, drive: Mapping[str, Any], voltage: str, mode: str
) -> bool:
    if branch is None or drive["type"] == "imported_ambiguous":
        return False
    if branch is drive:
        return drive["type"] == "imported_voltage" and _voltage_terminals(voltage) == (
            drive["p"].lower(),
            drive["n"].lower(),
        )
    return _series_meter_matches(deck, branch, drive, voltage, mode)


def _series_meter_matches(
    deck: str, meter: Mapping[str, Any], drive: Mapping[str, Any], voltage: str, mode: str
) -> bool:
    from .netlist_import import executable_netlist_lines, has_implicit_node_shunts, spice_card_fields

    if meter["type"] != "imported_voltage" or not meter["is_meter"]:
        return False
    meter_p, meter_n = meter["p"].lower(), meter["n"].lower()
    drive_nodes = {drive["p"].lower(), drive["n"].lower()}
    meter_nodes = {meter_p, meter_n}
    shared = drive_nodes & meter_nodes
    if len(shared) != 1 or len(drive_nodes | meter_nodes) != 3:
        return False
    junction = shared.pop()
    if junction in {"0", "gnd"}:
        return False
    other = (drive_nodes - {junction}).pop()
    # Branch current flows from the meter's p toward n. The junction and the
    # adjacent external terminal are equipotential for the zero-volt meter.
    ports = {(other, meter_n), (other, junction)} if meter_p == junction else {(meter_p, other), (junction, other)}
    if _voltage_terminals(voltage) not in ports or has_implicit_node_shunts(deck, mode=mode):
        return False

    # A proof needs an exclusive junction. An occurrence on any other card,
    # including a global declaration or unexpanded device, leaves it unproven.
    pair = {meter["name"].lower(), drive["name"].lower()}
    token = re.compile(rf"(?<![\w.-]){re.escape(junction)}(?![\w.-])", re.IGNORECASE)
    for line in executable_netlist_lines(deck, mode=mode):
        statement = line.statement
        if statement.startswith(("*", ";", "$")) or not token.search(statement):
            continue
        parts = spice_card_fields(statement)
        if not parts or not line.top_level or parts[0].lower() not in pair:
            return False
    return True
