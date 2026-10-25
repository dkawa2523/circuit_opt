"""Small input-boundary helpers; reject ambiguity before applying defaults."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

_UNITS = {
    "ohm": ("ohm", 1.0),
    "Ω": ("ohm", 1.0),
    "F": ("F", 1.0),
    "H": ("H", 1.0),
    "Hz": ("Hz", 1.0),
    "s": ("s", 1.0),
    "V": ("V", 1.0),
    "A": ("A", 1.0),
    "W": ("W", 1.0),
    "VA": ("VA", 1.0),
    "rad/s": ("rad/s", 1.0),
}
for _prefix, _factor in {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}.items():
    for _unit in ("ohm", "Ω", "F", "H", "Hz", "s", "V", "A", "W", "VA", "rad/s"):
        _UNITS[_prefix + _unit] = (_UNITS[_unit][0], _factor)


def si_value(value: Any, expected_unit: str | None = None) -> Any:
    """Convert an explicitly dimensioned value once; leave parameter references intact."""

    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµΩ/]+)\s*", value)
    if match is None:
        return value
    unit = match[2]
    if unit not in _UNITS:
        raise ValueError(f"unsupported unit {unit!r} in {value!r}; use an SI value or a supported unit")
    dimension, factor = _UNITS[unit]
    if expected_unit is not None and dimension != expected_unit:
        raise ValueError(f"{value!r} has dimension {dimension}, expected {expected_unit}")
    result = float(match[1]) * factor
    if not math.isfinite(result):
        raise ValueError(f"quantity must be finite: {value!r}")
    return result


def normalize_quantities(value: Any, unit: str | None = None) -> Any:
    """Normalize public numeric fields, preserving free-form evidence and SPICE text."""

    if isinstance(value, Mapping):
        annotation = value.get("unit")
        if isinstance(annotation, str) and annotation in _UNITS:
            unit = _UNITS[annotation][0]
        normalized = {}
        for key, item in value.items():
            dimension = next(
                (
                    suffix
                    for suffix in ("ohm", "Hz", "F", "H", "s", "V", "A", "W", "VA")
                    if re.search(r"_" + suffix + r"(?:_min|_max)?$", str(key))
                ),
                None,
            )
            if str(key).endswith("_rad_s"):
                dimension = "rad/s"
            if key == "value" and "ref" in value:
                dimension = {"R": "ohm", "C": "F", "L": "H"}.get(str(value["ref"])[:1].upper(), dimension)
            if key in {
                "metadata",
                "evidence",
                "description",
                "label",
                "title",
                "raw",
                "file",
                "waveform_file",
                "id",
                "scenario_id",
                "partition",
                "columns",
                "case_id",
                "schema",
                "type",
                "name",
                "ref",
                "n1",
                "n2",
                "p",
                "n",
                "output_node",
                "current_source",
                "source_voltage",
                "load_current",
                "voltage_node",
                "reference_plane",
                "unit",
                "builder",
            } or (value.get("type") in {"str", "string"} and key in {"default", "choices", "values"}):
                normalized[key] = item
            else:
                normalized[key] = normalize_quantities(item, dimension or unit)
        return normalized
    if isinstance(value, list):
        return [normalize_quantities(item, unit) for item in value]
    return si_value(value, unit)


def reject_unknown(values: Mapping[str, Any], allowed: set[str], path: str) -> None:
    if unknown := sorted(set(values) - allowed):
        raise ValueError(f"{path} contains unsupported fields {unknown}")


def parameter_spec(raw: Any, path: str, *, allow_grid: bool = False) -> dict[str, Any]:
    """Read a parameter domain once, including inline design/control domains."""

    spec = dict(raw) if isinstance(raw, Mapping) else {"default": raw}
    reject_unknown(
        spec,
        {
            "default",
            "bounds",
            "choices",
            "values",
            "type",
            "scale",
            "grid",
            "role",
            "unit",
            "description",
            "metadata",
            "extensions",
        },
        path,
    )
    if "choices" in spec and "values" in spec:
        raise ValueError(f"{path} must use only one of choices or values")
    annotation = spec.get("unit")
    if annotation is not None and not isinstance(annotation, str):
        raise ValueError(f"{path}.unit must be an SI unit annotation string")
    if annotation in _UNITS and _UNITS[annotation][1] != 1.0:
        raise ValueError(
            f"{path}.unit is SI metadata and does not convert values; use unit {_UNITS[annotation][0]!r} and SI numbers, or public unit-bearing values"
        )
    if "grid" in spec and ("bounds" not in spec or "choices" in spec or "values" in spec):
        raise ValueError(f"{path}.grid requires a bounds domain")
    if "grid" in spec and not allow_grid:
        raise ValueError(f"{path}.grid belongs in study.controls.variables; use choices for finite candidate values")
    for field in ("metadata", "extensions"):
        if field in spec and not isinstance(spec[field], Mapping):
            raise ValueError(f"{path}.{field} must be a mapping")
    return spec


def integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
    if isinstance(value, bool) or not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{path} must be an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        if minimum == 1:
            raise ValueError(f"{path} must be a positive integer")
        raise ValueError(f"{path} must be >= {minimum}")
    return result


def _unique_pairs(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate input field {key!r}")
        result[key] = value
    return result


def read_input_mapping(path: Path) -> dict[str, Any]:
    """Load authored YAML/JSON without last-key-wins field replacement."""

    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any) -> dict[Any, Any]:
        # Explicit fields must be unique. YAML merge defaults retain their
        # defined precedence, including an explicit override of an anchor.
        keys = [loader.construct_object(key) for key, _value in node.value if key.tag != "tag:yaml.org,2002:merge"]
        _unique_pairs([(key, None) for key in keys])
        return yaml.SafeLoader.construct_mapping(loader, node)

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    text = path.read_text(encoding="utf-8-sig")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.load(text, Loader=UniqueLoader)  # noqa: S506 - SafeLoader subclass checks duplicate fields.
        elif path.suffix.lower() == ".json":
            data = json.loads(text, object_pairs_hook=_unique_pairs)
        else:
            raise ValueError("case file must be YAML or JSON")
    except (yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"case file root must be a mapping: {path}")
    return data
