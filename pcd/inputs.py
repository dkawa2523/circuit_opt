"""Small input-boundary helpers; reject ambiguity before applying defaults."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_LOAD_REFERENCE_PLANE = "load_ports"

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


def load_plane_label(value: Any) -> str:
    """Resolve a display label independently of the actual load terminals."""

    label = "" if value is None else str(value).strip()
    return label or DEFAULT_LOAD_REFERENCE_PLANE


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


def _field_unit(name: str) -> str | None:
    if name.endswith("_rad_s"):
        return "rad/s"
    match = re.search(r"_(ohm|Hz|F|H|s|V|A|W|VA)(?:_min|_max)?$", name)
    return match[1] if match else None


def normalize_quantities(value: Any, unit: str | None = None) -> Any:
    """Normalize public numeric fields, preserving free-form evidence and SPICE text."""

    if isinstance(value, Mapping):
        annotation = value.get("unit")
        if isinstance(annotation, str) and annotation in _UNITS:
            unit = _UNITS[annotation][0]
        normalized = {}
        for key, item in value.items():
            dimension = _field_unit(str(key))
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
                "acquisition_group",
                "acquisition_group_column",
                "observations",
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
            } or (
                (key == "conditions" and "circuit" in value)
                or (value.get("type") in {"str", "string"} and key in {"default", "choices", "values"})
            ):
                normalized[key] = item
            else:
                normalized[key] = normalize_quantities(item, dimension or unit)
        return normalized
    if isinstance(value, list):
        return [normalize_quantities(item, unit) for item in value]
    return si_value(value, unit)


def condition_value(value: Any, name: str, spec: Mapping[str, Any], path: str, *, allow_units: bool = False) -> Any:
    """Use a declared domain before coercing scenario or observed-control values."""

    kind = spec.get("type")
    choices = spec.get("choices", spec.get("values"))
    if kind in {"str", "string"} or (
        kind is None and isinstance(choices, list) and choices and all(isinstance(item, str) for item in choices)
    ):
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return value
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{path} must be a boolean")
    if allow_units:
        annotation = spec.get("unit")
        unit = _UNITS[annotation][0] if isinstance(annotation, str) and annotation in _UNITS else _field_unit(name)
        value = normalize_quantities(value, unit)
    if kind == "int":
        return integer(value, path)
    if kind in {"float", "number"} or isinstance(value, str):
        try:
            if isinstance(value, bool):
                raise TypeError
            number = float(value)
        except (TypeError, ValueError) as exc:
            if kind in {"float", "number"}:
                raise ValueError(f"{path} must be numeric") from exc
            return value
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite")
        return number
    return value


def observation_metadata(
    rows: Iterable[Mapping[str, Any]], scenario_ids: Iterable[str], partition_column: str, group_column: str | None
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Resolve observation labels separately from the electrical values."""

    partitions: dict[str, list[str]] = {"fit": [], "holdout": []}
    groups: dict[str, str] = {}
    for row, name in zip(rows, scenario_ids, strict=True):
        partition = row.get(partition_column)
        if not isinstance(partition, str) or partition not in partitions:
            raise ValueError(f"observation {name!r} requires {partition_column}: fit or holdout")
        partitions[partition].append(name)
        if group_column is not None:
            group = row.get(group_column)
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"observation {name!r} requires a nonempty {group_column!r} acquisition group")
            groups[name] = group.strip()
    return partitions, groups


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
    if spec.get("scale", "linear") not in ("linear", "log"):
        raise ValueError(f"{path}.scale must be linear or log")
    if spec.get("type", "") not in ("", "float", "number", "int", "bool", "str", "string"):
        raise ValueError(f"{path}.type is not a supported parameter type")
    for field in ("choices", "values"):
        if field in spec:
            choices = spec[field]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"{path}.{field} must be a nonempty list")
            _check_finite_parameter_values(choices, f"{path}.{field}")
    if "default" in spec:
        _check_finite_parameter_values(spec["default"], f"{path}.default")
    if "bounds" in spec:
        _check_parameter_bounds(spec, path)
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


def _check_finite_parameter_values(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            _check_finite_parameter_values(item, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite_parameter_values(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")


def _check_parameter_bounds(spec: Mapping[str, Any], path: str) -> None:
    bounds = spec["bounds"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{path}.bounds must be a two-item list")
    try:
        low, high = map(float, bounds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path}.bounds must be numeric") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{path}.bounds must be finite and ordered")
    if spec.get("type") == "int" and math.ceil(low) > math.floor(high):
        raise ValueError(f"{path}.bounds must contain an integer")
    if spec.get("scale") == "log" and low <= 0:
        raise ValueError(f"{path}: log-scale bounds must be positive")


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
