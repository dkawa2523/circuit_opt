from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Case:
    """Small wrapper around a YAML/JSON case file.

    The case body intentionally remains a plain dict.  This avoids a heavy
    validation layer while keeping path handling and variable discovery in one
    place.
    """

    path: Path
    data: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def case_id(self) -> str:
        return str(self.data.get("case_id", self.path.stem))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_case(path: str | Path) -> Case:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"case file must be YAML or JSON: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"case file root must be a mapping: {path}")
    return Case(path=path, data=data)


def resolve_path(case: Case, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else case.base_dir / path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def yaml_dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    raise TypeError(type(obj).__name__)


# -----------------------------------------------------------------------------
# Design variables
# -----------------------------------------------------------------------------


def variable_specs(case: Case) -> dict[str, dict[str, Any]]:
    """Collect design variables from predictable, shallow YAML locations."""

    specs: dict[str, dict[str, Any]] = {}

    def add(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for name, spec in obj.items():
            specs[str(name)] = dict(spec) if isinstance(spec, dict) else {"default": spec}

    data = case.data
    add(data.get("variables", {}))
    if isinstance(data.get("source"), dict):
        add(data["source"].get("variables", {}))
    for src in data.get("sources", []) or []:
        if isinstance(src, dict):
            add(src.get("variables", {}))
    if isinstance(data.get("circuit"), dict):
        add(data["circuit"].get("variables", {}))
    if isinstance(data.get("load"), dict):
        add(data["load"].get("variables", {}))
    return specs


def default_params(case: Case) -> dict[str, Any]:
    return {k: v["default"] for k, v in variable_specs(case).items() if isinstance(v, dict) and "default" in v}


def fill_default_params(case: Case, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**default_params(case), **(params or {})}


def sample_param(rng: np.random.Generator, spec: dict[str, Any]) -> Any:
    if "choices" in spec:
        val = rng.choice(list(spec["choices"]))
        return val.item() if hasattr(val, "item") else val
    if spec.get("type") == "bool":
        return bool(rng.integers(0, 2))
    lo, hi = spec.get("bounds", [0.0, 1.0])
    if spec.get("type") == "int":
        return int(rng.integers(int(lo), int(hi) + 1))
    if spec.get("scale", "linear") == "log":
        lo_f, hi_f = float(lo), float(hi)
        if lo_f <= 0 or hi_f <= 0:
            raise ValueError(f"log-scale bounds must be positive: {spec}")
        return float(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f)))
    return float(rng.uniform(float(lo), float(hi)))


def grid_values(spec: dict[str, Any], levels: int = 3) -> list[Any]:
    if "choices" in spec:
        return list(spec["choices"])
    if "values" in spec:
        return list(spec["values"])
    if "default" in spec and "bounds" not in spec:
        return [spec["default"]]
    lo, hi = spec.get("bounds", [spec.get("default", 0.0), spec.get("default", 0.0)])
    n = max(1, int(spec.get("grid", levels)))
    if spec.get("type") == "int":
        vals = np.linspace(int(lo), int(hi), n)
        return sorted(set(int(round(v)) for v in vals))
    if spec.get("scale", "linear") == "log":
        lo_f, hi_f = float(lo), float(hi)
        if lo_f <= 0 or hi_f <= 0:
            raise ValueError(f"log-scale bounds must be positive: {spec}")
        return [float(v) for v in np.geomspace(lo_f, hi_f, n)]
    return [float(v) for v in np.linspace(float(lo), float(hi), n)]


def case_warnings(case: Case) -> list[str]:
    warnings: list[str] = []
    data = case.data
    if not data.get("source") and not data.get("sources"):
        warnings.append("no source/source list found; netlist may have no independent source")
    seen: dict[str, str] = {}

    def check(section: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for name in obj:
            key = str(name)
            if key in seen:
                warnings.append(f"design variable '{key}' appears in both {seen[key]} and {section}; later value wins")
            seen[key] = section

    check("variables", data.get("variables", {}))
    check("source.variables", data.get("source", {}).get("variables", {}) if isinstance(data.get("source"), dict) else {})
    for i, src in enumerate(data.get("sources", []) or []):
        if isinstance(src, dict):
            check(f"sources[{i}].variables", src.get("variables", {}))
    check("circuit.variables", data.get("circuit", {}).get("variables", {}) if isinstance(data.get("circuit"), dict) else {})
    check("load.variables", data.get("load", {}).get("variables", {}) if isinstance(data.get("load"), dict) else {})
    return warnings


# -----------------------------------------------------------------------------
# SPICE and YAML value helpers
# -----------------------------------------------------------------------------

_PLAIN_NAME = re.compile(r"^[A-Za-z_]\w*$")
_SPICE_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*$")
_SPECIAL_PARAM_NAMES = {
    "topology", "circuit_builder", "builder", "load_model", "load_name",
    "name_variable", "solver", "objective", "optimizer",
}


def is_special_param(name: str) -> bool:
    return name in _SPECIAL_PARAM_NAMES or name.endswith("_choice") or name.endswith("_name")


def is_spice_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("$"):
            return False
        return bool(_SPICE_NUMBER.match(text))
    return False


def should_emit_spice_param(name: str, value: Any) -> bool:
    return (not is_special_param(str(name))) and is_spice_literal(value)


def resolve_value(value: Any, params: dict[str, Any], default: Any | None = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        key = value[1:] if value.startswith("$") else value
        if key in params:
            return params[key]
    return value


# Friendly aliases used by method modules.
param_ref_or_value = resolve_value
plain_value = resolve_value


def pick_value(cfg: dict[str, Any], key: str, params: dict[str, Any], default: Any) -> Any:
    return resolve_value(cfg.get(key, default), params, default)


def spice_value(value: Any) -> str:
    """Render a Python/YAML value as a compact SPICE value."""

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        raise ValueError("SPICE value cannot be None")
    text = str(value).strip()
    if not text:
        return text
    if text.startswith("$") and _PLAIN_NAME.match(text[1:]):
        return "{" + text[1:] + "}"
    if text.startswith("{") and text.endswith("}"):
        return text
    if _SPICE_NUMBER.match(text):
        return text
    if _PLAIN_NAME.match(text):
        return "{" + text + "}"
    return text
