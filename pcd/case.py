"""Case loading, path resolution, and design-variable discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Case:
    """A loaded case plus its authored and deterministically resolved forms."""

    path: Path
    data: dict[str, Any]
    source_data: dict[str, Any] | None = None
    resolved_plan: dict[str, Any] | None = None

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def case_id(self) -> str:
        return str(self.data.get("case_id", self.path.stem))

    @property
    def authored_data(self) -> dict[str, Any]:
        return self.source_data if self.source_data is not None else self.data

    @property
    def is_resolved_rf(self) -> bool:
        return self.resolved_plan is not None


def load_case(path: str | Path) -> Case:
    """Load exactly one supported input schema and resolve public RF inputs."""

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

    # Local import avoids a case -> plan -> case cycle while keeping schema
    # routing at the only file-loading boundary.
    from .plan import EXECUTABLE_SCHEMA, PUBLIC_SCHEMA, compile_rf_case

    schema = str(data.get("schema", "")).strip()
    if schema == PUBLIC_SCHEMA:
        plan = compile_rf_case(data, path.parent)
        return Case(path=path, data=plan.case, source_data=data, resolved_plan=plan.to_dict())
    if schema == EXECUTABLE_SCHEMA:
        return Case(path=path, data=data)
    if not schema:
        raise ValueError(
            f"case schema is required; use {PUBLIC_SCHEMA!r} for RF studies "
            f"or {EXECUTABLE_SCHEMA!r} for advanced explicit cases: {path}"
        )
    raise ValueError(f"unsupported case schema {schema!r}; expected {PUBLIC_SCHEMA!r} or {EXECUTABLE_SCHEMA!r}: {path}")


def resolve_path(case: Case, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else case.base_dir / path


_VARIABLE_SECTIONS = ("variables", "source", "circuit", "load")


def _variable_sections(case: Case) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for section in _VARIABLE_SECTIONS:
        block = case.data.get(section)
        if section == "variables":
            variables = block
            label = "variables"
        elif isinstance(block, dict):
            variables = block.get("variables")
            label = f"{section}.variables"
        else:
            continue
        if isinstance(variables, dict):
            found.append((label, variables))

    for index, source in enumerate(case.data.get("sources") or []):
        if isinstance(source, dict) and isinstance(source.get("variables"), dict):
            found.append((f"sources[{index}].variables", source["variables"]))
    return found


def variable_specs(case: Case) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for _label, variables in _variable_sections(case):
        for name, spec in variables.items():
            specs[str(name)] = dict(spec) if isinstance(spec, dict) else {"default": spec}
    return specs


def default_params(case: Case) -> dict[str, Any]:
    return {name: spec["default"] for name, spec in variable_specs(case).items() if "default" in spec}


def fill_default_params(case: Case, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**default_params(case), **(params or {})}


NO_SOURCE_WARNING = "no source/source list found; netlist may have no independent source"


def case_warnings(case: Case) -> list[str]:
    """Return only concise warnings worth retaining with run artifacts."""

    warnings: list[str] = []
    if not case.data.get("source") and not case.data.get("sources"):
        warnings.append(NO_SOURCE_WARNING)

    declared_in: dict[str, str] = {}
    for label, variables in _variable_sections(case):
        for name in variables:
            key = str(name)
            if key in declared_in:
                warnings.append(
                    f"design variable '{key}' appears in both {declared_in[key]} and {label}; later value wins"
                )
            declared_in[key] = label
    return warnings
