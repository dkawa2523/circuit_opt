"""Preserve an authored SPICE deck while the case owns execution control.

The topology reader intentionally discards statements it cannot draw.  That
is the wrong boundary for simulation: model, parameter, include, conditional,
and subcircuit statements can all change the circuit.  This module therefore
does no topology interpretation.  It only removes commands that would compete
with the case's generated analysis block and records whether a line belongs to
the top-level circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NETLIST_IMPORT_MODES = frozenset({"fragment", "deck"})
SOURCE_POLICIES = frozenset({"preserve", "replace_named"})


@dataclass(frozen=True, slots=True)
class ImportedLine:
    text: str
    top_level: bool


def flatten_netlist_file(path: str | Path) -> tuple[str, tuple[Path, ...]]:
    """Inline readable include/library dependencies into one portable deck.

    The returned dependency paths are the exact source files that contributed
    content and can therefore be hashed separately in an input manifest.
    Missing or cyclic references remain as absolute directives so ngspice can
    report the problem without the archiver losing the run record.
    """

    source = Path(path).resolve()
    dependencies: list[Path] = []
    text = _flatten_netlist_text(
        source.read_text(encoding="utf-8"),
        source.parent,
        stack=(source,),
        dependencies=dependencies,
    )
    return text, tuple(dependencies)


def _flatten_netlist_text(
    text: str,
    source_dir: Path,
    *,
    stack: tuple[Path, ...],
    dependencies: list[Path],
) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        reference = _file_reference(raw, source_dir)
        if reference is None:
            lines.append(raw)
            continue
        dependency, section = reference
        if dependency not in dependencies:
            dependencies.append(dependency)
        if dependency in stack or not dependency.is_file():
            lines.append(_relocate_file_directive(raw.strip(), source_dir))
            continue
        included = dependency.read_text(encoding="utf-8")
        if section is not None:
            selected = _library_section(included, section)
            if selected is None:
                lines.append(_relocate_file_directive(raw.strip(), source_dir))
                continue
            included = selected
        lines.append(f"* begin inlined dependency: {dependency.name}")
        lines.extend(
            _flatten_netlist_text(
                included,
                dependency.parent,
                stack=(*stack, dependency),
                dependencies=dependencies,
            ).splitlines()
        )
        lines.append(f"* end inlined dependency: {dependency.name}")
    return "\n".join(lines) + "\n"


def _file_reference(line: str, source_dir: Path) -> tuple[Path, str | None] | None:
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    directive = parts[0].lower()
    if directive not in {".include", ".inc", ".lib"}:
        return None
    argument, remainder = _first_argument(parts[1])
    if not argument or (directive == ".lib" and not remainder):
        return None
    declared = Path(argument)
    dependency = (declared if declared.is_absolute() else source_dir / declared).resolve()
    section = _first_argument(remainder)[0] if directive == ".lib" else None
    return dependency, section


def _library_section(text: str, requested: str) -> str | None:
    selected: list[str] = []
    active = False
    for raw in text.splitlines():
        parts = raw.strip().split(maxsplit=1)
        directive = parts[0].lower() if parts else ""
        argument = _first_argument(parts[1])[0] if len(parts) == 2 else ""
        if not active and directive == ".lib" and argument.lower() == requested.lower():
            active = True
            continue
        if active and directive == ".endl":
            return "\n".join(selected) + "\n"
        if active:
            selected.append(raw)
    return None


# The case owns source-independent execution and artifact capture.  Everything
# else, including .param/.model/.include/.lib/.options/.save, is circuit input
# and must survive.
_CASE_OWNED_DIRECTIVES = {
    ".ac",
    ".dc",
    ".end",
    ".four",
    ".meas",
    ".measure",
    ".noise",
    ".op",
    ".plot",
    ".print",
    ".pz",
    ".sens",
    ".tf",
    ".tran",
}


def executable_netlist_lines(
    text: str,
    source_dir: str | Path | None = None,
    *,
    mode: str = "fragment",
) -> list[ImportedLine]:
    """Return semantically relevant deck lines in their authored order.

    ``.control`` blocks and top-level analysis/output statements are omitted
    because the platform emits one deterministic control block.  Relative
    first-level include/library paths are made absolute before the generated
    deck is executed from its run directory.  A complete SPICE ``deck`` has a
    mandatory first-line title, which is discarded because the generated deck
    supplies its own.  A ``fragment`` has no title convention, so every line is
    interpreted as circuit input.
    """

    base = Path(source_dir).resolve() if source_dir is not None else None
    result: list[ImportedLine] = []
    in_control = False
    subckt_depth = 0

    for raw in _circuit_body_lines(text, mode):
        line = raw.strip()
        if not line:
            continue
        statement = line.split(maxsplit=1)[0].lower()
        if statement == ".control":
            in_control = True
            continue
        if statement == ".endc":
            in_control = False
            continue
        if in_control:
            continue

        if statement == ".subckt":
            subckt_depth += 1
            result.append(ImportedLine(_relocate_file_directive(line, base), top_level=False))
            continue
        if statement == ".ends":
            result.append(ImportedLine(line, top_level=False))
            subckt_depth = max(0, subckt_depth - 1)
            continue

        top_level = subckt_depth == 0
        if top_level and statement in _CASE_OWNED_DIRECTIVES:
            continue
        result.append(ImportedLine(_relocate_file_directive(line, base), top_level=top_level))
    return result


def _circuit_body_lines(text: str, mode: str) -> list[str]:
    normalized = str(mode).strip().lower()
    if normalized not in NETLIST_IMPORT_MODES:
        raise ValueError(f"netlist_mode must be one of {sorted(NETLIST_IMPORT_MODES)}")
    lines = text.splitlines()
    return lines[1:] if normalized == "deck" else lines


def _relocate_file_directive(line: str, base: Path | None) -> str:
    """Resolve the filename of .include/.inc and file-form .lib statements."""

    if base is None:
        return line
    parts = line.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() not in {".include", ".inc", ".lib"}:
        return line
    argument, remainder = _first_argument(parts[1])
    if not argument:
        return line
    # A one-argument .lib may delimit a library section rather than name a
    # file.  Only relocate it when it visibly names a file or has a section
    # argument following it.
    candidate = Path(argument)
    if parts[0].lower() == ".lib" and not remainder and not (base / candidate).is_file():
        return line
    if candidate.is_absolute():
        return line
    resolved = (base / candidate).resolve().as_posix()
    suffix = f" {remainder}" if remainder else ""
    return f'{parts[0]} "{resolved}"{suffix}'


def _first_argument(text: str) -> tuple[str, str]:
    value = text.strip()
    if not value:
        return "", ""
    if value[0] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            return value[1:], ""
        return value[1:end], value[end + 1 :].strip()
    parts = value.split(maxsplit=1)
    return parts[0], parts[1].strip() if len(parts) == 2 else ""
