"""Preserve an authored SPICE deck while the case owns execution control.

The topology reader intentionally discards statements it cannot draw.  That
is the wrong boundary for simulation: model, parameter, include, conditional,
and subcircuit statements can all change the circuit. This module preserves
these statements, removes commands that would compete with the case's generated
analysis block, and identifies explicit terminal references in the import scope.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

NETLIST_IMPORT_MODES = frozenset({"fragment", "deck"})
SOURCE_POLICIES = frozenset({"preserve", "replace_named"})
_INLINE_COMMENT = re.compile(r""""[^"]*"|'[^']*'|(?P<comment>\s[;$].*)""")
_STOCHASTIC_FUNCTION = re.compile(r"\b(agauss|gauss|aunif|unif|sgauss|sunif|trrandom|trnoise)\s*\(", re.IGNORECASE)
_LITERAL_NODE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_PARAMETER_START = re.compile(r"[\s,]+(?=[^\s,=]+\s*=|params:(?:\s|$))", re.IGNORECASE)
_SPICE_CARD_FIELD = re.compile(r""""[^"]*"|'[^']*'|\{[^}]*\}|[^\s,()=]+""")
_DEVICE_NODE_COUNTS: dict[str, int] = {
    **dict.fromkeys("BCDEFGHILRVW", 2),
    **dict.fromkeys("JMQUZ", 3),
    **dict.fromkeys("OST", 4),
}


@dataclass(frozen=True, slots=True)
class ImportedLine:
    text: str
    subcircuit_depth: int

    @property
    def top_level(self) -> bool:
        return self.subcircuit_depth == 0

    @property
    def statement(self) -> str:
        """Read the statement's tokens without changing how it is emitted."""

        return _statement_text(self.text)


def _logical_statements(lines: Iterable[str]) -> Iterator[str]:
    statement: list[str] = []
    comments: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*"):
            comments.append(raw)
        elif line.startswith("+"):
            if not statement:
                raise ValueError("SPICE continuation has no preceding statement")
            statement.extend(comments)
            comments.clear()
            statement.append(raw)
        else:
            if statement:
                yield "\n".join(statement)
            yield from comments
            comments.clear()
            statement = [raw]
    if statement:
        yield "\n".join(statement)
    yield from comments


def _statement_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0].startswith("*"):
        return text.strip()
    content = [lines[0], *(line[1:].strip() for line in lines[1:] if line.startswith("+"))]
    parts = []
    for line in content:
        comment = next((match for match in _INLINE_COMMENT.finditer(line) if match["comment"] is not None), None)
        parts.append(line[: comment.start()].strip() if comment is not None else line)
    return " ".join(parts)


def spice_card_fields(statement: str) -> list[str]:
    """Read device fields with ngspice delimiters, keeping values in quotes or braces intact."""
    return _SPICE_CARD_FIELD.findall(statement)


def flatten_netlist_file(
    path: str | Path, *, mode: str = "fragment", captured: dict[Path, bytes] | None = None
) -> tuple[str, tuple[Path, ...]]:
    """Inline readable include/library dependencies into one portable deck.

    The returned dependency paths are the exact source files that contributed
    content and can therefore be hashed separately in an input manifest.
    In deck mode the first physical line is retained as an opaque title.
    Missing or cyclic references remain as absolute directives so ngspice can
    report the problem during ordinary inspection. A replay capture rejects
    unresolved dependencies instead of retaining pointers to live inputs.
    """

    source = Path(path).resolve()
    dependencies: list[Path] = []

    def read(file: Path) -> str:
        if captured is None:
            return file.read_text(encoding="utf-8")
        if file not in captured:
            captured[file] = file.read_bytes()
        return captured[file].decode("utf-8")

    title, body = _circuit_lines(read(source), mode)
    text = _flatten_netlist_lines(
        body,
        source.parent,
        stack=(source,),
        dependencies=dependencies,
        read=read,
        strict=captured is not None,
    )
    return (text if title is None else f"{title}\n{text}"), tuple(dependencies)


def _flatten_netlist_lines(
    physical_lines: Iterable[str],
    source_dir: Path,
    *,
    stack: tuple[Path, ...],
    dependencies: list[Path],
    read: Callable[[Path], str],
    strict: bool,
) -> str:
    lines: list[str] = []
    for raw in _logical_statements(physical_lines):
        statement = _statement_text(raw)
        reference = _file_reference(statement, source_dir)
        if reference is None:
            lines.append(raw)
            continue
        dependency, section = reference
        if dependency not in dependencies:
            dependencies.append(dependency)
        if dependency in stack:
            if strict:
                raise ValueError(f"cyclic SPICE dependency: {dependency}")
            lines.append(_relocate_file_directive(statement, source_dir))
            continue
        try:
            included = read(dependency)
        except FileNotFoundError:
            if strict:
                raise
            lines.append(_relocate_file_directive(statement, source_dir))
            continue
        if section is not None:
            selected = _library_section(included, section)
            if selected is None:
                if strict:
                    raise ValueError(f"SPICE library section {section!r} not found: {dependency}")
                lines.append(_relocate_file_directive(statement, source_dir))
                continue
            included = selected
        lines.append(f"* begin inlined dependency: {dependency.name}")
        lines.extend(
            _flatten_netlist_lines(
                included.splitlines(),
                dependency.parent,
                stack=(*stack, dependency),
                dependencies=dependencies,
                read=read,
                strict=strict,
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

    _title, body = _circuit_lines(text, mode)
    for line in _logical_statements(body):
        line = line.strip()
        if not line:
            continue
        content = _statement_text(line)
        statement = content.split(maxsplit=1)[0].lower()
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
            result.append(ImportedLine(line, subckt_depth))
            continue
        if statement == ".ends":
            result.append(ImportedLine(line, max(1, subckt_depth)))
            subckt_depth = max(0, subckt_depth - 1)
            continue

        top_level = subckt_depth == 0
        if top_level and statement in _CASE_OWNED_DIRECTIVES:
            continue
        relocated = _relocate_file_directive(content, base)
        result.append(ImportedLine(line if relocated == content else relocated, subckt_depth))
    return result


def top_level_spice_nodes(text: str, *, mode: str = "fragment") -> set[str]:
    """Return literal nodes supported by unconditional top-level terminal cards.

    Inspect a flattened DUT before any fixture is added. This is a positive
    inventory: unknown device syntax and condition-dependent connections do not
    establish terminals. Subcircuit instances can establish external pins, and
    ground is always available. Internal nodes and expression references are
    outside this inventory.
    """
    subcircuits: dict[str, int] = {}
    models: set[str] = set()
    devices: list[list[str]] = []
    conditional_depth = 0
    for line in executable_netlist_lines(text, mode=mode):
        statement = line.statement
        parts = spice_card_fields(statement)
        if not parts:
            continue
        head = parts[0].lower()
        if line.top_level and head == ".if":
            conditional_depth += 1
        elif line.top_level and head == ".endif":
            conditional_depth = max(0, conditional_depth - 1)
        elif not conditional_depth:
            if head == ".subckt" and line.subcircuit_depth == 1:
                positional = spice_card_fields(_PARAMETER_START.split(statement, maxsplit=1)[0])
                if len(positional) >= 2:
                    subcircuits[positional[1].lower()] = len(positional) - 2
            elif line.top_level and head == ".model" and len(parts) >= 3:
                models.add(parts[1].lower())
            elif line.top_level and head[0].isalpha():
                if head.startswith("x"):
                    parts = spice_card_fields(_PARAMETER_START.split(statement, maxsplit=1)[0])
                devices.append(parts)

    nodes = {"0"}
    for parts in devices:
        terminals = _device_terminals(parts, subcircuits, models)
        if all(_LITERAL_NODE.fullmatch(node) for node in terminals):
            nodes.update(node.lower() for node in terminals)
    return nodes


def _device_terminals(parts: list[str], subcircuits: dict[str, int], models: set[str]) -> list[str]:
    prefix = parts[0][0].upper()
    if prefix == "X":
        return parts[1:-1] if subcircuits.get(parts[-1].lower()) == len(parts) - 2 else []
    count = _DEVICE_NODE_COUNTS.get(prefix, 0)
    if prefix in {"M", "Q"}:
        # VDMOS, bulk MOSFETs and BJTs have model-dependent terminal counts.
        count = next((i - 1 for i in range(4, min(len(parts), 7)) if parts[i].lower() in models), count)
    elif prefix in {"E", "G"} and len(parts) >= 6:
        # Linear controlled sources have two extra nodes. Behavioral source
        # keywords and expressions occupy these positions in the other forms.
        control = parts[3:5]
        if all(_LITERAL_NODE.fullmatch(node) for node in control) and control[0].lower() not in {
            "poly",
            "value",
            "vol",
            "cur",
            "table",
            "laplace",
            "freq",
        }:
            count = 4
    minimum_fields = count + (1 if prefix in {"V", "I"} else 2)
    return parts[1 : count + 1] if count and len(parts) >= minimum_fields else []


def has_implicit_node_shunts(text: str, *, mode: str = "fragment") -> bool:
    """Identify ngspice options that attach elements to every analog node."""
    for line in executable_netlist_lines(text, mode=mode):
        fields = spice_card_fields(line.statement.lower())
        if fields and fields[0] in {".option", ".options"} and {"rshunt", "cshunt"}.intersection(fields[1:]):
            return True
    return False


def stochastic_netlist_function(text: str, *, mode: str = "fragment") -> str | None:
    """Find a known ngspice random function in an executable deck statement."""

    for line in executable_netlist_lines(text, mode=mode):
        statement = line.statement
        if statement.startswith(("*", ";", "$")) or statement.split(maxsplit=1)[0].lower() == ".title":
            continue
        if match := _STOCHASTIC_FUNCTION.search(statement):
            return match[1].lower()
    return None


def _circuit_lines(text: str, mode: str) -> tuple[str | None, list[str]]:
    normalized = str(mode).strip().lower()
    if normalized not in NETLIST_IMPORT_MODES:
        raise ValueError(f"netlist_mode must be one of {sorted(NETLIST_IMPORT_MODES)}")
    lines = text.splitlines()
    return (lines[0], lines[1:]) if lines and normalized == "deck" else (None, lines)


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
