"""Fixed-grid circuit primitives on top of Schemdraw.

Schemdraw two-terminal elements stretch to their requested endpoints.  This
wrapper keeps every R/L/C symbol at one declared endpoint length and fills any
remaining span with ordinary wire.  A fixed equal-aspect viewport then keeps a
schematic unit the same physical size across comparable panels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import schemdraw
import schemdraw.elements as elm

Point = tuple[float, float]
ElementFactory = Callable[[], Any]


@dataclass(frozen=True)
class CircuitStyle:
    """Physical drawing constants shared by all publication schematics."""

    unit: float = 2.2
    component_length: float = 1.15
    line_width: float = 1.25
    font_size: float = 8.2
    color: str = "#20262d"


@dataclass(frozen=True)
class CircuitViewport:
    """Explicit data-space viewport; comparable panels should share one."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass(frozen=True)
class PlacedComponent:
    """Auditable geometry for one fixed-length two-terminal symbol."""

    name: str
    span_start: Point
    symbol_start: Point
    symbol_end: Point
    span_end: Point

    @property
    def symbol_length(self) -> float:
        return math.dist(self.symbol_start, self.symbol_end)


class CircuitDiagram:
    """Draw an orthogonal, fixed-symbol-length circuit into one Matplotlib axis."""

    def __init__(self, axis: Any, style: CircuitStyle | None = None) -> None:
        self.axis = axis
        self.style = style or CircuitStyle()
        self.drawing = schemdraw.Drawing(canvas=axis, show=False)
        self.drawing.config(
            unit=self.style.unit,
            lw=self.style.line_width,
            fontsize=self.style.font_size,
            color=self.style.color,
        )
        self.anchors: dict[str, Point] = {}
        self.placements: list[PlacedComponent] = []

    def anchor(self, name: str, at: Point | None = None) -> Point:
        """Declare or retrieve a logical node coordinate by name."""

        if at is not None:
            existing = self.anchors.get(name)
            if existing is not None and not _same_point(existing, at):
                raise ValueError(f"anchor {name!r} was already declared at {existing}, not {at}")
            self.anchors[name] = at
        if name not in self.anchors:
            raise KeyError(f"unknown circuit anchor: {name}")
        return self.anchors[name]

    def wire(self, start: Point, end: Point) -> None:
        """Connect two named coordinates with a plain wire."""

        if not _same_point(start, end):
            self.drawing.add(elm.Line().at(start).to(end))

    def node(self, at: Point) -> None:
        """Draw an electrical junction."""

        self.drawing.add(elm.Dot().at(at))

    def port(self, at: Point, label: str, label_location: str) -> None:
        """Draw a named open terminal."""

        self.drawing.add(elm.Dot(open=True).at(at).label(label, loc=label_location))

    def ground(self, at: Point) -> None:
        """Draw a ground whose terminal anchor is *at*."""

        self.drawing.add(elm.Ground().at(at))

    def component(
        self,
        factory: ElementFactory,
        start: Point,
        end: Point,
        *,
        label: str | None = None,
        label_location: str | None = None,
        name: str | None = None,
    ) -> Any:
        """Place one fixed-length symbol centered in an orthogonal span.

        The component never absorbs layout distance.  Wires before and after
        it fill the span, so changing a branch height or node pitch cannot
        silently resize the electrical symbol.
        """

        dx, dy = end[0] - start[0], end[1] - start[1]
        if abs(dx) > 1e-10 and abs(dy) > 1e-10:
            raise ValueError(f"circuit components require an orthogonal span: {start} -> {end}")
        span_length = math.hypot(dx, dy)
        if span_length + 1e-10 < self.style.component_length:
            raise ValueError(
                f"component span {span_length:g} is shorter than fixed symbol length "
                f"{self.style.component_length:g}: {start} -> {end}"
            )
        if span_length <= 1e-12:
            raise ValueError("component span cannot be zero")

        ux, uy = dx / span_length, dy / span_length
        lead = 0.5 * (span_length - self.style.component_length)
        symbol_start = (start[0] + ux * lead, start[1] + uy * lead)
        symbol_end = (end[0] - ux * lead, end[1] - uy * lead)
        self.wire(start, symbol_start)
        element = factory().at(symbol_start).to(symbol_end)
        if label is not None:
            kwargs = {"loc": label_location} if label_location else {}
            element = element.label(label, **kwargs)
        self.drawing.add(element)
        self.wire(symbol_end, end)
        placement = PlacedComponent(
            name=name or getattr(factory, "__name__", type(element).__name__),
            span_start=start,
            symbol_start=symbol_start,
            symbol_end=symbol_end,
            span_end=end,
        )
        self.placements.append(placement)
        return element

    def finish(self, viewport: CircuitViewport) -> None:
        """Render with a fixed viewport and equal x/y physical scale."""

        self._audit_geometry()
        self.drawing.draw(show=False)
        self.axis.set_xlim(*viewport.xlim)
        self.axis.set_ylim(*viewport.ylim)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_axis_off()

    def _audit_geometry(self) -> None:
        expected = self.style.component_length
        for placement in self.placements:
            if not math.isclose(placement.symbol_length, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{placement.name} symbol length drifted: {placement.symbol_length:g} != {expected:g}")


def _same_point(first: Point, second: Point) -> bool:
    return math.isclose(first[0], second[0], abs_tol=1e-12) and math.isclose(first[1], second[1], abs_tol=1e-12)
