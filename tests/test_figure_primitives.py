"""Geometry contracts for circuit diagrams."""

from __future__ import annotations

import math

import pytest
import schemdraw.elements as elm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pcd.figures.circuit import CircuitDiagram, CircuitViewport


def _figure_with_axis(figsize: tuple[float, float] = (6.4, 4.8)):
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def test_component_symbol_length_does_not_follow_layout_span():
    _figure, axis = _figure_with_axis()
    circuit = CircuitDiagram(axis)
    resistor = circuit.component(elm.Resistor, (0.0, 0.0), (2.0, 0.0))
    inductor = circuit.component(elm.Inductor, (0.0, -1.0), (3.5, -1.0))

    lengths = [math.dist(element.absanchors["start"], element.absanchors["end"]) for element in (resistor, inductor)]
    assert lengths == pytest.approx([circuit.style.component_length] * 2)


def test_fixed_viewport_uses_equal_horizontal_and_vertical_scale():
    figure, axis = _figure_with_axis((3.0, 3.0))
    circuit = CircuitDiagram(axis)
    circuit.component(elm.Resistor, (0.5, 2.5), (2.5, 2.5))
    circuit.component(elm.Capacitor, (1.5, 2.5), (1.5, 0.5))
    circuit.finish(CircuitViewport(xlim=(0.0, 3.0), ylim=(0.0, 3.0)))
    figure.canvas.draw()

    origin = axis.transData.transform((0.0, 0.0))
    x_unit = axis.transData.transform((1.0, 0.0))
    y_unit = axis.transData.transform((0.0, 1.0))
    assert x_unit[0] - origin[0] == pytest.approx(y_unit[1] - origin[1], rel=1e-12)


def test_component_rejects_diagonal_or_too_short_spans():
    _figure, axis = _figure_with_axis()
    circuit = CircuitDiagram(axis)
    with pytest.raises(ValueError, match="orthogonal"):
        circuit.component(elm.Resistor, (0.0, 0.0), (2.0, 1.0))
    with pytest.raises(ValueError, match="shorter"):
        circuit.component(elm.Resistor, (0.0, 0.0), (0.5, 0.0))
