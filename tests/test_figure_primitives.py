"""Geometry contracts for the small publication-figure layer."""

from __future__ import annotations

import struct

import pytest
import schemdraw.elements as elm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bench.figures.generate import ROOT as FIGURE_ROOT
from bench.figures.generate import _repository_path
from pcd.figures import PAGE_SIZE, CircuitDiagram, CircuitViewport, assert_text_inside_canvas


def _figure_with_axis(figsize: tuple[float, float] = (6.4, 4.8)):
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def test_component_symbol_length_does_not_follow_layout_span():
    _figure, axis = _figure_with_axis()
    circuit = CircuitDiagram(axis)
    circuit.component(elm.Resistor, (0.0, 0.0), (2.0, 0.0), name="short-span")
    circuit.component(elm.Inductor, (0.0, -1.0), (3.5, -1.0), name="long-span")

    lengths = [placement.symbol_length for placement in circuit.placements]
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


def test_named_anchor_cannot_silently_move():
    _figure, axis = _figure_with_axis()
    circuit = CircuitDiagram(axis)
    assert circuit.anchor("load", (2.0, 0.0)) == (2.0, 0.0)
    assert circuit.anchor("load") == (2.0, 0.0)
    with pytest.raises(ValueError, match="already declared"):
        circuit.anchor("load", (2.1, 0.0))


def test_fixed_page_exports_exact_pixel_dimensions(tmp_path):
    figure = Figure(figsize=PAGE_SIZE)
    FigureCanvasAgg(figure)
    figure.text(0.5, 0.5, "inside", ha="center", va="center")
    assert_text_inside_canvas(figure)
    path = tmp_path / "page.png"
    figure.savefig(path, dpi=100)
    with path.open("rb") as handle:
        handle.seek(16)
        width, height = struct.unpack(">II", handle.read(8))
    assert (height, width) == (480, 725)


def test_committed_figure_provenance_uses_repository_relative_paths():
    assert _repository_path(FIGURE_ROOT / "runs" / "benchmark_suite") == "runs/benchmark_suite"

    with pytest.raises(ValueError, match="inside the repository"):
        _repository_path(FIGURE_ROOT.parent / "external-run")
