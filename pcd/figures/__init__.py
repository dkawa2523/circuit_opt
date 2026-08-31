"""Small, deterministic primitives for publication figures.

Benchmark-specific data and interpretation stay under :mod:`bench.figures`.
This package owns only the visual grammar shared by schematics and plots.
"""

from pcd.figures.circuit import CircuitDiagram, CircuitStyle, CircuitViewport, PlacedComponent
from pcd.figures.publication import (
    BLUE,
    BLUE_LIGHT,
    GOLD,
    GRID,
    INK,
    LIGHT,
    MUTED,
    ORANGE,
    ORANGE_LIGHT,
    PAGE_SIZE,
    WHITE,
    add_figure_footer,
    add_figure_title,
    add_panel_title,
    assert_text_inside_canvas,
    configure_publication_style,
)

__all__ = [
    "BLUE",
    "BLUE_LIGHT",
    "GOLD",
    "GRID",
    "INK",
    "LIGHT",
    "MUTED",
    "ORANGE",
    "ORANGE_LIGHT",
    "PAGE_SIZE",
    "WHITE",
    "CircuitDiagram",
    "CircuitStyle",
    "CircuitViewport",
    "PlacedComponent",
    "add_figure_footer",
    "add_figure_title",
    "add_panel_title",
    "assert_text_inside_canvas",
    "configure_publication_style",
]
