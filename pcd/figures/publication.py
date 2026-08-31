"""One restrained visual system for circuit and benchmark figures."""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import schemdraw.elements as elm

INK = "#20262d"
MUTED = "#68737d"
GRID = "#d8dde2"
LIGHT = "#eef1f3"
BLUE = "#2a6f97"
BLUE_LIGHT = "#dcebf3"
ORANGE = "#c66a12"
ORANGE_LIGHT = "#f5e1cd"
GOLD = "#a57d1b"
WHITE = "#ffffff"

# A fixed 7.25-inch column-spanning canvas is the pack's publication master.
PAGE_SIZE = (7.25, 4.8)


def configure_publication_style() -> None:
    """Apply the project-wide chart and IEEE circuit-symbol style."""

    elm.style(elm.STYLE_IEEE)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def add_panel_title(axis: Any, label: str, title: str) -> None:
    """Place a panel label and its neutral descriptive title on one baseline."""

    anchor = (0.0, 1.055)
    axis.annotate(
        label,
        anchor,
        xycoords="axes fraction",
        xytext=(-2, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    axis.annotate(
        title,
        anchor,
        xycoords="axes fraction",
        xytext=(32, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontweight="normal",
    )


def add_figure_title(figure: Any, title: str, subtitle: str) -> None:
    """Add the pack's shared title and evidence-context subtitle."""

    figure.text(0.01, 0.985, title, ha="left", va="top", fontsize=11.0, fontweight="bold")
    figure.text(0.01, 0.92, subtitle, ha="left", va="top", fontsize=7.7, color=MUTED)


def add_figure_footer(figure: Any, text: str) -> None:
    """Add a concise source or scope note inside the fixed canvas."""

    figure.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=6.6, color=MUTED)


def assert_text_inside_canvas(figure: Any, tolerance_px: float = 1.5) -> None:
    """Reject exported figures whose visible text is clipped by the page.

    This deliberately checks containment rather than pixel identity.  Font
    rasterization differs across systems, while text outside the fixed canvas
    is a real publication defect on every system.
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    page = figure.bbox
    offenders: list[str] = []
    for text in figure.findobj(match=lambda artist: isinstance(artist, mpl.text.Text)):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < page.x0 - tolerance_px
            or bounds.y0 < page.y0 - tolerance_px
            or bounds.x1 > page.x1 + tolerance_px
            or bounds.y1 > page.y1 + tolerance_px
        ):
            offenders.append(text.get_text().replace("\n", " ")[:80])
    if offenders:
        joined = "; ".join(offenders[:5])
        raise ValueError(f"figure text leaves the fixed canvas: {joined}")
