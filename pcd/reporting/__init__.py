"""Saved-run electrical summaries and standard engineering figures."""

from collections.abc import Callable
from typing import Any

from .run import analyze_run
from .study import analyze_saved

__all__ = ["analyze_run", "analyze_saved"]


def render_requested_output(render: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Keep a completed engineering result usable when its presentation fails."""

    try:
        return render()
    except Exception as exc:
        return {"status": "partial", "summary": None, "error": f"{type(exc).__name__}: {exc}"}
