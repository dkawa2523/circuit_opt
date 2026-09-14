"""Small, explicit maintenance operations for completed study generations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def prune_generations(study_root: str | Path, *, keep: int = 3, apply: bool = False) -> dict[str, Any]:
    """Plan or remove old generations while always preserving the active one.

    Raw/evaluation caches are deliberately outside this operation. The default
    is a dry run so retention remains an explicit operator decision.
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    root = Path(study_root).resolve()
    result_path = root / "study_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"study_result.json not found under {root}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    generation_dir = (root / "generations").resolve()
    declared = str((payload.get("artifacts") or {}).get("generation", ""))
    active = (root / declared).resolve() if declared else None
    if active is None or active.parent != generation_dir or not active.is_dir():
        raise ValueError("study_result.json does not declare an active generation under generations/")

    discovered = [path.resolve() for path in generation_dir.glob("g_*") if path.is_dir()]
    generations = sorted(
        (path for path in discovered if path.parent == generation_dir),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    retained = set(generations[:keep]) | {active}
    removable = [path for path in generations if path not in retained]
    if apply:
        for path in removable:
            shutil.rmtree(path)
    return {
        "study_root": str(root),
        "applied": apply,
        "active_generation": active.relative_to(root).as_posix(),
        "kept": [path.relative_to(root).as_posix() for path in generations if path in retained],
        "removed": [path.relative_to(root).as_posix() for path in removable],
    }
