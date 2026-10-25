"""Development checks in the uv-managed environment.

    uv run --frozen python -m nox -s quality-fast  # static checks and unit tests
    uv run --frozen python -m nox -s quality-pr    # all tests and coverage report
    uv run --frozen python -m nox -s format        # formatting and safe fixes
    uv run --frozen python -m nox -s clean         # generated build/cache files

Numerical regressions judge electrical behavior. Coverage is reported without
an arbitrary percentage gate that encourages tests tied to private structure.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["quality-fast"]

ROOT = Path(__file__).resolve().parent
SOURCES = ["pcd", "tests", "examples", "bench", "noxfile.py"]


def _static_checks(session: nox.Session) -> None:
    session.run(sys.executable, "-m", "ruff", "format", "--check", *SOURCES)
    session.run(sys.executable, "-m", "ruff", "check", *SOURCES)
    session.run(sys.executable, "-m", "pyrefly", "check")
    session.run(
        sys.executable,
        "-c",
        "from importlinter.cli import lint_imports_command; lint_imports_command()",
        "--config",
        ".importlinter",
    )


@nox.session(name="format", python=None)
def format_sources(session: nox.Session) -> None:
    """Apply the formatter and Ruff's safe fixes."""
    targets = session.posargs or SOURCES
    session.run(sys.executable, "-m", "ruff", "format", *targets)
    session.run(sys.executable, "-m", "ruff", "check", "--fix", *targets)


@nox.session(name="quality-fast", python=None)
def quality_fast(session: nox.Session) -> None:
    """Read-only static checks and unit tests."""
    _static_checks(session)
    session.run(sys.executable, "-m", "pytest", "-q", "-m", "not e2e", *session.posargs)


@nox.session(name="quality-pr", python=None)
def quality_pr(session: nox.Session) -> None:
    """Static boundaries, all available tests, and a coverage report."""
    _static_checks(session)
    session.run(sys.executable, "-m", "pytest", "-q", "--cov=pcd", "--cov-branch", *session.posargs)


@nox.session(name="quality-nightly", python=None)
def quality_nightly(session: nox.Session) -> None:
    """User journeys against a real ngspice installation."""
    session.run(sys.executable, "-m", "pytest", "-q", "-m", "e2e", *session.posargs)


@nox.session(name="benchmark-design", python=None)
def benchmark_design(session: nox.Session) -> None:
    """Reproduce independent positive and negative electrical cases."""
    args = session.posargs or ["--run-root", str(ROOT / "runs" / "benchmark_suite")]
    session.run(sys.executable, str(ROOT / "bench" / "run_suite.py"), *args)


@nox.session(python=None)
def clean(session: nox.Session) -> None:
    """Remove local build/cache files; retain studies and authored documents."""
    names = (
        ".coverage",
        ".hypothesis",
        ".import_linter_cache",
        ".grimp_cache",
        ".importlinter_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "htmlcov",
        "review_tmp",
        "tmp",
        ".codex-pptx-build",
    )
    targets = [ROOT / name for name in names]
    targets.extend(ROOT.glob("*.egg-info"))
    targets.extend(ROOT.glob(".chart-data-*"))
    for parent in (ROOT, ROOT / "pcd", ROOT / "tests", ROOT / "examples", ROOT / "bench"):
        targets.extend(parent.glob("__pycache__") if parent == ROOT else parent.rglob("__pycache__"))
    removed = 0
    for path in targets:
        if not path.absolute().is_relative_to(ROOT) or path == ROOT:
            raise ValueError(f"cleanup target is outside the workspace: {path}")
        if path.is_symlink() or path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"cleanup target resolves outside the workspace: {path}")
            shutil.rmtree(path)
            removed += 1
    session.log(f"removed {removed} generated paths")
