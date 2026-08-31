"""Quality entry points.

    uv run nox -s quality-fast       # while editing
    uv run nox -s quality-pr         # before finishing
    uv run nox -s quality-nightly    # scheduled end-to-end/property checks
    uv run nox -s benchmark-design   # engineering classification suite
    uv run nox -s quality-baseline   # record the coverage floor
    uv run nox -s clean              # remove generated local build state

Every gate is a direct tool invocation.  Each tool already exits non-zero on
findings, so there is no wrapper layer to read, debug, or trust.

Sessions run in the uv-managed ``.venv`` (``venv_backend="none"``); create it
once with ``uv sync --group dev``.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["quality-fast"]

ROOT = Path(__file__).resolve().parent
SOURCES = ["pcd", "tests", "examples", "bench", "noxfile.py"]

#: Repository branch-coverage floor.  Raise it with `quality-baseline`; never
#: lower it to make a run pass.
COVERAGE_FLOOR = ROOT / "quality" / "coverage-floor.txt"


@nox.session(python=None)
def clean(session: nox.Session) -> None:
    """Remove local caches, temporary work, and builds without touching studies."""

    targets = [
        ROOT / ".coverage",
        ROOT / ".hypothesis",
        ROOT / ".import_linter_cache",
        ROOT / ".nox",
        ROOT / ".pytest_cache",
        ROOT / ".ruff_cache",
        ROOT / "build",
        ROOT / "dist",
        ROOT / "htmlcov",
        ROOT / "review_tmp",
        ROOT / "tmp",
        *ROOT.glob("*.egg-info"),
        *ROOT.glob("__pycache__"),
        *(ROOT / "pcd").rglob("__pycache__"),
        *(ROOT / "tests").rglob("__pycache__"),
        *(ROOT / "examples").rglob("__pycache__"),
        *(ROOT / "bench").rglob("__pycache__"),
    ]
    removed = 0
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed += 1
        elif target.exists():
            target.unlink()
            removed += 1
    session.log(f"removed {removed} generated paths; runs/ was preserved")


def _floor() -> str:
    return COVERAGE_FLOOR.read_text(encoding="utf-8").strip() if COVERAGE_FLOOR.exists() else "0"


@nox.session(name="quality-fast", python=None)
def quality_fast(session: nox.Session) -> None:
    """Format, lint, type-check, and unit-test.  Safe fixes are applied."""

    targets = session.posargs or SOURCES
    session.run("ruff", "format", *targets, external=True)
    # Safe fixes only.  --unsafe-fixes can change behaviour and is never used.
    session.run("ruff", "check", "--fix", *targets, external=True)
    session.run("pyrefly", "check", external=True)
    session.run("pytest", "-q", "-m", "not e2e", external=True)


@nox.session(name="quality-pr", python=None)
def quality_pr(session: nox.Session) -> None:
    """The full gate.  Read-only: it reports, it never rewrites your code."""

    session.run("ruff", "format", "--check", *SOURCES, external=True)
    session.run("ruff", "check", *SOURCES, external=True)
    session.run("pyrefly", "check", external=True)
    # Note: on Windows, lint-imports exits 1 if its stdout is the NUL device,
    # so redirect this session to a file rather than /dev/null when scripting it.
    session.run("lint-imports", "--config", ".importlinter", external=True)
    session.run("bandit", "-c", "pyproject.toml", "-q", "-r", "pcd", external=True)
    # The pip-audit console-script shim can be blocked by Windows application
    # policy even when the installed module is allowed.  Running the same
    # entry point through this environment's interpreter is portable.
    # Audit dependencies declared by this project, not unrelated packages in
    # whichever host/runtime happens to launch nox.
    session.run(sys.executable, "-m", "pip_audit", ".", "--strict", "--progress-spinner=off", external=True)
    session.run("vulture", external=True)
    # Scoped to source: runs/ holds generated artifacts whose provenance SHA-256
    # digests are not secrets, and scanning them would need a large allowlist.
    session.run("detect-secrets-hook", *SOURCES, external=True)
    session.run("pytest", "-q", "--cov=pcd", "--cov-branch", "--cov-report=", external=True)
    # Checked with `coverage report`, not pytest's --cov-fail-under: the latter
    # compares the *rounded* percentage, so 90.53% would pass a 91% floor while
    # still printing a FAIL line.  Two decimals here, and the exit code agrees
    # with the message.
    session.run(
        "coverage",
        "report",
        "--show-missing",
        "--skip-covered",
        "--precision=2",
        f"--fail-under={_floor()}",
        external=True,
    )


@nox.session(name="quality-nightly", python=None)
def quality_nightly(session: nox.Session) -> None:
    """Scheduled end-to-end and property-based checks."""

    session.run("pytest", "-q", "-m", "e2e", external=True)
    session.run("pytest", "-q", "-m", "property", external=True)


@nox.session(name="benchmark-design", python=None)
def benchmark_design(session: nox.Session) -> None:
    """Reproduce the positive/negative design decisions with real ngspice."""

    run_root = ROOT / "runs" / "benchmark_suite"
    args = session.posargs or ["--run-root", str(run_root)]
    session.run(
        sys.executable,
        str(ROOT / "bench" / "run_suite.py"),
        *args,
        external=True,
    )


@nox.session(name="quality-baseline", python=None)
def quality_baseline(session: nox.Session) -> None:
    """Record the current branch coverage as the new floor.

    Deliberate and manual: no other session and no CI job writes this file.
    """

    session.run("pytest", "-q", "--cov=pcd", "--cov-branch", "--cov-report=", external=True)
    output = session.run(
        "coverage", "report", "--format=total", "--precision=2", external=True, silent=True, success_codes=[0, 2]
    )
    try:
        measured = float(str(output).strip())
    except ValueError:
        session.error(f"could not read a coverage total from: {output!r}")

    # Round *down* to a whole percent.  Recording the exact figure would make
    # the gate fail on a 0.01 pp wobble, and rounding up would record a floor
    # the checker can never meet.
    floor = math.floor(measured)
    COVERAGE_FLOOR.write_text(f"{floor}\n", encoding="utf-8")
    session.log(f"coverage is {measured:.2f}%; floor recorded as {floor}% -> {COVERAGE_FLOOR.relative_to(ROOT)}")
    session.log("Commit it as a deliberate change; never lower it to make CI pass.")
