"""Shared fixtures.

The platform's own artifacts (``case.yaml``, ``sim_manifest.json``,
``waveform.csv``) are the real fixtures, so these helpers only build minimal
valid instances of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pcd.case import Case, load_case
from tests import fakes as _test_fakes  # noqa: F401  # register the test-only solver

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def rc_case() -> Case:
    """RC low-pass driven by a pulse; the simplest complete case."""

    return load_case(EXAMPLES / "advanced" / "generic_rc_filter.yaml")


@pytest.fixture
def topology_case() -> Case:
    """Test-only categorical case for the advanced explicit extension API."""

    return load_case(FIXTURES / "advanced_case.yaml")


@pytest.fixture
def make_case(tmp_path: Path):
    """Write a case dict to disk and load it, so path resolution is exercised."""

    def _make(data: dict[str, Any], name: str = "case.yaml") -> Case:
        path = tmp_path / name
        payload = {"schema": "case_yaml.v1", **data}
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return load_case(path)

    return _make
