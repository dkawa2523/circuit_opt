"""pcd.registry — the name -> method lookup shared by both layers.

Tested once here, against a throwaway Registry, so the two layer registries
stay the three-line declarations they should be.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcd import metric_registry, search_registry, sim_registry
from pcd.registry import Registry, load_plugin_files


@pytest.fixture
def registry() -> Registry:
    """A registry whose built-ins module is a no-op stand-in."""

    return Registry(label="test", kinds=("widget",), builtins_module="pcd.case")


# --- registration ----------------------------------------------------------


def test_a_registered_method_can_be_looked_up(registry):
    @registry.register("widget", "spanner")
    def spanner():
        return "spanner"

    assert registry.get("widget", "spanner")() == "spanner"
    assert registry.available() == {"widget": ["spanner"]}


def test_an_unknown_kind_is_rejected_at_registration(registry):
    with pytest.raises(KeyError, match="unknown test method kind: gadget"):
        registry.register("gadget", "x")


def test_an_unknown_name_lists_what_is_available(registry):
    registry.register("widget", "spanner")(lambda: None)
    with pytest.raises(KeyError, match=r"unknown widget method: hammer. available=\['spanner'\]"):
        registry.get("widget", "hammer")


def test_registering_the_same_function_twice_is_harmless(registry):
    """Re-importing a plugin file must not raise or warn."""

    def spanner():
        return "spanner"

    registry.register("widget", "spanner")(spanner)
    registry.register("widget", "spanner")(spanner)
    assert registry.conflicts() == []


def test_a_genuine_clash_warns_and_keeps_the_first(registry):
    @registry.register("widget", "spanner")
    def first():
        return "first"

    with pytest.warns(RuntimeWarning, match="test method conflict for widget:spanner"):

        @registry.register("widget", "spanner")
        def second():
            return "second"

    assert registry.get("widget", "spanner")() == "first"
    assert registry.conflicts()[0]["name"] == "spanner"


# --- built-ins are loaded lazily, once -------------------------------------


def test_builtins_load_on_first_use_and_only_once(monkeypatch):
    imported: list[str] = []
    registry = Registry(label="test", kinds=("widget",), builtins_module="pcd.case")
    monkeypatch.setattr("pcd.registry.importlib.import_module", lambda name: imported.append(name))

    registry.available()
    registry.available()
    registry.conflicts()
    assert imported == ["pcd.case"]


# --- plugin files ----------------------------------------------------------


def _write_plugin(tmp_path: Path, body: str, name: str = "plug.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_plugin_file_is_executed(tmp_path):
    marker = tmp_path / "ran.txt"
    plugin = _write_plugin(tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    load_plugin_files([plugin.name], tmp_path)
    assert marker.read_text() == "ran"


def test_a_plugin_file_runs_only_once(tmp_path):
    counter = tmp_path / "count.txt"
    body = (
        "from pathlib import Path\n"
        f"p = Path({str(counter)!r})\n"
        "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1')\n"
    )
    plugin = _write_plugin(tmp_path, body, name="once.py")
    load_plugin_files([plugin.name], tmp_path)
    load_plugin_files([plugin.name], tmp_path)
    assert counter.read_text() == "1"


def test_a_missing_plugin_names_the_resolved_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="plugin not found"):
        load_plugin_files(["absent.py"], tmp_path)


def test_no_plugins_is_not_an_error(tmp_path):
    load_plugin_files(None, tmp_path)
    load_plugin_files([], tmp_path)


def test_an_absolute_plugin_path_is_used_as_given(tmp_path):
    marker = tmp_path / "abs.txt"
    plugin = _write_plugin(tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n", "abs.py")
    load_plugin_files([str(plugin)], Path("/nowhere"))
    assert marker.exists()


# --- application registries ------------------------------------------------


def test_application_registries_are_separate_and_complete():

    assert set(sim_registry.available()) == {"circuit", "load", "solver"}
    assert {"dummy", "ngspice_cli"} <= set(sim_registry.available()["solver"])
    assert "waveform_l2" in metric_registry.available()
    assert "random" in search_registry.available()
    assert "grid" in search_registry.available()
