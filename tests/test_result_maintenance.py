"""Explicit retention for study generations and disposable cache indices."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pcd.artifacts import exclusive_file_lock
from pcd.case import Case
from pcd.cli import main
from pcd.records import artifact_path
from pcd.results import prune_generations, prune_raw_cache, read_evaluation_table
from pcd.results.store import PUBLISHED_GENERATION_FILE, FileResultStore
from pcd.sim_core import archive_case_bundle
from pcd.study import run_case_study


def _study_with_generations(tmp_path):
    root = tmp_path / "study"
    generations = root / "generations"
    paths = [generations / f"g_{index:016x}" for index in range(4)]
    for path in paths:
        path.mkdir(parents=True)
    for path in paths[1:]:
        (path / PUBLISHED_GENERATION_FILE).touch()
    (root / "study_result.json").write_text(
        json.dumps({"artifacts": {"generation": f"generations\\{paths[1].name}"}}),
        encoding="utf-8",
    )
    return root, paths


def test_replay_bundle_publication_survives_a_transient_windows_directory_lock(tmp_path, monkeypatch):
    store = FileResultStore(tmp_path, "archive")
    generation = store.begin_generation()
    _, files = archive_case_bundle(Case(tmp_path / "input.yaml", {"case_id": "archive"}), generation)
    rename = Path.rename
    denied = False

    class SharingDenied(PermissionError):
        winerror = 5

    def deny_once(path, target):
        nonlocal denied
        if path.name == "bundle" and not denied:
            denied = True
            raise SharingDenied("simulated Windows access denied while publishing an archive")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", deny_once)
    archive = store.retain_case_archive(files)

    assert denied
    assert store.retain_case_archive(files) == archive
    assert (archive / files["case"]).read_bytes() == (generation / files["case"]).read_bytes()


def test_generation_prune_is_a_dry_run_and_preserves_the_active_generation(tmp_path):
    root, paths = _study_with_generations(tmp_path)

    plan = prune_generations(root, keep=1)

    assert not plan["applied"]
    assert paths[1].relative_to(root).as_posix() in plan["kept"]
    assert all(path.exists() for path in paths)


def test_generation_prune_cli_requires_apply_before_removal(tmp_path, capsys):
    root, paths = _study_with_generations(tmp_path)

    main(["result-prune", str(root), "--keep", "1", "--apply", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["applied"]
    assert paths[0].exists()  # Older in-flight or legacy generation.
    assert paths[1].exists()
    assert sum(path.exists() for path in paths) == 3


def test_generation_prune_preserves_a_busy_checkpoint_and_unmanaged_directory(tmp_path):
    root, paths = _study_with_generations(tmp_path)
    generation = paths[2]
    (generation / "checkpoint").mkdir()
    evidence = generation / "evaluations.csv"
    evidence.write_bytes(b"saved evidence")
    unrelated = generation.with_name("g_notes")
    unrelated.mkdir()
    (unrelated / PUBLISHED_GENERATION_FILE).touch()
    name = generation.relative_to(root).as_posix()

    with exclusive_file_lock(generation / "checkpoint" / ".lock"):
        result = prune_generations(root, keep=1, apply=True)

    assert result["busy"] == [name]
    assert name in result["kept"]
    assert name not in result["removed"]
    assert evidence.read_bytes() == b"saved evidence"
    assert (generation / PUBLISHED_GENERATION_FILE).is_file()
    assert unrelated.is_dir()
    assert not list(generation.parent.glob(".prune-*"))

    result = prune_generations(root, keep=1, apply=True)
    assert result["busy"] == []
    assert result["removed"] == [name]
    assert not generation.exists()
    assert unrelated.is_dir()


def test_pruning_preserves_cached_evaluation_inputs_and_reuses_the_archive(tmp_path, topology_case):
    kwargs = {"run_root": tmp_path, "n_trials": 1, "solver_override": "test_fake", "seed": 3}
    first = run_case_study(topology_case, **kwargs)
    second = run_case_study(topology_case, **kwargs)
    root = Path(second["run_root"])
    evaluation = read_evaluation_table(root).iloc[0]
    assert evaluation["from_cache"]
    manifest = root / evaluation["artifact.manifest"]
    case = artifact_path(manifest, "case")
    assert case is not None
    assert case.is_file()
    assert len(list((root / "archives").iterdir())) == 1

    result = prune_generations(root, keep=1, apply=True)

    assert result["removed"] == [first["artifacts"]["generation"]]
    assert case.is_file()
    assert manifest.is_file()


def test_raw_cache_retention_applies_both_limits_and_removes_only_indices(tmp_path):
    root = FileResultStore(tmp_path, "study").root
    indices = []
    for index, size in enumerate((2, 4, 12, 3)):
        path = root / "raw" / f"{index:024x}" / "raw_result.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * size)
        mtime = (index + 1) * 1_000_000_000
        os.utime(path, ns=(mtime, mtime))
        indices.append(path)
    unrelated = indices[2].with_name("evidence.log")
    unrelated.write_text("keep this file", encoding="utf-8")
    other = root / "raw" / "unmanaged" / "raw_result.json"
    other.parent.mkdir()
    other.write_text("keep this index", encoding="utf-8")

    def relative(paths):
        return [path.relative_to(root).as_posix() for path in paths]

    plan = prune_raw_cache(root, keep=2, max_bytes=9)
    assert not plan["applied"]
    assert plan["kept"] == relative([indices[3], indices[1]])
    assert plan["removed"] == relative([indices[2], indices[0]])
    assert (plan["kept_bytes"], plan["removed_bytes"]) == (7, 14)
    assert all(path.is_file() for path in indices)
    assert not (root / ".raw-cache.lock").exists()

    assert prune_raw_cache(root, keep=2, max_bytes=9, apply=True) == {**plan, "applied": True}
    assert [path.is_file() for path in indices] == [False, True, False, True]
    assert unrelated.is_file()
    assert other.is_file()
    assert all(path.parent.is_dir() for path in indices)


def test_raw_cache_eviction_preserves_evidence_and_next_run_recomputes(tmp_path, topology_case):
    options = {"run_root": tmp_path, "n_trials": 1, "solver_override": "test_fake", "seed": 3}
    run_case_study(topology_case, **options)
    cached = run_case_study(topology_case, **options)
    root = Path(cached["run_root"])
    row = read_evaluation_table(root).iloc[0]
    assert row["from_cache"]
    manifest = root / row["artifact.manifest"]
    case = artifact_path(manifest, "case")
    generation = root / cached["artifacts"]["generation"]
    result = (root / "study_result.json").read_bytes()
    original = manifest.read_bytes()

    evicted = prune_raw_cache(root, keep=0, apply=True)

    assert evicted["removed"]
    assert evicted["kept_bytes"] == 0
    assert (root / "study_result.json").read_bytes() == result
    assert generation.is_dir()
    assert manifest.read_bytes() == original
    assert case is not None
    assert case.is_file()
    assert read_evaluation_table(root).iloc[0]["from_cache"]
    run_case_study(topology_case, **options)
    assert not read_evaluation_table(root).iloc[0]["from_cache"]


def test_raw_cache_retention_requires_a_study_root_and_explicit_budget(tmp_path):
    with pytest.raises(ValueError, match="recognized study root"):
        prune_raw_cache(tmp_path, keep=0, apply=True)
    assert list(tmp_path.iterdir()) == []
    root = FileResultStore(tmp_path, "study").root
    with pytest.raises(ValueError, match="requires keep or max_bytes"):
        prune_raw_cache(root)
    with pytest.raises(ValueError, match="nonnegative integer"):
        prune_raw_cache(root, max_bytes=-1)


def test_raw_cache_retention_recognizes_legacy_study_root(tmp_path):
    root, _ = _study_with_generations(tmp_path)
    pointer = root / "study_result.json"
    result = json.loads(pointer.read_text(encoding="utf-8"))
    result.update({"schema": "study_result.v1", "study": {"study_id": "study"}})
    pointer.write_text(json.dumps(result), encoding="utf-8")
    path = root / "raw" / ("a" * 24) / "raw_result.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cache")

    plan = prune_raw_cache(root, max_bytes=0)

    assert plan["removed"] == [path.relative_to(root).as_posix()]
    assert plan["removed_bytes"] == 5
    assert path.is_file()
    assert not (root / ".study_id").exists()


def test_study_refuses_to_reuse_a_modified_replay_input(tmp_path, topology_case):
    options = {"run_root": tmp_path, "n_trials": 1, "solver_override": "test_fake", "seed": 3}
    first = run_case_study(topology_case, **options)
    root = Path(first["run_root"])
    archive = next((root / "archives").iterdir())
    next((archive / "inputs").iterdir()).write_text("changed", encoding="utf-8")
    committed = root / "study_result.json"
    previous = committed.read_bytes()

    with pytest.raises(ValueError, match="case archive content does not match"):
        run_case_study(topology_case, **options)

    assert committed.read_bytes() == previous


def test_study_ids_that_differ_by_case_use_distinct_result_roots(tmp_path, topology_case):
    def run(name):
        case = Case(topology_case.path, {**topology_case.data, "case_id": name})
        return run_case_study(case, tmp_path, n_trials=1, solver_override="test_fake")

    first = run("ReviewCase")
    first_root = Path(first["run_root"])
    pointer = first_root / "study_result.json"
    original = pointer.read_bytes()
    second = run("reviewcase")
    assert not first_root.samefile(second["run_root"])
    assert pointer.read_bytes() == original
    assert run("ReviewCase")["run_root"] == first["run_root"]


def test_existing_mixed_case_study_retains_its_legacy_root(tmp_path):
    legacy = tmp_path / "ReviewCase"
    legacy.mkdir()
    pointer = legacy / "study_result.json"
    pointer.write_text(json.dumps({"study": {"study_id": "ReviewCase"}}), encoding="utf-8")

    assert FileResultStore(tmp_path, "ReviewCase").root == legacy
    assert not FileResultStore(tmp_path, "reviewcase").root.samefile(legacy)
