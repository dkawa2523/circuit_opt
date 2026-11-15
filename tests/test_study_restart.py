"""Recovery at completed candidate boundaries, independently of the raw cache."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pcd import study as study_module
from pcd.artifacts import file_sha256, read_json
from pcd.core.pipeline import StudyRunner
from pcd.results import read_best_candidate, read_evaluation_table, selected_evaluation
from pcd.results import store as store_module
from pcd.results.journal import TrialJournal
from pcd.results.store import GENERATION_RESULT_FILE, PUBLISHED_GENERATION_FILE, FileResultStore
from pcd.search import RandomOptimizer
from pcd.search_registry import register
from pcd.study import run_case_study


@register("test_opaque_restart_optimizer")
def _opaque_optimizer(case, seed=None):
    return RandomOptimizer(case, seed)


def _case(make_case, optimizer, budget, *, two_conditions=False):
    capacitance = (
        {"role": "design", "choices": [1e-10, 1e-9, 1e-8]}
        if optimizer == "grid"
        else {
            "role": "design",
            "bounds": [1e-10, 1e-8],
            "default": 1e-9,
            "scale": "log",
        }
    )
    study = {}
    resistance: dict[str, Any] = {"default": 1000}
    if two_conditions:
        resistance["role"] = "operating"
        study["scenarios"] = [{"id": "low", "values": {"R": 1000}}, {"id": "high", "values": {"R": 2000}}]
    return make_case(
        {
            "case_id": "restart_rc",
            "source": {"type": "sine_voltage", "amplitude_V": 1, "frequency_Hz": 1e6},
            "variables": {"C": capacitance, "R": resistance},
            "circuit": {
                "components": [
                    {"ref": "R1", "n1": "src", "n2": "out", "value": "R"},
                    {"ref": "C1", "n1": "out", "n2": "0", "value": "C"},
                ]
            },
            "target": {
                "objective": "waveform_l2",
                "waveform_file": str(Path(__file__).parent / "fixtures" / "target_waveform.csv"),
            },
            "study": study,
            "run": {"trials": budget},
            "solver": {"name": "test_fake", "tran": {"step_s": 1e-8, "stop_s": 5e-6}},
            "optimizer": {"name": optimizer, "seed": 42},
        }
    )


def _interrupted_generation(root):
    (header,) = root.glob("restart_rc/generations/g_*/checkpoint/attempt.json")
    return header.parent.parent


@pytest.mark.parametrize(
    ("optimizer", "budget", "prefix"),
    [
        ("grid", 3, 1),
        ("random", 4, 2),
        ("differential_evolution", 10, 6),
    ],
)
def test_restart_replays_deterministic_candidate_prefix(make_case, tmp_path, monkeypatch, optimizer, budget, prefix):
    case = _case(make_case, optimizer, budget)
    baseline = run_case_study(case, tmp_path / "baseline")
    interrupted = tmp_path / "interrupted"
    original = StudyRunner.evaluate_candidate
    evaluated = []

    def interrupt(self, candidate):
        if len(evaluated) == prefix:
            raise RuntimeError("interrupted trial")
        evaluated.append(candidate.candidate_id)
        return original(self, candidate)

    with monkeypatch.context() as patch:
        patch.setattr(StudyRunner, "evaluate_candidate", interrupt)
        with pytest.raises(RuntimeError, match="interrupted trial"):
            run_case_study(case, interrupted)
    generation = _interrupted_generation(interrupted)
    assert not (generation.parent.parent / "study_result.json").exists()
    assert len(list((generation / "checkpoint").glob("trial_*.json"))) == prefix

    new_candidates = []

    def record(self, candidate):
        new_candidates.append(candidate.candidate_id)
        return original(self, candidate)

    monkeypatch.setattr(StudyRunner, "evaluate_candidate", record)
    resumed = run_case_study(case, interrupted, resume="auto")
    root = Path(resumed["run_root"])
    assert root / resumed["artifacts"]["generation"] == generation
    restart = resumed["execution"]["restart"]
    assert restart["invocations"] == 2
    assert restart["previous_incomplete_invocations"] == 1
    assert restart["replayed_candidates"] == restart["replayed_evaluations"] == prefix
    assert restart["new_candidates"] == restart["new_evaluations"] == budget - prefix
    assert new_candidates == [f"trial_{index:04d}" for index in range(prefix, budget)]
    history = read_json(root / resumed["artifacts"]["history"])
    expected = read_json(Path(baseline["run_root"]) / baseline["artifacts"]["history"])
    keys = ("params", "proposal", "rank", "aggregates")
    assert [[trial[key] for key in keys] for trial in history] == [[trial[key] for key in keys] for trial in expected]
    assert resumed["best"]["candidate"] == baseline["best"]["candidate"]
    assert len(read_evaluation_table(root)) == budget
    selected = read_best_candidate(root)
    assert all(not selected_evaluation(item)["from_cache"] for item in selected["scenarios"])
    committed = (root / "study_result.json").read_bytes()
    assert run_case_study(case, interrupted, resume=generation.name)["execution"]["restart"] == restart
    assert (root / "study_result.json").read_bytes() == committed
    assert len(new_candidates) == budget - prefix


def test_an_incomplete_candidate_uses_only_available_raw_cache(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "random", 2, two_conditions=True)
    original = StudyRunner._evaluate_control

    def interrupt(self, candidate, scenario, control, *, reuse_cached):
        if scenario.scenario_id == "high":
            raise RuntimeError("interrupted second condition")
        return original(self, candidate, scenario, control, reuse_cached=reuse_cached)

    with monkeypatch.context() as patch:
        patch.setattr(StudyRunner, "_evaluate_control", interrupt)
        with pytest.raises(RuntimeError, match="interrupted second condition"):
            run_case_study(case, tmp_path)
    resumed = run_case_study(case, tmp_path, resume="auto")
    restart = resumed["execution"]["restart"]
    assert restart["replayed_candidates"] == restart["replayed_evaluations"] == 0
    assert restart["new_candidates"] == 2
    assert restart["new_evaluations"] == 4
    assert restart["new_evaluations_from_raw_cache"] == 1
    assert resumed["verification"]["n_evaluations"] == 2
    assert not resumed["verification"]["cache_reused"]


def test_restart_executes_from_the_verified_case_archive(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "random", 2)
    baseline = run_case_study(case, tmp_path / "baseline")
    original = StudyRunner.evaluate_candidate

    def interrupt(self, candidate):
        if candidate.candidate_id == "trial_0001":
            raise RuntimeError("interrupted")
        return original(self, candidate)

    interrupted = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(StudyRunner, "evaluate_candidate", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            run_case_study(case, interrupted)
    generation = _interrupted_generation(interrupted)
    manifest = read_json(generation / "input_manifest.json")
    (target,) = (item["artifact"] for item in manifest["inputs"] if item["field"] == "$.target.waveform_file")
    (generation / target).write_text("time_s,voltage_V\n0,2\n2.5e-6,2\n5e-6,2\n", encoding="utf-8")

    resumed = run_case_study(case, interrupted, resume="auto")
    history = read_json(Path(resumed["run_root"]) / resumed["artifacts"]["history"])
    expected = read_json(Path(baseline["run_root"]) / baseline["artifacts"]["history"])
    assert resumed["execution"]["restart"]["replayed_candidates"] == 1
    assert [trial["aggregates"] for trial in history] == [trial["aggregates"] for trial in expected]


def test_restart_rejects_changed_settings_and_missing_trial_artifact(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "random", 2)
    original = StudyRunner.evaluate_candidate

    def interrupt(self, candidate):
        if candidate.candidate_id == "trial_0001":
            raise RuntimeError("interrupted")
        return original(self, candidate)

    with monkeypatch.context() as patch:
        patch.setattr(StudyRunner, "evaluate_candidate", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            run_case_study(case, tmp_path)
    generation = _interrupted_generation(tmp_path)
    with pytest.raises(ValueError, match="identity does not match"):
        run_case_study(case, tmp_path, seed=43, resume=generation.name)
    trial = read_json(generation / "checkpoint" / "trial_00000000.json")
    artifacts = trial["result"]["scenarios"][0]["trials"][0]["raw"]["artifacts"]
    study_root = generation.parent.parent
    (study_root / artifacts["waveform"]).unlink()
    with pytest.raises(ValueError, match="saved evaluation artifacts changed"):
        run_case_study(case, tmp_path, resume="auto")
    assert not (study_root / "study_result.json").exists()


def test_restart_waits_for_a_live_attempt_and_verifies_an_exhausted_search(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "random", 2)
    original = StudyRunner.verify_candidate

    def interrupt(self, candidate):
        del self, candidate
        raise RuntimeError("interrupted final verification")

    with monkeypatch.context() as patch:
        patch.setattr(StudyRunner, "verify_candidate", interrupt)
        with pytest.raises(RuntimeError, match="interrupted final verification"):
            run_case_study(case, tmp_path)
    generation = _interrupted_generation(tmp_path)
    journal = TrialJournal(FileResultStore(tmp_path, case.case_id), generation, {})
    with journal.locked(), pytest.raises(ValueError, match="is already running"):
        run_case_study(case, tmp_path, resume=generation.name)

    verifications = []

    def record(self, candidate):
        verifications.append(candidate.candidate.candidate_id)
        return original(self, candidate)

    monkeypatch.setattr(StudyRunner, "verify_candidate", record)
    resumed = run_case_study(case, tmp_path, resume=generation.name)
    assert resumed["execution"]["restart"]["replayed_candidates"] == 2
    assert resumed["execution"]["restart"]["new_evaluations"] == 0
    assert verifications == [resumed["verification"]["candidate_id"]]


def test_restart_of_published_inactive_generation_does_not_republish(make_case, tmp_path):
    case = _case(make_case, "grid", 3)
    first = run_case_study(case, tmp_path)
    root = Path(first["run_root"])
    first_generation = first["execution"]["restart"]["generation"]
    marker = root / first["artifacts"]["generation"] / PUBLISHED_GENERATION_FILE
    marker.unlink()
    assert run_case_study(case, tmp_path, resume=first_generation)["dataset"] == first["dataset"]
    assert marker.read_text(encoding="utf-8").strip() == file_sha256(marker.parent / GENERATION_RESULT_FILE)
    second = run_case_study(case, tmp_path)
    assert second["dataset"] != first["dataset"]
    active = (root / "study_result.json").read_bytes()

    recovered = run_case_study(case, tmp_path, resume=first_generation)

    assert recovered["dataset"] == first["dataset"]
    assert (root / "study_result.json").read_bytes() == active


def test_restart_rejects_missing_published_candidate_evidence(make_case, tmp_path):
    case = _case(make_case, "grid", 3)
    result = run_case_study(case, tmp_path)
    root = Path(result["run_root"])
    (root / result["artifacts"]["best_candidate"]).unlink()

    with pytest.raises(ValueError, match="saved evidence differs from its recorded best_candidate"):
        run_case_study(case, tmp_path, resume="auto")


def test_restart_after_rendering_retries_unpublished_outputs(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "grid", 3)
    case.data["output"] = {"plots": [], "schematic": False, "formats": ["svg"]}

    write_json = store_module.write_json

    def interrupt(path, payload):
        if path.name == "study_result.json":
            raise RuntimeError("interrupted publication")
        write_json(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "write_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted publication"):
            run_case_study(case, tmp_path)
    generation = _interrupted_generation(tmp_path)
    first_output = (generation / "output" / "summary.json").read_bytes()
    assert (generation / GENERATION_RESULT_FILE).is_file()
    assert not (generation.parent.parent / "study_result.json").exists()

    resumed = run_case_study(case, tmp_path, resume="auto")

    assert resumed["execution"]["restart"]["replayed_candidates"] == 3
    assert resumed["output"]["summary"].endswith("/output_001/summary.json")
    assert (generation / "output" / "summary.json").read_bytes() == first_output


def test_new_publication_seals_the_previously_active_checkpoint(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "grid", 3)
    atomic_write = store_module.atomic_write_text

    def interrupt(path, text):
        if path.name == PUBLISHED_GENERATION_FILE:
            raise RuntimeError("interrupted marker")
        atomic_write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "atomic_write_text", interrupt)
        with pytest.raises(RuntimeError, match="interrupted marker"):
            run_case_study(case, tmp_path)
    generation = _interrupted_generation(tmp_path)
    marker = generation / PUBLISHED_GENERATION_FILE
    root = generation.parent.parent
    assert not marker.exists()
    assert read_json(root / "study_result.json")["execution"]["restart"]["generation"] == generation.name

    latest = run_case_study(case, tmp_path)
    assert marker.is_file()
    assert latest["execution"]["restart"]["generation"] != generation.name

    assert run_case_study(case, tmp_path, resume="auto")["dataset"] == latest["dataset"]
    assert read_json(root / "study_result.json")["dataset"] == latest["dataset"]


def test_opaque_optimizer_keeps_raw_cache_only(make_case, tmp_path):
    case = _case(make_case, "test_opaque_restart_optimizer", 2)
    result = run_case_study(case, tmp_path)
    assert result["execution"]["restart"]["scope"] == "raw_cache_only"
    assert not list(tmp_path.glob("restart_rc/generations/g_*/checkpoint/attempt.json"))
    with pytest.raises(ValueError, match="does not support study checkpoint replay"):
        run_case_study(case, tmp_path, resume="auto")


def test_unresolved_solver_startup_allows_a_fresh_study_without_reuse(make_case, tmp_path, monkeypatch):
    case = _case(make_case, "grid", 3)
    monkeypatch.setattr(
        study_module,
        "bind_solver_settings",
        lambda settings: replace(settings, startup_fingerprint={"unresolved_sources": ["spinit:3"]}),
    )

    result = run_case_study(case, tmp_path)
    restart = result["execution"]["restart"]
    assert restart["scope"] == "unavailable"
    assert "startup uses a dependency path" in restart["reason"]
    assert not list(tmp_path.glob("restart_rc/raw/*/raw_result.json"))
    assert not list(tmp_path.glob("restart_rc/generations/g_*/checkpoint/attempt.json"))
    with pytest.raises(ValueError, match="startup uses a dependency path"):
        run_case_study(case, tmp_path, resume="auto")
