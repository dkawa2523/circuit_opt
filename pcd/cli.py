"""Command line entry point.

Structure: one ``_add_*`` function per command group builds the parser, and one
``_cmd_*`` handler implements each command.  Handlers return a process exit
code (0 for success) and ``main`` is the only place that calls ``sys.exit``.

Imports stay inside handlers so startup remains cheap and optional adapters are
loaded only by the command that needs them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _dump(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------


def _add_info_commands(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("list", help="list simulation, metric, and search methods")

    p = sub.add_parser("solver-diagnose", help="diagnose an external solver environment")
    p.add_argument("--solver", default="ngspice_cli")
    p.add_argument("--executable")
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--json", action="store_true")


def _add_study_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="run a design study from one case file")
    p.add_argument("case")
    p.add_argument("--output", default="runs", help="directory that receives the study; default: runs")
    p.add_argument("--solver")
    p.add_argument("--optimizer", help="advanced case only; public RF candidates are exact")
    p.add_argument("--trials", type=int, help="advanced case only; public RF candidate count is inferred")
    p.add_argument("--seed", type=int, help="advanced exploratory optimizer seed")
    p.add_argument(
        "--require-acceptance",
        action="store_true",
        help="exit nonzero unless the selected design meets every declared acceptance limit",
    )
    p.add_argument("--json", action="store_true", help="print the complete machine-readable result")

    p = sub.add_parser("result-summary", help="summarize candidates from a completed study")
    p.add_argument("study_root")
    p.add_argument("--out")

    p = sub.add_parser("result-prune", help="plan or remove inactive study generations")
    p.add_argument("study_root")
    p.add_argument("--keep", type=int, default=3, help="number of newest generations to retain; default: 3")
    p.add_argument("--apply", action="store_true", help="remove the listed generations; default is dry-run")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ml-prepare", help="prepare a leakage-aware holdout dataset from one completed study")
    p.add_argument("study_root")
    p.add_argument("--out", required=True, help="output directory for dataset.csv and manifest.json")
    p.add_argument(
        "--test-fraction", type=float, default=0.2, help="fixed-design groups assigned to test; default: 0.2"
    )
    p.add_argument("--seed", type=int, default=0, help="deterministic group split seed; default: 0")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ml-corpus", help="export graph-linked AC responses from completed studies")
    p.add_argument("study_roots", nargs="+", help="one or more completed study roots")
    p.add_argument("--out", required=True, help="output directory for the corpus artifacts")
    p.add_argument(
        "--test-fraction", type=float, default=0.2, help="whole design/condition groups assigned to test; default: 0.2"
    )
    p.add_argument("--seed", type=int, default=0, help="deterministic group split seed; default: 0")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ml-corpus-evaluate", help="compare fixed AC response models on declared corpus splits")
    p.add_argument("corpus_root", help="directory containing an ac_graph_corpus.v1 manifest")
    p.add_argument("--out", required=True, help="output directory for predictions.csv and evaluation.json")
    p.add_argument("--seed", type=int, default=0, help="deterministic neural initialization seed; default: 0")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ml-evaluate", help="compare a minimal surrogate with constant holdout baselines")
    p.add_argument("dataset_root", help="directory containing dataset.csv and manifest.json")
    p.add_argument("--out", required=True, help="output directory for predictions.csv and evaluation.json")
    p.add_argument(
        "--constraint-validation",
        help="separate prepared dataset used only for independent constraint classification",
    )
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("validate-case", help="validate a case file without simulation or metric evaluation")
    p.add_argument("case")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", action="store_true")


def _add_simulation_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sim-run", help="run one circuit simulation only; no objective scoring")
    p.add_argument("case")
    p.add_argument("--solver")
    p.add_argument("--run-root")
    p.add_argument(
        "--allow-failure",
        action="store_true",
        help="write a failed run record but exit zero (default is a nonzero failure exit)",
    )
    p.add_argument("--json", action="store_true", help="print the concise simulation summary")

    p = sub.add_parser("sim-netlist", help="generate one ngspice netlist without running a solver")
    p.add_argument("case")
    p.add_argument("--out", default="netlist.cir")

    p = sub.add_parser("visualize-netlist", help="render a simple topology schematic from a SPICE netlist")
    p.add_argument("netlist")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    p.add_argument("--summary-json")

    p = sub.add_parser("analyze", help="regenerate an electrical summary and standard figures from a saved run")
    p.add_argument("run", help="a run directory, summary.json, or debug/manifest.json")
    p.add_argument("--out", help="output directory; default: <run>/analysis")
    p.add_argument("--frequency-Hz", type=float, dest="frequency_hz", help="operating point; default: saved source")
    p.add_argument("--json", action="store_true", help="print the complete analysis summary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcd",
        description="Scenario-aware RF and electrical circuit design-study engine",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_info_commands(sub)
    _add_study_commands(sub)
    _add_simulation_commands(sub)
    return parser


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------


def _cmd_list(_args: argparse.Namespace) -> int:
    from .metric_registry import available as metric_available
    from .search_registry import available as optimizer_available
    from .sim_registry import available as sim_available

    _dump(
        {
            "simulation": sim_available(),
            "metrics": metric_available(),
            "optimizers": optimizer_available(),
        }
    )
    return 0


def _cmd_solver_diagnose(args: argparse.Namespace) -> int:
    from .solver import diagnose_solver

    diag = diagnose_solver(args.solver, executable=args.executable, timeout_s=args.timeout_s)
    if args.json:
        _dump(diag)
    else:
        for key, val in diag.items():
            print(f"{key}: {val}")
    return 0 if diag.get("batch_runnable", False) else 1


def _cmd_validate_case(args: argparse.Namespace) -> int:
    from .case import load_case
    from .validation import validate_case

    report = validate_case(load_case(args.case), strict=args.strict)
    if args.json:
        _dump(report.to_dict())
    else:
        print(report.format_text())
    return 0 if report.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from .case import load_case
    from .study import run_case_study

    result = run_case_study(
        load_case(args.case),
        run_root=args.output,
        n_trials=args.trials,
        optimizer_name=args.optimizer,
        solver_override=args.solver,
        seed=args.seed,
    )
    if args.json:
        _dump(result)
    else:
        _print_run_summary(result)
    failed = bool(result.get("n_failed_evaluations", 0))
    rejected = args.require_acceptance and (result.get("best") or {}).get("status") != "meets_declared_acceptance"
    return 1 if failed or rejected else 0


def _print_condition_summaries(conditions: list[dict]) -> None:
    show_all_conditions = len(conditions) <= 8
    for condition in conditions:
        if not show_all_conditions and condition.get("status") == "accepted":
            continue
        control = json.dumps(
            condition.get("selected_control") or {}, ensure_ascii=False, separators=(",", ":"), default=str
        )
        print(f"Condition {condition.get('scenario_id', '')}: {condition.get('status', '')}, control={control}")
        failed = condition.get("failed_constraints") or []
        if failed:
            print("  Failed constraints: " + ", ".join(str(item.get("name", "")) for item in failed))


def _print_pareto_summary(pareto: object) -> None:
    if not isinstance(pareto, dict):
        return
    scope = "declared grid" if pareto.get("scope") == "declared_grid" else "observed candidates"
    print(
        f"Pareto front ({scope}): {int(pareto.get('front_candidates', 0))}/"
        f"{int(pareto.get('eligible_candidates', 0))} eligible candidates"
    )


def _print_constraint_margins(best: Mapping[str, Any]) -> None:
    margins = best.get("constraint_margins") or {}
    if not margins:
        return
    formatted = ", ".join(f"{name}={float(value):.6g}" for name, value in margins.items())
    print(f"Worst constraint margins (positive=reserve): {formatted}")


def _print_electrical_context(study: Mapping[str, Any]) -> None:
    metadata = study.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return
    analysis = str(metadata.get("analysis", "")).upper()
    parts = [analysis] if analysis and analysis != "UNSPECIFIED" else []
    frequency = metadata.get("frequency_Hz")
    if frequency is not None:
        parts.append(f"{float(frequency) / 1e6:.6g} MHz")
    reference_plane = metadata.get("reference_plane")
    if reference_plane:
        parts.append(f"reference plane={reference_plane}")
    reference_impedance = metadata.get("reference_impedance_ohm")
    if reference_impedance is not None:
        parts.append(f"Z0={float(reference_impedance):.6g} ohm")
    if parts:
        print("Electrical context: " + ", ".join(parts))


def _print_run_summary(result: dict) -> None:
    best = result["best"]
    aggregates = best.get("aggregates") or {}
    study = result.get("study") or {}
    scenarios = study.get("scenarios") or []
    decision_status = best.get("status")
    feasible = decision_status == "meets_declared_acceptance" or (
        decision_status is None
        and float(best.get("feasible_fraction", 0.0)) == 1.0
        and float(best.get("success_fraction", 0.0)) == 1.0
    )
    print(f"Study: {study.get('study_id', '')}")
    _print_electrical_context(study)
    if decision_status == "incomplete_evidence":
        feasibility = "unknown (incomplete evidence)"
    else:
        feasibility = "yes" if feasible else ("no (tuning-margin limited)" if best.get("edge_limited") else "no")
    print(f"Feasible across all conditions: {feasibility}")
    if decision_status:
        print(f"Decision: {decision_status}")
    search_completeness = best.get("search_completeness")
    if search_completeness:
        print(f"Search completeness: {search_completeness}")
    candidate = best.get("candidate") or {}
    candidate_id = candidate.get("candidate_id", "")
    candidate_values = candidate.get("values") or {}
    candidate_suffix = (
        f" {json.dumps(candidate_values, ensure_ascii=False, separators=(',', ':'), default=str)}"
        if candidate_values
        else ""
    )
    print(f"Selected candidate: {candidate_id}{candidate_suffix}")
    print(
        f"Candidates: {result.get('n_candidates', 0)}  Conditions: {len(scenarios)}  "
        f"Electrical solves: {result.get('n_evaluations', 0)}"
    )
    coverage = best.get("coverage") or {}
    if coverage:
        total = int(coverage.get("conditions", 0))
        print(
            f"Condition coverage: accepted {int(coverage.get('accepted', 0))}/{total}, "
            f"solved {int(coverage.get('solved', 0))}/{total}"
        )
    if result.get("n_failed_evaluations", 0):
        print(f"Failed electrical solves: {result['n_failed_evaluations']}")
    for objective in study.get("objectives") or []:
        metric = str(objective.get("metric", ""))
        if metric in aggregates:
            value = aggregates[metric]
            formatted_value = f"{float(value):.6g}" if value is not None else "unavailable"
            print(
                f"Objective: {metric}={formatted_value} "
                f"({objective.get('aggregation', 'worst')}, {objective.get('direction', 'minimize')})"
            )
    _print_pareto_summary(result.get("pareto"))
    margin = best.get("control_margin")
    if margin is not None:
        print(f"Worst control margin: {float(margin):.1%} (0%=edge, 100%=center)")
    _print_constraint_margins(best)
    _print_condition_summaries(best.get("conditions") or [])
    run_root = Path(str(result.get("run_root", "")))
    print(f"Results: {run_root}")
    best_candidate = (result.get("artifacts") or {}).get("best_candidate")
    if best_candidate:
        print(f"Selected evidence: {run_root / str(best_candidate)}")


def _cmd_result_summary(args: argparse.Namespace) -> int:
    from .results import candidate_summary

    frame = candidate_summary(args.study_root)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(args.out)
    else:
        print(frame.to_string(index=False))
    return 0


def _cmd_result_prune(args: argparse.Namespace) -> int:
    from .results import prune_generations

    result = prune_generations(args.study_root, keep=args.keep, apply=args.apply)
    if args.json:
        _dump(result)
    else:
        action = "Removed" if args.apply else "Would remove"
        print(f"{action}: {len(result['removed'])} generation(s)")
        for path in result["removed"]:
            print(f"  {path}")
        if not args.apply and result["removed"]:
            print("Re-run with --apply to remove them.")
    return 0


def _feature_transforms_from_case(case_path: Path, columns: set[str]) -> dict[str, str]:
    from .case import load_case, variable_specs
    from .study_config import candidate_case

    case = load_case(case_path)
    all_specs = variable_specs(case)
    design_specs = variable_specs(candidate_case(case))
    transforms: dict[str, str] = {}
    for column in sorted(columns):
        prefix, separator, name = column.partition(".")
        if not separator or prefix not in {"design", "scenario", "control"}:
            continue
        spec = (design_specs if prefix == "design" else all_specs).get(name, {})
        scale = str(spec.get("scale", "linear"))
        if scale not in {"linear", "log"}:
            raise ValueError(f"feature {column!r} scale must be linear or log")
        transforms[column] = "log10" if scale == "log" else "linear"
    return transforms


def _cmd_ml_prepare(args: argparse.Namespace) -> int:
    import pandas as pd

    from .artifacts import atomic_write_text, file_sha256, read_json, write_json
    from .ml import prepare_evaluation_dataset
    from .results import study_artifact_path

    study_root = Path(args.study_root)
    result_path = study_root / "study_result.json"
    if not result_path.is_file():
        raise ValueError(f"completed study result not found: {result_path}")
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("study_result.json must contain a mapping")
    result_artifacts = result.get("artifacts")
    if not isinstance(result_artifacts, dict):
        raise ValueError("study_result.json must contain an artifacts mapping")
    evaluation_path = study_artifact_path(study_root, "evaluation_table")
    if evaluation_path is None or not evaluation_path.is_file():
        raise ValueError("committed study does not contain an evaluation table")
    case_path = study_artifact_path(study_root, "case")
    if case_path is None or not case_path.is_file():
        raise ValueError("committed study does not contain its executable case")

    output_dir = Path(args.out)
    resolved_output = output_dir.resolve()
    generation_root = evaluation_path.parent.resolve()
    if resolved_output == generation_root or generation_root in resolved_output.parents:
        raise ValueError("ML export must not modify the immutable study generation")

    evaluations = pd.read_csv(evaluation_path)
    prepared = prepare_evaluation_dataset(
        evaluations,
        result,
        test_fraction=args.test_fraction,
        seed=args.seed,
        feature_transforms=_feature_transforms_from_case(case_path, set(evaluations.columns)),
    )
    dataset_path = output_dir / "dataset.csv"
    manifest_path = output_dir / "manifest.json"
    atomic_write_text(dataset_path, prepared.frame.to_csv(index=False))

    declared_table = str(result_artifacts.get("evaluation_table", ""))
    manifest: dict[str, Any] = {
        **prepared.manifest,
        "source_artifacts": {
            "study_result": {"path": "study_result.json", "sha256": file_sha256(result_path)},
            "evaluation_table": {"path": declared_table, "sha256": file_sha256(evaluation_path)},
        },
        "artifacts": {
            "dataset": {"path": "dataset.csv", "sha256": file_sha256(dataset_path)},
            "manifest": "manifest.json",
        },
    }
    write_json(manifest_path, manifest)
    if args.json:
        _dump(manifest)
    else:
        rows = manifest["split"]["rows"]
        groups = manifest["split"]["groups"]
        print(f"ML dataset: {output_dir}")
        print(f"Rows: train={rows['train']}, test={rows['test']}, total={rows['total']}")
        print(f"Fixed-design groups: train={groups['train']}, test={groups['test']}, total={groups['total']}")
        print("Features: " + ", ".join(manifest["roles"]["features"]))
        print("Objectives: " + ", ".join(manifest["roles"]["objective_targets"]))
    return 0


def _cmd_ml_corpus(args: argparse.Namespace) -> int:
    from .corpus import export_ac_graph_corpus

    manifest = export_ac_graph_corpus(
        args.study_roots,
        args.out,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    if args.json:
        _dump(manifest)
    else:
        counts = manifest["counts"]
        print(f"AC graph corpus: {Path(args.out).resolve()}")
        print(
            f"Studies: {counts['source_studies']}  Graphs: {counts['graphs']}  "
            f"Samples: {counts['samples']}  Component responses: {counts['component_response_rows']}"
        )
        if counts["skipped_evaluations"]:
            details = ", ".join(f"{name}={count}" for name, count in counts["skipped_by_reason"].items())
            print(f"Skipped evaluations: {counts['skipped_evaluations']} ({details})")
        print("Artifacts: graphs.jsonl, samples.csv, component_responses.csv, manifest.json")
    return 0


def _cmd_ml_corpus_evaluate(args: argparse.Namespace) -> int:
    from .corpus_evaluation import evaluate_ac_graph_corpus

    result = evaluate_ac_graph_corpus(args.corpus_root, args.out, seed=args.seed)
    if args.json:
        _dump(result)
    else:
        print(f"AC model comparison: {Path(args.out).resolve()}")
        for name, protocol in result["protocols"].items():
            if protocol["status"] in {"evaluated", "partial"}:
                print(f"{name}: {protocol['status']} winner={protocol['winner']}")
            else:
                print(f"{name}: unavailable ({protocol.get('reason', 'insufficient evidence')})")
        decision = result["decision"]
        print(f"Decision: {decision['status']}")
    return 0


def _read_ml_dataset(root: Path) -> tuple[Any, dict[str, Any], Path, Path, str]:
    import pandas as pd

    from .artifacts import file_sha256, read_json

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"ML dataset manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a mapping")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts mapping")
    dataset_artifact = artifacts.get("dataset")
    if not isinstance(dataset_artifact, dict):
        raise ValueError("manifest dataset artifact must be a mapping")
    declared_path = dataset_artifact.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("manifest dataset artifact must declare a path")
    dataset_path = (root / declared_path).resolve()
    if root != dataset_path.parent and root not in dataset_path.parents:
        raise ValueError("manifest dataset artifact must remain inside the dataset directory")
    if not dataset_path.is_file():
        raise ValueError(f"prepared dataset not found: {dataset_path}")
    actual_hash = file_sha256(dataset_path)
    if actual_hash != dataset_artifact.get("sha256"):
        raise ValueError("dataset.csv SHA-256 does not match manifest.json")
    return pd.read_csv(dataset_path), manifest, manifest_path, dataset_path, str(actual_hash)


def _separate_derived_output(source: Path, destination: Path) -> None:
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("surrogate evaluation output must not overlap the prepared dataset directory")


def _print_ml_evaluation(output_dir: Path, summary: Mapping[str, Any]) -> None:
    print(f"Surrogate evaluation: {output_dir}")
    for name, result in summary["regression"].items():
        if result["status"] != "evaluated":
            print(f"Objective {name}: {result['status']}")
            continue
        improvement = 100.0 * float(result["relative_rmse_improvement"])
        print(f"Objective {name}: RMSE improvement over mean={improvement:.1f}%")
    for name, result in summary["classification"].items():
        print(f"Classification {name}: {result['status']}")
    independent = summary["independent_constraint_validation"]
    if independent["status"] == "evaluated":
        for name, result in independent["classification"].items():
            print(f"Independent classification {name}: {result['status']}")
    print(f"Next evidence step: {summary['evidence_gate']['recommendation']}")


def _cmd_ml_evaluate(args: argparse.Namespace) -> int:
    from .artifacts import atomic_write_text, file_sha256, write_json
    from .ml import evaluate_surrogate_dataset

    dataset_root = Path(args.dataset_root).resolve()
    dataset, manifest, manifest_path, dataset_path, dataset_hash = _read_ml_dataset(dataset_root)
    output_dir = Path(args.out).resolve()
    _separate_derived_output(dataset_root, output_dir)
    validation_dataset = None
    validation_manifest = None
    validation_artifacts: dict[str, Any] | None = None
    if args.constraint_validation:
        validation_root = Path(args.constraint_validation).resolve()
        (
            validation_dataset,
            validation_manifest,
            validation_manifest_path,
            validation_dataset_path,
            validation_dataset_hash,
        ) = _read_ml_dataset(validation_root)
        _separate_derived_output(validation_root, output_dir)
        validation_artifacts = {
            "manifest": {
                "path": str(validation_manifest_path),
                "sha256": file_sha256(validation_manifest_path),
            },
            "dataset": {"path": str(validation_dataset_path), "sha256": validation_dataset_hash},
        }
    evaluated = evaluate_surrogate_dataset(
        dataset,
        manifest,
        constraint_validation_dataset=validation_dataset,
        constraint_validation_manifest=validation_manifest,
    )
    predictions_path = output_dir / "predictions.csv"
    evaluation_path = output_dir / "evaluation.json"
    atomic_write_text(predictions_path, evaluated.predictions.to_csv(index=False))
    source_artifacts: dict[str, Any] = {
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "dataset": {"path": str(dataset_path), "sha256": dataset_hash},
    }
    artifacts: dict[str, Any] = {
        "predictions": {"path": "predictions.csv", "sha256": file_sha256(predictions_path)},
        "evaluation": "evaluation.json",
    }
    if validation_artifacts is not None and evaluated.constraint_validation_predictions is not None:
        validation_predictions_path = output_dir / "constraint_validation_predictions.csv"
        atomic_write_text(
            validation_predictions_path,
            evaluated.constraint_validation_predictions.to_csv(index=False),
        )
        source_artifacts["constraint_validation"] = validation_artifacts
        artifacts["constraint_validation_predictions"] = {
            "path": "constraint_validation_predictions.csv",
            "sha256": file_sha256(validation_predictions_path),
        }
    summary: dict[str, Any] = {
        **evaluated.summary,
        "source_artifacts": source_artifacts,
        "artifacts": artifacts,
    }
    write_json(evaluation_path, summary)
    if args.json:
        _dump(summary)
    else:
        _print_ml_evaluation(output_dir, summary)
    return 0


def _cmd_sim_run(args: argparse.Namespace) -> int:
    from .case import load_case
    from .sim_core import simulate_case

    rec = simulate_case(load_case(args.case), run_root=args.run_root, solver_override=args.solver)
    summary = rec.summary()
    if args.json:
        _dump(summary)
    else:
        _print_sim_summary(summary)
    return 1 if rec.status != "ok" and not args.allow_failure else 0


def _print_sim_summary(summary: dict) -> None:
    version = f" ({summary['solver_version']})" if summary.get("solver_version") else ""
    print(f"Simulation: {summary.get('case_id', '')}")
    print(f"Status: {summary.get('status', '')}")
    print(f"Solver: {summary.get('solver', '')}{version}")
    if summary.get("error"):
        print(f"Error: {summary['error']}")
    print(f"Results: {summary.get('run_dir', '')}")
    artifacts = summary.get("artifacts") or {}
    available = [name for name in ("frequency_response", "waveform") if name in artifacts]
    if available:
        print("Artifacts: " + ", ".join(f"{name}={artifacts[name]}" for name in available))


def _cmd_sim_netlist(args: argparse.Namespace) -> int:
    from .case import default_params, load_case
    from .netlist import build_circuit, build_load_subckt, render_ngspice_netlist
    from .simulation_input import resolve_simulation_case

    case = load_case(args.case)
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    simulation = resolve_simulation_case(case, params)
    Path(args.out).write_text(render_ngspice_netlist(case, circuit, load, params, simulation), encoding="utf-8")
    print(args.out)
    return 0


def _cmd_visualize_netlist(args: argparse.Namespace) -> int:
    from .artifacts import write_json
    from .netlist_parse import netlist_summary
    from .netlist_viz import render_netlist_schematic

    render_netlist_schematic(args.netlist, args.out, title=args.title)
    if args.summary_json:
        write_json(args.summary_json, netlist_summary(args.netlist))
    print(args.out)
    return 0


def _print_analysis_summary(summary: dict[str, Any]) -> None:
    analyses = summary.get("analyses") or {}
    print(f"Analysis: {summary.get('case_id', '')}")
    print(f"Results: {summary.get('output_dir', '')}")
    if isinstance(analyses, dict) and isinstance(analyses.get("ac"), dict):
        _print_ac_analysis(analyses["ac"])
    if isinstance(analyses, dict) and isinstance(analyses.get("transient"), dict):
        _print_transient_analysis(analyses["transient"])
    artifacts = summary.get("artifacts") or {}
    if isinstance(artifacts, dict):
        print("Artifacts: " + ", ".join(f"{name}={path}" for name, path in artifacts.items()))


def _print_ac_analysis(ac: dict[str, Any]) -> None:
    print(
        f"AC: {float(ac['frequency_Hz']) / 1e6:.6g} MHz, "
        f"Z={float(ac['resistance_ohm']):.6g}{float(ac['reactance_ohm']):+.6g}j ohm, "
        f"|Gamma|={float(ac['reflection_magnitude']):.6g}"
    )
    sweep = ac.get("sweep")
    if isinstance(sweep, dict):
        _print_sweep_analysis(sweep)


def _print_sweep_analysis(sweep: dict[str, Any]) -> None:
    best = float(sweep["sampled_best_match_frequency_Hz"]) / 1e6
    resonance = sweep.get("half_power_resonance")
    if not isinstance(resonance, dict):
        print(f"Sweep: {sweep['sample_count']} points, best sampled match={best:.6g} MHz; -3 dB band not bracketed")
        return
    print(
        f"Sweep: {sweep['sample_count']} points, "
        f"power peak={float(resonance['resonant_frequency_Hz']) / 1e6:.6g} MHz, "
        f"-3 dB BW={float(resonance['bandwidth_Hz']) / 1e6:.6g} MHz, "
        f"loaded Q={float(resonance['loaded_quality_factor']):.6g} "
        f"({resonance['basis']})"
    )


def _print_transient_analysis(transient: dict[str, Any]) -> None:
    periodic = transient.get("periodic") or {}
    status = periodic.get("status") if isinstance(periodic, dict) else "not_available"
    print(
        f"Transient: {transient.get('samples', 0)} samples, "
        f"Vpeak={float(transient['waveform_voltage_peak_V']):.6g} V, periodic={status}"
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .reporting import analyze_run

    summary = analyze_run(args.run, out_dir=args.out, frequency_hz=args.frequency_hz)
    if args.json:
        _dump(summary)
    else:
        _print_analysis_summary(summary)
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": _cmd_list,
    "solver-diagnose": _cmd_solver_diagnose,
    "validate-case": _cmd_validate_case,
    "run": _cmd_run,
    "result-summary": _cmd_result_summary,
    "result-prune": _cmd_result_prune,
    "ml-prepare": _cmd_ml_prepare,
    "ml-corpus": _cmd_ml_corpus,
    "ml-corpus-evaluate": _cmd_ml_corpus_evaluate,
    "ml-evaluate": _cmd_ml_evaluate,
    "sim-run": _cmd_sim_run,
    "sim-netlist": _cmd_sim_netlist,
    "visualize-netlist": _cmd_visualize_netlist,
    "analyze": _cmd_analyze,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        exit_code = HANDLERS[args.cmd](args)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if getattr(args, "json", False):
            _dump({"status": "invalid", "error": str(exc)})
        else:
            print(f"Input error: {exc}")
        exit_code = 2
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
