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

    p = sub.add_parser("identify", help="fit declared latent circuit parameters and verify held-out observations")
    p.add_argument("case")
    p.add_argument("--output", default="runs", help="directory that receives the identification; default: runs")
    p.add_argument("--solver")
    p.add_argument("--optimizer", help="bounded latent-parameter optimizer override")
    p.add_argument("--trials", type=int, help="latent-parameter trial count override")
    p.add_argument("--seed", type=int, help="optimizer seed override")
    p.add_argument("--json", action="store_true", help="print the complete machine-readable result")

    p = sub.add_parser("result-summary", help="summarize candidates from a completed study")
    p.add_argument("study_root")
    p.add_argument("--out")

    p = sub.add_parser("result-prune", help="plan or remove inactive study generations")
    p.add_argument("study_root")
    p.add_argument("--keep", type=int, default=3, help="number of newest generations to retain; default: 3")
    p.add_argument("--apply", action="store_true", help="remove the listed generations; default is dry-run")
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
    verification = result.get("verification") or {}
    failed = bool(result.get("n_failed_evaluations", 0)) or bool(verification.get("n_failed_evaluations", 0))
    rejected = args.require_acceptance and (result.get("best") or {}).get("status") != "meets_declared_acceptance"
    return 1 if failed or rejected else 0


def _cmd_identify(args: argparse.Namespace) -> int:
    from .case import load_case
    from .identification import run_identification

    result = run_identification(
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
        print(f"Identification: {result['case_id']}")
        print(f"Status: {result['status']}")
        print(
            "Estimated latent parameters: "
            + json.dumps(result["estimated_latent"], ensure_ascii=False, separators=(",", ":"), default=str)
        )
        fit, holdout = result["fit"], result["holdout"]
        print(f"Fit: {fit['metric']}={float(fit['value']):.6g} (limit {float(fit['limit']):.6g})")
        print(f"Holdout: {holdout['metric']}={float(holdout['value']):.6g} (limit {float(holdout['limit']):.6g})")
        identifiable = result["identifiability"]
        condition = identifiable.get("condition_number")
        condition_text = "unavailable" if condition is None else f"{float(condition):.6g}"
        print(
            f"Local sensitivity: rank {identifiable['rank']}/{identifiable['required_rank']}, "
            f"condition={condition_text}"
        )
        print(f"Results: {result['run_root']}")
    return 0 if result.get("status") == "identified" else 1


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


def _print_constant_inputs(parameters: Any) -> None:
    if not isinstance(parameters, Mapping):
        return
    constants = parameters.get("constant_values")
    if not isinstance(constants, Mapping):
        return
    for role in ("fixed", "calibration", "latent"):
        values = constants.get(role)
        if isinstance(values, Mapping) and values:
            label = role.replace("_", " ").capitalize()
            print(f"{label} inputs: {json.dumps(dict(values), ensure_ascii=False, separators=(',', ':'), default=str)}")


def _print_final_verification(verification: object) -> None:
    if not isinstance(verification, Mapping) or not verification:
        return
    print(
        "Final candidate replay: "
        f"{verification.get('status', 'unknown')} "
        f"({int(verification.get('n_evaluations', 0))} fresh solve(s), cache reused=no)"
    )


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
    _print_constant_inputs(result.get("parameters"))
    print(
        f"Candidates: {result.get('n_candidates', 0)}  Conditions: {len(scenarios)}  "
        f"Electrical solves: {result.get('n_evaluations', 0)}"
    )
    _print_final_verification(result.get("verification"))
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
    "identify": _cmd_identify,
    "result-summary": _cmd_result_summary,
    "result-prune": _cmd_result_prune,
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
