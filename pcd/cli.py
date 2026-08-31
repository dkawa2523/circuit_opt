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
from collections.abc import Callable
from pathlib import Path


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
    p.add_argument("--optimizer")
    p.add_argument("--trials", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--json", action="store_true", help="print the complete machine-readable result")

    p = sub.add_parser("result-summary", help="summarize candidates from a completed study")
    p.add_argument("study_root")
    p.add_argument("--out")

    p = sub.add_parser("validate-case", help="validate a case file without simulation or metric evaluation")
    p.add_argument("case")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", action="store_true")


def _add_simulation_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sim-run", help="run one circuit simulation only; no objective scoring")
    p.add_argument("case")
    p.add_argument("--solver")
    p.add_argument("--run-root")
    p.add_argument("--strict-exit", action="store_true")

    p = sub.add_parser("sim-netlist", help="generate one ngspice netlist without running a solver")
    p.add_argument("case")
    p.add_argument("--out", default="netlist.cir")

    p = sub.add_parser("visualize-netlist", help="render a simple topology schematic from a SPICE netlist")
    p.add_argument("netlist")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    p.add_argument("--summary-json")

    p = sub.add_parser("visualize-response", help="plot a run's waveform and, if present, its Smith chart")
    p.add_argument("run_dir", help="a run directory, or the sim_manifest.json inside it")
    p.add_argument("--out", required=True)
    p.add_argument("--title")
    p.add_argument("--marker-Hz", type=float, dest="marker_hz", help="frequency to mark; default is the source's")


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
    return 1 if result.get("n_failed_evaluations", 0) else 0


def _print_run_summary(result: dict) -> None:
    best = result["best"]
    aggregates = best.get("aggregates") or {}
    feasible = float(best.get("feasible_fraction", 0.0)) == 1.0 and float(best.get("success_fraction", 0.0)) == 1.0
    scenarios = (result.get("study") or {}).get("scenarios") or []
    print(f"Study: {(result.get('study') or {}).get('study_id', '')}")
    feasibility = "yes" if feasible else ("no (tuning-margin limited)" if best.get("edge_limited") else "no")
    print(f"Feasible across all conditions: {feasibility}")
    print(
        f"Candidates: {result.get('n_candidates', 0)}  Conditions: {len(scenarios)}  "
        f"Electrical solves: {result.get('n_evaluations', 0)}"
    )
    if result.get("n_failed_evaluations", 0):
        print(f"Failed electrical solves: {result['n_failed_evaluations']}")
    reflection = aggregates.get("reflection_magnitude")
    if reflection is not None:
        gamma = float(reflection)
        print(f"Worst reflection: |Gamma|={gamma:.6g}, reflected power={gamma**2:.3%}")
    margin = best.get("control_margin")
    if margin is not None:
        print(f"Worst control margin: {float(margin):.1%} (0%=edge, 100%=center)")
    print(f"Results: {result.get('run_root')}")


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


def _cmd_sim_run(args: argparse.Namespace) -> int:
    from .case import load_case
    from .sim_core import simulate_case

    rec = simulate_case(load_case(args.case), run_root=args.run_root, solver_override=args.solver)
    _dump(rec.manifest())
    return 1 if args.strict_exit and rec.status != "ok" else 0


def _cmd_sim_netlist(args: argparse.Namespace) -> int:
    from .case import default_params, load_case
    from .netlist import build_circuit, build_load_subckt, render_ngspice_netlist

    case = load_case(args.case)
    params = default_params(case)
    _, circuit = build_circuit(case, params)
    _, load = build_load_subckt(case, params)
    Path(args.out).write_text(render_ngspice_netlist(case, circuit, load, params), encoding="utf-8")
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


def _cmd_visualize_response(args: argparse.Namespace) -> int:
    from .analysis import DEFAULT_Z0, read_ac
    from .case import load_case
    from .records import frequency_response_path, load_waveform, read_sim_record
    from .response_plot import render_response
    from .spice import fundamental_hz

    record = read_sim_record(args.run_dir)
    ac_path = frequency_response_path(record)
    # A run archives the case it ran, so the plot describes that run and not
    # whatever the case file happens to say now.
    case_path = Path(record["run_dir"]) / "case.yaml"
    case = load_case(case_path) if case_path.exists() else None
    marker = args.marker_hz
    if marker is None and case is not None:
        marker = fundamental_hz(case, record.get("params") or {})

    render_response(
        waveform=load_waveform(record),
        ac=read_ac(ac_path) if ac_path else None,
        out=args.out,
        title=args.title or str(record.get("case_id", "")),
        z0=float((record.get("measurement") or {}).get("reference_impedance_ohm", DEFAULT_Z0)),
        marker_hz=marker,
    )
    print(args.out)
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "list": _cmd_list,
    "solver-diagnose": _cmd_solver_diagnose,
    "validate-case": _cmd_validate_case,
    "run": _cmd_run,
    "result-summary": _cmd_result_summary,
    "sim-run": _cmd_sim_run,
    "sim-netlist": _cmd_sim_netlist,
    "visualize-netlist": _cmd_visualize_netlist,
    "visualize-response": _cmd_visualize_response,
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
