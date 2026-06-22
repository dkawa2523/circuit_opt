from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pcd", description="Separated circuit simulation and ML/design workflow platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list simulation and ML methods")

    p = sub.add_parser("solver-diagnose", help="diagnose an external solver environment")
    p.add_argument("--solver", default="ngspice_cli"); p.add_argument("--executable"); p.add_argument("--timeout-s", type=float, default=300.0); p.add_argument("--json", action="store_true")

    p = sub.add_parser("validate-case", help="validate a case file without running simulation or ML")
    p.add_argument("case"); p.add_argument("--strict", action="store_true"); p.add_argument("--json", action="store_true")

    p = sub.add_parser("sim-run", help="run one circuit simulation only; no objective scoring")
    p.add_argument("case"); p.add_argument("--solver"); p.add_argument("--run-root"); p.add_argument("--strict-exit", action="store_true")

    p = sub.add_parser("sim-prepare", help="generate simulation artifacts without running a solver")
    p.add_argument("case"); p.add_argument("--run-root")

    p = sub.add_parser("sim-netlist", help="generate one ngspice netlist without running a solver")
    p.add_argument("case"); p.add_argument("--out", default="netlist.cir")

    p = sub.add_parser("visualize-netlist", help="render a simple topology schematic from a SPICE netlist")
    p.add_argument("netlist"); p.add_argument("--out", required=True); p.add_argument("--title"); p.add_argument("--summary-json")

    p = sub.add_parser("sim-batch", help="run simulations for candidate CSV rows; no scoring")
    p.add_argument("case"); p.add_argument("candidates_csv"); p.add_argument("--solver"); p.add_argument("--run-root", required=True); p.add_argument("--strict-exit", action="store_true")

    p = sub.add_parser("ml-propose", help="generate candidate parameter rows; no simulation")
    p.add_argument("case"); p.add_argument("--optimizer"); p.add_argument("--n", type=int, default=10); p.add_argument("--seed", type=int); p.add_argument("--out", default="candidates.csv")

    p = sub.add_parser("ml-score", help="score existing simulation records; no solver execution")
    p.add_argument("case"); p.add_argument("run_root"); p.add_argument("--strict-exit", action="store_true")

    p = sub.add_parser("ml-fit-surrogate", help="fit a lightweight ridge surrogate from existing records")
    p.add_argument("run_root"); p.add_argument("--target-col", default="loss"); p.add_argument("--out", default="surrogate.json"); p.add_argument("--exclude-failed", action="store_true")
    p.add_argument("--exclude-infeasible", action="store_true"); p.add_argument("--constraint-col", default="constraint_penalty")
    p.add_argument("--target-transform", choices=["none", "log1p"], default="none"); p.add_argument("--clip-target-quantile", type=float)

    p = sub.add_parser("ml-predict", help="predict candidate loss using a saved surrogate")
    p.add_argument("surrogate_json"); p.add_argument("candidates_csv"); p.add_argument("--out", default="predicted_candidates.csv"); p.add_argument("--strict-schema", action="store_true")

    p = sub.add_parser("data-summary", help="summarize simulation records and metrics")
    p.add_argument("run_root"); p.add_argument("--out")

    p = sub.add_parser("workflow-optimize", help="closed-loop optimize: ask -> simulate -> score -> tell")
    p.add_argument("case"); p.add_argument("--optimizer"); p.add_argument("--solver"); p.add_argument("--n-trials", type=int, default=10); p.add_argument("--seed", type=int); p.add_argument("--run-root", required=True); p.add_argument("--strict-exit", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        from .ml_registry import available as ml_available
        from .sim_registry import available as sim_available
        print(json.dumps({"simulation": sim_available(), "ml": ml_available()}, indent=2, ensure_ascii=False)); return

    if args.cmd == "solver-diagnose":
        from .sim_core import diagnose_solver
        diag = diagnose_solver(args.solver, executable=args.executable, timeout_s=args.timeout_s)
        if args.json:
            print(json.dumps(diag, indent=2, ensure_ascii=False))
        else:
            for key, val in diag.items():
                print(f"{key}: {val}")
        if not diag.get("batch_runnable", False):
            sys.exit(1)
        return

    if args.cmd == "validate-case":
        from .common import load_case
        from .validation import validate_case
        report = validate_case(load_case(args.case), strict=args.strict)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.format_text())
        if not report.ok:
            sys.exit(1)
        return

    if args.cmd == "sim-run":
        from .common import load_case
        from .sim_core import simulate_case
        rec = simulate_case(load_case(args.case), run_root=args.run_root, solver_override=args.solver)
        print(json.dumps(rec.manifest(), indent=2, ensure_ascii=False))
        if args.strict_exit and rec.status != "ok":
            sys.exit(1)
        return

    if args.cmd == "sim-prepare":
        from .common import load_case
        from .sim_core import prepare_case
        rec = prepare_case(load_case(args.case), run_root=args.run_root)
        print(json.dumps(rec.manifest(), indent=2, ensure_ascii=False)); return

    if args.cmd == "sim-netlist":
        from .common import default_params, load_case
        from .sim_core import build_circuit, build_load_subckt, render_ngspice_netlist
        case = load_case(args.case); params = default_params(case)
        _, circuit = build_circuit(case, params); _, load = build_load_subckt(case, params)
        Path(args.out).write_text(render_ngspice_netlist(case, circuit, load, params), encoding="utf-8")
        print(args.out); return

    if args.cmd == "visualize-netlist":
        from .common import write_json
        from .netlist_viz import netlist_summary, render_netlist_schematic
        render_netlist_schematic(args.netlist, args.out, title=args.title)
        if args.summary_json:
            write_json(args.summary_json, netlist_summary(args.netlist))
        print(args.out); return

    if args.cmd == "sim-batch":
        import pandas as pd
        from .common import load_case
        from .sim_core import simulate_candidate_table
        records = simulate_candidate_table(load_case(args.case), pd.read_csv(args.candidates_csv), args.run_root, solver_override=args.solver)
        print(json.dumps({"n": len(records), "run_root": args.run_root}, indent=2, ensure_ascii=False))
        if args.strict_exit and any(r.get("status") != "ok" for r in records):
            sys.exit(1)
        return

    if args.cmd == "ml-propose":
        from .common import load_case
        from .ml_core import propose_candidates, save_candidates
        df = propose_candidates(load_case(args.case), args.n, optimizer_name=args.optimizer, seed=args.seed)
        save_candidates(df, args.out); print(args.out); return

    if args.cmd == "ml-score":
        from .common import load_case
        from .ml_core import score_run_root
        df = score_run_root(load_case(args.case), args.run_root)
        print(df.to_string(index=False))
        if args.strict_exit and (not df.empty) and ("status" in df.columns) and (df["status"] != "ok").any():
            sys.exit(1)
        return

    if args.cmd == "ml-fit-surrogate":
        from .common import write_json
        from .ml_core import build_learning_table, fit_ridge_surrogate
        table = build_learning_table(args.run_root, include_failed=not args.exclude_failed)
        write_json(args.out, fit_ridge_surrogate(
            table,
            target_col=args.target_col,
            exclude_infeasible=args.exclude_infeasible,
            constraint_col=args.constraint_col,
            target_transform=args.target_transform,
            clip_target_quantile=args.clip_target_quantile,
        )); print(args.out); return

    if args.cmd == "ml-predict":
        import pandas as pd
        from .common import read_json
        from .ml_core import predict_candidates_with_surrogate
        pred = predict_candidates_with_surrogate(read_json(args.surrogate_json), pd.read_csv(args.candidates_csv), strict_schema=args.strict_schema)
        pred.to_csv(args.out, index=False); print(args.out); return

    if args.cmd == "data-summary":
        from .records import save_summary
        df = save_summary(args.run_root, args.out)
        print(args.out if args.out else df.to_string(index=False)); return

    if args.cmd == "workflow-optimize":
        from .common import load_case
        from .workflow import optimize_closed_loop
        result = optimize_closed_loop(load_case(args.case), args.n_trials, args.run_root, optimizer_name=args.optimizer, solver_override=args.solver, seed=args.seed)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if args.strict_exit and result.get("n_failed_trials", 0):
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
