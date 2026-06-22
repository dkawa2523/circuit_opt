from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

PACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcd.common import Case, load_case, write_json
from pcd.ml_core import build_learning_table, fit_ridge_surrogate
from pcd.records import save_summary
from pcd.workflow import optimize_closed_loop


DEFAULT_PROFILE_FILE = PACK_DIR / "ngspice_benchmark_profiles.json"


def _load_profiles(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case_for_ngspice(case_file: str, executable: str | None, timeout_s: float | None, objective: str | None = None):
    case = load_case(PACK_DIR / case_file)
    data = copy.deepcopy(case.data)
    solver_cfg = data.setdefault("solver", {})
    solver_cfg["name"] = "ngspice_cli"
    if executable:
        solver_cfg["executable"] = executable
    elif solver_cfg.get("executable") in {"", None}:
        solver_cfg.pop("executable", None)
    if timeout_s is not None:
        solver_cfg["timeout_s"] = float(timeout_s)
    if objective:
        data.setdefault("target", {})["objective"] = objective
    return Case(path=case.path, data=data)


def _fit_surrogate_if_possible(case_run_dir: Path) -> dict[str, Any] | None:
    table = build_learning_table(case_run_dir, include_failed=True)
    if table.empty:
        return None
    model = fit_ridge_surrogate(table, target_col="loss")
    write_json(case_run_dir / "surrogate.json", model)
    return model


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    profiles = _load_profiles(Path(args.profile_file))
    profile = profiles["profiles"][args.profile]
    solver_cfg = profiles.get("solver", {}) or {}
    executable = args.executable if args.executable is not None else solver_cfg.get("executable")
    timeout_s = args.timeout_s if args.timeout_s is not None else profile.get("timeout_s")
    selected_cases = args.cases or profile.get("cases") or list(profiles["cases"].keys())
    objective = args.objective or profile.get("objective")
    label = args.label or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{args.profile}"
    output_root = Path(args.run_root or profiles.get("output_root", "runs/ccp_ngspice_reeval"))
    run_dir = (REPO_ROOT / output_root if not output_root.is_absolute() else output_root) / label

    manifest: dict[str, Any] = {
        "schema": "ccp_ngspice_benchmark_run.v1",
        "profile": args.profile,
        "n_trials": int(args.n_trials or profile["n_trials"]),
        "solver": {"name": "ngspice_cli", "executable": executable, "timeout_s": timeout_s},
        "objective": objective,
        "run_dir": str(run_dir),
        "cases": {},
    }
    if args.dry_run:
        for case_key in selected_cases:
            spec = profiles["cases"][case_key]
            manifest["cases"][case_key] = {
                "case_file": spec["case_file"],
                "run_dir": str(run_dir / spec.get("output_dir", case_key)),
                "seed": int(args.seed if args.seed is not None else spec["seed"]),
            }
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)
    for case_key in selected_cases:
        spec = profiles["cases"][case_key]
        case = _load_case_for_ngspice(spec["case_file"], executable, timeout_s, objective=objective)
        case_run_dir = run_dir / spec.get("output_dir", case_key)
        result = optimize_closed_loop(
            case,
            n_trials=int(args.n_trials or profile["n_trials"]),
            run_root=case_run_dir,
            optimizer_name=args.optimizer,
            solver_override="ngspice_cli",
            seed=int(args.seed if args.seed is not None else spec["seed"]),
        )
        summary = save_summary(case_run_dir, case_run_dir / "summary.csv")
        surrogate = _fit_surrogate_if_possible(case_run_dir)
        manifest["cases"][case_key] = {
            "case_id": case.case_id,
            "case_file": spec["case_file"],
            "run_dir": str(case_run_dir),
            "n_trials": int(result.get("n_trials", 0)),
            "n_failed_trials": int(result.get("n_failed_trials", 0)),
            "summary_csv": str(case_run_dir / "summary.csv"),
            "summary_rows": int(len(summary)),
            "surrogate_json": str(case_run_dir / "surrogate.json") if surrogate else None,
            "best": result.get("best"),
        }
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CCP Level 2/3 benchmark profiles with ngspice_cli.")
    parser.add_argument("--profile-file", default=str(DEFAULT_PROFILE_FILE))
    parser.add_argument("--profile", choices=["smoke", "standard", "extended", "category_extended", "harmonic_focused", "surrogate_feasible"], default="smoke")
    parser.add_argument("--cases", nargs="+", choices=["level2_timevarying_plasma", "level3_topology_load_choice"])
    parser.add_argument("--run-root", help="Override output root. Defaults to the profile output_root.")
    parser.add_argument("--label", help="Run directory label below the output root.")
    parser.add_argument("--n-trials", type=int, help="Override profile trial count.")
    parser.add_argument("--seed", type=int, help="Override every case seed.")
    parser.add_argument("--optimizer", default="random")
    parser.add_argument("--objective", help="Override target.objective for every selected case.")
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--executable", help="Optional ngspice executable override, for example ngspice_con.exe.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-exit", action="store_true", help="Exit nonzero if any trial fails.")
    args = parser.parse_args(argv)
    manifest = run_profile(args)
    if args.strict_exit:
        failed = sum(int(case.get("n_failed_trials", 0)) for case in manifest.get("cases", {}).values())
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
