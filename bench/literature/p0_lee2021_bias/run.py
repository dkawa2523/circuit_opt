"""Execute the Lee 2021 P0 source-fidelity and reference-plane benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pcd.analysis import input_impedance  # noqa: E402
from pcd.case import Case, load_case  # noqa: E402
from pcd.solver import ngspice_cli  # noqa: E402
from pcd.study import run_case_study  # noqa: E402


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return data


def _table_impedance(source: dict[str, Any]) -> complex:
    """Independent phasor evaluation of the Table-I equivalent circuit."""

    frequency = float(source["operating_condition"]["frequency_Hz"])
    values = source["table_i"]["equivalent_circuit"]
    omega = 2.0 * math.pi * frequency
    c_fixed = float(values["fixed_structure_capacitance_F"])
    c_sheath = float(values["sheath_capacitance_F"])
    r_plasma = float(values["plasma_resistance_ohm"])
    l_plasma = float(values["plasma_inductance_H"])
    z_plasma = complex(r_plasma, omega * l_plasma)
    z_parallel = 1.0 / (1.0 / z_plasma + 1j * omega * c_fixed)
    return 1.0 / (1j * omega * c_sheath) + z_parallel


def _pi_input_impedance(load: complex, frequency_hz: float, c1: float, l1: float, c2: float) -> complex:
    """Independent ideal-network oracle; it does not call a PCD circuit function."""

    omega = 2.0 * math.pi * frequency_hz
    load_with_c2 = 1.0 / (1.0 / load + 1j * omega * c2)
    series_branch = 1j * omega * l1 + load_with_c2
    return 1.0 / (1j * omega * c1 + 1.0 / series_branch)


def _gamma(impedance: complex, reference_ohm: float = 50.0) -> complex:
    return (impedance - reference_ohm) / (impedance + reference_ohm)


def _table_spice_netlist(values: dict[str, Any], frequency: float) -> str:
    """Render the independent Table-I netlist with the current AC artifact contract."""

    return "\n".join(
        [
            "* Lee 2021 Table I: Csheath series [Cfixed parallel (Rp series Lp)]",
            "Vsrc src 0 AC 1",
            f"Csheath src bulk {float(values['sheath_capacitance_F']):.12g}",
            f"Cfixed bulk 0 {float(values['fixed_structure_capacitance_F']):.12g}",
            f"Rplasma bulk rl {float(values['plasma_resistance_ohm']):.12g}",
            f"Lplasma rl 0 {float(values['plasma_inductance_H']):.12g}",
            ".control",
            f"ac lin 1 {frequency:.12g} {frequency:.12g}",
            # Source V/I are followed by the declared output voltage.  The
            # corrected terminal being closed here is the source port itself.
            "wrdata ac.csv v(src) i(Vsrc) v(src)",
            "quit",
            ".endc",
            ".end",
            "",
        ]
    )


def _table_spice(source: dict[str, Any], root: Path) -> complex:
    frequency = float(source["operating_condition"]["frequency_Hz"])
    values = source["table_i"]["equivalent_circuit"]
    run_dir = root / "table_i_closure"
    run_dir.mkdir(parents=True, exist_ok=True)
    netlist = run_dir / "table_i_closure.cir"
    netlist.write_text(_table_spice_netlist(values, frequency), encoding="utf-8")
    case = Case(
        path=netlist,
        data={
            "case_id": "literature_p0_lee2021_table_i_closure",
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": frequency}},
        },
    )
    simulation = ngspice_cli(netlist, run_dir, case, {})
    if simulation.status != "ok" or simulation.frequency_response is None:
        raise RuntimeError(f"Table-I ngspice run failed: {simulation.diagnostics}\n{simulation.log}")
    row = input_impedance(simulation.frequency_response).iloc[0]
    return complex(float(row["resistance_ohm"]), float(row["reactance_ohm"]))


def _matching_case(path: Path, root: Path) -> dict[str, Any]:
    authored = _read_yaml(path)
    case = load_case(path)
    study = run_case_study(case, run_root=root, n_trials=1, solver_override="ngspice_cli", seed=0)
    candidate_id = str(study["best"]["candidate"]["candidate_id"])
    candidate_path = Path(study["run_root"]) / "candidates" / f"{candidate_id}.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    selected = candidate["scenarios"][0]["selected"]
    metrics = selected["metrics"]

    frequency = float(authored["frequency_Hz"])
    load_cfg = authored["load"]
    network = authored["network"]["fixed"]
    oracle_zin = _pi_input_impedance(
        complex(float(load_cfg["resistance_ohm"]), float(load_cfg["reactance_ohm"])),
        frequency,
        float(network["C1"]),
        float(network["L1"]),
        float(network["C2"]),
    )
    oracle_gamma = abs(_gamma(oracle_zin))
    spice_zin = complex(float(metrics["resistance_ohm"]), float(metrics["reactance_ohm"]))
    return {
        "case_file": path.name,
        "case_id": case.case_id,
        "reference_plane": str(load_cfg["reference_plane"]),
        "load_ohm": {"resistance": float(load_cfg["resistance_ohm"]), "reactance": float(load_cfg["reactance_ohm"])},
        "oracle_input_ohm": {"resistance": oracle_zin.real, "reactance": oracle_zin.imag},
        "spice_input_ohm": {"resistance": spice_zin.real, "reactance": spice_zin.imag},
        "oracle_reflection_magnitude": oracle_gamma,
        "spice_reflection_magnitude": float(metrics["reflection_magnitude"]),
        "spice_reflected_power_fraction": float(metrics["reflection_magnitude"]) ** 2,
        "feasible": float(candidate["feasible_fraction"]) == 1.0 and float(candidate["success_fraction"]) == 1.0,
        "solver_status": str(selected["raw"]["status"]),
    }


def execute(run_root: Path) -> dict[str, Any]:
    source = _read_yaml(HERE / "source.yaml")
    expected = _read_yaml(HERE / "expectations.yaml")
    oracle_table = _table_impedance(source)
    spice_table = _table_spice(source, run_root)
    target = expected["table_i_closure"]["paper_target_ohm"]
    paper_tolerance = expected["table_i_closure"]["paper_rounding_tolerance_ohm"]
    numeric_tolerance = expected["table_i_closure"]["ngspice_vs_independent_oracle_tolerance_ohm"]

    table_checks = {
        "oracle_resistance_matches_rounded_table": abs(oracle_table.real - float(target["resistance"]))
        <= float(paper_tolerance["resistance_abs"]),
        "oracle_reactance_matches_rounded_table": abs(oracle_table.imag - float(target["reactance"]))
        <= float(paper_tolerance["reactance_abs"]),
        "ngspice_resistance_matches_oracle": abs(spice_table.real - oracle_table.real)
        <= float(numeric_tolerance["resistance_abs"]),
        "ngspice_reactance_matches_oracle": abs(spice_table.imag - oracle_table.imag)
        <= float(numeric_tolerance["reactance_abs"]),
    }
    table_result = {
        "paper_target_ohm": target,
        "independent_oracle_ohm": {"resistance": oracle_table.real, "reactance": oracle_table.imag},
        "ngspice_ohm": {"resistance": spice_table.real, "reactance": spice_table.imag},
        "checks": table_checks,
        "passed": all(table_checks.values()),
    }

    match_tolerance = expected["matching_numeric_tolerance"]
    matching_results = []
    for filename, case_expected in expected["matching_cases"].items():
        result = _matching_case(HERE / filename, run_root / "matching")
        lower, upper = (float(value) for value in case_expected["expected_reflection_magnitude"])
        oracle_z = result["oracle_input_ohm"]
        spice_z = result["spice_input_ohm"]
        checks = {
            "solver_succeeded": result["solver_status"] == "ok",
            "feasibility_classification": result["feasible"] is bool(case_expected["expected_feasible"]),
            "reflection_in_expected_range": lower <= result["spice_reflection_magnitude"] <= upper,
            "input_resistance_matches_oracle": abs(float(spice_z["resistance"]) - float(oracle_z["resistance"]))
            <= float(match_tolerance["input_resistance_abs_ohm"]),
            "input_reactance_matches_oracle": abs(float(spice_z["reactance"]) - float(oracle_z["reactance"]))
            <= float(match_tolerance["input_reactance_abs_ohm"]),
            "reflection_matches_oracle": abs(
                float(result["spice_reflection_magnitude"]) - float(result["oracle_reflection_magnitude"])
            )
            <= float(match_tolerance["reflection_magnitude_abs"]),
        }
        result.update({"role": case_expected["role"], "checks": checks, "passed": all(checks.values())})
        matching_results.append(result)

    design_regression_passed = all(item["passed"] for item in matching_results)
    passed = bool(table_result["passed"]) and design_regression_passed
    return {
        "schema": "pcd.literature_benchmark_result.v2",
        "benchmark_id": str(source["benchmark_id"]),
        "source_doi": str(source["source"]["doi"]),
        "passed": passed,
        "benchmark_integrity_passed": passed,
        "source_fidelity": {
            "passed": bool(table_result["passed"]),
            "claim": "Table-I equivalent-circuit arithmetic at 13.56 MHz",
            "table_i_closure": table_result,
        },
        "platform_design_challenge": {
            "regression_passed": design_regression_passed,
            "claim": "response of one frozen matcher-output fixture to explicitly named reference-plane substitutions",
            "outcomes": matching_results,
            "reference_plane_rule": (
                "the published impedances are alternate plane descriptions, not simultaneous operating scenarios; "
                "no de-embedding transform is inferred or validated"
            ),
        },
        "claim_boundary": {
            "establishes": [
                "Table-I equivalent-circuit arithmetic at 13.56 MHz",
                "agreement of independent phasor algebra and current PCD/ngspice execution",
                "the circuit consequence of substituting a downstream impedance into a frozen matcher-output fixture",
            ],
            "does_not_establish": [
                "reproduction of the paper's 720 mm transmission-line de-embedding",
                "automatic detection of a mismatched reference plane",
                "ICP source-coil model validity or microscopic plasma-parameter validity",
                "a universal 10 percent reflected-power requirement",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, help="Keep PCD/ngspice artifacts at this path instead of a temporary directory."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path.")
    args = parser.parse_args()

    if args.run_root:
        args.run_root.mkdir(parents=True, exist_ok=True)
        payload = execute(args.run_root.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="pcd_p0_lee2021_") as temporary:
            payload = execute(Path(temporary))

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
