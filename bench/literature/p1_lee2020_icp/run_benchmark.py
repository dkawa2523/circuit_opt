"""Lee et al. (2020) Eq. 18/19 algebra benchmark for the ICP load model.

The paper does not publish a machine-readable table of its density-dependent
Rp, Lp, and M curves.  The cases in this directory are therefore deliberately
dimensionally valid algebra cases, not reconstructed experimental golden data.
They test the exact circuit reduction shared by the paper and PCD while leaving
the paper's geometry-to-parameter plasma calculation outside the public API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcd.rf_loads import icp_effective_impedance  # noqa: E402

PAPER_DOI = "https://doi.org/10.1063/1.5133862"
PYTHON_REL_TOL = 1.0e-12
PYTHON_ABS_TOL_OHM = 1.0e-12
NGSPICE_REL_TOL = 2.0e-5
NGSPICE_ABS_TOL_OHM = 1.0e-7
POWER_REL_TOL = 5.0e-5
POWER_ABS_TOL_W = 1.0e-10
SPICE_COIL_REGULARIZATION_OHM = 1.0e-6


@dataclass(frozen=True)
class AlgebraCase:
    case_id: str
    regime: str
    frequency_hz: float
    current_peak_a: float
    coil_inductance_h: float
    plasma_resistance_ohm: float
    plasma_inductance_h: float
    electron_inertia_inductance_h: float
    mutual_inductance_h: float


@dataclass(frozen=True)
class Oracle:
    impedance_ohm: complex
    plasma_resistance_seen_ohm: float
    plasma_reactance_seen_ohm: float
    absorbed_power_w: float
    absorbed_power_eq19_w: float
    secondary_inductance_h: float
    reflected_inductance_h: float
    damping_rate_rad_s: float
    damping_ratio: float
    collision_rate_rad_s: float
    coupling: float


@dataclass(frozen=True)
class NgspiceResult:
    impedance_ohm: complex
    absorbed_power_w: float


def _finite_positive(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def load_cases(path: Path) -> list[AlgebraCase]:
    cases: list[AlgebraCase] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cases.append(
                AlgebraCase(
                    case_id=row["case_id"],
                    regime=row["regime"],
                    frequency_hz=float(row["frequency_Hz"]),
                    current_peak_a=float(row["I_RF_peak_A"]),
                    coil_inductance_h=float(row["L_coil_H"]),
                    plasma_resistance_ohm=float(row["R_p_ohm"]),
                    plasma_inductance_h=float(row["L_p_H"]),
                    electron_inertia_inductance_h=float(row["L_e_H"]),
                    mutual_inductance_h=float(row["M_H"]),
                )
            )
    if not cases:
        raise ValueError(f"no algebra cases in {path}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError(f"case_id values must be unique in {path}")
    return cases


def paper_oracle(case: AlgebraCase) -> Oracle:
    """Evaluate Eq. 18/19 without calling a PCD load-model function.

    Equation 18 is translated to the passive input-port convention required by
    its positive absorbed-power Eq. 19.  The reflected plasma term then has a
    positive real part and a negative reactive correction:

        Z = jw*Lc + w^2*M^2 / (Rp + jw*(Lp + Le)).

    Le is Rp/nu in the paper.  Keeping Le in the fixture avoids introducing
    collision physics into PCD's public input while still checking Eq. 19.
    """

    frequency = _finite_positive("frequency_Hz", case.frequency_hz)
    current_peak = _finite_positive("I_RF_peak_A", case.current_peak_a)
    coil_inductance = _finite_positive("L_coil_H", case.coil_inductance_h)
    plasma_resistance = _finite_positive("R_p_ohm", case.plasma_resistance_ohm)
    plasma_inductance = _finite_positive("L_p_H", case.plasma_inductance_h)
    inertia_inductance = _finite_positive("L_e_H", case.electron_inertia_inductance_h)
    mutual_inductance = _finite_positive("M_H", case.mutual_inductance_h)

    omega = 2.0 * math.pi * frequency
    secondary_inductance = plasma_inductance + inertia_inductance
    denominator = plasma_resistance**2 + (omega * secondary_inductance) ** 2
    reflected_resistance = omega**2 * mutual_inductance**2 * plasma_resistance / denominator
    reflected_reactance = -(omega**3 * mutual_inductance**2 * secondary_inductance) / denominator
    impedance = complex(reflected_resistance, omega * coil_inductance + reflected_reactance)

    collision_rate = plasma_resistance / inertia_inductance
    absorbed_power = 0.5 * current_peak**2 * reflected_resistance
    absorbed_power_eq19 = (
        0.5
        * current_peak**2
        * omega**2
        * mutual_inductance**2
        * plasma_resistance
        / (plasma_resistance**2 + omega**2 * (plasma_resistance / collision_rate + plasma_inductance) ** 2)
    )
    reflected_inductance = mutual_inductance**2 / secondary_inductance
    damping_rate = plasma_resistance / secondary_inductance
    coupling = mutual_inductance / math.sqrt(coil_inductance * secondary_inductance)
    return Oracle(
        impedance_ohm=impedance,
        plasma_resistance_seen_ohm=reflected_resistance,
        plasma_reactance_seen_ohm=reflected_reactance,
        absorbed_power_w=absorbed_power,
        absorbed_power_eq19_w=absorbed_power_eq19,
        secondary_inductance_h=secondary_inductance,
        reflected_inductance_h=reflected_inductance,
        damping_rate_rad_s=damping_rate,
        damping_ratio=damping_rate / omega,
        collision_rate_rad_s=collision_rate,
        coupling=coupling,
    )


def _close(actual: float, expected: float, relative: float, absolute: float) -> bool:
    return abs(actual - expected) <= absolute + relative * abs(expected)


def _complex_close(actual: complex, expected: complex, relative: float, absolute: float) -> bool:
    return _close(actual.real, expected.real, relative, absolute) and _close(
        actual.imag, expected.imag, relative, absolute
    )


def _complex_vector_close(actual: complex, expected: complex, relative: float, absolute: float) -> bool:
    """Compare an AC vector using the same scaling as the repository E2E test.

    NGSpice solves voltage/current and PCD derives impedance by division.  When
    Re(Z) is much smaller than Im(Z), a component-wise relative tolerance would
    amplify harmless solver digits in source current.  Absorbed power is checked
    separately from the secondary-resistor current, so this vector norm does not
    conceal an error in plasma loss.
    """

    return abs(actual - expected) <= absolute + relative * abs(expected)


def _pcd_impedance(case: AlgebraCase, oracle: Oracle) -> complex:
    return icp_effective_impedance(
        case.frequency_hz,
        0.0,
        case.coil_inductance_h,
        oracle.reflected_inductance_h,
        oracle.damping_rate_rad_s,
        0.0,
    )


def _ngspice_available() -> bool:
    return bool(shutil.which("ngspice_con.exe") or shutil.which("ngspice"))


def _ngspice_result(case: AlgebraCase, oracle: Oracle, working_root: Path) -> NgspiceResult:
    case_root = working_root / case.case_id
    case_root.mkdir(parents=True, exist_ok=False)
    netlist_path = case_root / "netlist.cir"
    output_path = case_root / "ac.csv"
    log_path = case_root / "solver.log"
    # This netlist is intentionally written from the paper elements instead of
    # using PCD's load-netlist builder.  It is the independent circuit oracle;
    # the PCD Python implementation is evaluated through _pcd_impedance.
    netlist_path.write_text(
        f"""* Lee et al. 2020 Eq. 18 transformer algebra case: {case.case_id}
Vsrc src 0 DC 0 AC 1
Rcoil_regularization src coil {SPICE_COIL_REGULARIZATION_OHM:.15g}
Lcoil coil 0 {case.coil_inductance_h:.15g}
Lsecondary ns nr {oracle.secondary_inductance_h:.15g}
Rsecondary nr ns {case.plasma_resistance_ohm:.15g}
Rsecondary_ref ns 0 1e15
Kload Lcoil Lsecondary {oracle.coupling:.15g}
.save v(src) i(Vsrc) i(Lsecondary)
.control
set numdgt=15
ac lin 3 {case.frequency_hz / 2.0:.15g} {case.frequency_hz * 1.5:.15g}
wrdata ac.csv v(src) i(Vsrc) i(Lsecondary)
quit
.endc
.end
""",
        encoding="utf-8",
    )

    executable = shutil.which("ngspice_con.exe") or shutil.which("ngspice")
    if executable is None:
        raise RuntimeError("ngspice disappeared from PATH after availability check")
    completed = subprocess.run(
        [executable, "-b", "-o", str(log_path), str(netlist_path)],
        cwd=case_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300.0,
    )
    if completed.returncode != 0 or not output_path.exists():
        solver_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        raise RuntimeError(
            f"ngspice run failed for {case.case_id}:\n{solver_log}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    rows = [[float(field) for field in line.split()] for line in output_path.read_text().splitlines() if line.strip()]
    if not rows or any(len(row) < 9 for row in rows):
        raise RuntimeError(f"ngspice AC output is incomplete for {case.case_id}")
    row = min(rows, key=lambda values: abs(values[0] - case.frequency_hz))
    voltage = complex(row[1], row[2])
    primary_current = -complex(row[4], row[5])
    secondary_current = complex(row[7], row[8])
    if abs(primary_current) == 0.0:
        raise RuntimeError(f"ngspice returned zero primary current for {case.case_id}")
    impedance = voltage / primary_current - complex(SPICE_COIL_REGULARIZATION_OHM, 0.0)
    # Scale the simulated secondary loss to the paper's specified peak primary
    # current so the result can be compared directly with Eq. 19.
    current_ratio_squared = (abs(secondary_current) / abs(primary_current)) ** 2
    absorbed_power = 0.5 * case.current_peak_a**2 * current_ratio_squared * case.plasma_resistance_ohm
    return NgspiceResult(impedance_ohm=impedance, absorbed_power_w=absorbed_power)


def _complex_json(value: complex | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"real": value.real, "imag": value.imag}


def evaluate_case(
    case: AlgebraCase,
    *,
    use_ngspice: bool,
    ngspice_root: Path | None,
) -> dict[str, Any]:
    oracle = paper_oracle(case)
    pcd_impedance = _pcd_impedance(case, oracle)
    ngspice_result = _ngspice_result(case, oracle, ngspice_root) if use_ngspice and ngspice_root else None
    ngspice_impedance = ngspice_result.impedance_ohm if ngspice_result else None

    checks = {
        "mutual_inductance_bound": oracle.coupling <= 1.0 + 1.0e-12,
        "passive_plasma_resistance": oracle.plasma_resistance_seen_ohm > 0.0,
        "plasma_reactive_correction_is_negative": oracle.plasma_reactance_seen_ohm < 0.0,
        "total_port_is_passive": oracle.impedance_ohm.real >= 0.0,
        "eq18_eq19_power": _close(
            oracle.absorbed_power_w,
            oracle.absorbed_power_eq19_w,
            PYTHON_REL_TOL,
            POWER_ABS_TOL_W,
        ),
        "pcd_impedance_matches_eq18": _complex_close(
            pcd_impedance,
            oracle.impedance_ohm,
            PYTHON_REL_TOL,
            PYTHON_ABS_TOL_OHM,
        ),
    }
    pcd_power = 0.5 * case.current_peak_a**2 * pcd_impedance.real
    checks["pcd_power_matches_eq19"] = _close(
        pcd_power,
        oracle.absorbed_power_eq19_w,
        PYTHON_REL_TOL,
        POWER_ABS_TOL_W,
    )

    ngspice_power: float | None = None
    if ngspice_result is not None and ngspice_impedance is not None:
        ngspice_power = ngspice_result.absorbed_power_w
        checks["ngspice_impedance_matches_eq18"] = _complex_vector_close(
            ngspice_impedance,
            oracle.impedance_ohm,
            NGSPICE_REL_TOL,
            NGSPICE_ABS_TOL_OHM,
        )
        checks["ngspice_power_matches_eq19"] = _close(
            ngspice_power,
            oracle.absorbed_power_eq19_w,
            POWER_REL_TOL,
            POWER_ABS_TOL_W,
        )

    return {
        "case": asdict(case),
        "derived": {
            "secondary_inductance_H": oracle.secondary_inductance_h,
            "reflected_inductance_H": oracle.reflected_inductance_h,
            "secondary_damping_rate_rad_s": oracle.damping_rate_rad_s,
            "gamma_over_omega": oracle.damping_ratio,
            "collision_rate_rad_s": oracle.collision_rate_rad_s,
            "coupling": oracle.coupling,
            "plasma_resistance_seen_ohm": oracle.plasma_resistance_seen_ohm,
            "plasma_reactance_seen_ohm": oracle.plasma_reactance_seen_ohm,
        },
        "oracle_impedance_ohm": _complex_json(oracle.impedance_ohm),
        "pcd_impedance_ohm": _complex_json(pcd_impedance),
        "ngspice_impedance_ohm": _complex_json(ngspice_impedance),
        "absorbed_power_eq19_W": oracle.absorbed_power_eq19_w,
        "pcd_absorbed_power_W": pcd_power,
        "ngspice_absorbed_power_W": ngspice_power,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _coverage_checks(results: list[dict[str, Any]]) -> dict[str, bool]:
    ratios = [float(result["derived"]["gamma_over_omega"]) for result in results]
    frequencies = {float(result["case"]["frequency_hz"]) for result in results}
    transition = min(abs(math.log10(ratio)) for ratio in ratios)
    return {
        "weak_damping_covered": min(ratios) <= 0.01 * (1.0 + 1.0e-12),
        "transition_covered": transition <= 1.0e-12,
        "strong_damping_covered": max(ratios) >= 100.0 * (1.0 - 1.0e-12),
        "multiple_frequencies_covered": len(frequencies) >= 2,
    }


def _symmetry_checks(results: list[dict[str, Any]]) -> dict[str, bool]:
    by_id = {result["case"]["case_id"]: result for result in results}

    def same_power(left: str, right: str) -> bool:
        return _close(
            float(by_id[left]["absorbed_power_eq19_W"]),
            float(by_id[right]["absorbed_power_eq19_W"]),
            PYTHON_REL_TOL,
            POWER_ABS_TOL_W,
        )

    return {
        "reciprocal_ratio_power_0.01_100": same_power("weak_damping_13M56", "strong_damping_13M56"),
        "reciprocal_ratio_power_0.1_10": same_power("underdamped_13M56", "overdamped_13M56"),
    }


def run(*, cases_path: Path, run_ngspice: bool, require_ngspice: bool) -> dict[str, Any]:
    available = _ngspice_available()
    if require_ngspice and not available:
        raise RuntimeError("ngspice is required but neither ngspice_con.exe nor ngspice is on PATH")
    use_ngspice = run_ngspice and available

    cases = load_cases(cases_path)
    with tempfile.TemporaryDirectory(prefix="pcd-lee2020-") as temporary:
        working_root = Path(temporary)
        results = [evaluate_case(case, use_ngspice=use_ngspice, ngspice_root=working_root) for case in cases]
    coverage = _coverage_checks(results)
    symmetry = _symmetry_checks(results)
    passed = all(result["passed"] for result in results) and all(coverage.values()) and all(symmetry.values())
    return {
        "benchmark": "p1_lee2020_icp_equations_18_19",
        "source": {
            "title": "A simple model of solenoidal inductively coupled plasma sources considering finite size",
            "doi": PAPER_DOI,
            "equations": [18, 19],
            "evidence_class": "paper_equation_algebra_case",
            "experimental_golden": False,
        },
        "scope": {
            "validates": [
                "Eq. 18 passive-port impedance reduction",
                "Eq. 19 absorbed-power identity with peak-current convention",
                "mapping Lref=M^2/(Lp+Le), gamma=Rp/(Lp+Le)",
                "PCD analytic implementation and an independent optional ngspice transformer",
            ],
            "does_not_validate": [
                "geometry-to-circuit plasma model",
                "electron-density reconstruction",
                "experimental chamber accuracy",
                "matching-network performance",
            ],
        },
        "tolerances": {
            "python_relative": PYTHON_REL_TOL,
            "python_absolute_ohm": PYTHON_ABS_TOL_OHM,
            "ngspice_relative": NGSPICE_REL_TOL,
            "ngspice_absolute_ohm": NGSPICE_ABS_TOL_OHM,
            "power_relative": POWER_REL_TOL,
            "power_absolute_W": POWER_ABS_TOL_W,
        },
        "ngspice": {
            "requested": run_ngspice,
            "available": available,
            "executed": use_ngspice,
            "required": require_ngspice,
        },
        "coverage_checks": coverage,
        "symmetry_checks": symmetry,
        "cases": results,
        "summary": {
            "case_count": len(results),
            "passed_case_count": sum(bool(result["passed"]) for result in results),
            "failed_case_ids": [result["case"]["case_id"] for result in results if not result["passed"]],
            "passed": passed,
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    status = "PASS" if report["summary"]["passed"] else "FAIL"
    ngspice = report["ngspice"]
    print(f"Lee 2020 ICP Eq.18/19 benchmark: {status}")
    print(
        f"cases: {report['summary']['passed_case_count']}/{report['summary']['case_count']} passed; "
        f"ngspice: {'executed' if ngspice['executed'] else 'skipped'}"
    )
    print("case_id                         gamma/omega        Re(Z) ohm        Im(Z) ohm       Pabs W")
    for result in report["cases"]:
        case = result["case"]
        derived = result["derived"]
        impedance = result["oracle_impedance_ohm"]
        marker = "PASS" if result["passed"] else "FAIL"
        print(
            f"{case['case_id']:<31} {derived['gamma_over_omega']:>11.5g} "
            f"{impedance['real']:>16.9g} {impedance['imag']:>16.9g} "
            f"{result['absorbed_power_eq19_W']:>12.7g}  {marker}"
        )
    failed_global = [
        name
        for group in (report["coverage_checks"], report["symmetry_checks"])
        for name, passed in group.items()
        if not passed
    ]
    if failed_global:
        print("failed global checks: " + ", ".join(failed_global))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.csv")
    parser.add_argument("--json", action="store_true", help="print the full machine-readable report")
    parser.add_argument("--output", type=Path, help="also write the full report as JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-ngspice", action="store_true", help="run only Eq.18/19 vs the PCD Python function")
    group.add_argument("--require-ngspice", action="store_true", help="fail if the ngspice comparison cannot run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(
            cases_path=args.cases.resolve(),
            run_ngspice=not args.no_ngspice,
            require_ngspice=args.require_ngspice,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"benchmark setup failed: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
