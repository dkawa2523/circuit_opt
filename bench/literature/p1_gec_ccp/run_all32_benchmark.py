"""Validate all 32 central CCP points in Hargis et al. Tables III/IV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "raw_tables_iii_iv.csv"
DERIVED_PATH = HERE / "derived_impedance_all32.csv"
DERIVED_66PA_PATH = HERE / "derived_impedance_66pa.csv"
SPREAD_66PA_PATH = HERE / "reported_spread_envelope.csv"

DRIVE_FREQUENCY_HZ = 13_560_000.0
PRESSURES_PA = (13, 33, 66, 133)
DRIVE_LEVELS_VPP = (75, 100, 150, 200)
TABLE_GROUPS = {"III": (149, 24), "IV": (150, 34)}
POWER_RELATIVE_TOLERANCE = 0.05
NGSPICE_RELATIVE_TOLERANCE = 1.0e-8
NGSPICE_ABSOLUTE_TOLERANCE_OHM = 1.0e-6

SOURCE_FIELDS = (
    "source_table",
    "journal_page",
    "resonance_group_MHz",
    "drive_frequency_Hz",
    "pressure_Pa",
    "external_drive_vpp_V",
    "electrode_voltage_peak_V",
    "electrode_voltage_spread_V",
    "phase_deg",
    "phase_spread_deg",
    "electrode_current_peak_A",
    "electrode_current_spread_A",
    "dc_bias_V",
    "dc_bias_spread_V",
    "reported_power_W",
    "reported_power_spread_W",
    "reference_plane",
)

NUMERIC_FIELDS = SOURCE_FIELDS[1:-1]
DERIVED_FIELDS = (
    "scenario_id",
    "frequency_Hz",
    "resistance_ohm",
    "reactance_ohm",
    "weight",
    "impedance_magnitude_ohm",
    "impedance_phase_deg",
    "fundamental_power_W",
    "reported_power_W",
    "power_relative_difference",
    "resonance_group_MHz",
    "pressure_Pa",
    "external_drive_vpp_V",
    "reference_plane",
    "source_table",
    "source_journal_page",
)
SPREAD_FIELDS = (
    "scenario_id",
    "frequency_Hz",
    "resistance_ohm",
    "reactance_ohm",
    "weight",
    "central_scenario_id",
    "magnitude_bound",
    "phase_bound",
    "impedance_magnitude_ohm",
    "impedance_phase_deg",
    "resonance_group_MHz",
    "pressure_Pa",
    "external_drive_vpp_V",
    "reference_plane",
    "meaning",
)


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    for row in rows:
        for field in NUMERIC_FIELDS:
            value = row[field]
            if value == "":
                continue
            row[field] = float(value)
    return rows, fields


def _key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["source_table"]),
        int(row["resonance_group_MHz"]),
        int(row["pressure_Pa"]),
        int(row["external_drive_vpp_V"]),
    )


def _scenario_id(row: dict[str, Any]) -> str:
    pressure = int(row["pressure_Pa"])
    group = int(row["resonance_group_MHz"])
    vpp = int(row["external_drive_vpp_V"])
    return f"gec{pressure}_r{group}_vpp{vpp:03d}"


def _derive(row: dict[str, Any]) -> dict[str, Any]:
    voltage = float(row["electrode_voltage_peak_V"])
    current = float(row["electrode_current_peak_A"])
    phase_deg = float(row["phase_deg"])
    magnitude = voltage / current
    phase_rad = math.radians(phase_deg)
    impedance = magnitude * complex(math.cos(phase_rad), math.sin(phase_rad))
    power = 0.5 * voltage * current * math.cos(phase_rad)
    reported_power = float(row["reported_power_W"])
    return {
        "scenario_id": _scenario_id(row),
        "frequency_Hz": float(row["drive_frequency_Hz"]),
        "resistance_ohm": impedance.real,
        "reactance_ohm": impedance.imag,
        "weight": 1,
        "impedance_magnitude_ohm": abs(impedance),
        "impedance_phase_deg": math.degrees(math.atan2(impedance.imag, impedance.real)),
        "fundamental_power_W": power,
        "reported_power_W": reported_power,
        "power_relative_difference": abs(power - reported_power) / reported_power,
        "resonance_group_MHz": float(row["resonance_group_MHz"]),
        "pressure_Pa": float(row["pressure_Pa"]),
        "external_drive_vpp_V": float(row["external_drive_vpp_V"]),
        "reference_plane": row["reference_plane"],
        "source_table": row["source_table"],
        "source_journal_page": float(row["journal_page"]),
    }


def _spread_corners(row: dict[str, Any]) -> list[dict[str, Any]]:
    voltage = float(row["electrode_voltage_peak_V"])
    voltage_spread = float(row["electrode_voltage_spread_V"])
    current = float(row["electrode_current_peak_A"])
    current_spread = float(row["electrode_current_spread_A"])
    magnitude_bounds = {
        "low": (voltage - voltage_spread) / (current + current_spread),
        "high": (voltage + voltage_spread) / (current - current_spread),
    }
    phase_bounds = {
        "low": float(row["phase_deg"]) - float(row["phase_spread_deg"]),
        "high": float(row["phase_deg"]) + float(row["phase_spread_deg"]),
    }
    central_id = _scenario_id(row)
    result = []
    for magnitude_name, magnitude in magnitude_bounds.items():
        for phase_name, phase_deg in phase_bounds.items():
            impedance = magnitude * complex(math.cos(math.radians(phase_deg)), math.sin(math.radians(phase_deg)))
            result.append(
                {
                    "scenario_id": f"{central_id}__z{magnitude_name}_phase_{phase_name}",
                    "frequency_Hz": float(row["drive_frequency_Hz"]),
                    "resistance_ohm": impedance.real,
                    "reactance_ohm": impedance.imag,
                    "weight": 1,
                    "central_scenario_id": central_id,
                    "magnitude_bound": magnitude_name,
                    "phase_bound": phase_name,
                    "impedance_magnitude_ohm": magnitude,
                    "impedance_phase_deg": phase_deg,
                    "resonance_group_MHz": float(row["resonance_group_MHz"]),
                    "pressure_Pa": float(row["pressure_Pa"]),
                    "external_drive_vpp_V": float(row["external_drive_vpp_V"]),
                    "reference_plane": row["reference_plane"],
                    "meaning": "four-corner envelope from the table's reported group spread",
                }
            )
    return result


def _committed_view_matches(path: Path, fields: tuple[str, ...], expected: list[dict[str, Any]]) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
        if tuple(reader.fieldnames or ()) != fields or len(actual) != len(expected):
            return False
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for field in fields:
            left = actual_row[field]
            right = expected_row[field]
            if isinstance(right, (int, float)):
                if not math.isclose(float(left), float(right), rel_tol=2.0e-11, abs_tol=1.0e-8):
                    return False
            elif left != str(right):
                return False
    return True


def _ngspice_replay(derived_rows: list[dict[str, Any]]) -> dict[str, Any]:
    executable = shutil.which("ngspice_con.exe") or shutil.which("ngspice")
    if executable is None:
        return {"available": False, "executed": False, "rows": []}

    frequency = DRIVE_FREQUENCY_HZ
    omega = 2.0 * math.pi * frequency
    elements: list[str] = []
    vectors: list[str] = []
    for index, row in enumerate(derived_rows, start=1):
        resistance = float(row["resistance_ohm"])
        reactance = float(row["reactance_ohm"])
        capacitance = -1.0 / (omega * reactance)
        elements.extend(
            (
                f"Vsrc{index} n{index} 0 DC 0 AC 1",
                f"Rload{index} n{index} m{index} {resistance:.15g}",
                f"Cload{index} m{index} 0 {capacitance:.15g}",
            )
        )
        vectors.extend((f"v(n{index})", f"i(Vsrc{index})"))

    with tempfile.TemporaryDirectory(prefix="pcd_gec_all32_") as temporary:
        root = Path(temporary)
        netlist_path = root / "replay.cir"
        output_path = root / "ac.csv"
        log_path = root / "solver.log"
        netlist_path.write_text(
            "\n".join(
                [
                    "* Hargis 1994 Tables III/IV series-RC replay",
                    *elements,
                    ".control",
                    "set numdgt=15",
                    f"ac lin 3 {frequency / 2.0:.15g} {frequency * 1.5:.15g}",
                    f"wrdata ac.csv {' '.join(vectors)}",
                    "quit",
                    ".endc",
                    ".end",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [executable, "-b", "-o", str(log_path), str(netlist_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=300.0,
        )
        if completed.returncode != 0 or not output_path.exists():
            solver_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(
                f"ngspice replay failed:\n{solver_log}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        rows = [
            [float(field) for field in line.split()]
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    expected_columns = 6 * len(derived_rows)
    if not rows or any(len(row) < expected_columns for row in rows):
        raise RuntimeError("ngspice replay output has an unexpected wrdata layout")
    values = min(rows, key=lambda row: abs(row[0] - frequency))
    results = []
    for index, expected in enumerate(derived_rows):
        offset = 6 * index
        voltage = complex(values[offset + 1], values[offset + 2])
        source_current = complex(values[offset + 4], values[offset + 5])
        actual = voltage / -source_current
        target = complex(float(expected["resistance_ohm"]), float(expected["reactance_ohm"]))
        absolute_error = abs(actual - target)
        relative_error = absolute_error / abs(target)
        results.append(
            {
                "scenario_id": expected["scenario_id"],
                "impedance_ohm": {"real": actual.real, "imag": actual.imag},
                "absolute_error_ohm": absolute_error,
                "relative_error": relative_error,
                "pass": absolute_error <= NGSPICE_ABSOLUTE_TOLERANCE_OHM + NGSPICE_RELATIVE_TOLERANCE * abs(target),
            }
        )
    return {"available": True, "executed": True, "rows": results}


def _range(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"min": min(values), "max": max(values)}


def run(*, run_ngspice: bool, require_ngspice: bool) -> dict[str, Any]:
    source_rows, source_fields = _load_rows(SOURCE_PATH)
    expected_keys = {
        (table, group, pressure, vpp)
        for table, (_, group) in TABLE_GROUPS.items()
        for pressure, vpp in itertools.product(PRESSURES_PA, DRIVE_LEVELS_VPP)
    }
    observed_keys = [_key(row) for row in source_rows]
    nonempty = all(all(str(row.get(field, "")) != "" for field in SOURCE_FIELDS) for row in source_rows)
    finite = all(
        math.isfinite(float(row[field])) for row in source_rows for field in NUMERIC_FIELDS if row[field] != ""
    )
    source_checks = {
        "exact_unit_bearing_columns": source_fields == SOURCE_FIELDS,
        "all_cells_nonempty": nonempty,
        "all_numeric_values_finite": finite,
        "exact_32_point_cartesian_grid": set(observed_keys) == expected_keys and len(observed_keys) == 32,
        "no_duplicate_points": len(observed_keys) == len(set(observed_keys)),
        "table_page_and_resonance_group": all(
            (int(row["journal_page"]), int(row["resonance_group_MHz"])) == TABLE_GROUPS[str(row["source_table"])]
            for row in source_rows
        ),
        "drive_is_13_56_MHz": all(float(row["drive_frequency_Hz"]) == DRIVE_FREQUENCY_HZ for row in source_rows),
        "powered_electrode_surface_reference_plane": all(
            row["reference_plane"] == "powered_electrode_surface" for row in source_rows
        ),
        "positive_amplitudes_and_power": all(
            float(row[field]) > 0.0
            for row in source_rows
            for field in ("electrode_voltage_peak_V", "electrode_current_peak_A", "reported_power_W")
        ),
        "nonnegative_reported_spreads": all(
            float(row[field]) >= 0.0
            for row in source_rows
            for field in (
                "electrode_voltage_spread_V",
                "phase_spread_deg",
                "electrode_current_spread_A",
                "dc_bias_spread_V",
                "reported_power_spread_W",
            )
        ),
        "published_voltage_phase_is_negative": all(float(row["phase_deg"]) < 0.0 for row in source_rows),
    }

    derived_rows = [_derive(row) for row in source_rows]
    derived_66pa = [row for row in derived_rows if int(row["pressure_Pa"]) == 66]
    spread_66pa = [corner for row in source_rows if int(row["pressure_Pa"]) == 66 for corner in _spread_corners(row)]
    committed_view_checks = {
        "all32_view_matches_derivation": _committed_view_matches(DERIVED_PATH, DERIVED_FIELDS, derived_rows),
        "66Pa_view_matches_all32_subset": _committed_view_matches(DERIVED_66PA_PATH, DERIVED_FIELDS, derived_66pa),
        "66Pa_spread_view_matches_derivation": _committed_view_matches(SPREAD_66PA_PATH, SPREAD_FIELDS, spread_66pa),
    }
    identity_rows = []
    power_rows = []
    for source, derived in zip(source_rows, derived_rows, strict=True):
        impedance = complex(float(derived["resistance_ohm"]), float(derived["reactance_ohm"]))
        power_from_resistance = 0.5 * float(source["electrode_current_peak_A"]) ** 2 * impedance.real
        identity_rows.append(
            {
                "scenario_id": derived["scenario_id"],
                "magnitude_error_ohm": abs(
                    abs(impedance)
                    - float(source["electrode_voltage_peak_V"]) / float(source["electrode_current_peak_A"])
                ),
                "phase_error_deg": abs(
                    math.degrees(math.atan2(impedance.imag, impedance.real)) - float(source["phase_deg"])
                ),
                "power_identity_error_W": abs(power_from_resistance - float(derived["fundamental_power_W"])),
            }
        )
        absolute_difference = abs(float(derived["fundamental_power_W"]) - float(source["reported_power_W"]))
        power_rows.append(
            {
                "scenario_id": derived["scenario_id"],
                "fundamental_power_W": derived["fundamental_power_W"],
                "reported_five_harmonic_power_W": source["reported_power_W"],
                "absolute_difference_W": absolute_difference,
                "relative_difference": derived["power_relative_difference"],
                "within_5_percent": float(derived["power_relative_difference"]) <= POWER_RELATIVE_TOLERANCE,
                "within_reported_spread": absolute_difference <= float(source["reported_power_spread_W"]),
            }
        )

    physical_checks = {
        "positive_resistance": all(float(row["resistance_ohm"]) > 0.0 for row in derived_rows),
        "capacitive_reactance": all(float(row["reactance_ohm"]) < 0.0 for row in derived_rows),
        "phase_between_minus_90_and_0_deg": all(
            -90.0 < float(row["impedance_phase_deg"]) < 0.0 for row in derived_rows
        ),
        "independent_impedance_and_power_identities": all(
            row["magnitude_error_ohm"] <= 1.0e-10
            and row["phase_error_deg"] <= 1.0e-12
            and row["power_identity_error_W"] <= 1.0e-12
            for row in identity_rows
        ),
        "fundamental_power_closure": all(
            bool(row["within_5_percent"]) and bool(row["within_reported_spread"]) for row in power_rows
        ),
    }

    spice: dict[str, Any] = (
        _ngspice_replay(derived_rows) if run_ngspice else {"available": False, "executed": False, "rows": []}
    )
    if require_ngspice and not spice["executed"]:
        raise RuntimeError("ngspice replay is required but ngspice is not available")
    spice_pass = not spice["executed"] or all(bool(row["pass"]) for row in spice["rows"])
    checks = {
        "source_integrity": all(source_checks.values()),
        "committed_views": all(committed_view_checks.values()),
        "physical_and_power_consistency": all(physical_checks.values()),
        "ngspice_replay": spice_pass,
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": "literature_table_benchmark.v1",
        "benchmark_id": "p1_gec_ccp_tables_iii_iv_all32",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "source_checks": source_checks,
        "committed_view_checks": committed_view_checks,
        "physical_checks": physical_checks,
        "counts": {
            "source_rows": len(source_rows),
            "derived_rows": len(derived_rows),
            "pressures": len({int(row["pressure_Pa"]) for row in source_rows}),
            "resonance_groups": len({int(row["resonance_group_MHz"]) for row in source_rows}),
            "external_drive_levels_per_group_pressure": 4,
            "derived_66Pa_rows": len(derived_66pa),
            "reported_spread_66Pa_rows": len(spread_66pa),
        },
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "ranges": {
            "resistance_ohm": _range(derived_rows, "resistance_ohm"),
            "reactance_ohm": _range(derived_rows, "reactance_ohm"),
            "impedance_magnitude_ohm": _range(derived_rows, "impedance_magnitude_ohm"),
            "fundamental_power_W": _range(derived_rows, "fundamental_power_W"),
        },
        "maximum_fundamental_power_relative_difference": max(float(row["relative_difference"]) for row in power_rows),
        "power_checks": power_rows,
        "identity_checks": identity_rows,
        "ngspice": {
            "requested": run_ngspice,
            "required": require_ngspice,
            "available": spice["available"],
            "executed": spice["executed"],
            "maximum_absolute_error_ohm": max(
                (float(row["absolute_error_ohm"]) for row in spice["rows"]), default=None
            ),
            "maximum_relative_error": max((float(row["relative_error"]) for row in spice["rows"]), default=None),
            "rows": spice["rows"],
        },
        "interpretation": {
            "24_and_34_MHz": "empty-cell resonance groups; every row is driven at 13.56 MHz",
            "fundamental_power": "consistency check against reported power summed through the fifth harmonic",
            "spreads": "reported standard deviations of group means, not instrument-only uncertainty",
            "derived_views": "committed immutable views checked against the single 32-row source table",
            "scope": "GEC reference-cell terminal-load evidence, not production-chamber or microscopic-plasma validation",
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-ngspice", action="store_true", help="skip the series-RC circuit replay")
    group.add_argument("--require-ngspice", action="store_true", help="fail if the circuit replay cannot run")
    args = parser.parse_args()
    result = run(run_ngspice=not args.no_ngspice, require_ngspice=args.require_ngspice)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"P1 GEC CCP Tables III/IV all 32 central points: {result['status']}")
    print(f"  source/derived points: {result['counts']['source_rows']}/{result['counts']['derived_rows']}")
    print(
        "  R range: "
        f"{result['ranges']['resistance_ohm']['min']:.3f} to "
        f"{result['ranges']['resistance_ohm']['max']:.3f} ohm"
    )
    print(
        "  X range: "
        f"{result['ranges']['reactance_ohm']['min']:.3f} to "
        f"{result['ranges']['reactance_ohm']['max']:.3f} ohm"
    )
    print(f"  max |P1-P_reported|/P_reported: {100.0 * result['maximum_fundamental_power_relative_difference']:.3f}%")
    print(
        "  ngspice replay: "
        f"{'executed' if result['ngspice']['executed'] else 'skipped'}; "
        f"max relative error={result['ngspice']['maximum_relative_error']}"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
