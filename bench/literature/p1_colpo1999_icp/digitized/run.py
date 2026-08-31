"""Qualify and replay the digitized Colpo 1999 plasma-on impedance window."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from pcd.analysis import at_frequency, input_impedance, read_ac
from pcd.artifacts import write_json
from pcd.case import Case
from pcd.sim_core import simulate_case

HERE = Path(__file__).resolve().parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _axis_value(pixel: float, anchors: list[dict[str, float]]) -> float:
    first, second = anchors
    p0, p1 = float(first["pixel"]), float(second["pixel"])
    v0, v1 = float(first["value"]), float(second["value"])
    return v0 + (float(pixel) - p0) * (v1 - v0) / (p1 - p0)


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def _replay_case(row: dict[str, str], run_root: Path, relative_tolerance: float) -> dict[str, Any]:
    scenario_id = row["scenario_id"]
    frequency = float(row["frequency_Hz"])
    resistance = float(row["resistance_ohm"])
    reactance = float(row["reactance_ohm"])
    case = Case(
        path=HERE / f"{scenario_id}.yaml",
        data={
            "schema": "case_yaml.v1",
            "case_id": f"digitized_{scenario_id}",
            "source": {
                "type": "sine_voltage",
                "name": "Vsrc",
                "p": "src",
                "n": 0,
                "amplitude_V": 1.0,
                "frequency_Hz": frequency,
            },
            "circuit": {
                "builder": "from_yaml",
                "output_node": "port",
                "components": [{"raw": "Rfixture src port 1e-9"}],
            },
            "load": {
                "name": "impedance_point",
                "ports": {"p": "port", "n": 0},
                "reference_plane": row["reference_plane"],
                "characterization": {
                    "origin": "digitized_published_measurement",
                    "doi": row["source_doi"],
                    "figures": [2, 3],
                },
                "resistance_ohm": resistance,
                "reactance_ohm": reactance,
                "model_frequency_Hz": frequency,
            },
            "solver": {"name": "ngspice_cli", "ac": {"frequency_Hz": frequency}},
            "target": {"objective": "impedance_match"},
        },
    )
    record = simulate_case(case, run_root=run_root, run_id=f"replay_{scenario_id}")
    if record.status != "ok" or record.frequency_response_file is None:
        return {"scenario_id": scenario_id, "passed": False, "error": record.warnings}
    response = input_impedance(read_ac(record.run_dir / record.frequency_response_file))
    point = at_frequency(response, frequency)
    actual = complex(float(point["resistance_ohm"]), float(point["reactance_ohm"]))
    expected = complex(resistance, reactance)
    relative_error = abs(actual - expected) / abs(expected)
    return {
        "scenario_id": scenario_id,
        "expected_ohm": {"resistance": expected.real, "reactance": expected.imag},
        "spice_ohm": {"resistance": actual.real, "reactance": actual.imag},
        "relative_complex_error": relative_error,
        "passed": relative_error <= relative_tolerance,
    }


def run(run_root: Path) -> dict[str, Any]:
    manifest = yaml.safe_load((HERE / "axis_calibration.yaml").read_text(encoding="utf-8"))
    rows = _read_csv(HERE / "qualified_impedance.csv")
    excluded = _read_csv(HERE / "excluded_points.csv")
    checks: list[dict[str, Any]] = []

    source = manifest["source"]
    scope = manifest["operating_scope"]
    digitization = manifest["digitization"]
    replay_thresholds = manifest["qualification"]["replay_thresholds"]
    pixel_value_tolerance = float(replay_thresholds["pixel_value_round_trip_tolerance_ohm"])
    spice_relative_tolerance = float(replay_thresholds["spice_relative_complex_error_max"])
    resistance_axis = manifest["figures"]["resistance"]
    reactance_axis = manifest["figures"]["reactance"]
    expected_keys = {
        (int(pressure), int(power)) for pressure in scope["pressure_mTorr"] for power in scope["paired_rf_power_W"]
    }
    actual_keys = {(int(row["argon_pressure_mTorr"]), float(row["reported_rf_power_W"])) for row in rows}
    scenario_ids = [row["scenario_id"] for row in rows]
    checks.append(
        _check(
            "complete_same_condition_pairing",
            len(rows) == 15
            and len(scenario_ids) == len(set(scenario_ids))
            and len(actual_keys) == len(rows)
            and actual_keys == expected_keys,
        )
    )

    hashes = [
        resistance_axis["working_image"]["sha256"],
        reactance_axis["working_image"]["sha256"],
    ]
    checks.append(
        _check(
            "figure_identity_and_raster_hashes_recorded",
            resistance_axis["paper_figure"] == 2
            and reactance_axis["paper_figure"] == 3
            and resistance_axis["visible_axis_label"] == "R_m"
            and reactance_axis["visible_axis_label"] == "X_m"
            and all(len(value) == 64 and all(char in "0123456789ABCDEF" for char in value) for value in hashes),
        )
    )

    power_tolerance = float(digitization["power_pixel_tolerance_W"])
    r_uncertainty = float(digitization["uncertainty_half_width"]["resistance_ohm"])
    x_uncertainty = float(digitization["uncertainty_half_width"]["reactance_ohm"])
    pixel_ok = True
    provenance_ok = True
    uncertainty_ok = True
    passive_ok = True
    max_power_residual = 0.0
    max_r_residual = 0.0
    max_x_residual = 0.0
    by_pressure: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        power = float(row["reported_rf_power_W"])
        resistance = float(row["resistance_ohm"])
        reactance = float(row["reactance_ohm"])
        power_from_r = _axis_value(float(row["resistance_pixel_x"]), resistance_axis["x_axis"]["anchors"])
        power_from_x = _axis_value(float(row["reactance_pixel_x"]), reactance_axis["x_axis"]["anchors"])
        resistance_from_pixel = _axis_value(float(row["resistance_pixel_y"]), resistance_axis["y_axis"]["anchors"])
        reactance_from_pixel = _axis_value(float(row["reactance_pixel_y"]), reactance_axis["y_axis"]["anchors"])
        power_residual = max(abs(power_from_r - power), abs(power_from_x - power))
        resistance_residual = abs(resistance_from_pixel - resistance)
        reactance_residual = abs(reactance_from_pixel - reactance)
        max_power_residual = max(max_power_residual, power_residual)
        max_r_residual = max(max_r_residual, resistance_residual)
        max_x_residual = max(max_x_residual, reactance_residual)
        pixel_ok &= power_residual <= power_tolerance
        # The published values are stored to 0.1 ohm after averaging two
        # independently selected marker centres.  Reversing that rounded mean
        # through either reader's pixel scale can differ by slightly more than
        # half a display unit; this check detects transcription/axis mistakes,
        # not the much wider 6/7-ohm evidence uncertainty.
        pixel_ok &= resistance_residual <= pixel_value_tolerance
        pixel_ok &= reactance_residual <= pixel_value_tolerance
        provenance_ok &= row["source_doi"] == source["doi"]
        provenance_ok &= int(row["source_resistance_figure"]) == 2
        provenance_ok &= int(row["source_reactance_figure"]) == 3
        provenance_ok &= row["reference_plane"] == scope["reference_plane"]
        provenance_ok &= math.isclose(float(row["frequency_Hz"]), float(scope["frequency_Hz"]))
        provenance_ok &= row["qualified"].lower() == "true"
        uncertainty_ok &= int(row["reader_count"]) == 2
        uncertainty_ok &= float(row["digitization_uncertainty_R_ohm"]) >= r_uncertainty
        uncertainty_ok &= float(row["digitization_uncertainty_X_ohm"]) >= x_uncertainty
        uncertainty_ok &= float(row["reader_difference_R_ohm"]) <= 2.0 * r_uncertainty
        uncertainty_ok &= float(row["reader_difference_X_ohm"]) <= 2.0 * x_uncertainty
        passive_ok &= resistance > 0.0 and reactance > 0.0
        by_pressure[int(row["argon_pressure_mTorr"])].append((int(power), resistance, reactance))
    checks.append(
        _check(
            "pixel_transforms_reproduce_values",
            pixel_ok,
            maximum_power_residual_W=max_power_residual,
            maximum_resistance_residual_ohm=max_r_residual,
            maximum_reactance_residual_ohm=max_x_residual,
            tolerance_ohm=pixel_value_tolerance,
        )
    )
    checks.append(_check("provenance_and_reference_plane", provenance_ok))
    checks.append(_check("two_reader_uncertainty_preserved", uncertainty_ok))
    observed_max_reader_r = max(float(row["reader_difference_R_ohm"]) for row in rows)
    observed_max_reader_x = max(float(row["reader_difference_X_ohm"]) for row in rows)
    checks.append(
        _check(
            "reader_difference_summary_matches_manifest",
            math.isclose(
                observed_max_reader_r,
                float(digitization["independent_reader_maximum_difference"]["resistance_ohm"]),
            )
            and math.isclose(
                observed_max_reader_x,
                float(digitization["independent_reader_maximum_difference"]["reactance_ohm"]),
            ),
            observed_maximum_R_ohm=observed_max_reader_r,
            observed_maximum_X_ohm=observed_max_reader_x,
        )
    )
    checks.append(_check("passive_inductive_points", passive_ok))

    excluded_pressures = {int(row["argon_pressure_mTorr"]) for row in excluded}
    excluded_ids = [row["scenario_id"] for row in excluded]
    excluded_ok = (
        len(excluded) == 3
        and len(excluded_ids) == len(set(excluded_ids))
        and not set(excluded_ids).intersection(scenario_ids)
        and excluded_pressures == set(scope["pressure_mTorr"])
        and all(
            row["qualified"].lower() == "false"
            and float(row["reported_rf_power_W"]) == 500.0
            and float(row["frequency_Hz"]) == float(scope["frequency_Hz"])
            and row["available_quantity"] == "R_m"
            and row["missing_quantity"] == "X_m"
            for row in excluded
        )
    )
    checks.append(_check("unpaired_500W_points_remain_excluded", excluded_ok))

    monotone_ok = True
    for values in by_pressure.values():
        ordered = sorted(values)
        monotone_ok &= all(left[1] >= right[1] and left[2] >= right[2] for left, right in pairwise(ordered))
    checks.append(_check("reported_R_and_X_decrease_with_power", monotone_ok))
    five_kw_r = [value[1] for values in by_pressure.values() for value in values if value[0] == 5000]
    observations = [
        {
            "name": "reported_resistance_converges_near_50_ohm",
            "consistent_with_paper_text": len(five_kw_r) == 3 and all(35.0 <= value <= 65.0 for value in five_kw_r),
            "resistance_ohm": five_kw_r,
            "gating": False,
        }
    ]

    replay = [_replay_case(row, run_root / "replay", spice_relative_tolerance) for row in rows]
    maximum_replay_error = max(float(item.get("relative_complex_error", math.inf)) for item in replay)
    checks.append(
        _check(
            "pcd_ngspice_one_port_replay",
            all(item["passed"] for item in replay),
            maximum_relative_complex_error=maximum_replay_error,
            relative_tolerance=spice_relative_tolerance,
        )
    )
    resistances = [float(row["resistance_ohm"]) for row in rows]
    reactances = [float(row["reactance_ohm"]) for row in rows]
    raw_gamma = [
        abs((complex(r, x) - 50.0) / (complex(r, x) + 50.0)) for r, x in zip(resistances, reactances, strict=True)
    ]
    payload: dict[str, Any] = {
        "benchmark_id": "p1_colpo1999_plasma_on_digitized_window",
        "source": source["doi"],
        "evidence_class": "two_reader_central_point_plot_transcription",
        "point_count": len(rows),
        "frequency_Hz": float(scope["frequency_Hz"]),
        "R_range_ohm": [min(resistances), max(resistances)],
        "X_range_ohm": [min(reactances), max(reactances)],
        "digitization_uncertainty_half_width_ohm": {
            "resistance": r_uncertainty,
            "reactance": x_uncertainty,
        },
        "digitization_uncertainty_scope": digitization["uncertainty_scope"],
        "uncertainty_propagated_into_design_evaluation": False,
        "descriptive_raw_50ohm_gamma_range": [min(raw_gamma), max(raw_gamma)],
        "raw_gamma_is_acceptance_metric": False,
        "maximum_spice_replay_relative_error": maximum_replay_error,
        "checks": checks,
        "observations": observations,
        "replay": replay,
        "limitations": manifest["qualification"]["not_qualified_use"],
    }
    payload["passed"] = all(check["passed"] for check in checks)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/literature/colpo1999_digitized"))
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    payload = run(args.run_root.resolve())
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
