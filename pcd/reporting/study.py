"""Study/identification views read published numeric artifacts, never re-solve."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from pcd.artifacts import (
    create_run_directory,
    evidence_digest,
    file_sha256,
    package_source_sha256,
    resolve_artifact_path,
    write_json,
)
from pcd.case import Case
from pcd.inputs import load_plane_label, read_input_mapping
from pcd.output import output_settings
from pcd.records import artifact_path, load_waveform, read_sim_record
from pcd.results import selected_evaluation
from pcd.results.record import SavedResult, read_saved_result
from pcd.results.table import read_evaluation_csv
from pcd.simulation import SOURCE_VOLTAGE_COLUMN, VOLTAGE_COLUMN

from .figures import _headless_pyplot
from .recipes import save_figure
from .run import analyze_run


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:  # Windows paths on separate volumes cannot be made relative.
        return path.resolve().as_posix()


def _artifact_hashes(
    payload: dict[str, Any], root: Path, destination: Path, names: tuple[str, ...]
) -> dict[str, str | None]:
    artifacts = payload["artifacts"]
    paths = ((name, resolve_artifact_path(root, artifacts[name])) for name in names if name in artifacts)
    return {_relative(path, destination): evidence_digest(payload, name, path) for name, path in paths}


def _analysis_generation(record: SavedResult) -> Path:
    if record.generation is None:
        raise ValueError(f"saved study record does not declare a generation: {record.path}")
    required = ("best_candidate", "history", "evaluation_table", "case")
    if missing := [name for name in required if record.artifact_path(name) is None]:
        raise ValueError(f"saved study record lacks aggregate analysis artifacts: {missing}")
    return record.generation


def analyze_saved(
    record_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    frequency_hz: float | None = None,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate a simulation, study, or identification report from saved evidence."""

    record = read_saved_result(Path(record_path).resolve())
    if record is None:
        from . import analyze_run as analyze_simulation

        return analyze_simulation(record_path, out_dir=out_dir, frequency_hz=frequency_hz, output=output)
    if frequency_hz is not None:
        raise ValueError("--frequency-Hz selects an AC operating point for one saved simulation")
    root, payload = record.root, record.payload
    generation = _analysis_generation(record) if payload["schema"] == "study_result.v1" else None
    if output is None:
        case = record.artifact_path("case") or root / "input" / "case.yaml"
        output = read_input_mapping(case).get("output")
    settings = output_settings(output)
    requested = (Path(out_dir) if out_dir is not None else root / "analysis").resolve()
    if any(
        requested.is_relative_to((root / name).resolve())
        for name in ("generations", "raw", "archives", "artifacts", "input", "fit", "holdout", "sensitivity")
    ):
        raise ValueError("analysis output must be outside the saved result's evidence directories")
    destination = create_run_directory(requested.parent, requested.name)
    source = {"record": _relative(record.path, destination), "sha256": record.sha256}
    if generation is not None:
        source["generation"] = _relative(generation, destination)
        source["aggregate_integrity"] = "verified" if "artifact_sha256" in payload else "unverified_legacy"
        return _study_outputs(payload, root, destination, settings, source=source)
    return _identification_outputs(payload, root, destination, settings, source=source)


def _selected(payload: dict[str, Any], root: Path, partition: str = "verification") -> list[dict[str, Any]]:
    path = resolve_artifact_path(root, payload["artifacts"]["best_candidate"])
    best = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scenario in best["scenarios"]:
        evaluation = selected_evaluation(scenario)
        manifest = evaluation["raw"]["artifacts"].get("manifest")
        rows.append(
            {
                "candidate_id": best["candidate"]["candidate_id"],
                "scenario_id": scenario["scenario"]["scenario_id"],
                "phase": partition,
                "control": json.dumps(evaluation["request"]["control"]["values"], sort_keys=True),
                "status": evaluation["raw"]["status"],
                "manifest": str(resolve_artifact_path(root, manifest)) if manifest else "",
            }
        )
    return rows


def _search(payload: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    frame = read_evaluation_csv(resolve_artifact_path(root, payload["artifacts"]["evaluation_table"]))
    rows = []
    for row in frame.to_dict(orient="records"):
        manifest = row.get("artifact.manifest")
        control = row.get("control")
        if not isinstance(control, str):
            # Older evaluation tables only stored flattened controls.
            control = json.dumps(
                {
                    str(key).removeprefix("control."): value
                    for key, value in row.items()
                    if str(key).startswith("control.") and pd.notna(value)
                },
                sort_keys=True,
                allow_nan=False,
            )
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "scenario_id": row["scenario_id"],
                "phase": "search",
                "control": control,
                "status": row["status"],
                "manifest": str(resolve_artifact_path(root, manifest)) if isinstance(manifest, str) else "",
            }
        )
    return rows


def _export_traces(rows: list[dict[str, Any]], destination: Path) -> dict[str, Any]:
    def context_column(name: str) -> str:
        return name if name.startswith("parameter.") else f"context.{name}"

    index = []
    inputs: dict[str, str | None] = {}
    missing = []
    for row in rows:
        entry = dict(row)
        manifest = Path(row["manifest"]) if row["manifest"] else None
        if manifest is not None and manifest.is_file():
            record = read_sim_record(manifest)
            entry["manifest"] = _relative(manifest, destination)
            measurement = record.get("measurement") or {}
            load_ports = measurement.get("load_ports") or {}
            entry.update(
                drive_basis=measurement.get("drive_basis"),
                output_voltage_vector=measurement.get("output_voltage_vector"),
                load_reference_plane=load_plane_label(measurement.get("load_reference_plane")),
                impedance_reference_plane=measurement.get("impedance_reference_plane"),
                load_terminal_p=load_ports.get("p"),
                load_terminal_n=load_ports.get("n"),
            )
            inputs[entry["manifest"]] = file_sha256(manifest)
            for name in ("waveform", "frequency_response", "signals"):
                path = artifact_path(record, name)
                if path is not None and path.is_file():
                    entry[name] = _relative(path, destination)
                    inputs[entry[name]] = evidence_digest(record, name, path)
                elif path is not None or (name == "waveform" and row["status"] == "ok"):
                    missing.append({"manifest": entry["manifest"], "artifact": name})
            entry.update({f"parameter.{name}": value for name, value in record.get("params", {}).items()})
        elif row["status"] == "ok":
            missing.append({"scenario_id": row["scenario_id"], "phase": row["phase"], "artifact": "manifest"})
        index.append(entry)
    pd.DataFrame(index).rename(columns=context_column).to_csv(destination / "observations.csv", index=False)
    exports = {"index": "observations.csv"}
    samples: dict[str, int] = {}
    # Header discovery reads no samples; the second pass streams one run at a time.
    for source, name in (("waveform", "transient"), ("frequency_response", "ac")):
        context = list(
            dict.fromkeys(key for row in index for key in row if key not in {"waveform", "frequency_response"})
        )
        columns = [context_column(key) for key in context]
        for row in index:
            if source in row:
                columns.extend(
                    f"signal.{column}"
                    for column in pd.read_csv(destination / row[source], nrows=0).columns
                    if f"signal.{column}" not in columns
                )
        wrote = False
        count = 0
        file = destination / f"{name}_observations.csv"
        for row in index:
            if source not in row:
                continue
            for data in pd.read_csv(destination / row[source], chunksize=10000):
                if data.empty:
                    continue
                data = data.add_prefix("signal.")
                for key in context:
                    if key in row:
                        data[context_column(key)] = row[key]
                data.reindex(columns=columns).to_csv(file, mode="a" if wrote else "w", header=not wrote, index=False)
                wrote = True
                count += len(data)
        if wrote:
            exports[name] = file.name
            samples[name] = count
    return {
        "files": exports,
        "evaluations": len(index),
        "samples": samples,
        "source_hashes": inputs,
        "missing_source_artifacts": missing,
        "column_namespaces": {"context": "context.", "parameters": "parameter.", "signals": "signal."},
    }


def _point_outputs(rows: list[dict[str, Any]], destination: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for index, row in enumerate(rows):
        manifest = Path(row["manifest"]) if row["manifest"] else None
        context = {**row, "manifest": _relative(manifest, destination) if manifest else ""}
        if row["status"] != "ok":
            points.append({**context, "output_status": "not_available", "reason": "selected solve failed"})
            continue
        if manifest is None or not manifest.is_file():
            points.append({**context, "output_status": "not_available", "reason": "saved simulation manifest missing"})
            continue
        try:
            analysis = analyze_run(manifest, out_dir=destination / f"point_{index + 1:03d}", output=settings)
        except (FileNotFoundError, ValueError) as exc:
            points.append({**context, "output_status": "not_available", "reason": str(exc)})
            continue
        points.append(
            {
                **context,
                "output_status": analysis["status"],
                "summary": _relative(Path(analysis["output_dir"]) / "summary.json", destination),
            }
        )
    return points


def _comparison(points: list[dict[str, Any]], destination: Path, formats: list[str]) -> dict[str, Any]:
    """Overlay only voltages with the same observation terminals and drive basis."""
    groups: dict[tuple[str, str, str, str], list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {}
    unavailable = []
    for point in points:
        context = {name: point[name] for name in ("phase", "scenario_id", "control")}
        if "summary" not in point:
            unavailable.append({**context, "reason": point["reason"]})
            continue
        report_path = destination / point["summary"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifacts = report["artifacts"]
        catalog = json.loads((report_path.parent / artifacts["signals"]).read_text(encoding="utf-8"))
        channel = catalog["signals"][VOLTAGE_COLUMN]
        terminals = channel.get("terminals")
        scope = "observed output"
        if terminals == catalog["load_ports"]:
            scope = catalog["load_reference_plane"]
        elif terminals == catalog["signals"][SOURCE_VOLTAGE_COLUMN].get("terminals"):
            scope = catalog["impedance_reference_plane"]
        drive = catalog["drive_basis"]
        for domain in report["analyses"]:
            columns = channel["columns"].get(domain)
            if not terminals or not columns:
                unavailable.append(
                    {**context, "domain": domain, "reason": "output voltage terminals or data unavailable"}
                )
                continue
            axis = catalog["axes"][domain]["column"]
            if domain == "ac":
                data = pd.read_csv(report_path.parent / artifacts["frequency_metrics"], usecols=[axis, *columns])
                values = np.hypot(data[columns[0]].to_numpy(float), data[columns[1]].to_numpy(float))
            else:
                data = load_waveform(destination / point["manifest"])
                values = data[columns[0]].to_numpy(float)
            observed = np.isfinite(values)
            if not observed.any():
                unavailable.append({**context, "domain": domain, "reason": "no finite output voltage samples"})
                continue
            vector = f"V({terminals['p']}, {terminals['n']})"
            key = (domain, vector, scope, drive)
            coverage = {**context, "samples": len(values), "finite_samples": int(observed.sum())}
            groups.setdefault(key, []).append(
                (data[axis].to_numpy(float), np.where(observed, values, np.nan), coverage)
            )
    if not groups:
        return {"figures": [], "panels": [], "unavailable": unavailable}
    plt = _headless_pyplot()
    figure, axes = plt.subplots(len(groups), 1, figsize=(9, 3.4 * len(groups)), squeeze=False)
    panels = []
    for index, ((domain, vector, scope, drive), traces) in enumerate(groups.items()):
        axis = axes[index, 0]
        for x, y, coverage in traces:
            label = f"{coverage['phase']}: {coverage['scenario_id']} {coverage['control']}"
            axis.plot(x, y, marker="o" if len(x) == 1 else None, label=label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        drive_label = "1 V peak reference" if drive == "normalized_1V_peak" else drive
        axis.set(
            xlabel="time [s]" if domain == "transient" else "frequency [Hz]",
            ylabel="voltage [V]" if domain == "transient" else "voltage magnitude [V peak]",
            title=f"{vector} · {scope} · {drive_label}",
        )
        if domain == "ac":
            axis.set_xscale("log")
            if all(len(x) == 1 and x[0] == traces[0][0][0] for x, _, _ in traces):
                frequency = float(traces[0][0][0])
                axis.set_xticks([frequency], [f"{frequency:g}"])
                axis.minorticks_off()
        panels.append(
            {
                "domain": domain,
                "voltage": vector,
                "plane": scope,
                "drive_basis": drive,
                "traces": [coverage for _, _, coverage in traces],
            }
        )
    return {
        "figures": save_figure(figure, destination, "condition_comparison", formats),
        "panels": panels,
        "unavailable": unavailable,
    }


def render_study_outputs(payload: dict[str, Any], case: Case, generation_root: Path) -> dict[str, Any]:
    destination = create_run_directory(generation_root, "output")
    root = Path(payload["run_root"])
    result = _study_outputs(payload, root, destination, output_settings(case.data["output"]))
    return {"status": result["status"], "summary": _relative(destination / "summary.json", root)}


def _study_outputs(
    payload: dict[str, Any],
    root: Path,
    destination: Path,
    settings: dict[str, Any],
    *,
    source: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected = _selected(payload, root)
    all_waveforms = settings["data"]["waveforms"] == "all"
    rows = [*_search(payload, root), *selected] if all_waveforms else selected
    export = _export_traces(rows, destination)
    inputs = (
        ("case", "best_candidate", "history", "evaluation_table")
        if all_waveforms
        else ("case", "best_candidate", "history")
    )
    export["source_hashes"].update(_artifact_hashes(payload, root, destination, inputs))
    points = _point_outputs(selected, destination, settings)
    plt = _headless_pyplot()
    history_path = resolve_artifact_path(root, payload["artifacts"]["history"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    objectives = payload["study"]["objectives"]
    conditions = payload["best"]["conditions"]
    limits = sorted(
        {name for condition in conditions for name in condition["constraint_margins"]}
        | {limit["name"] for condition in conditions for limit in condition.get("failed_constraints", ())}
    )
    figure, axes = plt.subplots(
        max(1, len(objectives) + len(limits)),
        1,
        figsize=(9, 2.6 * max(1, len(objectives) + len(limits))),
        squeeze=False,
    )
    for index, objective in enumerate(objectives):
        key = objective["metric"]
        axis = axes[index, 0]
        axis.plot(range(1, len(history) + 1), [row["aggregates"].get(key) for row in history], ".-")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axis.set(
            xlabel="candidate evaluation",
            ylabel=key,
            title=f"{objective['direction']} · {objective['aggregation']} reduction",
        )
        axis.grid(alpha=0.3)
    for index, key in enumerate(limits, start=len(objectives)):
        axis = axes[index, 0]
        names = [item["scenario_id"] for item in conditions]
        margins = np.asarray([item["constraint_margins"].get(key, np.nan) for item in conditions], dtype=float)
        present = np.isfinite(margins)
        axis.set_xticks(range(len(names)), names)
        if present.any():
            axis.bar(np.flatnonzero(present), margins[present])
            axis.axhline(0, color="black", lw=0.8)
            missing = np.flatnonzero(~present)
            if len(missing):
                axis.scatter(
                    missing,
                    np.full(len(missing), 0.92),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    color="tab:red",
                    label="Unavailable",
                )
                axis.legend(fontsize=7)
        else:
            axis.set_yticks([])
            axis.text(0.5, 0.5, "Unavailable in every condition", transform=axis.transAxes, ha="center")
        axis.set(ylabel=key, title="Signed engineering margin (positive is feasible reserve)")
        axis.tick_params(axis="x", rotation=25)
    overview = save_figure(figure, destination, "design_history_and_margins", settings["formats"])
    comparison = _comparison(points, destination, settings["formats"])
    result = {
        "schema": "study_analysis.v1",
        "case_id": payload["study"]["study_id"],
        "status": _output_status(points, export),
        "output_dir": str(destination),
        "artifacts": {"summary": "summary.json", **export["files"]},
        "settings": settings,
        "points": points,
        "data": export,
        "figures": [*overview, *comparison["figures"]],
        "voltage_comparison": comparison,
        "scope": "saved search evaluations and selected verification results",
        "engineering_status": payload["best"]["status"],
        "conditions": conditions,
        **({"source": source} if source is not None else {}),
        "source_sha256": package_source_sha256(),
        "rendering_environment": {name: version(name) for name in ("matplotlib", "schemdraw")},
    }
    write_json(destination / "summary.json", result)
    return result


def _output_status(points: list[dict[str, Any]], export: dict[str, Any]) -> str:
    return (
        "complete"
        if points
        and all(row["output_status"] == "complete" for row in points)
        and not export["missing_source_artifacts"]
        else "partial"
    )


def render_identification_outputs(payload: dict[str, Any], case: Case) -> dict[str, Any]:
    root = Path(payload["run_root"])
    destination = root / "output"
    destination.mkdir()
    result = _identification_outputs(payload, root, destination, output_settings(case.data["output"]))
    return {"status": result["status"], "summary": "output/summary.json"}


def _identification_phase(
    path: Path, result: dict[str, Any], root: Path, destination: Path, partition: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    record = read_saved_result(path, study_only=True)
    if record is None:
        raise ValueError(f"saved {partition} phase does not declare a study result: {path}")
    phase_root, payload = record.root, record.payload
    generation = _analysis_generation(record)
    if phase_root != root / partition:
        raise ValueError(f"saved {partition} study is outside the identification's phase directory")
    evidence = result[partition]
    best = payload["best"]
    if [item["scenario_id"] for item in best["conditions"]] != evidence["scenario_ids"]:
        raise ValueError(f"saved {partition} conditions do not match the identification record")
    if evidence["value"] is not None and (
        best["candidate"]["values"] != result["estimated_latent"]
        or best["aggregates"].get(evidence["metric"]) != evidence["value"]
    ):
        raise ValueError(f"saved {partition} values do not match the identification record")
    return (
        payload,
        phase_root,
        {
            "record": _relative(record.path, destination),
            "sha256": record.sha256,
            "generation": _relative(generation, destination),
            "evidence": evidence,
        },
    )


def _identification_outputs(
    payload: dict[str, Any],
    root: Path,
    destination: Path,
    settings: dict[str, Any],
    *,
    source: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    observations = []
    phases = {}
    missing = []
    input_hashes = _artifact_hashes(payload, root, destination, ("case",))
    uncertainty = payload.get("uncertainty")
    if uncertainty is not None:
        for realization in uncertainty["realizations"]:
            if realization["name"] == "nominal":
                continue
            key = realization["artifact"]
            reference = payload["artifacts"].get(key)
            path = resolve_artifact_path(root, reference) if reference else None
            if path is not None and path.is_file():
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"saved {realization['name']} result is outside the identification directory")
                input_hashes[_relative(path, destination)] = evidence_digest(payload, key, path)
                record = read_saved_result(path)
                if record is None or record.payload["schema"] != "identification_result.v1":
                    raise ValueError(f"saved {realization['name']} result is not an identification record")
                children = ["case", "fit_result", "fit_observations"]
                if record.payload["holdout"]["status"] != "not_run":
                    children.extend(("holdout_result", "holdout_observations"))
                for child_key in children:
                    child = record.artifact_path(child_key)
                    if child is not None and child.is_file():
                        if not child.resolve().is_relative_to(record.root.resolve()):
                            raise ValueError(f"saved {realization['name']} {child_key} is outside its result directory")
                        input_hashes[_relative(child, destination)] = evidence_digest(record.payload, child_key, child)
                    else:
                        missing.append({"phase": realization["name"], "artifact": child_key})
            else:
                missing.append({"phase": realization["name"], "artifact": key})
    all_waveforms = settings["data"]["waveforms"] == "all"
    for partition in ("fit", "holdout"):
        evidence = payload[partition]
        expected = evidence.get("status") != "not_run"
        phases[partition] = {"evidence": evidence}
        result_key = f"{partition}_result"
        result_ref = payload["artifacts"].get(result_key)
        path = resolve_artifact_path(root, result_ref).resolve() if result_ref else None
        if path is not None and path.is_file():
            input_hashes[_relative(path, destination)] = evidence_digest(payload, result_key, path)
            phase, phase_root, phases[partition] = _identification_phase(path, payload, root, destination, partition)
            rows.extend(_selected(phase, phase_root, partition))
            sources = ("best_candidate", "evaluation_table") if all_waveforms else ("best_candidate",)
            input_hashes.update(_artifact_hashes(phase, phase_root, destination, sources))
            if all_waveforms:
                rows.extend({**row, "phase": partition + "_search"} for row in _search(phase, phase_root))
        elif expected or result_ref:
            missing.append({"phase": partition, "artifact": result_key})
        observations_key = f"{partition}_observations"
        observations_ref = payload["artifacts"].get(observations_key)
        path = resolve_artifact_path(root, observations_ref) if observations_ref else None
        if path is not None and path.is_file():
            table = pd.read_csv(path, converters={"scenario_id": str, "acquisition_group": str})
            if table["scenario_id"].tolist() != evidence["scenario_ids"]:
                raise ValueError(f"saved {partition} observation rows do not match the identification record")
            input_hashes[_relative(path, destination)] = evidence_digest(payload, observations_key, path)
            table["partition"] = partition
            observations.append(table)
        elif expected or observations_ref:
            missing.append({"phase": partition, "artifact": observations_key})
    selected = [row for row in rows if not row["phase"].endswith("_search")]
    export = _export_traces(rows, destination)
    export["missing_source_artifacts"].extend(missing)
    export["source_hashes"].update(input_hashes)
    points = _point_outputs(selected, destination, settings)
    combined = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    combined.to_csv(destination / "terminal_observations.csv", index=False)
    metrics = [str(key).removeprefix("target.") for key in combined if str(key).startswith("target.")]
    comparison = _comparison(points, destination, settings["formats"])
    figures = list(comparison["figures"])
    plt = _headless_pyplot()
    if metrics:
        figure, axes = plt.subplots(len(metrics), 2, figsize=(10, 3.5 * len(metrics)), squeeze=False)
        for index, metric in enumerate(metrics):
            for partition, marker in (("fit", "o"), ("holdout", "^")):
                frame = combined.loc[combined["partition"] == partition]
                target, predicted = frame[f"target.{metric}"], frame[f"predicted.{metric}"]
                axes[index, 0].scatter(target, predicted, label=partition, marker=marker)
                x = frame.get("input.frequency_Hz", frame.index)
                axes[index, 1].scatter(x, frame[f"residual.{metric}"], label=partition, marker=marker)
            values = pd.concat([combined[f"target.{metric}"], combined[f"predicted.{metric}"]]).dropna()
            if not values.empty:
                low, high = float(values.min()), float(values.max())
                padding = 0.05 * max(high - low, abs(low), abs(high), 1e-12)
                limits = [low - padding, high + padding]
                axes[index, 0].plot(limits, limits, "k--", lw=0.8)
                axes[index, 0].set(xlim=limits, ylim=limits, aspect="equal")
                axes[index, 0].ticklabel_format(useOffset=False)
            axes[index, 0].set(xlabel=f"observed {metric}", ylabel=f"predicted {metric}")
            axes[index, 1].set(
                xlabel="frequency [Hz]" if "input.frequency_Hz" in combined else "condition index",
                ylabel=f"predicted - observed {metric}",
            )
            axes[index, 1].axhline(0, color="black", lw=0.8)
            for axis in axes[index]:
                axis.grid(alpha=0.3)
                axis.legend()
        figures.extend(save_figure(figure, destination, "observed_predicted_and_residuals", settings["formats"]))
    sensitivity = payload["identifiability"]
    if sensitivity.get("normalized_matrix"):
        matrix = np.asarray(sensitivity["normalized_matrix"], dtype=float)
        figure, axis = plt.subplots(figsize=(7, max(3, 0.3 * matrix.shape[0])))
        limit = max(float(np.abs(matrix).max()), 1e-30)
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_xticks(range(matrix.shape[1]), sensitivity["parameters"], rotation=20)
        axis.set_yticks(range(matrix.shape[0]), sensitivity["observations"])
        axis.set_title(
            f"Local normalized sensitivity; rank {sensitivity['rank']}/{sensitivity['required_rank']}\nNot a confidence interval or global identifiability proof"
        )
        figure.colorbar(image, ax=axis)
        figures.extend(save_figure(figure, destination, "local_sensitivity", settings["formats"]))
    if source is not None:
        integrity = "unverified_legacy" if "artifact_sha256" not in payload else "incomplete" if missing else "verified"
        source = {**source, "aggregate_integrity": integrity}
    result = {
        "schema": "identification_analysis.v1",
        "case_id": payload["case_id"],
        "status": _output_status(points, export),
        "output_dir": str(destination),
        "artifacts": {
            "summary": "summary.json",
            "terminal_observations": "terminal_observations.csv",
            **export["files"],
        },
        "settings": settings,
        "points": points,
        "data": export,
        "terminal_observations": "terminal_observations.csv",
        "figures": figures,
        "voltage_comparison": comparison,
        "scope": (
            "saved nominal fit, holdout and local sensitivity with separate declared realization summaries"
            if uncertainty is not None
            else "saved fit, holdout and local sensitivity evidence"
        ),
        "identification_status": payload["status"],
        "phases": phases,
        **({"uncertainty": uncertainty} if uncertainty is not None else {}),
        **({"source": source} if source is not None else {}),
        "source_sha256": package_source_sha256(),
        "rendering_environment": {name: version(name) for name in ("matplotlib", "schemdraw")},
    }
    write_json(destination / "summary.json", result)
    return result
