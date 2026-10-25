"""Study/identification views read published numeric artifacts, never re-solve."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pcd.artifacts import file_sha256, package_source_sha256, write_json
from pcd.case import Case
from pcd.output import output_settings
from pcd.records import artifact_path, load_frequency_response, load_waveform, read_sim_record
from pcd.results import selected_evaluation

from .figures import _headless_pyplot
from .recipes import save_figure
from .run import analyze_run


def _path(root: Path, declared: str) -> Path:
    path = Path(declared)
    return path if path.is_absolute() else root / path


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _selected(payload: dict[str, Any], partition: str = "verification") -> list[dict[str, Any]]:
    root = Path(payload["run_root"])
    best = json.loads(_path(root, payload["artifacts"]["best_candidate"]).read_text(encoding="utf-8"))
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
                "manifest": str(_path(root, manifest)) if manifest else "",
            }
        )
    return rows


def _search(payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(payload["run_root"])
    frame = pd.read_csv(_path(root, payload["artifacts"]["evaluation_table"]))
    rows = []
    for row in frame.to_dict(orient="records"):
        manifest = row.get("artifact.manifest")
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "scenario_id": row["scenario_id"],
                "phase": "search",
                "control": json.dumps(
                    {
                        str(key).removeprefix("control."): value
                        for key, value in row.items()
                        if str(key).startswith("control.")
                    },
                    sort_keys=True,
                ),
                "status": row["status"],
                "manifest": str(_path(root, manifest)) if isinstance(manifest, str) else "",
            }
        )
    return rows


def _export_traces(rows: list[dict[str, Any]], destination: Path) -> dict[str, Any]:
    def context_column(name: str) -> str:
        return name if name.startswith("parameter.") else f"context.{name}"

    index = []
    inputs: dict[str, str | None] = {}
    for row in rows:
        entry = dict(row)
        manifest = Path(row["manifest"]) if row["manifest"] else None
        if manifest is not None and manifest.is_file():
            record = read_sim_record(manifest)
            entry["manifest"] = _relative(manifest, destination)
            entry["drive_basis"] = record.get("measurement", {}).get("drive_basis", "declared_or_imported")
            inputs[entry["manifest"]] = file_sha256(manifest)
            for name in ("waveform", "frequency_response", "signals"):
                path = artifact_path(record, name)
                if path is not None and path.is_file():
                    entry[name] = _relative(path, destination)
                    inputs[entry[name]] = file_sha256(path)
            entry.update({f"parameter.{name}": value for name, value in record.get("params", {}).items()})
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
        "column_namespaces": {"context": "context.", "parameters": "parameter.", "signals": "signal."},
    }


def _point_outputs(rows: list[dict[str, Any]], destination: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for index, row in enumerate(rows):
        if row["status"] != "ok" or not row["manifest"]:
            points.append({**row, "output_status": "not_available", "reason": "selected solve failed"})
            continue
        analysis = analyze_run(row["manifest"], out_dir=destination / f"point_{index + 1:03d}", output=settings)
        points.append(
            {
                **row,
                "output_status": analysis["status"],
                "summary": _relative(Path(analysis["output_dir"]) / "summary.json", destination),
            }
        )
    return points


def _comparison(rows: list[dict[str, Any]], destination: Path, formats: list[str]) -> list[str]:
    traces: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {"transient": [], "ac": []}
    for row in rows:
        if row["status"] != "ok" or not row["manifest"]:
            continue
        record = read_sim_record(row["manifest"])
        label = f"{row['phase']}: {row['scenario_id']} {row['control']}"
        if record.get("measurement", {}).get("drive_basis") == "normalized_1V_peak":
            label += " [1 V peak reference]"
        waveform = load_waveform(record)
        if not waveform.empty:
            traces["transient"].append(
                (waveform["time_s"].to_numpy(float), waveform["voltage_V"].to_numpy(float), label)
            )
        if artifact_path(record, "frequency_response") is not None:
            ac = load_frequency_response(record)
            if {"voltage_V_re", "voltage_V_im"} <= set(ac):
                values = np.hypot(ac["voltage_V_re"].to_numpy(float), ac["voltage_V_im"].to_numpy(float))
                traces["ac"].append((ac["frequency_Hz"].to_numpy(float), values, label))
    domains = [domain for domain, data in traces.items() if data]
    if not domains:
        return []
    plt = _headless_pyplot()
    figure, axes = plt.subplots(len(domains), 1, figsize=(9, 3.4 * len(domains)), squeeze=False)
    for index, domain in enumerate(domains):
        axis = axes[index, 0]
        for x, y, label in traces[domain]:
            axis.plot(x, y, marker="o" if len(x) == 1 else None, label=label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
        axis.set(
            xlabel="time [s]" if domain == "transient" else "frequency [Hz]",
            ylabel="observed voltage [V]" if domain == "transient" else "observed voltage peak [V]",
            title=f"Selected condition {domain} responses",
        )
        if domain == "ac":
            axis.set_xscale("log")
    return save_figure(figure, destination, "condition_comparison", formats)


def render_study_outputs(payload: dict[str, Any], case: Case, generation_root: Path) -> dict[str, Any]:
    destination = generation_root / "output"
    destination.mkdir()
    settings = output_settings(case.data["output"])
    root = Path(payload["run_root"])
    selected = _selected(payload)
    rows = [*_search(payload), *selected] if settings["data"]["waveforms"] == "all" else selected
    export = _export_traces(rows, destination)
    points = _point_outputs(selected, destination, settings)
    plt = _headless_pyplot()
    history = json.loads(_path(root, payload["artifacts"]["history"]).read_text(encoding="utf-8"))
    objectives = payload["study"]["objectives"]
    limits = sorted({key for item in payload["best"]["conditions"] for key in item["constraint_margins"]})
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
        axis.set(
            xlabel="candidate evaluation", ylabel=key, title=f"{objective['direction']}; {objective['aggregation']}"
        )
        axis.grid(alpha=0.3)
    for index, key in enumerate(limits, start=len(objectives)):
        conditions = payload["best"]["conditions"]
        axis = axes[index, 0]
        axis.bar(
            [item["scenario_id"] for item in conditions],
            [item["constraint_margins"].get(key, np.nan) for item in conditions],
        )
        axis.axhline(0, color="black", lw=0.8)
        axis.set(ylabel=key, title="Signed engineering margin (positive is feasible reserve)")
        axis.tick_params(axis="x", rotation=25)
    overview = save_figure(figure, destination, "design_history_and_margins", settings["formats"])
    comparisons = _comparison(selected, destination, settings["formats"])
    result = {
        "status": "complete" if points and all(row["output_status"] == "complete" for row in points) else "partial",
        "settings": settings,
        "points": points,
        "data": export,
        "figures": [*overview, *comparisons],
        "scope": "evaluated candidates and freshly replayed selected conditions",
        "source_sha256": package_source_sha256(),
        "rendering_environment": {name: version(name) for name in ("matplotlib", "schemdraw")},
    }
    write_json(destination / "summary.json", result)
    return {"status": result["status"], "summary": _relative(destination / "summary.json", root)}


def render_identification_outputs(payload: dict[str, Any], case: Case) -> dict[str, Any]:
    root = Path(payload["run_root"])
    destination = root / "output"
    destination.mkdir()
    settings = output_settings(case.data["output"])
    rows: list[dict[str, Any]] = []
    observations = []
    for partition in ("fit", "holdout"):
        artifact = payload["artifacts"].get(f"{partition}_result")
        if artifact:
            phase = json.loads((root / artifact).read_text(encoding="utf-8"))
            rows.extend(_selected(phase, partition))
            if settings["data"]["waveforms"] == "all":
                rows.extend({**row, "phase": partition + "_search"} for row in _search(phase))
        artifact = payload["artifacts"].get(f"{partition}_observations")
        if artifact:
            table = pd.read_csv(root / artifact)
            table["partition"] = partition
            observations.append(table)
    selected = [row for row in rows if not row["phase"].endswith("_search")]
    export = _export_traces(rows, destination)
    points = _point_outputs(selected, destination, settings)
    combined = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    combined.to_csv(destination / "terminal_observations.csv", index=False)
    metrics = [str(key).removeprefix("target.") for key in combined if str(key).startswith("target.")]
    figures = _comparison(selected, destination, settings["formats"])
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
    result = {
        "status": "complete" if points and all(row["output_status"] == "complete" for row in points) else "partial",
        "settings": settings,
        "points": points,
        "data": export,
        "terminal_observations": "terminal_observations.csv",
        "figures": figures,
        "source_sha256": package_source_sha256(),
        "rendering_environment": {name: version(name) for name in ("matplotlib", "schemdraw")},
    }
    write_json(destination / "summary.json", result)
    return {"status": result["status"], "summary": "output/summary.json"}
