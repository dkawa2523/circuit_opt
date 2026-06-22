"""Generate explanatory figures for the CCP ngspice benchmark report.

The script intentionally reads only curated result CSVs under
``ccp_benchmark_pack/results`` and writes compact PNG figures that can be
embedded from ``report.md``. Raw benchmark runs remain outside versioned docs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PACK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACK_DIR / "results"
STANDARD_DIR = RESULTS_DIR / "run_20260622_084121_parser_fixed"
EXTENDED_DIR = RESULTS_DIR / "run_20260622_091247_extended"
FIGURE_DIR = RESULTS_DIR / "figures"


COLORS = {
    "blue": "#3b6ea8",
    "teal": "#2a9d8f",
    "orange": "#e76f51",
    "yellow": "#e9c46a",
    "gray": "#6c757d",
    "light_gray": "#eef1f4",
    "dark": "#1f2933",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dfe4ea", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_problem_map() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.05, 0.60, 0.22, 0.25, "Problem setup", "L2: tune circuit values\nL3: choose topology/load\nFixed target waveform"),
        (0.39, 0.60, 0.22, 0.25, "Simulation", "ngspice transient run\nBatch CLI execution\nNo GUI required"),
        (0.73, 0.60, 0.22, 0.25, "Objective metrics", "loss\nconstraint penalty\nharmonic error\nvoltage/current risk"),
        (0.22, 0.12, 0.25, 0.29, "Primary judgement", "feasible median\npenalty rate\np90/max tail risk\ncategory stability"),
        (0.56, 0.12, 0.25, 0.29, "Engineering reading", "best loss is secondary\ninfeasible low loss is risky\ntopology/load = risk profile"),
    ]

    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#c7ced6",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + 0.018, y + h - 0.055, title, fontsize=13, weight="bold", color=COLORS["dark"])
        ax.text(x + 0.018, y + h - 0.105, body, fontsize=10.2, va="top", color="#344054", linespacing=1.3)

    arrows = [
        ((0.27, 0.73), (0.39, 0.73)),
        ((0.61, 0.73), (0.73, 0.73)),
        ((0.84, 0.60), (0.69, 0.41)),
        ((0.50, 0.60), (0.35, 0.41)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["gray"]),
        )

    ax.text(
        0.05,
        0.94,
        "CCP ngspice benchmark: what is being evaluated",
        fontsize=16,
        weight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.05,
        0.89,
        "The benchmark asks whether the optimizer can find physically feasible circuit candidates that reproduce the target waveform.",
        fontsize=11,
        color="#475467",
    )

    _save(fig, "benchmark_problem_map.png")


def plot_feasibility_summary() -> None:
    standard = pd.read_csv(STANDARD_DIR / "feasibility_summary.csv")
    extended = pd.read_csv(EXTENDED_DIR / "feasibility_summary.csv")

    rows = [
        (
            "L2 standard\n30 trials",
            standard[(standard["case"] == "level2_timevarying_plasma") & (standard["bucket"] == "all")].iloc[0],
        ),
        (
            "L3 standard\n30 trials",
            standard[(standard["case"] == "level3_topology_load_choice") & (standard["bucket"] == "all")].iloc[0],
        ),
        (
            "L3 extended\n100 trials",
            extended[(extended["case"] == "level3_topology_load_choice") & (extended["bucket"] == "all")].iloc[0],
        ),
    ]

    labels = [label for label, _ in rows]
    total = [int(row["count"]) for _, row in rows]
    penalty_rate = [float(row["penalty_rate"]) for _, row in rows]
    infeasible = [round(t * r) for t, r in zip(total, penalty_rate)]
    feasible = [t - i for t, i in zip(total, infeasible)]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(labels))
    ax.bar(x, feasible, color=COLORS["teal"], label="Feasible")
    ax.bar(x, infeasible, bottom=feasible, color=COLORS["orange"], label="Infeasible")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("trial count")
    ax.set_title("Feasibility is the first benchmark gate")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)

    for i, (f_count, inf_count, rate) in enumerate(zip(feasible, infeasible, penalty_rate)):
        ax.text(i, f_count + inf_count + 2, f"penalty {rate:.0%}", ha="center", fontsize=10, color=COLORS["dark"])

    _save(fig, "benchmark_feasibility.png")


def plot_topology_load_risk() -> None:
    stats = pd.read_csv(EXTENDED_DIR / "category_stats.csv")
    filtered = stats[stats["group_type"].isin(["topology", "load"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, group_type, title in [
        (axes[0], "topology", "Topology risk profile"),
        (axes[1], "load", "Load model risk profile"),
    ]:
        rows = filtered[filtered["group_type"] == group_type].sort_values("median_loss")
        x = range(len(rows))
        ax.bar(x, rows["median_loss"], color=COLORS["blue"], label="median loss")
        ax.scatter(x, rows["p90_loss"], color=COLORS["orange"], s=70, zorder=3, label="p90 loss")
        ax.set_xticks(list(x), rows["group"], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("loss (log scale)")
        ax.set_title(title)
        _style_axes(ax)

        for idx, (_, row) in enumerate(rows.iterrows()):
            ax.text(
                idx,
                row["p90_loss"] * 1.18,
                f"{row['penalty_rate']:.0%}",
                ha="center",
                fontsize=9,
                color=COLORS["dark"],
            )

    axes[0].legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01, "Labels above markers show penalty rate.", ha="center", fontsize=10, color="#475467")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "benchmark_topology_load_risk.png")


def plot_harmonic_ratios() -> None:
    harmonics = pd.read_csv(EXTENDED_DIR / "harmonic_amplitudes.csv")
    best = harmonics[harmonics["source"] == "ngspice_best"].iloc[0]
    names = ["A1", "A2", "A3"]
    ratios = [best[f"{name}_target_ratio"] for name in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, ratios, color=[COLORS["teal"], COLORS["blue"], COLORS["blue"]])
    ax.axhline(1.0, color=COLORS["orange"], linewidth=1.4, linestyle="--", label="target ratio")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("amplitude / target amplitude")
    ax.set_title("Best L3 candidate matches A1, but not A2/A3")
    ax.legend(frameon=False, loc="upper right")
    _style_axes(ax)

    for bar, value in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}", ha="center", fontsize=10)

    _save(fig, "benchmark_harmonic_ratios.png")


def plot_engineering_gate_counts() -> None:
    """Show simple pass/fail gates using counts rather than statistics."""
    comparison = pd.read_csv(STANDARD_DIR / "comparison_summary.csv")
    labels = {
        "level2_timevarying_plasma": "L2 standard",
        "level3_topology_load_choice": "L3 standard",
    }
    gates = [
        ("Feasible", lambda row, total: round(total * (1.0 - row["ngspice_penalty_rate"]))),
        ("Loss < 2", lambda row, total: int(row["loss_lt_2"])),
        ("Vpeak <= 1000 V", lambda row, total: total - int(row["v_peak_gt_1000"])),
        ("Irms <= 20 A", lambda row, total: total - int(row["i_rms_gt_20"])),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x_positions = []
    values = []
    tick_labels = []
    colors = []
    width = 0.34

    for case_idx, (_, row) in enumerate(comparison.iterrows()):
        total = 30
        base = case_idx * (len(gates) + 1)
        for gate_idx, (gate_name, fn) in enumerate(gates):
            x_positions.append(base + gate_idx)
            values.append(fn(row, total))
            tick_labels.append(gate_name)
            colors.append(COLORS["teal"] if gate_idx < 2 else COLORS["blue"])

        center = base + (len(gates) - 1) / 2
        ax.text(center, 33, labels[row["case"]], ha="center", va="bottom", fontsize=11, weight="bold")

    bars = ax.bar(x_positions, values, width=width, color=colors)
    ax.axhline(30, color=COLORS["gray"], linewidth=1.1, linestyle="--", label="30 trials")
    ax.set_ylim(0, 36)
    ax.set_ylabel("passing trial count")
    ax.set_title("Engineering gates for the 30-trial standard benchmark")
    ax.set_xticks(x_positions, tick_labels, rotation=25, ha="right")
    ax.legend(frameon=False, loc="upper right")
    _style_axes(ax)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, str(value), ha="center", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Each gate is counted independently; these bars are not a cumulative funnel.",
        ha="center",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "benchmark_engineering_gates.png")


def plot_topology_load_matrix() -> None:
    """Map topology/load combinations as a design lookup table."""
    stats = pd.read_csv(EXTENDED_DIR / "category_stats.csv")
    combos = stats[stats["group_type"] == "topology_load"].copy()
    combos[["topology", "load"]] = combos["group"].str.split(" + ", regex=False, expand=True)

    topologies = ["l_match", "pi_match", "pi_match_harmonic"]
    loads = ["plasma_state_rlc", "plasma_fixed_rlc", "electrode_stray"]
    matrix = []
    labels = []
    for topology in topologies:
        row_values = []
        row_labels = []
        for load in loads:
            item = combos[(combos["topology"] == topology) & (combos["load"] == load)].iloc[0]
            row_values.append(item["median_loss"])
            row_labels.append(f"{item['median_loss']:.2f}\n{item['penalty_rate']:.0%}")
        matrix.append(row_values)
        labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(loads)), loads, rotation=20, ha="right")
    ax.set_yticks(range(len(topologies)), topologies)
    ax.set_title("L3 topology/load map: median loss and penalty rate")

    for i in range(len(topologies)):
        for j in range(len(loads)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=10, color=COLORS["dark"])

    cbar = fig.colorbar(image, ax=ax, shrink=0.78)
    cbar.set_label("median loss, lower is better")
    fig.text(
        0.5,
        0.01,
        "Cell text is median loss on the first line and penalty rate on the second line.",
        ha="center",
        fontsize=10,
        color="#475467",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "benchmark_topology_load_matrix.png")


def plot_candidate_safety_window() -> None:
    candidates = pd.read_csv(EXTENDED_DIR / "top_candidates.csv")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    for bucket, color, marker, label in [
        ("feasible", COLORS["teal"], "o", "feasible top candidates"),
        ("risky_low_loss", COLORS["orange"], "s", "low-loss but infeasible"),
    ]:
        rows = candidates[candidates["bucket"] == bucket]
        ax.scatter(
            rows["v_peak_abs_V"],
            rows["i_rms_A"],
            s=72,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for _, row in rows.head(5).iterrows():
            ax.text(row["v_peak_abs_V"] + 18, row["i_rms_A"] + 0.35, str(row["rank"]), fontsize=9)

    ax.axvline(1000, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.axhline(20, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.text(1006, 1.0, "1000 V", rotation=90, va="bottom", fontsize=9, color=COLORS["gray"])
    ax.text(210, 20.35, "20 A", fontsize=9, color=COLORS["gray"])
    ax.set_xlabel("peak electrode voltage |V| [V]")
    ax.set_ylabel("RMS current [A]")
    ax.set_title("Top candidates in the voltage/current safety window")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "benchmark_candidate_safety_window.png")


def plot_loss_driver_bars() -> None:
    correlations = pd.read_csv(EXTENDED_DIR / "spearman_correlations.csv")
    labels = {
        "metric.constraint_penalty": "constraint penalty",
        "metric.i_rms_A": "RMS current",
        "metric.normalized_rmse": "waveform error",
        "metric.harmonic_error": "harmonic error",
        "metric.v_peak_abs_V": "peak voltage",
        "metric.v_rms_V": "RMS voltage",
        "metric.power_error": "power error",
        "param.Vsrc_amp": "source amplitude",
        "param.C1": "C1",
    }
    rows = correlations[correlations["feature"].isin(labels)].copy()
    rows["label"] = rows["feature"].map(labels)
    rows = rows.sort_values("abs_corr", ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.barh(rows["label"], rows["abs_corr"], color=COLORS["blue"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("alignment with loss (0 weak, 1 strong)")
    ax.set_title("What most strongly moved with loss in L3 extended")
    _style_axes(ax)

    for y, value in enumerate(rows["abs_corr"]):
        ax.text(value + 0.02, y, f"{value:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "benchmark_loss_drivers.png")


def plot_metric_gate() -> None:
    standard = pd.read_csv(STANDARD_DIR / "feasibility_summary.csv")
    rows = standard[standard["bucket"].isin(["feasible", "infeasible"])].copy()
    rows["label"] = (
        rows["case"]
        .map({"level2_timevarying_plasma": "L2", "level3_topology_load_choice": "L3"})
        + " "
        + rows["bucket"]
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(rows))
    ax.bar(
        x,
        rows["median_loss"],
        color=[COLORS["teal"] if g == "feasible" else COLORS["orange"] for g in rows["bucket"]],
    )
    ax.scatter(x, rows["p90_loss"], color=COLORS["dark"], s=55, label="p90")
    ax.set_yscale("log")
    ax.set_xticks(list(x), rows["label"], rotation=20, ha="right")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Feasible and infeasible trials must be read separately")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)

    for idx, (_, row) in enumerate(rows.iterrows()):
        ax.text(idx, row["median_loss"] * 1.18, f"n={int(row['count'])}", ha="center", fontsize=9)

    fig.tight_layout()
    _save(fig, "benchmark_metric_gate.png")


def main() -> None:
    plot_problem_map()
    plot_feasibility_summary()
    plot_metric_gate()
    plot_topology_load_risk()
    plot_harmonic_ratios()
    plot_engineering_gate_counts()
    plot_topology_load_matrix()
    plot_candidate_safety_window()
    plot_loss_driver_bars()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
