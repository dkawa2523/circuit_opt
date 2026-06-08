# CCP Benchmark Pack Execution Report

## Overview

`ccp_benchmark_pack` contains a three-level GEC-like argon CCP benchmark:

- **Level 1: fixed/state RLC** (`ccp_gec_level1_fixed_match.yaml`)  
  Fixed matching topology with a state-derived plasma RLC load.
- **Level 2: time-varying plasma table** (`ccp_gec_level2_timevarying_plasma.yaml`)  
  Harmonic matching topology with `plasma_table_rlcq` driven by the synthetic plasma table.
- **Level 3: topology/load choice** (`ccp_gec_level3_topology_and_load_choice.yaml`)  
  Mixed categorical/continuous search over topology, load model, and component/process parameters.

All three cases validated successfully in non-strict mode.  The only validation warning was the expected `dummy` solver warning: this run evaluates the workflow and screening behavior, not physical plasma fidelity.

## Execution

Evaluation output root:

```text
runs/ccp_benchmark_eval
```

Each benchmark was run with:

- optimizer: `random`
- solver: `dummy`
- trials per case: `30`
- failed trials: `0` for all cases

Generated artifacts:

- Summary CSV: `runs/ccp_benchmark_eval/benchmark_summary.csv`
- Surrogate CSV: `runs/ccp_benchmark_eval/surrogate_summary.csv`
- Per-case summaries: `runs/ccp_benchmark_eval/<level>/summary.csv`
- Figures:
  - `runs/ccp_benchmark_eval/figures/plasma_table_rlc.png`
  - `runs/ccp_benchmark_eval/figures/loss_curves.png`
  - `runs/ccp_benchmark_eval/figures/best_waveform_overlays.png`
  - `runs/ccp_benchmark_eval/figures/best_metric_summary.png`
  - `runs/ccp_benchmark_eval/figures/level3_category_performance.png`

## Results

| Case | Trials | Failed | Best loss | Norm. RMSE | Harmonic error | Power error | Best peak V |
|---|---:|---:|---:|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 30 | 0 | 0.093730 | 0.038031 | 0.037997 | 0.999996 | 311.42 V |
| Level 2 time-varying plasma | 30 | 0 | 0.491199 | 0.271650 | 0.678226 | 0.999847 | 315.83 V |
| Level 3 topology/load choice | 30 | 0 | 0.501831 | 0.284901 | 0.667723 | 0.999998 | 280.68 V |

Surrogate diagnostics:

| Case | Train rows | Features | Train RMSE | Train R2 | CV RMSE |
|---|---:|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 30 | 7 | 0.185419 | 0.112457 | 0.274372 |
| Level 2 time-varying plasma | 30 | 7 | 0.122149 | 0.319327 | 0.225863 |
| Level 3 topology/load choice | 30 | 18 | 0.093336 | 0.540628 | 0.214386 |

Level 3 categorical trends:

| Category | Count | Min loss | Mean loss |
|---|---:|---:|---:|
| topology `l_match` | 11 | 0.501831 | 0.638352 |
| topology `pi_match` | 12 | 0.502475 | 0.640974 |
| topology `pi_match_harmonic` | 7 | 0.522930 | 0.697173 |
| load `plasma_fixed_rlc` | 16 | 0.501831 | 0.688387 |
| load `plasma_state_rlc` | 5 | 0.511038 | 0.611989 |
| load `electrode_stray` | 9 | 0.512192 | 0.613293 |

## Evaluation

Level 1 is the strongest result in this screening run.  Its best normalized RMSE is about `0.038`, and harmonic error is also low.  The fixed/state-derived RLC case is therefore a good smoke test for the platform: netlist generation, scoring, optimization history, manifest provenance, and surrogate fitting all complete cleanly.

Level 2 is materially harder.  The time-varying plasma table is generated and plotted successfully, but the best dummy-solver waveform remains far from the tailored target in harmonic content.  This is expected: the dummy solver does not actually solve the time-varying plasma load dynamics.  This level is useful as an integration and boundary-artifact test, not as a physical validation result.

Level 3 exercises the data-science path most strongly.  It produces the widest feature schema (`18` encoded features), including topology and load categories.  The best categorical result came from `l_match` with `plasma_fixed_rlc`, but the category means are close enough that 30 random trials are not sufficient for a robust design conclusion.  The surrogate has better train R2 than Levels 1/2, but CV RMSE remains large relative to best-loss differences, so it should be used for ranking support only.

Power error is approximately `1.0` in all three cases.  This indicates that the current dummy-solver current proxy is not suitable for power-delivery evaluation.  The voltage waveform terms can still support workflow screening, but power-related conclusions require `ngspice_cli` or a higher-fidelity coupled solver.

## Recommendation

Use this benchmark pack in two modes:

1. **Workflow regression / platform QA**  
   Run the same dummy-solver benchmark to verify validation, records, metrics, plots, surrogate outputs, and categorical handling.

2. **Physical design evaluation**  
   Replace `dummy` with `ngspice_cli`, keep `--strict-exit`, inspect solver diagnostics, and compare against measured, fluid, global-model, or PIC/MCC plasma outputs.  Do not use the dummy-solver power proxy for physical design decisions.

## Ngspice Netlist And Circuit Visualization Check

Additional output root:

```text
runs/ccp_ngspice_visual_eval
```

This check generated ngspice-ready netlists, expanded the `load_model` subcircuits into Schemdraw-based schematic diagrams, and attempted `ngspice_cli` execution for each CCP case.

Generated artifacts:

- Netlists: `runs/ccp_ngspice_visual_eval/netlists/*.cir`
- Netlist summaries: `runs/ccp_ngspice_visual_eval/netlists/*_netlist_summary.json`
- Schematics:
  - `runs/ccp_ngspice_visual_eval/figures/level1_fixed_match_schematic.png`
  - `runs/ccp_ngspice_visual_eval/figures/level2_timevarying_plasma_schematic.png`
  - `runs/ccp_ngspice_visual_eval/figures/level3_topology_load_choice_schematic.png`
- Component count plot: `runs/ccp_ngspice_visual_eval/figures/component_counts.png`
- Ngspice attempt status plot: `runs/ccp_ngspice_visual_eval/figures/ngspice_attempt_status.png`
- Attempt summary: `runs/ccp_ngspice_visual_eval/ngspice_attempt_summary.csv`

Ngspice execution status:

| Case | Circuit | Load | Nodes | Expanded components | ngspice status | Cause |
|---|---|---|---:|---:|---|---|
| Level 1 fixed/state RLC | `pi_match` | `plasma_state_rlc` | 5 | 8 | failed | `ngspice` executable not found |
| Level 2 time-varying plasma | `pi_match_harmonic` | `plasma_table_rlcq` | 6 | 10 | failed | `ngspice` executable not found |
| Level 3 topology/load choice | `pi_match` | `plasma_state_rlc` | 5 | 8 | failed | `ngspice` executable not found |

Expanded component counts:

| Case | V source | Capacitors | Inductors | Resistors |
|---|---:|---:|---:|---:|
| Level 1 fixed/state RLC | 1 | 3 | 2 | 2 |
| Level 2 time-varying plasma | 1 | 4 | 3 | 2 |
| Level 3 topology/load choice | 1 | 3 | 2 | 2 |

Evaluation:

- The platform can generate ngspice-ready netlists for all three CCP cases.
- The visualization path can parse the netlist, expand the `load_model` subcircuit, and render conventional schematic symbols for the RF source, matching network, electrode node, ground-referenced branches, and plasma-load internals.
- Actual ngspice solving was not completed in this environment because `ngspice` is not installed or not visible on `PATH`.
- The failed ngspice attempts were recorded correctly as failed simulation records with `loss=1e30`, `metrics_status=failed`, and `metrics_reason=simulation_failed`.
- Once ngspice is installed, the same netlists and `ngspice_cli` path can be used for physical waveform generation and scoring.
