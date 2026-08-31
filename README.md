# PCD — semiconductor-equipment RF circuit studies

PCD evaluates whether one fixed RF matching network remains electrically
acceptable over a declared chamber-load window after only the tuner settings
available in the equipment are adjusted.

It is a circuit-model foundation, not a plasma solver. Chamber or plasma
information enters as a qualified electrical one-port at a named reference
plane: measured or supplied R+jX points, an effective CCP R-L-C fit, or an
effective ICP transformer fit. PCD does not infer density, sheath geometry,
chemistry, species power, temperature, or self-consistent plasma/circuit
feedback.

## Smallest useful study

```yaml
schema: pcd.rf.v1
case_id: production_window
frequency_Hz: 13560000

network:
  type: pi_match
  fixed: {L1: 6.43e-7}
  tuning:
    C1: [4.8e-10, 6.7e-10, 8.9e-10]
    C2: [4.5e-11, 1.6e-10, 2.1e-10]

load:
  type: impedance_table
  file: load_points.csv
  reference_plane: electrode_terminal

acceptance:
  reflected_power_fraction_max: 0.10
```

`load_points.csv` has one stable shape:

```csv
scenario_id,resistance_ohm,reactance_ohm,weight
nominal,25,-80.77,1
high_R,50,-80.77,1
low_R,12.5,-80.77,1
```

Run it with one command:

```powershell
uv sync --group dev
uv run pcd run case.yaml
```

Results default to `runs/`. The terminal shows the selected fixed candidate,
condition coverage, selected tuner state per condition, declared objectives,
failed limits, and the result directory. `study_result.json` retains the same
compact decision together with every detailed result reference. Use `--json`
for the machine-readable result or `--output path` to choose the root
directory.

## What users specify

| input | engineering meaning | changes when |
|---|---|---|
| `network.fixed` | selected hardware | a new candidate is built |
| `network.search` | hardware values PCD may search | a new candidate is proposed |
| `network.tuning` | discrete equipment settings | independently for each load condition |
| load table / `conditions` | external electrical conditions | imposed on the candidate |
| `acceptance` | pass/fail engineering limits | never optimized away |

The internal Candidate, Scenario, and ControlState types enforce those roles,
but users do not need to write them. A parameter cannot belong to more than one
role.

PCD enumerates every declared tuning combination for every condition. It does
not sample an incomplete tuner grid and call the result infeasible. The
default safety limit is 250 states; a deliberately larger study may set
`execution.control_state_limit`.

The same rule applies to hardware selection: every `network.search.values`
combination is compared exactly once and the candidate count is inferred.
The public RF input does not turn a continuous range into an arbitrary sample
and present that sample as a completed design decision. Exploratory continuous
optimization remains available only in the advanced explicit extension path.

## What PCD resolves automatically

The public input is compiled once before validation or simulation. The
resolved plan explicitly contains:

- the RF source, standard nodes, load ports, and 50-ohm measurement plane;
- the same frequency for source, impedance anchor, and one-point AC solve;
- scenario-column mappings and a complete tuning-state budget;
- reflection objective, engineering constraints, solver, and the exact
  candidate count;
- component probes and effective series-loss elements only when requested.

No numerical layer reinterprets the short input independently. Every run
archives `input_case.yaml`, `resolved_plan.yaml`, and executable `case.yaml`,
so inferred defaults remain reviewable and replayable. Command-line execution
overrides are applied before validation, hashing, and archival; the stored
resolved plan therefore describes the run that actually occurred.

See [the RF input reference](docs/input-format.md) for all supported fields and
[RF load models](docs/rf-load-models.md) for model responsibility and limits.

## Component stress and effective loss

Absolute stress requires a declared drive amplitude:

```yaml
drive_peak_V: 100
network:
  type: pi_match
  fixed: {C1: 2.58e-10, L1: 1.20e-6, C2: 7.45e-12}
  loss_ohm: {C1: 0.1, L1: 0.5, C2: 0.1}
acceptance:
  reflected_power_fraction_max: 0.10
  component_limits:
    L1: {current_rms_A_max: 1.0, loss_W_max: 0.5}
  loss_balance_fraction_max: 1.0e-5
```

PCD reports terminal RMS/peak voltage and current, effective loss, network
efficiency, source-terminal RMS current/apparent power, and electrical loss
closure. With an explicit drive, every named component in the public matching
topology is observed automatically; limits remain optional. Effective ESR/DCR
is not an internal temperature or lifetime model. Details are in
[component loss and stress](docs/component-loss-stress.md).

## RF load choices

| public load type | appropriate input | boundary |
|---|---|---|
| `impedance_table` | independent R+jX points, optionally with `frequency_Hz` per row | no interpolation between supplied points |
| `impedance_point` | one R+jX value at one frequency | exact only at that anchor |
| `ccp_lumped` | qualified effective series R-L-C parameters | no sheath state or species-power inference |
| `icp_transformer` | qualified coil plus identifiable reflected-loading fit | terminal model only; no density or plasma-power split |

`reference_plane` is required. `evidence` is optional so exploratory work can
run, but strict validation warns when applicability has not been documented.
Referenced scenario, target-waveform, and external-netlist files are archived
once per study by content hash, so the executable `case.yaml` replays from the
study bundle rather than depending on the original file location.

## Core electrical benchmarks

The core suite contains twelve concise public RF cases and four explicit
advanced/boundary cases:

- A1-A3: independent complex-impedance goldens for every public topology;
- A4: multi-frequency E2E for the effective CCP R-L-C port;
- A5: multi-frequency public-input E2E for the effective ICP transformer port;
- B1-B3: fixed, limited-control, and full-control design decisions;
- B4: independent synthetic frequency-point replay;
- B5: match passes but high-drive component limits fail;
- B6: complete three-value hardware search;
- B7: Candidate x Scenario x Control orthogonal enumeration;
- B8: deterministic full-factorial component-value corner stress;
- D1-D3: equivalent lossy reference-plane representations and a fixture
  double-counting negative control.

Expectations and explanatory text live separately in
`bench/expectations.yaml`. Reproduce all 411 real-ngspice evaluations with:

```powershell
uv run python bench/run_suite.py --run-root runs/benchmark_suite
```

These cases establish circuit-pipeline behavior and bounded electrical design
decisions, not a qualified reactor process window.

The cross-disciplinary case report for semiconductor-equipment, thermal, and
data-analysis readers is stored at
[`output/reports/benchmark-case-report-artifact.json`](output/reports/benchmark-case-report-artifact.json).
See [`bench/reports/README.md`](bench/reports/README.md) for its reproducible
generation sequence and interpretation boundary.

## Code structure

```text
pcd/plan.py       public RF input -> explicit resolved execution plan
pcd/case.py       schema routing, case paths, and design-variable discovery
pcd/study_config.py advanced Candidate/Scenario/Control translation
pcd/core/         role types, scenario/control selection, aggregation
pcd/study.py      evaluation execution, caching, and study result assembly
pcd/search.py     exact public candidate grids and advanced exploratory search
pcd/sim_core.py   one simulation and its immutable run artifacts
pcd/netlist.py    circuit IR and ngspice rendering
pcd/sim_methods.py named circuit, load, and solver implementations
pcd/spice.py      SPICE parameter resolution and value formatting
pcd/artifacts.py  run serialization and implementation identity
pcd/analysis.py   AC/transient electrical measurements
pcd/metrics.py    metrics and engineering-limit evaluation
pcd/rf_loads.py   electrical RF-load equations and parameter validation
pcd/results/      content-addressed result storage and summaries
pcd/signals/      reusable time-series and phasor primitives
```

The explicit `case_yaml.v1` form remains an advanced extension path for custom
netlists, plugins, transient studies, and generic waveform objectives. Both
public and advanced studies use `pcd run`; `sim-run` is reserved for one
simulation without scoring. New RF matching studies should use `pcd.rf.v1`.

## Verification

```powershell
uv run pytest -q
uv run nox -s quality-pr
uv run python bench/run_suite.py --run-root runs/benchmark_suite
```

Runtime output belongs under `runs/`; examples document syntax; tests verify
software and numerical behavior; benchmarks support only their stated
electrical decisions.
