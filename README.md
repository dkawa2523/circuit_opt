# Circuit Design Platform v6

This is a compact platform for circuit design studies where the simulation layer and the data/ML layer are deliberately separated.

```text
Simulation layer:  pcd.sim_core + pcd.sim_methods + pcd.sim_registry
  case.yaml -> circuit/load -> ngspice netlist -> solver -> waveform artifacts
  It does not compute objective metrics and does not import ML code.

Data/ML layer:     pcd.records + pcd.ml_core + pcd.ml_methods + pcd.ml_registry
  existing artifacts or external waveforms -> metrics -> learning table -> candidates/surrogate
  It does not import ngspice solvers or circuit builders.

Workflow layer:    pcd.workflow
  optional glue: ask -> simulate -> score -> tell
```

The boundary artifact is:

```text
sim_manifest.json + waveform.csv
```

Simulation commands never write `metrics.json`.  ML commands read existing records and write `metrics.json`, `scores.csv`, and optional surrogate outputs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional Optuna:

```bash
pip install -e '.[optuna]'
```

## Simulation only

```bash
pcd sim-run examples/rf_plasma_fixed.yaml --solver dummy --run-root runs/sim_only
pcd sim-netlist examples/rf_plasma_fixed.yaml --out netlist.cir
pcd visualize-netlist netlist.cir --out circuit_schematic.png --summary-json circuit_schematic.json
```

`visualize-netlist` uses Schemdraw to render conventional schematic symbols
for resistors, capacitors, inductors, sources, and grounds.  Subcircuits are
expanded where possible so load-model internals remain visible.

## Research mode vs production-safe mode

Research mode keeps exploratory loops moving.  It allows the `dummy` solver and
scores failed simulations with an explicit large penalty so optimizers can keep
collecting observations:

```bash
pcd workflow-optimize examples/topology_choice_pipeline.yaml \
  --optimizer random \
  --solver dummy \
  --n-trials 10 \
  --run-root runs/research_loop
```

Production-safe mode adds validation, strict process exit behavior, failed-run
exclusion for surrogate training, and a bounded external solver timeout:

```bash
pcd validate-case examples/rf_plasma_fixed.yaml --strict
pcd sim-run examples/rf_plasma_fixed.yaml --solver ngspice_cli --run-root runs/prod_sim --strict-exit
pcd ml-score examples/rf_plasma_fixed.yaml runs/prod_sim --strict-exit
pcd ml-fit-surrogate runs/prod_sim --exclude-failed --out runs/prod_sim/surrogate.json
```

Set `solver.timeout_s` in the case file to override the default 300 second
timeout used by `ngspice_cli`.

## ML/data only

```bash
pcd ml-propose examples/topology_choice_pipeline.yaml --n 20 --out candidates.csv
pcd ml-score examples/topology_choice_pipeline.yaml runs/sim_only
pcd ml-fit-surrogate runs/sim_only --out surrogate.json
pcd ml-predict surrogate.json candidates.csv --out predicted_candidates.csv
```

External measured or plasma-coupled waveforms can be imported with `pcd.records.import_external_waveform()` and then scored by the ML layer without running ngspice.

## Explicit closed-loop workflow

```bash
pcd workflow-optimize examples/topology_choice_pipeline.yaml \
  --optimizer random \
  --solver dummy \
  --n-trials 10 \
  --run-root runs/closed_loop
```

## Built-in simulation methods

Circuit builders:

```text
from_yaml, l_match, pi_match, pi_match_harmonic
```

Load models:

```text
none, resistor, parallel_rc, series_rlc, electrode_stray,
from_yaml, plasma_fixed_rlc, plasma_state_rlc, plasma_table_rlcq
```

Solvers:

```text
dummy, ngspice_cli
```

## Built-in ML methods

Objectives:

```text
waveform_l2, waveform_l2_harmonics
```

Optimizers:

```text
random, optuna
```

## Plugin pattern

```python
from pcd.sim_registry import register as sim_register
from pcd.ml_registry import register as ml_register
```

Keep plugin functions small.  A circuit builder returns `Circuit`; a load builder returns a `load_model` subckt string; an objective consumes a saved waveform and returns a metrics dict containing `loss`.
