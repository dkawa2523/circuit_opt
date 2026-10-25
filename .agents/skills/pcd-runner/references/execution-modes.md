# PCD execution modes

Run commands from the repository root. Replace placeholders in braces with
the user's paths. Prefer forward slashes in arguments so the examples work in
PowerShell and POSIX shells.

## Common preflight

Check the public entry point and solver without changing the lock file:

```text
uv run --frozen python -m pcd --help
uv run --frozen python -m pcd solver-diagnose --json
```

If the environment has not been created yet, `uv run` may materialize it from
`uv.lock`. If locked dependencies are genuinely missing, use
`uv sync --frozen --group dev`; do not regenerate the lock file.

List registered circuit, load, solver, metric, and optimizer implementations:

```text
uv run --frozen python -m pcd list
```

## Validate or inspect without running ngspice

Validate normal executable input:

```text
uv run --frozen python -m pcd validate-case {case.yaml} --json
```

Use `validate-case {case.yaml} --plan` to inspect the resolved roles, condition
IDs, control counts, observations, defaults, and input files without solving.

Add `--strict` only when warnings must fail a requested release or
qualification check.

Generate the resolved ngspice deck without solving it:

```text
uv run --frozen python -m pcd sim-netlist {case.yaml} --out runs/netlists/{case-name}.cir
```

Render a topology-oriented view of a generated or authored netlist:

```text
uv run --frozen python -m pcd visualize-netlist {netlist.cir} --out runs/visualizations/{name}.svg
```

Netlist visualization is explanatory parsing; it is not proof that ngspice
accepted or solved the deck.

## Run one simulation

Use this when the user wants waveforms or AC response for exactly one resolved
case, without candidate ranking or objective scoring:

```text
uv run --frozen python -m pcd sim-run {case.yaml} --run-root runs
```

Public examples `examples/circuit_external.yaml` and
`examples/circuit_transient.yaml` are minimal single-point inputs. Cases with
multiple conditions, candidate domains, or control states require explicit
`--scenario`, `--params` (SI JSON), and/or `--controls`, or the `run` workflow.
Never silently reduce a requested study to a default point. `sim-netlist`
remains an explicitly labelled default-point preview.

The command prints the run directory. Read `summary.json` first. Canonical
waveform and AC data are under `data/`; the exact netlist, solver log, replay
case, and detailed `manifest.json` are under `debug/`. Resolve paths from the
summary or debug manifest rather than assuming filenames. Historical root
manifests are not part of the current read contract. A solver-reported failure
is nonzero by default but retains its run record; use `--allow-failure` only when that
known result is intentionally being collected. Invalid input exits with code 2
and does not manufacture a failed waveform. Unexpected implementation errors
propagate instead of being reclassified as simulation results.

Regenerate engineering measurements and standard figures from an existing
successful simulation directory or manifest, without rerunning ngspice:

```text
uv run --frozen python -m pcd analyze {run-dir-or-manifest} --out runs/analysis/{name}
```

Use `--frequency-Hz` only when the user supplies or requests a meaningful AC
operating point. Inspect `summary.json` and the applicable impedance/Smith,
transient, power, component-stress, and harmonic figures. A multi-point AC
sweep may also contain a power-response resonance, bracketed -3 dB bandwidth,
and loaded Q. Do not interpret that value as unloaded component Q, and do not
manufacture it for a single point or an unbracketed sweep. This mode reports
electrical measurements; use a study when engineering acceptance is required.

PCD accepts `output` in the same case YAML. `output: {}` automatically
renders applicable figures and a schematic after a successful solve/study/fit;
omitting it retains data-only execution. Use `analyze RUN --case case.yaml` to
read only revised output settings and regenerate figures from saved observations.
Inspect `output.status` separately from solver/engineering status: requested
missing signals or partial schematics produce a nonzero CLI exit and saved reasons.
See `docs/manual-ja.md`, sections 9–11, for recipes, signal units/signs, and export scope.
From 0.22, combined CSV columns distinguish `context.phase` and `parameter.*`
from `signal.*`. Read plot `coverage` and nullable power ratios with their saved
reasons; an unavailable efficiency cannot satisfy an engineering limit.

## Run a design study

Use this for public `pcd.rf.v1` studies and advanced `case_yaml.v1` studies
that require Candidate x Scenario x Control evaluation:

```text
uv run --frozen python -m pcd run {case.yaml} --output runs --json
```

Add `--require-acceptance` for CI, unattended qualification, or when the user
explicitly wants an unacceptable design to produce a nonzero exit. An exact
finite candidate grid cannot be truncated or changed by CLI overrides. For
continuous public circuit design, use its declared exploratory optimizer,
budget, and seed unless the user requests overrides.

After completion, inspect the printed `run_root` and its `study_result.json`.
Important fields are:

- `best`: selected candidate and aggregate decision;
- `study`: scenarios, objectives, and model metadata;
- `dataset`: dataset ID, table schema, and runtime/solver fingerprints;
- `artifacts.generation`: active immutable generation;
- `artifacts.evaluation_table`: complete generation-relative audit table;
- `artifacts.pareto_front`: optional complete-evidence nondominated candidates
  for a study with two or more objectives;
- `artifacts.snapshot_response`: optional selected-candidate time table for an
  `impedance_profile` study;
- `artifacts.best_candidate`: structured control and solver evidence for the
  selected design;
- `n_evaluations` and `n_failed_evaluations`: execution completeness.

For `pareto_front`, report `study_result.json.pareto.scope`. An
`observed_candidates` front covers only evaluated samples; only
`declared_grid` represents the complete declared discrete candidate set.

The selected candidate stores each control trial once and selects it by index.
Use `pcd.results.selected_evaluation` when reading `best_candidate.json`. Use
`study_history.json` for all-candidate summaries and `evaluations.csv` for
Candidate x Scenario x Control detail. The root `raw/` area is reusable
physical cache, not a second published decision table.

## Identify effective terminal parameters

Use this only when the case declares bounded `latent` parameters, calibrated
fixed circuitry, complex terminal V/I observations, disjoint fit and holdout
scenarios, and identifiability limits:

```text
uv run --frozen python -m pcd identify {case.yaml} --output runs --json
```

Read `identification_result.json` first. Check `validation.scope`: unseen
declared conditions and same-condition replicates make different claims.
Repeated IDs/observations do not establish independent acquisition. Expected
unavailable evidence returns `not_identified` with a phase-specific reason;
follow only artifacts listed for phases actually executed. It links separate `fit/` and
`holdout/` studies, aligned observation tables, local-sensitivity evidence,
and each phase's cache-free selected-candidate replay. An `identified` result
means the declared electrical fit/holdout and local-rank checks passed. It does
not identify plasma chemistry, density, sheath geometry, or process yield.
`run` proposes only `design` values; `identify` proposes only `latent` values.

Produce a compact tabular summary when requested:

```text
uv run --frozen python -m pcd result-summary {study-root}
```

## Inspect or prune generations

Never guess a generation path from timestamps. Start from the study root's
`study_result.json`, then follow the recorded artifact paths.

Preview retention:

```text
uv run --frozen python -m pcd result-prune {study-root} --keep 3 --json
```

Only after explicit deletion authorization and review of that exact preview:

```text
uv run --frozen python -m pcd result-prune {study-root} --keep 3 --apply --json
```

The active and newest generations are preserved; the reusable raw cache is
outside this pruning operation. The former interpreted-evaluation cache no
longer exists.

## Reproduce validation suites and curated outputs

Run these only when the user explicitly asks for the full benchmark,
regression evidence, or publication artifacts. The core suite performs 411
real-ngspice evaluations when no matching cache is available.

PCD user-workflow release acceptance uses a new root. The exact count depends
on numerical-method convergence; read it from `release_result.json` rather than
assuming the historical 0.16.0 count:

```text
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
```

Read `release_result.json` and `REPORT.md`. A suite PASS qualifies the expected
circuit workflow reproduction only. Machine learning is outside this gate.
Never reuse an existing run root to turn cached evaluations into release
evidence.

Core electrical suite:

```text
uv run --frozen python bench/run_suite.py --run-root runs/benchmark_suite
```

Literature/source-conformance suite:

```text
uv run --frozen python bench/literature/run_suite.py --run-root runs/literature/final_evaluation
```

Regenerate the report and publication figures only after both source suites
complete successfully:

```text
uv run --frozen python bench/reports/generate_case_report.py
uv run --frozen python bench/figures/generate.py --run-root runs/benchmark_suite
```

Check `passed` in `runs/benchmark_suite/benchmark_result.json` and both
`passed` and `benchmark_integrity_passed` in the literature evaluation. A
benchmark case may intentionally be electrically infeasible and still PASS by
reproducing its expected classification.

For a requested code-quality gate after implementation work, use:

```text
uv run --frozen python -m nox -s quality-pr
```

Do not run the complete quality gate for an ordinary case execution unless the
user asks or source code changed as part of the same task.

## Case selection when the user asks for a demonstration

- Minimal real-ngspice smoke: `bench/cases/topology_l_match_golden.yaml`.
- Public matching study: `examples/rf_impedance_point_study.yaml`.
- Independent frequency points: `examples/rf_impedance_frequency_table.yaml`.
- Quasi-static time-ordered R+jX snapshots: `examples/rf_quasi_static_profile.yaml`.
- Component stress and effective loss: `examples/rf_component_stress.yaml`.
- Effective CCP one-port: `examples/rf_ccp_lumped.yaml`.
- Effective ICP terminal fit: `examples/rf_icp_transformer.yaml`.
- Effective CCP terminal-parameter identification: `examples/terminal_identification.yaml`.
- Advanced constrained continuous sizing: `examples/advanced/generic_rc_filter.yaml`.
- Advanced transient port measurement: `examples/advanced/rf_port_transient.yaml`.

Examples demonstrate syntax and model behavior. They are not qualified chamber
data or process recipes.
