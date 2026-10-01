# PCD v1 user-workflow acceptance

This executable matrix closes the P1 user-path gate and is the circuit part of
the P4 release decision. It checks that an engineer can start from a supported
case, run the real ngspice backend, and reach a concise electrical result plus
replay evidence. It is not semiconductor-equipment or plasma-model
qualification; those require apparatus-specific measured data.

Verified on 2026-09-30 with ngspice 46. The final P4 evidence is generated under
the new root supplied with `--run-root` and can be reproduced by `run_suite.py`. The
runner uses only the public CLI, validates all seven inputs in strict mode,
requires a new run root, and rejects cached study evaluations. A workflow
passes when its execution and required evidence are present; an intentionally
infeasible design remains a valid workflow result only when its named failed
limits and coverage are reproduced.

| workflow | input | required user result | verified result |
|---|---|---|---|
| saved AC/transient analysis | `examples/advanced/rf_port_transient.yaml` | canonical waveform, reference plane, periodic power, standard figures, replay manifest | PASS: 8,008 samples, 24.9998 W load power, zero warnings, transient/power/harmonic figures |
| Candidate x Scenario x Control study | `bench/cases/role_factorial_search.yaml` plus `examples/rf_impedance_point_study.yaml` negative control | complete finite enumeration, selected control per condition, coverage, failed limits, selected-candidate evidence | PASS: 2 x 2 x 2 = 8 solves accepted; negative control reports 1/3 accepted and two `max_reflection_magnitude` failures |
| frequency sweep | `bench/literature/p1_colpo1999_icp/dummy_resonance.yaml` | impedance, Smith trace, bracketed half-power resonance, -3 dB bandwidth, loaded Q | PASS: 13.88725 MHz, 1.039898 MHz bandwidth, loaded Q 13.3544, zero warnings |
| prescribed time variation | `examples/rf_quasi_static_profile.yaml` and `examples/advanced/time_varying_resistor.yaml` | independent R+jX snapshots and a separate one-way R(t) transient without claiming plasma feedback | PASS: 15/15 snapshot solves and `snapshot_response.csv`; 258-point R(t) waveform marked periodic `not_applicable` |
| target-waveform sizing | `examples/advanced/generic_rc_filter.yaml` | feasibility-first decision, objectives, seed, Pareto table, selected real-ngspice evidence | PASS: 24/24 solves, seed 3, RMSE 0.527345, peak 3.97504 V, 17-point observed Pareto front, selected manifest re-analyzed |

## Reproduce

Run the complete circuit acceptance matrix with one command:

```powershell
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
```

To produce the full P4 decision, first reproduce the fixed P3 ranking protocol,
then attach its result without changing its model, pools, seeds, or threshold:

```powershell
uv run --frozen python bench/ml/run_ranking_benchmark.py --run-root runs/p3_ranking
uv run --frozen python bench/release/run_suite.py `
  --run-root runs/release_closure `
  --ml-ranking-result runs/p3_ranking/ranking_evaluation.json
```

`release_result.json` and `REPORT.md` separate three decisions: circuit
foundation readiness, ML candidate-proposal readiness, and the full originally
requested scope. The suite exits nonzero for a broken workflow or invalid
attached evidence. A valid preregistered ML failure remains valid evidence but
keeps the ML and full-scope decisions at NO-GO.

For a study, `study_result.json` reports analysis type, frequency when fixed,
reference plane, Z0, acceptance coverage, and `artifacts.best_candidate`.
That candidate file links each selected condition to its exact ngspice
manifest, waveform/AC table, netlist, and solver log. `analyze` can consume the
manifest directly without another solver run.

The 2026-09-30 P4 run reproduced all seven cases with 131 fresh ngspice
evaluations and zero cache hits. Its release decisions are circuit foundation
**GO**, ML candidate proposal **NO-GO**, and full originally requested scope
**NO-GO**. No new analysis model or ML proposal loop is authorized by this
result.
