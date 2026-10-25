# PCD user-workflow acceptance

This matrix calls the public CLI with real ngspice and checks electrical
results plus replay evidence. It qualifies the calculation workflow, rather
than a semiconductor process or apparatus-specific plasma model.

| Workflow | Input | Required result |
|---|---|---|
| Saved transient analysis | `examples/advanced/rf_port_transient.yaml` | Waveform, periodic power, figures, archived replay inputs |
| Candidate x Scenario x Control | `bench/cases/role_factorial_search.yaml` | Complete enumeration and selected controls for every condition |
| Declared design rejection | `examples/rf_impedance_point_study.yaml` | Named failed limits and incomplete acceptance coverage |
| Frequency sweep | `bench/literature/p1_colpo1999_icp/dummy_resonance.yaml` | Impedance, Smith trace, bracketed half-power bandwidth and loaded Q |
| Quasi-static profile | `examples/rf_quasi_static_profile.yaml` | One result per independent load/time snapshot |
| Prescribed time variation | `examples/advanced/time_varying_resistor.yaml` | Archived R(t), transient response, appropriate periodic applicability |
| Target-waveform design | `examples/advanced/generic_rc_filter.yaml` | Feasibility-first selection, objective history, observed Pareto front |
| Effective terminal identification | `examples/terminal_identification.yaml` | Bounded latent fit, unseen-frequency holdout, fresh replay, local rank |

## Reproduce

Use a new output root. The runner validates all eight inputs in strict mode,
rejects cached search evaluations, and includes selected-control replays in
its solver counts.

```powershell
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
```

Read `release_result.json` and `REPORT.md`. Candidate and solver counts are
measured, rather than frozen to the former 0.16.0 DE identification budget.
The current terminal example uses bounded least squares and can stop at
convergence before its maximum candidate count.

A workflow can pass while its intentionally infeasible design is rejected.
The suite must reproduce the named constraint failures and condition coverage;
solver success alone is insufficient. Historical numerical summaries in Git
and publication exports are not a current product acceptance result.

Numerical edge cases, including solver-resolved narrow pulse stress and source
versus load reference planes, are covered by independent `tests/test_e2e.py`
journeys. Local sensitivity evidence is not a parameter confidence interval.
Machine learning is outside this gate.
