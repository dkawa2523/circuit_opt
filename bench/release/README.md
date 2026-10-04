# PCD v1 user-workflow acceptance

This executable matrix checks that an engineer can start from a supported
case, run the real ngspice backend, and reach a concise electrical result plus
replay evidence. It is not semiconductor-equipment or plasma-model
qualification; those require apparatus-specific measured data.

Verified on 2026-10-04 with ngspice 46. The final evidence is generated under
the new root supplied with `--run-root` and can be reproduced by `run_suite.py`. The
runner uses only the public CLI, validates all eight inputs in strict mode,
requires a new run root, and rejects cached study evaluations. A workflow
passes when its execution and required evidence are present; an intentionally
infeasible design remains a valid workflow result only when its named failed
limits and coverage are reproduced.
For studies, the reported ngspice count includes both search evaluations and
the cache-free selected-control replay for every scenario.

| workflow | input | required user result | verified result |
|---|---|---|---|
| saved AC/transient analysis | `examples/advanced/rf_port_transient.yaml` | canonical waveform, reference plane, periodic power, standard figures, replay manifest | PASS: 8,008 samples, 24.9998 W load power, zero warnings, transient/power/harmonic figures |
| Candidate x Scenario x Control study | `bench/cases/role_factorial_search.yaml` plus `examples/rf_impedance_point_study.yaml` negative control | complete finite enumeration, selected control per condition, coverage, failed limits, selected-candidate evidence | PASS: 2 x 2 x 2 = 8 solves accepted; negative control reports 1/3 accepted and two `max_reflection_magnitude` failures |
| frequency sweep | `bench/literature/p1_colpo1999_icp/dummy_resonance.yaml` | impedance, Smith trace, bracketed half-power resonance, -3 dB bandwidth, loaded Q | PASS: 13.88725 MHz, 1.039898 MHz bandwidth, loaded Q 13.3544, zero warnings |
| prescribed time variation | `examples/rf_quasi_static_profile.yaml` and `examples/advanced/time_varying_resistor.yaml` | independent R+jX snapshots and a separate one-way R(t) transient without claiming plasma feedback | PASS: 15/15 snapshot solves and `snapshot_response.csv`; 258-point R(t) waveform marked periodic `not_applicable` |
| target-waveform sizing | `examples/advanced/generic_rc_filter.yaml` | feasibility-first decision, objectives, seed, Pareto table, selected real-ngspice evidence | PASS: 24 unique search solves plus one replay, seed 3, nRMSE 0.340510 to 0.180607, peak 4.81899 V, 15-point observed Pareto front |
| effective terminal-parameter identification | `examples/advanced/ccp_terminal_identification.yaml` | latent-only fit, unseen-frequency holdout, cache-free replay, local sensitivity rank, parameter recovery | PASS: R = 17.2353 ohm (4.25% from known 18 ohm), C = 121.135 pF (0.95% from known 120 pF), fit error 0.01459, holdout error 0.00985, rank 2/2 |

## Reproduce

Run the complete circuit acceptance matrix with one command:

```powershell
uv run --frozen python bench/release/run_suite.py --run-root runs/release_acceptance
```

`release_result.json` and `REPORT.md` report the circuit-foundation decision.
The suite exits nonzero for a broken workflow. Machine learning is not part of
this release gate and cannot change its result.

For a study, `study_result.json` reports analysis type, frequency when fixed,
reference plane, Z0, acceptance coverage, and `artifacts.best_candidate`.
That candidate file links each selected condition to its exact ngspice
manifest, waveform/AC table, netlist, and solver log. `analyze` can consume the
manifest directly without another solver run.

The 2026-10-04 run reproduced all eight cases with 353 fresh ngspice
evaluations and zero cache hits. Its circuit-analysis, deterministic-sizing,
and effective-terminal-identification decision was **GO**.
This does not qualify a chamber process or authorize an ML proposal loop.
