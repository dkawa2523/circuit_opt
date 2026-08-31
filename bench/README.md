# Core electrical benchmarks

This suite answers six separate questions about the generic circuit platform:

1. Does each published network topology reproduce independent circuit algebra?
2. Does a declared one-port load model survive planning, SPICE execution, and
   metric extraction without changing its electrical meaning?
3. Are fixed hardware (Candidate), external conditions (Scenario), and
   per-condition settings (Control) kept separate during a design decision?
4. Is a load represented at one reference plane exactly once?
5. Can the result distinguish electrical reachability, tuning reserve,
   component stress, and source-terminal loading?
6. Can fixed hardware be checked at every declared realized-component-value
   corner without treating those corners as new designs?

These are software, model-form, and decision-workflow benchmarks. A case can
correctly return an infeasible design and still pass its benchmark. None of the
synthetic inputs qualifies a reactor, process window, plasma-state model, or
vendor component.

Literature transcription and apparatus-specific design challenges are kept in
[`literature/`](literature/README.md). They are not folded into one universal
load envelope.

## Benchmark map

### A — engine and load-model conformance

| ID | case | independent check | expected design result |
|---|---|---|---|
| A1 | `topology_l_match_golden.yaml` | closed-form complex input Z | infeasible |
| A2 | `topology_pi_match_golden.yaml` | closed-form complex input Z | feasible |
| A3 | `topology_pi_match_harmonic_golden.yaml` | closed-form complex input Z including the series-LC shunt branch | infeasible |
| A4 | `ccp_lumped_frequency_conformance.yaml` | `R + jwL + 1/(jwC)` followed by independent L-match algebra at 10, 13.56, and 20 MHz | infeasible at all three points |
| A5 | `icp_transformer_frequency_conformance.yaml` | identifiable reflected-loading equation, shunt capacitance, and independent L-match algebra at 10, 13.56, and 20 MHz | matched only at the nominal point |

A1-A3 use the same `30-j20 ohm` one-port and fixed values so each topology is
checked against a small frozen complex-number oracle. Their different
feasibility results are consequences of those arbitrary values; they are not a
topology ranking.

A4 is the missing public-schema, multi-frequency E2E for `ccp_lumped`. Its
three parameters are effective electrical inputs. The benchmark establishes
their frequency continuation and wiring, not their applicability to a physical
CCP or their identifiability from a single measured impedance.

A5 provides the corresponding public-schema E2E for `icp_transformer`. Its
fixed network is analytically matched at 13.56 MHz so frequency substitution,
the optional shunt capacitance, reflected resistance/reactance, and the design
classification are all exercised without adding plasma-state inputs.

### B — design-decision behavior

| ID | case | Candidate | Scenario | Control | expected design result |
|---|---|---|---|---|---|
| B1 | `match_fixed_nominal.yaml` | one fixed pi network | five R+jX points | none | infeasible |
| B2 | `match_limited_tuner.yaml` | one fixed inductor | same five points | 25-state C1/C2 grid | electrically reachable but three points fail 20% reserve |
| B3 | `match_full_tuner.yaml` | same fixed inductor | same five points | extended 49-state C1/C2 grid | feasible with at least 20% reserve |
| B4 | `match_independent_frequency_points.yaml` | one fixed pi network | three independently supplied frequency points | none | infeasible |
| B5 | `match_high_drive_stress.yaml` | one lossy pi network | low/high drive | none | infeasible from high-drive stress |
| B6 | `match_discrete_hardware_search.yaml` | three exact C1 choices | one nominal point | none | one feasible candidate |
| B7 | `role_factorial_search.yaml` | two exact L1 choices | two constructed R+jX points | two exact C1 settings | one robust candidate |
| B8 | `component_value_corner_stress.yaml` | one selected nominal pi network | eight full-factorial realized-value corners | none | five corners infeasible |

B1-B3 retain one electrical envelope and vary only control authority. B2 now
separates an edge-reachable match from an accepted design, while B3 adds real
range beyond every selected setting. B4 calls
its points synthetic and independent: it neither claims measurement provenance
nor interpolates them into a broadband plasma model. B5 uses `high_drive`, not
`production_drive`, because no production tool is being qualified.

B7 is the compact orthogonal role check that was previously missing. It runs
all `2 Candidate x 2 Scenario x 2 Control = 8` evaluations. Only `L1=1 uH`
covers both scenarios, selecting `C1=20 pF` for one and `C1=80 pF` for the
other. This proves role separation without adding fields to the user schema.

B8 keeps the selected nominal C1/L1/C2 BOM in Candidate and places the three
uncontrollable realized-value factors in Scenario. Its deliberately broad
`+/-15%` full-factorial envelope
crosses the match boundary, proving that deterministic component corners can
reject a nominal design. The percentage is synthetic benchmark stimulus, not
a vendor tolerance or yield distribution. The advanced input form is used so
the concise public schema does not gain a tolerance-specific workflow.
Its `3/8` feasible fraction is only equal-weight corner coverage; it is not a
37.5% yield estimate, and checking the vertices does not prove the interior of
the continuous tolerance box. B6 already checks the same nominal hardware
value, so B8 does not duplicate it as a ninth scenario.

### D — reference-plane boundary

| ID | representation | expected |
|---|---|---|
| D1 | plasma-terminal load with a lossy series R-L two-port explicit in the circuit | feasible |
| D2 | the same lossy two-port folded once into an electrode-terminal one-port | feasible and same source Z as D1 |
| D3 | the D2 one-port plus the same explicit two-port again | infeasible and different source Z |

D1-D3 use the explicit internal case format because they test the boundary
implementation itself. The concise public RF input is intentionally not
expanded with fixture-de-embedding syntax. D1 and D2 must agree within
`0.02 ohm`; D3 must move by at least `10 ohm` and cross the match limit.

The negative control proves that double counting is electrically consequential.
It does not claim that a reference-plane string can detect undocumented fixture
content in externally supplied data.

## Acceptance and golden data

Every case uses:

```text
|Gamma| <= sqrt(0.1) = 0.316227766
```

so reflected incident-power fraction must be at most 10%. B2/B3 also require
20% normalized tuning headroom. B5 limits L1 current/effective loss and source
RMS current/apparent power. B8 enumerates all declared component-value corners.
A candidate is feasible only if every declared
scenario has a successful control setting satisfying every applicable limit.

Executable cases contain only their engineering inputs. Frozen
classifications, complex input-impedance goldens, selected controls, and
cross-case reference-plane invariants live in
[`expectations.yaml`](expectations.yaml). The analytic derivations are repeated
with plain complex arithmetic in `tests/test_benchmark_cases.py`; the benchmark
runner compares those frozen values with the real ngspice result.

The shared files are deliberately small:

- [`load_scenarios.csv`](load_scenarios.csv): five synthetic 13.56 MHz corners
  used only by B1-B3;
- [`frequency_scenarios.csv`](frequency_scenarios.csv): three independent
  synthetic frequency rows used only by B4;
- [`role_factorial_scenarios.csv`](role_factorial_scenarios.csv): two
  analytically constructed role-separation rows used only by B7.
- [`component_value_corners.csv`](component_value_corners.csv): the complete
  synthetic three-axis corner table used only by B8.

## Run

```powershell
uv run python bench/run_suite.py --run-root runs/benchmark_suite
```

The suite performs 411 deterministic ngspice evaluations:

```text
A1-A3  3
A4     3
A5     3
B1     5
B2   125
B3   245
B4     3
B5     2
B6     3
B7     8
B8     8
D1-D3  3
total 411
```

`benchmark_result.json` separates each benchmark's own PASS/FAIL from its
design feasibility. `REPORT.md` includes complex input Z, reflection, selected
control, violated limits, and the reference-plane cross-case checks. The runner
exits nonzero if a frozen electrical result, enumeration count, selected
hardware/control, expected infeasible-scenario set, or cross-case invariant
changes.

Publication-style circuit diagrams and result graphs, together with their
source and interpretation notes, are generated by
[`figures/generate.py`](figures/generate.py) and documented in
[`figures/README.md`](figures/README.md).

The reviewed 33-question technical report is generated by
[`reports/generate_case_report.py`](reports/generate_case_report.py). Its
audience, canonical inputs, reproducible command sequence, and committed output
are documented in [`reports/README.md`](reports/README.md).

## Scope boundary

The suite establishes:

- correct public topology wiring at the fundamental frequency;
- correct effective CCP R-L-C continuation across declared frequencies;
- correct effective ICP transformer continuation through the public input;
- complete finite Candidate and Control enumeration;
- scenario/control separation and worst-case aggregation;
- separation of electrical reachability from declared tuning reserve;
- exact replay of independent frequency points;
- component and source-terminal gating independent of match quality;
- equivalence of two correctly transformed lossy reference-plane representations;
- rejection of a deliberate fixture-double-counting example.

It does not establish:

- a qualified plasma or semiconductor-process window;
- microscopic sheath, density, chemistry, or species-power predictions;
- nonlinear harmonics or self-consistent circuit/plasma feedback;
- component temperature or vendor power qualification;
- tuner dynamics, controller stability, or global optimality outside declared
  finite candidate sets.
