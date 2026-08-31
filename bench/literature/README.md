# Literature benchmarks

This suite keeps three questions separate:

1. **Source fidelity** — can a published table, equation, or qualified plot
   transcription be reproduced without changing its reference plane?
2. **Model conformance** — does the implemented terminal model reproduce the
   declared equation?
3. **Platform design challenge** — what decision does one explicitly declared
   PCD matcher fixture make when those loads are supplied?

A suite PASS means the source/equation checks and expected decision
classifications were reproduced. An infeasible matcher is a valid design
outcome, not a benchmark failure. No result qualifies an unspecified
production chamber.

Run from the repository root:

```powershell
uv run python bench/literature/run_suite.py --run-root runs/literature/final_evaluation
```

All generated JSON, reports, logs, netlists, and intermediate inputs are kept
under the selected run root. The committed literature tables are immutable
source or derived views and are checked, not regenerated in place.

## Executable evidence

| class | source | narrow executable claim |
|---|---|---|
| source fidelity | [Lee 2021](https://doi.org/10.1063/6.0000883) | Table-I bias-path circuit closure at the corrected plasma terminal |
| source fidelity | [Hargis 1994](https://doi.org/10.1063/1.1144770) | 32 V/I/phase rows, derived views, power closure and one-port replay |
| model conformance | [Lee 2020](https://doi.org/10.1063/1.5133862) | ICP transformer Eqs. 18/19 over synthetic damping regimes |
| source fidelity | [Colpo 1999](https://doi.org/10.1063/1.369268) | fixture resonances, graphite dummy and 15 paired digitized centers |
| design challenge | Hargis / Sobolewski | control authority and separately reported operating, apparatus-spread and phase-model families |
| design challenge | Colpo digitization | fixed and bounded fixtures over four reading corners per 15 parent conditions |

Hargis's 24/34 MHz labels are empty-cell apparatus groups; every load is
driven at 13.56 MHz. Central operating conditions, reported apparatus spread,
and common-mode phase sensitivities are never flattened into one weighted
score. In particular, the suite produces no 80-row winner.

The Colpo `+/-6 ohm` R and `+/-7 ohm` X widths describe digitization and
marker-center ambiguity only. The resulting 60 corners are outcomes within 15
pressure-power parent conditions, not 60 independent physical observations.

## Scope boundary

The following remain references, rather than unfinished goldens:

- Cao 2020 lacks phase-resolved complex terminal impedance.
- Metze 1986 and Saikia 2018 require nonlinear sheath/self-bias dynamics.
- Howling/Guittienne require distributed antenna-mode coupling.
- Qu 2020 requires time-varying pulsed matching.
- Lafleur 2013 is a probe-calibration method, not a reusable load table.

The public `pcd.rf.v1` input remains unchanged. Paper-specific provenance,
digitization, and sensitivity construction stay inside this directory. The
10% reflected-power limit and matcher grids are platform fixtures, not limits
or hardware published by the cited papers.
