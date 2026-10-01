# P1 — Lee et al. (2020) ICP equation benchmark

This benchmark checks the transformer reduction in Eqs. 18 and 19 of:

> J. J. Lee, S. J. Kim, K. K. Kim, Y. S. Lee, and S. J. You,
> “A simple model of solenoidal inductively coupled plasma sources considering
> finite size,” *AIP Advances* 10, 035008 (2020),
> <https://doi.org/10.1063/1.5133862> (CC BY 4.0).

## Evidence class and responsibility

`cases.csv` contains **paper-equation algebra cases**, not measured chamber
data and not digitized values from Figs. 5–7.  The paper gives curves rather
than a machine-readable table of the density-dependent `Rp`, `Lp`, and `M`
values.  Inventing precise experimental golden values from those plots would
overstate the available evidence.

The paper's geometry, electron-density, field, and collision calculations stay
outside the platform's public input.  They may prepare terminal parameters, but
this benchmark starts at the circuit boundary.  It therefore keeps normal PCD
input simple and checks only the circuit reduction needed by this repository.

## Mapping under test

Let

```text
Ls     = Lp + Le
Lref   = M^2 / Ls
gamma  = Rp / Ls
Le     = Rp / nu
```

In the passive input-port convention consistent with the positive absorbed
power in Eq. 19, Eq. 18 is evaluated as

```text
Z_RF = j*w*Lc + w^2*M^2 / (Rp + j*w*Ls)
```

and the plasma power for **peak** coil-current amplitude is

```text
Pabs = 0.5 * I_RF_peak^2 * Re(Z_RF)
```

The `0.5` factor would not be used for RMS current.  This explicit convention
prevents an otherwise easy factor-of-two error.

The benchmark compares three independently reached results:

1. a direct rectangular-form implementation of Eqs. 18 and 19 in the runner;
2. `pcd.rf_loads.icp_effective_impedance` after the `Lref`, `gamma` mapping;
3. when installed, an independent NGSpice transformer built directly from the
   paper elements `Lc`, `Ls`, `Rp`, and `M`.

The ideal paper circuit has zero coil resistance.  A literal zero-ohm
`Rcoil` is internally replaced by NGSpice and makes the small real part poorly
conditioned.  The solver fixture therefore uses a known `1e-6 ohm` series
regularization and subtracts it from reported impedance.  This is numerical
fixture handling, not a new plasma parameter or a public-input requirement.
NGSpice absorbed power is independently obtained from the paper transformer's
secondary-resistor current, rather than from the small terminal real part.

The NGSpice netlist deliberately does not call PCD's netlist builder.  Otherwise
the solver comparison could repeat the implementation under test and pass for
the same mistake.  PCD's public analytic implementation is the second path;
the paper-topology NGSpice circuit is the third path.

It also checks passivity, the negative plasma reactive correction, the mutual
inductance energy bound, reciprocal damping-ratio symmetry, and coverage of
`gamma/w = 0.01, 0.1, 1, 10, 100`.  A second frequency prevents the case set
from being a single-frequency identity only.

`gamma` is an identifiable secondary damping rate.  It is **not** generally the
electron-neutral collision rate `nu`; this benchmark reports both so an
accidental `gamma = nu` implementation is visible.

## Run

From the repository root:

```powershell
uv run python bench/literature/p1_lee2020_icp/run_benchmark.py --require-ngspice
```

For the equation and Python implementation only:

```powershell
uv run python bench/literature/p1_lee2020_icp/run_benchmark.py --no-ngspice
```

Use `--json` for machine-readable stdout or `--output path.json` to save a full
report.  Temporary YAML, netlists, and solver artifacts are deleted after each
run.

## Acceptance

| Comparison | Relative tolerance | Absolute tolerance |
|---|---:|---:|
| Eq. 18/19 vs PCD Python | `1e-12` | `1e-12 ohm` |
| Eq. 18 vs NGSpice complex-vector norm | `2e-5` | `1e-7 ohm` |
| Eq. 19 power vs NGSpice-derived power | `5e-5` | `1e-10 W` |

The Python tolerance tests algebra and floating-point implementation.  The
looser NGSpice tolerance follows the existing physical-solver integration
criterion and allows solver/output precision without weakening sign or
passivity checks.

## What passing does not mean

A pass does not validate the paper's finite-geometry plasma calculation,
electron-density inference, a semiconductor chamber measurement, high-density
or low-density plot values, or a matching network.  It establishes that once
`Rp`, `Lp`, `Le`, and `M` have been supplied by a qualified external source,
the platform preserves the paper's terminal impedance and absorbed-power
algebra across the tested regimes.
