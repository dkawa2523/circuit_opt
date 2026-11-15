# Measured two-port transfer across VNA acquisitions

This benchmark fits a passive equivalent to one acquisition of an on-wafer 10 dB
attenuator and predicts complex transmission on a second acquisition of the same
device. The fitted equivalent represents attenuation and propagation delay. Its
remaining reflection error bounds where the equivalent is useful.

## Source and partition

The source is version 1 of the NPL, MPI and Anritsu dataset from Ausden, Ridler,
Rumiantsev, Martens and Shang, licensed under CC BY 4.0. Both archived Touchstone
files contain multiline-TRL calibrated 50 Ω S-parameters. The repository keeps
two files byte-for-byte, with local names identifying the measurement system.
The publisher's `data.zip` MD5 is `3580300746156d05dbc5de8957f0b911`.
The runner verifies each archived file's SHA-256, row count and frequency grid.
[Dataset and license](https://zenodo.org/records/22658069) ·
[Measurement paper](https://zenodo.org/records/17425562) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Archived file | Original path in `data/Main Results/` | Usage |
|---|---|---|
| `source/broadband.s2p` | `Broadband Single-Sweep/10dB_Attenuator_0603__CORR.S2P` | Fit 41 rows at 2.5–12.5 GHz. Ignore the remaining 830 rows. |
| `source/coaxial.s2p` | `Coaxial/10dB_Attenuator_0603__CORR.S2P` | Hold out all 191 rows. Report 41 overlapping-frequency rows and 150 rows above the fit band, through 50 GHz. |

Both systems measured the same device in the same laboratory. This is validation
across acquisitions and across the declared frequency extension for that device.

## Circuit and comparison

- Take the mean magnitude of the training acquisition's S21 and S12 for the
  attenuation. Fit one nonnegative delay to their unwrapped phase against
  angular frequency, through zero phase at DC. Keep both fitted values fixed
  for every holdout row.
- Use a matched resistive T-pad at the fitted attenuation for both models. The
  delay model adds a matched, lossless SPICE transmission line. The resistive
  baseline uses the same attenuation and has zero propagation delay.
- Use the same DUT and 50 Ω terminations in both excitation states. The public
  `run_network` workflow saves both port voltages and currents and solves the
  full complex `B = S A` system in the declared 50 Ω power-wave references.
  The benchmark compares S11, S21, S12 and S22 at every measured frequency.
- Check the saved public Bode plot data against the complex wave calculation.
  The ratio displayed by the Bode recipe uses the generator voltage, which is
  twice the incident voltage for this matched source. The benchmark applies
  that known normalization when comparing the two.

The authored regression limits are 0.10 dB RMSE for transmission magnitude,
0.005 complex RMSE for transmission and 0.04 complex RMSE for reflection. Errors
are pooled across the two excitation directions within each frequency band.
These are explicit benchmark targets. Instrument uncertainty and hardware
acceptance require separate specifications.

| Coaxial holdout region | Pad with delay, transfer complex RMSE | Resistive pad, transfer complex RMSE | Pad with delay, reflection complex RMSE |
|---|---:|---:|---:|
| 2.5–12.5 GHz | 0.000745 | 0.0154 | 0.0607 |
| >12.5–50 GHz | 0.00297 | 0.0626 | 0.0758 |

Both models meet the magnitude target. The delay model passes the complex
transmission target in both bands and misses the reflection target. The
resistive model fails the complex transmission target in the extension band.
The benchmark succeeds when it reproduces these distinctions. It exercises
matched-port wave interpretation and output reporting relevant to RF fixtures.
Plasma-state inference requires other evidence.

## Reproduce

```powershell
uv run --frozen python bench/literature/p2_ausden2025_two_port/run.py --run-root runs/measured_two_port
```

The runner requires a fresh output directory. It saves `benchmark_result.json`,
`comparison.csv`, `comparison.png`, the two complete network results, and the
four PCD simulation and Bode records. The figure separates transmission
magnitude, phase and reflection, with a vertical line at the fit-band boundary.
