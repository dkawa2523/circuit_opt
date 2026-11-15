# Measured common-mode impedance at two fixture planes

This benchmark uses Marc Kopf's [published VNA data, version 1](https://doi.org/10.5281/zenodo.10016478) and the fixture description in the [Kopf and Roc'h study](https://doi.org/10.1109/EMCEurope57790.2023.10274348). The two files under `source/` are redistributed unchanged under CC BY 4.0. The runner verifies their SHA-256 checksums before use. Zenodo publishes these MD5 digests for the source files.

| File | MD5 |
|---|---|
| `scenario1_phantomside_30k_3G_V2.s1p` | `63bb3ef7acabdf81a0691b83f09fc874` |
| `SCENARIO1_pulserside_30k_3G.s1p` | `fb40409703aad117bc0867821c63e3a5` |

The two common-mode measurements use different calibrated SMA fixture planes. In scenario 1, the cable is attached to a conductive plane, and the pulser-side measurement uses a fixture that represents part of the chassis. Each source provides 10,001 S11 observations from 30 kHz to 3 GHz, with magnitude in dB, phase in degrees and a 50 Ω reference. The two curves therefore describe two fixture measurements. They cannot be used interchangeably as the plasma gap impedance or as two ports of one measured network.

## Checks

1. Read six source frequencies closest to 3, 5, 8, 13.56, 20 and 27.12 MHz at each plane. Each passive `R+jX` point is represented by a series equivalent valid at that exact frequency. The reference comparison uses the calibrated S11 and the independent relation `Z = 50 (1 + S11)/(1 - S11)`.
2. Run AC studies at each plane with fresh selected-candidate verification. Pass 50 Ω and 75 Ω through the public case setting in separate studies at **the same plane**. Compare complex impedance and reflection from all saved SPICE phasors with the original measurement and its direct 75 Ω transformation. Separately verify that the reference and reflection magnitude reported by the configured study match the expected values for every point.
3. Require the unchanged 50 Ω coefficient to disagree with the 75 Ω reference at one or more replayed frequencies. This is a negative control for conflating the reference impedance and the electrical plane. The raw source is retained, including samples with `|S11| > 1`. Such points are counted and the selected replay points must all admit a passive equivalent.

The checks evaluate conversion, sign conventions and exact-frequency replay. Each point uses its own measured impedance. This workflow makes no broadband, cross-plane, operating-plasma, or discharge-chemistry prediction. The 75 Ω comparison changes the calculation reference and does not add a 75 Ω resistor to the circuit.

## Reproduce

```powershell
uv run --frozen python bench/literature/p2_kopf2023_common_mode/run.py --run-root runs/common_mode_vna
```

`comparison.csv` lists the measured frequency, impedance and both reflection references with the replay errors. `comparison.png` compares the two planes and distinguishes the measured sweep from the replay markers. `benchmark_result.json` records the source hashes, passive-disk coverage, tolerances' outcomes and the two study roots. The output root must be new.
