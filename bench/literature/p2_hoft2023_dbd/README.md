# Höft 2023 dielectric barrier discharge transfer

The published experiment varies the drive amplitude of a 10 kHz dielectric
barrier discharge in virtually pure nitrogen. The two conditions share a
1 mm electrode gap and a nominal pressure of 1 bar. Each published electrical
curve averages 1,000 HV periods. The electrical current is the total current
through the cell, including displacement current.

| Role | Published condition | Independent optical observation |
|---|---|---|
| Fit | Figure 6, 12.6 kV peak-to-peak | Diffuse emission |
| Holdout | Figure 8(a), 14.0 kV peak-to-peak | Overlay of diffuse discharge and filament emission in phase bin VI |

The 12.6 kVpp current is the only response used to estimate R and C by least
squares in `I = V/R + C dV/dt`. Each drive is reconstructed from its own
voltage trace as a 10 kHz sinusoid with fitted amplitude and phase. The
14.0 kVpp current enters only the prediction comparison. The reconstruction
keeps the derivative free of pixel-scale steps and has about 4% relative L2
error on the digitized voltage.

The ngspice replay uses the same frozen R and C for both conditions. Its
current is checked against the circuit equation before comparing it with the
published traces. The diffuse relative L2 error is about 5.9%. On the held-out
hybrid acquisition it rises to about 27%. The predicted current in phase bin
VI (50 to 60 µs) is about 57 µA less negative than even the upper edge of the
published red trace near 58.2 µs. The benchmark passes when it reproduces this
observed failure of the fitted terminal model. The 40 µA benchmark threshold
is a regression margin for this particular trace and circuit comparison.

As a drive sensitivity check, an analytic prediction uses the same R and C
with DC and the first three harmonics fitted to the 14.0 kVpp voltage alone.
This lowers the drive error to about 2% and leaves about 56 µA of current
underestimate beyond the ink span. The missed excursion persists for both
drive reconstructions.

The ICCD images accumulate 1,000 exposures per 10 µs phase bin. Figure 8(a)
overlays the diffuse discharge and filament emission in the averaged window.
Figure 12(a) shows the individual filament morphology, and Figure 13 shows
streak recordings that distinguish hybrid cycles from diffuse cycles. The
paper reports that filaments occur roughly every fifth period in this gas.
The averaged benchmark has phase-bin resolution for optical comparison.
Individual pulse amplitudes, per-shot electrical and optical alignment, and
plasma parameters such as electron density remain outside its evidence.
The fitted values are effective circuit parameters at the declared terminal.

## Run

```powershell
uv run --frozen python bench/literature/p2_hoft2023_dbd/run.py --run-root runs/literature/hoft2023_dbd
```

The run directory holds the actual ngspice cases, both aligned comparison
traces and the analytic sensitivity trace, a plot with the optical phase bin
marked, and `benchmark_result.json`.

## Source and extraction

`digitized.csv` adapts Figures 6 and 8(a) of Hans Höft, Markus M. Becker,
Manfred Kettlitz, Simon Dap, Nicolas Naudé, Ronny Brandenburg and Klaus-Dieter
Weltmann, *Exploring the mechanisms leading to diffuse and filamentary modes
in dielectric barrier discharges in N₂ with N₂O admixtures*, European Physical
Journal D 77, 35 (2023). © The authors 2023. [Article and attribution](https://doi.org/10.1140/epjd/s10053-023-00601-z),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The article figures supply the digitized trace used in this benchmark. The
authors list numerical files in the [INPTDAT deposit](https://doi.org/10.34711/inptdat.667).

The extraction calibrates each axis by the figure ticks, removes covered
columns at the optical markers and the right-axis ticks, and takes the center
of the red stroke with its endpoints plus one pixel. This span includes the
visible trace thickness and the vertical extent of steep trace segments in a
column. It describes plot-reading resolution only. The hybrid panel has about
0.10 µs and 3.8 µA per pixel, and its voltage axis has about 79 V per pixel.
CSV decimal places make the extraction reproducible.

The embedded raster bytes from the publisher and TIB copies agree. Regenerate
the CSV into a new file and compare with the committed fixture using either
PDF. The optional dependency is used only for source extraction.

```powershell
uv run --with pymupdf python bench/literature/p2_hoft2023_dbd/extract.py article.pdf --output hoft_digitized.csv
Get-FileHash hoft_digitized.csv, bench/literature/p2_hoft2023_dbd/digitized.csv
```

[Publisher PDF](https://link.springer.com/content/pdf/10.1140/epjd/s10053-023-00601-z.pdf),
[TIB repository PDF](https://oa.tib.eu/renate/server/api/core/bitstreams/87a157bf-3800-429e-a158-43688a69e4a2/content).
