# Publication figure system

## Purpose

The figure layer turns a frozen circuit-study result into diagrams that a
third party can read without knowing the implementation. It does not perform
simulation and it does not infer plasma physics. Benchmark-specific questions
and data extraction remain in `bench/figures`; the reusable geometry and visual
rules live in `pcd/figures`.

```text
frozen case + candidate + ngspice artifacts
                    |
                    v
        benchmark-specific extraction
                    |
                    v
 pcd.figures publication style + circuit geometry
                    |
                    v
 fixed-size SVG + 300 dpi PNG + multipage PDF
```

## Renderer decision

Schemdraw 0.23 remains the primary circuit renderer. Its official placement
API provides element anchors and exact endpoints, and it renders directly into
the same Matplotlib axes used for benchmark plots. See the
[Schemdraw placement documentation](https://schemdraw.readthedocs.io/en/stable/usage/placement.html)
and [style documentation](https://schemdraw.readthedocs.io/en/stable/usage/styles.html).

CircuiTikZ is a strong optional target for a future LaTeX-native paper, but it
would add a TeX toolchain and a second layout/font pipeline to the current
Python report. Lcapy generates CircuiTikZ using semi-automatic placement and
its own documentation notes that component orientations and sizing hints are
still required. Neither replaces the need for an explicit layout contract.
See the [CircuiTikZ project](https://circuitikz.github.io/circuitikz/) and
[Lcapy schematic documentation](https://github.com/mph-/lcapy/blob/master/doc/schematics.rst).

KiCad and SKiDL remain appropriate for production schematics, ERC, PCB, and
manufacturing workflows. They are intentionally outside this explanatory
one-port and benchmark-figure responsibility. Graphviz may be used for a
workflow graph, but not as an electrical-symbol renderer.

## Geometry contract

`CircuitDiagram` enforces the following rules:

1. Every logical connection point has a named anchor. Redeclaring an anchor at
   a different coordinate is an error.
2. Every R/L/C or two-terminal block has one fixed endpoint length. A long
   layout span is filled by wire before and after the symbol; the symbol itself
   is never stretched to absorb whitespace.
3. Component spans are orthogonal. A diagonal or shorter-than-symbol span is an
   error.
4. Comparable panels use the same explicit `CircuitViewport`.
5. `finish()` applies an equal x/y physical scale. A vertical component and a
   horizontal component therefore have the same physical length.
6. Junctions with three or more branches are explicit dots; port and ground
   symbols come from Schemdraw rather than separate hand-drawn variants.
7. IEEE symbol style, line width, font, and component length are set once in
   `pcd/figures`.

These rules address the main failure modes of direct `.to((x, y))` drawing:
Schemdraw two-terminal elements stretch to requested endpoints, and a
Matplotlib axis with unequal x/y scale makes vertical symbols appear larger
than horizontal ones.

## Page and export contract

- Every figure starts at `PAGE_SIZE = 7.25 x 4.8 inch`.
- SVG, PNG, and PDF are exported from that same canvas; no post-layout resize
  is allowed.
- Publication PNG is always `2175 x 1440 px` at 300 dpi.
- Production export does not use `bbox_inches="tight"`; content cannot change
  the page size.
- SVG keeps text editable and PDF uses TrueType font embedding.
- `figure_data.json` records Python, Matplotlib, Schemdraw, layout schema,
  symbol standard, page size, and source hashes.
- Every visible text box is checked against the fixed page before export.

The lock file is the reproducible environment. The package dependency also
keeps Schemdraw within the verified 0.23 series so a new placement contract is
not accepted silently.

## AC waveform convention

An AC result is a complex steady-state fundamental, not a transient waveform.
When a figure needs one RF cycle, it is reconstructed only for display as

```text
x(theta) = Re{X exp(j theta)}
```

where `X` is the archived peak phasor. The figure must say “reconstructed from
AC phasors” and must not claim switching transients, harmonics, ignition,
ringing, or settling. Source current is sign-corrected because ngspice reports
current into the positive terminal of a voltage source. Peak-phasor real power
is checked with

```text
P = 0.5 Re{V conj(I)}
```

New AC simulations always save the declared output voltage in addition to
source voltage/current. Optional load current remains available when the case
requests it. Derived cycle samples are never written back to `waveform.csv`, so
transient evidence and phasor reconstruction stay distinguishable.

## Minimal QA

The test suite checks geometry rather than image pixels:

- fixed symbol length across different layout spans;
- equal horizontal and vertical physical scale;
- anchor immutability and invalid-span rejection;
- exact fixed-canvas PNG dimensions;
- figure text containment;
- phasor-derived B5 source/load power against archived metrics.

Pixel-perfect golden images are intentionally omitted because font
rasterization varies by platform and does not test electrical meaning.

## Adding a figure

1. State the engineering question and the one defensible reading.
2. Read only frozen artifacts and record their hashes.
3. Use a dot/bar/matrix for a small set of independent conditions; do not draw
   a continuous line unless the solver actually produced an ordered sweep.
4. Use `CircuitDiagram` rather than direct Schemdraw calls for circuit symbols.
5. Include units, reference plane, solver/analysis mode, and scope limitation in
   title/subtitle/footer.
6. Generate all three formats, inspect the PDF-rendered page, and run the
   geometry and full test suites.
