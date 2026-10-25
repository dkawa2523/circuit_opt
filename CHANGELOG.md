# Changelog

## 0.22.1 — 2026-10-26

- Consolidate the input, output, load-model, component-stress, architecture,
  and figure specifications into one Japanese engineering manual. Include
  complete YAML examples, measurement conventions, result interpretation,
  and reproducible numerical demonstrations.
- Replace the long implementation plan with a current roadmap; organize
  individual studies and research notes separately, and repair their links.
- Ship the same manual, generated SVG figures and runnable examples in the
  wheel under `share/pcd`, preserving their relative paths without a second copy.
- Keep numerical behavior unchanged. Retain frozen benchmark data and their
  unique provenance documents separately from the current specification.

## 0.22.0 — 2026-10-26

- Pass the exact selected periodic window through power/RMS and harmonic
  measurements. Probe spectra inherit the case's cycle count, settling policy,
  and harmonic count; persist the actual window with the numerical tables.
- Separate study export columns into `context.*`, `parameter.*`, and `signal.*`.
  This changes the combined observation CSV headers and prevents arbitrary
  probe names from replacing condition, phase, or candidate information.
- Report plotted-data coverage. Entirely unavailable curves are `not_available`,
  gaps are `partial`, and requested incomplete output exits nonzero while
  preserving the numerical table and explaining the missing data on the figure.
- Return null with a reason for undefined efficiency and normalized loss balance.
  Share the finite power-ratio calculation across AC and transient measurements;
  keep finite ratios at small power levels and reject unavailable constrained metrics.
- Replace the cycle-count-only check with closed-form window regressions and
  exercise arbitrary export aliases and unavailable transfer data without
  introducing another schema or diagnostic framework.

## 0.21.0 — 2026-10-25

- Add one optional YAML `output` block for formats, physical/instrumented
  schematics, waveforms, probe AC responses, explicit Bode ratios, and harmonics.
  Replot saved data with `analyze --case` without another solver invocation.
- Save the selected display voltage in AC as `voltage_V_re/im`, retaining
  legacy `voltage_re/im` for source-port voltage. Allow named-network probes
  and save signal catalogs with units, vectors, terminals, signs, and drive basis.
- Export shared numerical tables for plots, frequency/operating metrics,
  instantaneous power, and harmonics; preserve raw adaptive samples and
  distinguish normalized 1 V peak reference excitation in figures.
- Replace lossy main-path schematic extraction with complete branch/pin
  representation, explicit partial status for unresolved topology, and real
  current-meter symbols. Fix source-symbol endpoint orientation and polarity.
- Add selected-condition, objective/margin, fit/holdout residual, and local
  sensitivity views, plus streaming observation exports retaining evaluation context.
- Separate numerical success from requested output completion; expose solver
  vector failures at the public error field and return nonzero for partial output.
- Remove obsolete path-layout code and tests tied to those private helpers;
  consolidate current output documentation and preserve numerical regressions.

## 0.20.0 — 2026-10-25

- Extend `pcd.rf.v1` to component circuits, external SPICE, continuous sizing,
  and compact terminal identification through the existing execution engine.
- Group fixed/calibration/operating/design/control/latent parameters by owner;
  reject tolerance ranges placed on fixed inputs and ambiguous scaled unit metadata.
- Resolve explicit unit literals to SI once, with known-field/annotation dimension
  checks. Keep raw SPICE syntax intact and make wave-reference Z0 configurable.
- Accept inline impedance points and optional row IDs alongside CSV, preserving
  explicit rows and pinned control settings without implicit Cartesian products.
- Make `sim-run` and the supported Python API require explicit point selection
  from multi-condition/candidate/control input; label netlist previews and saved
  simulation scope. Add `validate-case --plan` using the shared resolved case.
- Observe unambiguous imported voltage-source ports, suppress false missing-source
  warnings, and reject duplicate preserved sources. Ambiguous imports require
  explicit observations rather than guessed topology.
- Compile standard complex V/I observations and explicit partition columns into
  fit/holdout input. Declare error scales once and archive original input data
  before phase splitting. Move known-truth acceptance to benchmarks.
- Add minimal imported/fixed-transient/design examples, promote the compact fit
  example, and remove the obsolete verbose fit input and duplicate CSV examples.
- Reject engineering-acceptance gates with no declared engineering limits.
- Replace misleading `best.search_completeness`/`search_limitation` output
  with `evaluation_evidence`/`evidence_limitation`, and print optimizer
  termination separately from evidence completeness and engineering acceptance.

## 0.19.0 — 2026-10-25

- Reject voltage/source columns used as load-current bindings; require a declared
  current vector or the inserted load ammeter. Record the exact vector and
  whether its direction is supplied by the user. Keep source voltage at the
  selected source port and reject current vectors in voltage bindings.
- Check parameter-domain fields at the shared boundary, including inline design
  and control specifications, so misspelled bounds cannot become a fixed search.
- Preserve observed controls across identification partitions. Separate unseen
  conditions from explicit same-condition replicates and report duplicate V/I.
- Save unavailable fit, holdout, and sensitivity evidence through one result path.
  Treat open predicted impedance as unavailable terminal-fit evidence.
- Render nullable open-circuit measurements consistently in text and JSON.
- Preserve earlier analyses and record analyzer/environment/options/input hashes
  separately from original simulation provenance.
- Remove fit-only failure serialization, redundant partition checks, and three
  private-helper-only tests; retain numerical and user-workflow regressions.
- Use an electrically accepted role-factorial case for the wheel CI acceptance
  check; a topology conformance golden intentionally violates its matching limit.

## 0.18.0 — 2026-10-25

- Reject duplicate YAML/JSON fields, unused owned settings, conflicting aliases,
  missing parameter references, and fractional execution budgets.
- Preserve declared scenario-table defaults and weights. Pin observed controls
  without overwriting them or manufacturing tuning headroom.
- Decide complete acceptance from individual conditions, independently of
  floating-point rounding in weighted coverage.
- Separate display voltage, differential load voltage, and source-port impedance
  in canonical observations, measurements, figures, and saved metadata.
- Separate AC marker frequency, periodic fundamental, and AC excitation source.
  Require explicit voltage vectors for raw/imported sources.
- Render current sources with current units/names and measure actual AC current
  through a series ammeter; preserve numeric input precision in SPICE.
- Validate successful adapter responses against requested time/frequency domains
  and correctly execute ngspice's two-point linear-sweep edge case.
- Align terminal least squares with declared weighted RMS and persist optimizer
  termination separately from acceptance. Save unavailable fits without estimates.
- Retry bounded Windows file-sharing conflicts during atomic result publication.
- Refresh editable-install metadata when the dynamic version source or README
  changes; verify imported and distribution versions agree.
- Remove the residual cyclomatic-complexity gate and replace obsolete formatting
  assumptions in tests with numeric and workflow assertions.
- Analysis metadata now uses `load_reference_plane`, `impedance_reference_plane`,
  `ac_frequency_Hz`, and `periodic_fundamental_Hz` in place of ambiguous plane/frequency fields.

## 0.17.0 — 2026-10-25

- Preserve native adaptive samples in periodic measurements and cycle comparisons;
  prevent narrow pulses from disappearing from stress and power decisions.
- Share complete AC and periodic operating-point measurements between studies
  and saved-run reports; calculate ratios after complex interpolation and use
  the declared reference impedance for wave power.
- Share authored CSV structure checks across load tables, scenarios, profiles,
  and target waveforms; reject duplicate headers and ambiguous target data.
- Normalize weights before coverage, mean, and CVaR aggregation, including very
  small and very large common weight scales.
- Add bounded complex-terminal least squares through the same study evaluator;
  count finite-difference proposals against the candidate budget and retain
  feasibility-first selection and fresh final replay.
- Apply an absolute floor to normalized local sensitivity rank, state its local
  scope, require fixed observed controls, and isolate repeated identification
  invocations so previous evidence remains stable.
- Stream flat evaluation rows on disk and retain detailed trials only for the
  selected candidate and observed Pareto front.
- Allocate simulation and identification directories atomically.
- Keep AC load voltage tied to the load port when a different transient output
  node is selected, and retain optional missing source channels in periodic data.
- Record Python, platform, and numerical-library versions with result provenance.
- Keep time integration compatible with the declared NumPy 1.23 lower bound.
- Reduce quality checks to Ruff, Pyrefly, import boundaries, and meaningful
  behavior tests; remove redundant diagnostic dependencies and percentage gates.
- Check built wheels outside the checkout, including minimum dependency minors.
- Replace stale workflow/planning text and remove obsolete build/cache files
  and retired documentation. Historical publication evidence remains explicitly
  labeled as such; pending local export cleanup is recorded separately.

## 0.16.0

- Distinguish complete SPICE decks from title-less circuit fragments and replace
  imported sources by exact name.
- Keep fixed hardware, external conditions, and per-condition controls separate.
- Preserve unavailable current as missing data and require declared RF cycles.
- Expose the `pcd.api` extension surface with detached plugin inputs.
- Record dataset identities and runtime/solver fingerprints.
- Resolve saved-run artifacts centrally and provide preview-first generation
  retention without deleting reusable raw solves.
- Add the repository-local `pcd-runner` skill.

Intermediate development experiments and retired interfaces are available in
Git history. Current behavior is described by README and the input reference.
