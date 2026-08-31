# Benchmark case report

`benchmark-case-report-artifact.json` is the durable, third-party-readable
summary of the 16 core and 17 literature-derived validation questions. It
keeps benchmark reproduction separate from engineering feasibility and states
which apparatus, process, thermal, and lifetime claims are not evaluated.

The committed report is:

- [`output/reports/benchmark-case-report-artifact.json`](../../output/reports/benchmark-case-report-artifact.json)

## Reproduce

Run the two source suites, then build the report:

```powershell
uv run python bench/run_suite.py --run-root runs/benchmark_suite
uv run python bench/literature/run_suite.py --run-root runs/literature/final_evaluation
uv run python bench/reports/generate_case_report.py
```

The report generator discovers the B5 candidate detail from the completed core
suite; callers do not need to know a content-addressed study directory. Use
`--core-result`, `--literature-result`, `--b5-candidate-result`, or `--output`
only when intentionally reviewing a different snapshot.

`runs/` is ignored because it contains reproducible solver artifacts. The
reviewed report, publication figures, figure provenance manifest, and PDF pack
are the curated versioned outputs.
