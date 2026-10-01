# Benchmark report

The report is a projection of the completed core and literature suite
summaries. It does not reopen candidate files or reinterpret solver results.
The committed outputs are:

- [`benchmark-report.md`](../../output/reports/benchmark-report.md): readable outcome, case coverage, failed limits, and scope boundary;
- [`benchmark-summary.json`](../../output/reports/benchmark-summary.json): compact status, counts, source hashes, and artifact paths;
- [`core-cases.csv`](../../output/reports/core-cases.csv): one row per core circuit case;
- [`literature-evidence.csv`](../../output/reports/literature-evidence.csv): source, model, design, reference-only, and execution evidence rows.

Benchmark reproduction and electrical design feasibility are deliberately
separate columns. A negative-control design can therefore be electrically
infeasible while its benchmark reproduction passes.

## Reproduce

Run the two source suites, then build the report:

```powershell
uv run --frozen python bench/run_suite.py --run-root runs/benchmark_suite
uv run --frozen python bench/literature/run_suite.py --run-root runs/literature/final_evaluation
uv run --frozen python bench/reports/generate_case_report.py
```

Use `--core-result`, `--literature-result`, or `--output-dir` only when
intentionally reviewing a different snapshot. The source file hashes produce
a deterministic snapshot ID; report generation itself does not add a clock
value.

`runs/` is ignored because it contains reproducible solver artifacts. The
reviewed report, publication figures, figure provenance manifest, and PDF pack
are the curated versioned outputs.
