# Architecture and ownership

PCD has one numerical path for public RF studies and advanced explicit cases:

```text
authored YAML/JSON
        |
        v
case.py -> plan.py (pcd.rf.v1 only) -> executable Case
        |                                  |
        |                                  v
        |                         study_config.py
        |                                  |
        v                                  v
netlist.py -> solver.py -> sim_core.py -> core/StudyRunner
                         |                    |
                         v                    v
                    records.py          metrics.py
                         \___________________/
                                  |
                                  v
                    results/ + evaluations.csv
```

The arrows show data flow, not every Python import. Public RF input is compiled
once; simulation, caching, metrics, and reporting consume the same resolved
case rather than interpreting shorthand independently.

## Directory responsibilities

| path | owns | does not own |
|---|---|---|
| `pcd/core/` | immutable study roles, evaluation order, feasibility-first aggregation | circuits, ngspice, search algorithms |
| `pcd/signals/` | cleaned numerical series, periodic windows, phasors, time-weighted power | case schemas or engineering limits |
| `pcd/` | application planning, circuits, execution, measurements, persistence | apparatus qualification data |
| `examples/` | runnable syntax and extension examples | regression truth |
| `tests/` | software, numerical, and boundary regression checks | scientific qualification claims |
| `bench/` | reproducible electrical and literature-derived evidence | runtime package behavior |
| `docs/` | supported inputs, interpretation, and model limits | generated run artifacts |
| `runs/` | ignored, reproducible execution/cache data | reviewed publication outputs |
| `output/` | reviewed, versioned report and figure products | mutable run caches |
| `quality/` | the deliberate coverage floor | custom test frameworks |

## Module boundaries

- `case.py` loads one schema and owns path resolution. `plan.py` is the only
  public-RF shorthand compiler.
- `study_config.py` translates an executable case into Candidate, Scenario,
  ControlState, Objective, and Constraint objects.
- `netlist.py` builds the in-memory circuit and renders deterministic ngspice
  text. `netlist_import.py` preserves executable external statements;
  `netlist_parse.py` intentionally extracts only the topology needed to draw.
- `solver.py` executes a solver and returns arrays. `sim_core.py` owns run
  directories, manifests, and failure records. `records.py` is the read-only
  artifact boundary.
- `analysis.py` plans probes and converts AC/transient outputs into electrical
  quantities. `metrics.py` selects a named metric and evaluates engineering
  limits.
- `results/` owns study generations and caches. The root `study_result.json`
  is the commit pointer written only after a complete generation exists.

## Supported extension boundary

Plugins and external Python integrations import from `pcd.api`. A circuit
builder returns `Circuit`, a solver returns `SimulationResult`, a metric returns
a non-empty mapping, and an optimizer derives from `BaseOptimizer`. Extension
calls receive detached case/parameter data so they cannot mutate the cached
study definition accidentally.

The advanced `case_yaml.v1` schema remains an explicit integration format; the
smaller `pcd.rf.v1` schema is the stable user format for RF matching studies.
Adding a new plasma-equivalent model belongs in the load-model boundary only
when its terminal equation, parameters, reference plane, and applicability
evidence are well defined.
