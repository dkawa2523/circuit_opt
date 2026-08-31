# Examples

The files in this directory are short, user-facing `pcd.rf.v1` studies. They
show what question each load representation can answer; they are executable
documentation, not qualified chamber data.

| example | question answered | boundary |
|---|---|---|
| `rf_impedance_point_study.yaml` | which candidate pi network best matches supplied R+jX conditions? | one supplied point per row |
| `rf_impedance_frequency_table.yaml` | how does one fixed network perform at independent frequency/R+jX points? | no interpolation or broadband fit |
| `rf_component_stress.yaml` | does a selected network meet match, terminal stress, and effective-loss limits? | no temperature or lifetime prediction |
| `rf_ccp_lumped.yaml` | how does a network interact with a qualified effective CCP R-L-C one-port? | no sheath-state or species-power inference |
| `rf_icp_transformer.yaml` | how does a network interact with a qualified effective ICP coil-loading fit? | uses only the two reflected terms identifiable at the coil port |

Run any public example with one command:

```powershell
uv run pcd run examples/rf_impedance_point_study.yaml
```

The `advanced/` directory is intentionally separate. Its explicit
`case_yaml.v1` files cover custom circuits, plugins, transient probes, and a
generic waveform objective for developers extending the platform:

| advanced example | purpose |
|---|---|
| `advanced/generic_rc_filter.yaml` | minimal custom transient and waveform-objective smoke case |
| `advanced/rf_port_transient.yaml` | exact load-port current, power flow, harmonics, and settling |
| `advanced/plugin_case.yaml` | custom circuit and metric registration |

Use `bench/`—not examples—for reproducible pass/fail decision cases.
