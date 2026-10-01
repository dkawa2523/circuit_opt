# Circuit benchmark report

Snapshot: `1ffcfe72e2aa8c73`

## Outcome

| check | result |
|---|---:|
| overall benchmark reproduction | yes |
| core suite | yes |
| literature suite | yes |
| literature integrity | yes |
| core cases reproduced | 16/16 |
| core electrical evaluations | 411 |

Benchmark reproduction and engineering feasibility are separate columns below. An intentionally infeasible
electrical design can be a successful negative-control benchmark.

## Core circuit cases

| ID | role | reproduced | design feasible | condition coverage | evaluations | worst Γ magnitude | failed limits |
|---|---|---:|---:|---:|---:|---:|---|
| A1_topology_l_match | engine_conformance | yes | no | 0/1 | 1 | 0.4849 | max_reflection_magnitude |
| A2_topology_pi_match | engine_conformance | yes | yes | 1/1 | 1 | 0.259 | none |
| A3_topology_pi_match_harmonic | engine_conformance | yes | no | 0/1 | 1 | 0.3871 | max_reflection_magnitude |
| A4_ccp_lumped_frequency_conformance | model_conformance | yes | no | 0/3 | 3 | 0.8458 | max_reflection_magnitude |
| A5_icp_transformer_frequency_conformance | model_conformance | yes | no | 1/3 | 3 | 0.9999 | max_reflection_magnitude |
| B1_fixed_nominal | negative_control | yes | no | 1/5 | 5 | 0.7672 | max_reflection_magnitude |
| B2_limited_tuner | negative_control | yes | no | 2/5 | 125 | 0.302 | min_control_margin |
| B3_full_tuner | positive_control | yes | yes | 5/5 | 245 | 0.03884 | none |
| B4_independent_frequency_points | negative_control | yes | no | 1/3 | 3 | 0.8203 | max_reflection_magnitude |
| B5_high_drive_stress | negative_control | yes | no | 1/2 | 2 | 0.01104 | max_component_L1_current_rms_A;max_component_L1_loss_W;max_source_apparent_power_VA;max_source_current_rms_A |
| B6_discrete_hardware_search | positive_control | yes | yes | 1/1 | 3 | 0.001022 | none |
| B7_role_factorial_search | positive_control | yes | yes | 2/2 | 8 | 1.583e-12 | none |
| B8_component_value_corner_stress | negative_control | yes | no | 3/8 | 8 | 0.3725 | max_reflection_magnitude |
| D1_reference_plane_explicit | boundary_conformance | yes | yes | 1/1 | 1 | 0.0383 | none |
| D2_reference_plane_embedded | boundary_conformance | yes | yes | 1/1 | 1 | 0.0383 | none |
| D3_reference_plane_double_counted | negative_control | yes | no | 0/1 | 1 | 0.3418 | max_reflection_magnitude |

## Literature-derived evidence

| category | ID/source | reproduced | design feasible | condition coverage | worst Γ magnitude |
|---|---|---:|---:|---:|---:|
| source_fidelity | lee2021_bias_table_i | yes | n/a | n/a | n/a |
| source_fidelity | hargis1994_tables_iii_iv | yes | n/a | n/a | n/a |
| source_fidelity | colpo1999_fixture_and_graphite | yes | n/a | n/a | n/a |
| source_fidelity | colpo1999_digitized_centers | yes | n/a | n/a | n/a |
| source_fidelity | colpo1999_digitization_derivation | yes | n/a | n/a | n/a |
| model_conformance | lee2020_icp_transformer_equations | yes | n/a | n/a | n/a |
| design_challenges | literature_p0_lee2021_matcher_output_probe | yes | yes | 1/1 | 5.035e-10 |
| design_challenges | literature_p0_lee2021_post_coax_plane_sensitivity | yes | no | 0/1 | 0.8145 |
| design_challenges | literature_p0_lee2021_plasma_terminal_plane_sensitivity | yes | no | 0/1 | 0.7248 |
| design_challenges | P1_CCP_fixed_central | yes | no | 7/8 | 0.525 |
| design_challenges | P1_CCP_limited_central | yes | no | 7/8 | 0.4875 |
| design_challenges | P1_CCP_full_central | yes | yes | 8/8 | 0.08035 |
| design_challenges | P1_CCP_reported_spread | yes | no | 30/32 | 0.5281 |
| design_challenges | P1_CCP_phase_minus6 | yes | no | 7/8 | 0.3658 |
| design_challenges | P1_CCP_phase_plus6 | yes | yes | 8/8 | 0.1644 |
| design_challenges | central_operating_conditions__L1_1.4uH | yes | no | 17/32 | 0.8576 |
| design_challenges | central_operating_conditions__L1_1.5uH | yes | no | 22/32 | 0.6623 |
| design_challenges | central_operating_conditions__L1_1.6uH | yes | no | 26/32 | 0.6426 |
| design_challenges | reported_apparatus_spread__L1_1.4uH | yes | no | 26/32 | 0.7591 |
| design_challenges | reported_apparatus_spread__L1_1.5uH | yes | no | 30/32 | 0.5281 |
| design_challenges | reported_apparatus_spread__L1_1.6uH | yes | no | 20/32 | 0.5186 |
| design_challenges | phase_model_minus6__L1_1.4uH | yes | no | 5/8 | 0.4399 |
| design_challenges | phase_model_minus6__L1_1.5uH | yes | no | 7/8 | 0.3658 |
| design_challenges | phase_model_minus6__L1_1.6uH | yes | no | 4/8 | 0.5873 |
| design_challenges | phase_model_plus6__L1_1.4uH | yes | no | 7/8 | 0.4235 |
| design_challenges | phase_model_plus6__L1_1.5uH | yes | yes | 8/8 | 0.1644 |
| design_challenges | phase_model_plus6__L1_1.6uH | yes | yes | 8/8 | 0.2204 |
| design_challenges | literature_colpo1999_fixed_digitization_corners | yes | no | 35/60 | 0.7538 |
| design_challenges | literature_colpo1999_bounded_digitization_corners | yes | yes | 60/60 | 0.1957 |
| reference_inventory | Lee 2021 Figures 4-7 | n/a | n/a | n/a | n/a |
| reference_inventory | Cao 2020 commercial planar ICP | n/a | n/a | n/a | n/a |
| reference_inventory | Metze 1986 / Saikia 2018 | n/a | n/a | n/a | n/a |
| reference_inventory | Howling / Guittienne planar ICP | n/a | n/a | n/a | n/a |
| reference_inventory | Qu 2020 pulsed ICP | n/a | n/a | n/a | n/a |

## Interpretation boundary

These results establish circuit implementation, equation/data reproduction, and declared electrical design decisions only. They do not qualify plasma chemistry, chamber process windows, thermal lifetime, apparatus yield, or self-consistent plasma-circuit dynamics.

## Source artifacts

| source | schema | SHA-256 | path |
|---|---|---|---|
| core | design_benchmark_suite.v1 | `305aa5efa6c501fab4176fe5624977c955141ecb4c0b7d1fa758321245e04778` | `runs/benchmark_suite/benchmark_result.json` |
| literature | pcd.literature_benchmark_suite.v4 | `4a7cb1dc2f0115293c4421853aa6a804491bd8798508f7a94531fd86582119b4` | `runs/literature/final_evaluation/evaluation.json` |
