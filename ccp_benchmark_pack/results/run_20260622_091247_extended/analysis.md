# ngspice benchmark analysis v2

Run root: `C:\Users\user\Desktop\circuit_design_platform_v6_final\runs\ccp_ngspice_reeval\run_20260622_091247_extended`

## Interpretation Rules

- Treat best loss as a secondary indicator.
- Prefer feasible median, penalty rate, p90/max, and topology/load risk profile.
- Do not call waveform tailoring successful when A2/A3 target ratios are low.
- Treat surrogate results as diagnostics until feasible-only CV metrics are acceptable.

## Dummy vs ngspice summary

| case                        | dummy_best_loss | dummy_median_loss | dummy_p90_loss | dummy_penalty_rate | ngspice_best_loss | ngspice_median_loss | ngspice_p90_loss | ngspice_mean_loss | ngspice_max_loss | ngspice_failed | ngspice_penalty_rate | feasible_median_loss | infeasible_median_loss | v_peak_gt_1000 | i_rms_gt_20 | i_rms_gt_25 | loss_lt_1 | loss_lt_2 |
| --------------------------- | --------------- | ----------------- | -------------- | ------------------ | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- | -------------- | -------------------- | -------------------- | ---------------------- | -------------- | ----------- | ----------- | --------- | --------- |
| level3_topology_load_choice | 0.501831        | 0.634929          | 0.791512       | 0                  | 0.502765          | 2.53198             | 236.887          | 609.287           | 32321.6          | 0              | 0.5                  | 1.39867              | 14.6121                | 31             | 35          | 28          | 17        | 39        |

## Feasibility summary

| case                        | bucket     | count | best_loss | median_loss | mean_loss | p90_loss | max_loss | v_peak_max | i_rms_max | penalty_rate |
| --------------------------- | ---------- | ----- | --------- | ----------- | --------- | -------- | -------- | ---------- | --------- | ------------ |
| level3_topology_load_choice | all        | 100   | 0.502765  | 2.53198     | 609.287   | 236.887  | 32321.6  | 20125.6    | 630.338   | 0.5          |
| level3_topology_load_choice | feasible   | 50    | 0.502765  | 1.39867     | 1.40903   | 2.17402  | 3.31288  | 964.999    | 15.6844   | 0            |
| level3_topology_load_choice | infeasible | 50    | 1.47836   | 14.6121     | 1217.16   | 2223.56  | 32321.6  | 20125.6    | 630.338   | 1            |

## Category risk

| case                        | group_type    | group                                | count | min_loss | median_loss | mean_loss | p75_loss | p90_loss | max_loss | penalty_rate |
| --------------------------- | ------------- | ------------------------------------ | ----- | -------- | ----------- | --------- | -------- | -------- | -------- | ------------ |
| level3_topology_load_choice | topology      | l_match                              | 34    | 0.502765 | 1.93595     | 462.596   | 3.65308  | 688.22   | 6284.69  | 0.294118     |
| level3_topology_load_choice | topology      | pi_match                             | 31    | 0.561769 | 3.22222     | 307.262   | 14.6121  | 189.323  | 8026.13  | 0.612903     |
| level3_topology_load_choice | topology      | pi_match_harmonic                    | 35    | 0.551579 | 4.72111     | 1019.29   | 33.2975  | 106.531  | 32321.6  | 0.6          |
| level3_topology_load_choice | load          | electrode_stray                      | 36    | 0.591182 | 2.81951     | 1285.85   | 76.4957  | 508.993  | 32321.6  | 0.5          |
| level3_topology_load_choice | load          | plasma_fixed_rlc                     | 39    | 0.511977 | 2.62031     | 293.268   | 13.2695  | 63.6046  | 8026.13  | 0.512821     |
| level3_topology_load_choice | load          | plasma_state_rlc                     | 25    | 0.502765 | 2.1571      | 128.027   | 18.9844  | 97.3809  | 2188.8   | 0.48         |
| level3_topology_load_choice | topology_load | l_match + electrode_stray            | 15    | 0.591182 | 1.91182     | 862.966   | 3.84009  | 3833.97  | 6284.69  | 0.333333     |
| level3_topology_load_choice | topology_load | l_match + plasma_fixed_rlc           | 12    | 0.511977 | 1.7613      | 48.406    | 2.573    | 10.6019  | 554.087  | 0.166667     |
| level3_topology_load_choice | topology_load | l_match + plasma_state_rlc           | 7     | 0.502765 | 2.1571      | 314.7     | 4.55387  | 878.726  | 2188.8   | 0.428571     |
| level3_topology_load_choice | topology_load | pi_match + electrode_stray           | 9     | 0.657815 | 3.22222     | 53.6076   | 8.46515  | 205.914  | 272.28   | 0.555556     |
| level3_topology_load_choice | topology_load | pi_match + plasma_fixed_rlc          | 13    | 0.561769 | 4.53461     | 626.811   | 14.5916  | 51.9275  | 8026.13  | 0.769231     |
| level3_topology_load_choice | topology_load | pi_match + plasma_state_rlc          | 9     | 0.777154 | 2.05582     | 99.3457   | 47.5575  | 244.435  | 699.791  | 0.444444     |
| level3_topology_load_choice | topology_load | pi_match_harmonic + electrode_stray  | 12    | 0.764148 | 9.40837     | 2738.63   | 89.7412  | 222.231  | 32321.6  | 0.666667     |
| level3_topology_load_choice | topology_load | pi_match_harmonic + plasma_fixed_rlc | 14    | 0.551579 | 3.09715     | 193.433   | 19.9188  | 61.7819  | 2536.33  | 0.571429     |
| level3_topology_load_choice | topology_load | pi_match_harmonic + plasma_state_rlc | 9     | 0.627786 | 4.84132     | 11.5168   | 18.9844  | 32.572   | 38.7377  | 0.555556     |

## Top Spearman correlations

| case                        | feature                   | spearman_loss |
| --------------------------- | ------------------------- | ------------- |
| level3_topology_load_choice | metric.constraint_penalty | 0.919859      |
| level3_topology_load_choice | metric.i_rms_A            | 0.852085      |
| level3_topology_load_choice | metric.normalized_rmse    | 0.812373      |
| level3_topology_load_choice | metric.rmse_V             | 0.812373      |
| level3_topology_load_choice | metric.harmonic_error     | 0.777258      |
| level3_topology_load_choice | metric.v_peak_abs_V       | 0.717816      |
| level3_topology_load_choice | metric.v_rms_V            | 0.690297      |
| level3_topology_load_choice | param.Vsrc_amp            | 0.53793       |

## Harmonic amplitudes and phase

| case                        | source       | mean_V    | rms_V   | v_peak_abs_V | A1_V    | P1_deg   | A2_V    | P2_deg   | A3_V    | P3_deg   | A1_target_ratio | P1_error_deg | A2_target_ratio | P2_error_deg | A3_target_ratio | P3_error_deg |
| --------------------------- | ------------ | --------- | ------- | ------------ | ------- | -------- | ------- | -------- | ------- | -------- | --------------- | ------------ | --------------- | ------------ | --------------- | ------------ |
| level3_topology_load_choice | target       | -19.9918  | 206.53  | 352.492      | 279.976 | -89.6611 | 69.9881 | 0.715751 | 34.9856 | -134.073 |                 |              |                 |              |                 |              |
| level3_topology_load_choice | ngspice_best | -0.133236 | 199.884 | 283.599      | 281.719 | -96.2285 | 1.70323 | 87.675   | 1.10066 | 90.0611  | 1.00623         | -6.56737     | 0.024336        | 86.9592      | 0.0314604       | -135.866     |

## Surrogate diagnostics

| case                        | variant        | schema             | n_train | n_features | training_rmse | training_r2 | cv_rmse  | dropped_rows | clipped_rows | target_transform |
| --------------------------- | -------------- | ------------------ | ------- | ---------- | ------------- | ----------- | -------- | ------------ | ------------ | ---------------- |
| level3_topology_load_choice | all            | ridge_surrogate.v2 | 100     | 18         | 3013.28       | 0.216584    | 4065.99  | 0            | 0            | none             |
| level3_topology_load_choice | feasible_only  | ridge_surrogate.v2 | 50      | 18         | 0.477042      | 0.53747     | 0.810643 | 50           | 0            | none             |
| level3_topology_load_choice | log1p_clip_p90 | ridge_surrogate.v2 | 100     | 18         | 71.2532       | 0.111082    | 82.2907  | 0            | 10           | log1p            |
