# ngspice benchmark analysis v2

Run root: `C:\Users\user\Desktop\circuit_design_platform_v6_final\runs\ccp_ngspice_reeval\run_20260622_084121_parser_fixed`

## Interpretation Rules

- Treat best loss as a secondary indicator.
- Prefer feasible median, penalty rate, p90/max, and topology/load risk profile.
- Do not call waveform tailoring successful when A2/A3 target ratios are low.
- Treat surrogate results as diagnostics until feasible-only CV metrics are acceptable.

## Dummy vs ngspice summary

| case                        | dummy_best_loss | dummy_median_loss | dummy_p90_loss | dummy_penalty_rate | ngspice_best_loss | ngspice_median_loss | ngspice_p90_loss | ngspice_mean_loss | ngspice_max_loss | ngspice_failed | ngspice_penalty_rate | feasible_median_loss | infeasible_median_loss | v_peak_gt_1000 | i_rms_gt_20 | i_rms_gt_25 | loss_lt_1 | loss_lt_2 |
| --------------------------- | --------------- | ----------------- | -------------- | ------------------ | ----------------- | ------------------- | ---------------- | ----------------- | ---------------- | -------------- | -------------------- | -------------------- | ---------------------- | -------------- | ----------- | ----------- | --------- | --------- |
| level2_timevarying_plasma   | 0.491199        | 0.687434          | 0.860495       | 0                  | 0.577952          | 2.64847             | 121.461          | 60.6871           | 753.367          | 0              | 0.566667             | 1.39033              | 11.0501                | 9              | 10          | 8           | 3         | 11        |
| level3_topology_load_choice | 0.501831        | 0.634929          | 0.791512       | 0                  | 0.502765          | 3.60308             | 704.383          | 383.789           | 6284.69          | 0              | 0.6                  | 1.52943              | 46.0012                | 11             | 13          | 10          | 2         | 10        |

## Feasibility summary

| case                        | bucket     | count | best_loss | median_loss | mean_loss | p90_loss | max_loss | v_peak_max | i_rms_max | penalty_rate |
| --------------------------- | ---------- | ----- | --------- | ----------- | --------- | -------- | -------- | ---------- | --------- | ------------ |
| level2_timevarying_plasma   | all        | 30    | 0.577952  | 2.64847     | 60.6871   | 121.461  | 753.367  | 6813       | 126.602   | 0.566667     |
| level2_timevarying_plasma   | feasible   | 13    | 0.577952  | 1.39033     | 1.56017   | 2.47628  | 2.8123   | 887.973    | 12.892    | 0            |
| level2_timevarying_plasma   | infeasible | 17    | 1.54565   | 11.0501     | 105.902   | 370.636  | 753.367  | 6813       | 126.602   | 1            |
| level3_topology_load_choice | all        | 30    | 0.502765  | 3.60308     | 383.789   | 704.383  | 6284.69  | 15429.7    | 243.281   | 0.6          |
| level3_topology_load_choice | feasible   | 12    | 0.502765  | 1.52943     | 1.62682   | 2.57373  | 3.31288  | 942.271    | 15.6844   | 0            |
| level3_topology_load_choice | infeasible | 18    | 1.47836   | 46.0012     | 638.563   | 1282.89  | 6284.69  | 15429.7    | 243.281   | 1            |

## Category risk

| case                        | group_type    | group                                | count | min_loss | median_loss | mean_loss | p75_loss | p90_loss | max_loss | penalty_rate |
| --------------------------- | ------------- | ------------------------------------ | ----- | -------- | ----------- | --------- | -------- | -------- | -------- | ------------ |
| level3_topology_load_choice | topology      | l_match                              | 11    | 0.502765 | 1.96007     | 691.527   | 282.75   | 745.706  | 6284.69  | 0.363636     |
| level3_topology_load_choice | topology      | pi_match                             | 12    | 1.04083  | 3.83824     | 99.5441   | 56.2114  | 263.984  | 699.791  | 0.75         |
| level3_topology_load_choice | topology      | pi_match_harmonic                    | 7     | 1.41278  | 18.9844     | 387.476   | 74.5481  | 1060.18  | 2536.33  | 0.714286     |
| level3_topology_load_choice | load          | electrode_stray                      | 9     | 1.23319  | 76.0782     | 841.804   | 272.28   | 1853.5   | 6284.69  | 0.666667     |
| level3_topology_load_choice | load          | plasma_fixed_rlc                     | 16    | 0.511977 | 3.22737     | 200.96    | 11.5191  | 313.553  | 2536.33  | 0.625        |
| level3_topology_load_choice | load          | plasma_state_rlc                     | 5     | 0.502765 | 1.73732     | 144.411   | 18.9844  | 427.469  | 699.791  | 0.4          |
| level3_topology_load_choice | topology_load | l_match + electrode_stray            | 4     | 1.23319  | 373.676     | 1758.32   | 2130.45  | 4622.99  | 6284.69  | 0.5          |
| level3_topology_load_choice | topology_load | l_match + plasma_fixed_rlc           | 5     | 0.511977 | 3.31288     | 114.257   | 11.4118  | 337.017  | 554.087  | 0.4          |
| level3_topology_load_choice | topology_load | l_match + plasma_state_rlc           | 2     | 0.502765 | 1.12004     | 1.12004   | 1.42868  | 1.61387  | 1.73732  | 0            |
| level3_topology_load_choice | topology_load | pi_match + electrode_stray           | 3     | 1.3891   | 189.323     | 154.331   | 230.801  | 255.688  | 272.28   | 0.666667     |
| level3_topology_load_choice | topology_load | pi_match + plasma_fixed_rlc          | 7     | 1.47836  | 3.14186     | 4.38636   | 4.69225  | 7.64628  | 11.8409  | 0.857143     |
| level3_topology_load_choice | topology_load | pi_match + plasma_state_rlc          | 2     | 1.04083  | 350.416     | 350.416   | 525.104  | 629.916  | 699.791  | 0.5          |
| level3_topology_load_choice | topology_load | pi_match_harmonic + electrode_stray  | 2     | 3.89328  | 39.9857     | 39.9857   | 58.0319  | 68.8597  | 76.0782  | 1            |
| level3_topology_load_choice | topology_load | pi_match_harmonic + plasma_fixed_rlc | 4     | 1.41278  | 37.8191     | 653.344   | 688.845  | 1797.33  | 2536.33  | 0.5          |
| level3_topology_load_choice | topology_load | pi_match_harmonic + plasma_state_rlc | 1     | 18.9844  | 18.9844     | 18.9844   | 18.9844  | 18.9844  | 18.9844  | 1            |

## Top Spearman correlations

| case                        | feature                   | spearman_loss |
| --------------------------- | ------------------------- | ------------- |
| level3_topology_load_choice | metric.constraint_penalty | 0.935478      |
| level2_timevarying_plasma   | metric.constraint_penalty | 0.906209      |
| level3_topology_load_choice | metric.i_rms_A            | 0.866963      |
| level2_timevarying_plasma   | metric.normalized_rmse    | 0.835818      |
| level2_timevarying_plasma   | metric.rmse_V             | 0.835818      |
| level2_timevarying_plasma   | metric.harmonic_error     | 0.779755      |
| level3_topology_load_choice | metric.normalized_rmse    | 0.769077      |
| level3_topology_load_choice | metric.rmse_V             | 0.769077      |
| level3_topology_load_choice | metric.harmonic_error     | 0.74594       |
| level2_timevarying_plasma   | metric.i_rms_A            | 0.725918      |
| level3_topology_load_choice | metric.v_rms_V            | 0.716574      |
| level3_topology_load_choice | metric.v_peak_abs_V       | 0.716129      |
| level2_timevarying_plasma   | metric.v_peak_abs_V       | 0.713459      |
| level2_timevarying_plasma   | metric.v_rms_V            | 0.680089      |
| level3_topology_load_choice | param.C1                  | 0.610234      |
| level2_timevarying_plasma   | metric.power_error        | 0.557286      |

## Harmonic amplitudes and phase

| case                        | source       | mean_V    | rms_V   | v_peak_abs_V | A1_V    | P1_deg   | A2_V    | P2_deg   | A3_V    | P3_deg   | A1_target_ratio | P1_error_deg | A2_target_ratio | P2_error_deg | A3_target_ratio | P3_error_deg |
| --------------------------- | ------------ | --------- | ------- | ------------ | ------- | -------- | ------- | -------- | ------- | -------- | --------------- | ------------ | --------------- | ------------ | --------------- | ------------ |
| level2_timevarying_plasma   | target       | -19.9918  | 206.53  | 352.492      | 279.976 | -89.6611 | 69.9881 | 0.715751 | 34.9856 | -134.073 |                 |              |                 |              |                 |              |
| level2_timevarying_plasma   | ngspice_best | -21.9732  | 199.806 | 335.242      | 277.43  | -97.162  | 5.71943 | 71.2455  | 4.16566 | -158.85  | 0.990905        | -7.50089     | 0.08172         | 70.5298      | 0.119068        | -24.7772     |
| level3_topology_load_choice | target       | -19.9918  | 206.53  | 352.492      | 279.976 | -89.6611 | 69.9881 | 0.715751 | 34.9856 | -134.073 |                 |              |                 |              |                 |              |
| level3_topology_load_choice | ngspice_best | -0.133236 | 199.884 | 283.599      | 281.719 | -96.2285 | 1.70323 | 87.675   | 1.10066 | 90.0611  | 1.00623         | -6.56737     | 0.024336        | 86.9592      | 0.0314604       | -135.866     |

## Surrogate diagnostics

| case                        | variant        | schema             | n_train | n_features | training_rmse | training_r2 | cv_rmse | dropped_rows | clipped_rows | target_transform |
| --------------------------- | -------------- | ------------------ | ------- | ---------- | ------------- | ----------- | ------- | ------------ | ------------ | ---------------- |
| level2_timevarying_plasma   | all            | ridge_surrogate.v2 | 30      | 7          | 135.305       | 0.304466    | 198.847 | 0            | 0            | none             |
| level2_timevarying_plasma   | feasible_only  | ridge_surrogate.v2 | 13      | 7          | 0.34464       | 0.732431    | 2.21466 | 17           | 0            | none             |
| level2_timevarying_plasma   | log1p_clip_p90 | ridge_surrogate.v2 | 30      | 7          | 34.9329       | 0.228481    | 44.3874 | 0            | 3            | log1p            |
| level3_topology_load_choice | all            | ridge_surrogate.v2 | 30      | 18         | 585.559       | 0.760704    | 2344.02 | 0            | 0            | none             |
| level3_topology_load_choice | feasible_only  | ridge_surrogate.v2 | 12      | 18         | 7.18517e-08   | 1           | 1.00952 | 18           | 0            | none             |
| level3_topology_load_choice | log1p_clip_p90 | ridge_surrogate.v2 | 30      | 18         | 191.333       | 0.408767    | 292001  | 0            | 3            | log1p            |
