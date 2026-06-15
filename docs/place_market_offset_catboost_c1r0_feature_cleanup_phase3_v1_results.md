# C1R0 Feature Cleanup Phase3 Results

## Paired Bootstrap Summary
| unit      | metric        |   point_estimate_delta |   bootstrap_mean_delta |    ci_lower |    ci_upper |   prob_delta_below_zero |   prob_delta_above_zero | better_model   | decision   |
|:----------|:--------------|-----------------------:|-----------------------:|------------:|------------:|------------------------:|------------------------:|:---------------|:-----------|
| race_id   | delta_logloss |            0.000551475 |            0.000553962 | 0.000406376 | 0.000700588 |                  0      |                  1      | raw            | raw_clear  |
| race_id   | delta_brier   |            0.000155054 |            0.000155841 | 0.000103978 | 0.00020842  |                  0      |                  1      | raw            | raw_clear  |
| race_id   | delta_ece     |            0.00378882  |            0.00254552  | 0.000994135 | 0.00376969  |                  0.001  |                  0.999  | raw            | raw_clear  |
| race_date | delta_logloss |            0.000551475 |            0.000551926 | 0.000404581 | 0.000702385 |                  0      |                  1      | raw            | raw_clear  |
| race_date | delta_brier   |            0.000155054 |            0.000155431 | 0.000100515 | 0.000212499 |                  0      |                  1      | raw            | raw_clear  |
| race_date | delta_ece     |            0.00378882  |            0.00252789  | 0.00103491  | 0.00370984  |                  0.0004 |                  0.9996 | raw            | raw_clear  |

## Cumulative Starts Decision
{
  "selected_starts": "raw",
  "selected_model_key": "C1R0_fixed300_ablation_drop_person_codes",
  "improved_years_logloss": 0,
  "reason": "Race-level paired bootstrap CI shows clip_p99_log1p worsens Logloss and Brier, so raw starts are retained."
}

## Additional Ablation Comparison
| model_key                                |   combined_logloss |   combined_brier |   combined_ece |   calibration_slope |   calibration_intercept |   worst_year_logloss |   worst_year_brier |   residual_std |   residual_std_cv |   abs_residual_p90 |   abs_residual_p95 |   abs_residual_p99 |   abs_residual_p95_cv |   ev_ge_1_count_sum |   ev_ge_1_count_cv |   ev_roi_spearman |   ev_ge_1_roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p975 |
|:-----------------------------------------|-------------------:|-----------------:|---------------:|--------------------:|------------------------:|---------------------:|-------------------:|---------------:|------------------:|-------------------:|-------------------:|-------------------:|----------------------:|--------------------:|-------------------:|------------------:|--------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|
| C1R0_fixed300_ablation_drop_person_codes |           0.405716 |         0.130356 |     0.00336398 |            0.999845 |            -0.000184642 |             0.410444 |           0.13171  |       0.160968 |         0.0754061 |           0.247503 |           0.316674 |           0.506951 |             0.080014  |                1376 |           0.516279 |          0.495238 |       111.485 |           104.961  |            94.0579 |            85.1832 |             66.5603 |              78.4426 |              146.415 |
| C1R0_300_cleanbase_no_meeting_admin      |           0.40572  |         0.130372 |     0.00402122 |            0.999832 |            -0.000239214 |             0.410557 |           0.131735 |       0.161726 |         0.0741483 |           0.25039  |           0.318825 |           0.502706 |             0.0772039 |                1424 |           0.438169 |          0.490476 |       112.034 |           106.29   |            96.3595 |            88.48   |             71.1162 |              79.7637 |              147.98  |
| C1R0_300_cleanbase_no_raw_time           |           0.405777 |         0.130372 |     0.00401169 |            0.999861 |            -0.000200719 |             0.410617 |           0.131748 |       0.160018 |         0.0541611 |           0.246308 |           0.314349 |           0.505942 |             0.0558061 |                1404 |           0.428428 |          0.67619  |       116.108 |           109.612  |            98.8243 |            89.8933 |             71.4233 |              82.2449 |              151.966 |
| C1R0_300_cleanbase_no_raw_body_weight    |           0.40591  |         0.130447 |     0.00376711 |            0.999821 |            -0.000207656 |             0.410541 |           0.131715 |       0.157152 |         0.0641812 |           0.243486 |           0.310266 |           0.493619 |             0.0721322 |                1398 |           0.478953 |          0.27619  |       101.46  |            95.1993 |            85.0057 |            75.9243 |             56.5761 |              68.5094 |              137.288 |

## Selected Feature Set
{
  "selected_model_key": "C1R0_fixed300_ablation_drop_person_codes",
  "selected_additional": null,
  "reason": "No additional ablation had clear individual benefit under 2020-2024 priority metrics.",
  "selected_row": {
    "model_key": "C1R0_fixed300_ablation_drop_person_codes",
    "combined_logloss": 0.40571562844863307,
    "combined_brier": 0.13035585431001873,
    "combined_ece": 0.0033639802296566027,
    "calibration_slope": 0.9998446787085726,
    "calibration_intercept": -0.00018464215427187338,
    "worst_year_logloss": 0.4104435070777986,
    "worst_year_brier": 0.13170983082372525,
    "residual_std": 0.1609678441732988,
    "residual_std_cv": 0.07540610839467246,
    "abs_residual_p90": 0.24750314543120683,
    "abs_residual_p95": 0.31667447228394563,
    "abs_residual_p99": 0.5069508442752244,
    "abs_residual_p95_cv": 0.08001398781237062,
    "ev_ge_1_count_sum": 1376,
    "ev_ge_1_count_cv": 0.5162787321750154,
    "ev_roi_spearman": 0.49523809523809526,
    "ev_ge_1_roi": 111.48526590221279,
    "top1_removed_roi": 104.96121419929281,
    "top3_removed_roi": 94.05790606982893,
    "top5_removed_roi": 85.18321185639763,
    "top10_removed_roi": 66.56031760791916,
    "bootstrap_roi_p025": 78.44259534772384,
    "bootstrap_roi_p975": 146.4152102892279
  }
}

## 2025/2026 Diagnostic
| model_key                                | period              |   Year |   rows |   positives |   logloss |    brier |      auc |        ece |       mce |   calibration_slope |   calibration_intercept |
|:-----------------------------------------|:--------------------|-------:|-------:|------------:|----------:|---------:|---------:|-----------:|----------:|--------------------:|------------------------:|
| C1R0_fixed300_ablation_drop_person_codes | latest_holdout_2026 |   2026 |  21276 |        4500 |  0.383668 | 0.122487 | 0.835785 | 0.00799261 | 0.0364075 |             1.05757 |             0.0380643   |
| C1R0_fixed300_ablation_drop_person_codes | test_2025           |   2025 |  47497 |       10276 |  0.401109 | 0.1288   | 0.820061 | 0.0049654  | 0.020754  |             1.0043  |            -0.000244553 |

## 2025/2026 EV
| model_key                                | period              |   Year |   rows |   ev_ge_1_count |   ev_ge_1_rate |   market_only_ev_ge_1_count |   market_lt1_to_final_ge1 |   market_ge1_to_final_lt1 |   ev_roi_spearman |
|:-----------------------------------------|:--------------------|-------:|-------:|----------------:|---------------:|----------------------------:|--------------------------:|--------------------------:|------------------:|
| C1R0_fixed300_ablation_drop_person_codes | latest_holdout_2026 |   2026 |  21276 |             132 |     0.00620417 |                          46 |                       105 |                        19 |          0.571429 |
| C1R0_fixed300_ablation_drop_person_codes | test_2025           |   2025 |  47497 |             197 |     0.00414763 |                          64 |                       165 |                        32 |         -0.190476 |

## 2025/2026 ROI
| model_key                                | period              |   Year |   bets |      roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p500 |   bootstrap_roi_p975 |
|:-----------------------------------------|:--------------------|-------:|-------:|---------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| C1R0_fixed300_ablation_drop_person_codes | latest_holdout_2026 |   2026 |    132 | 104.015  |            92.2137 |            75.1938 |            62.2835 |             37.9508 |              63.6043 |             102.554  |              150.411 |
| C1R0_fixed300_ablation_drop_person_codes | test_2025           |   2025 |    197 |  73.3503 |            65.051  |            54.1753 |            45.2083 |             26.738  |              44.4524 |              72.6864 |              109.088 |

## Reuse Training Log
| model              | fold      | action   | reason   |
|:-------------------|:----------|:---------|:---------|
| no_meeting_admin   | fold_2020 | reuse    | ok       |
| no_meeting_admin   | fold_2021 | reuse    | ok       |
| no_meeting_admin   | fold_2022 | reuse    | ok       |
| no_meeting_admin   | fold_2023 | reuse    | ok       |
| no_meeting_admin   | fold_2024 | reuse    | ok       |
| no_raw_time        | fold_2020 | reuse    | ok       |
| no_raw_time        | fold_2021 | reuse    | ok       |
| no_raw_time        | fold_2022 | reuse    | ok       |
| no_raw_time        | fold_2023 | reuse    | ok       |
| no_raw_time        | fold_2024 | reuse    | ok       |
| no_raw_body_weight | fold_2020 | reuse    | ok       |
| no_raw_body_weight | fold_2021 | reuse    | ok       |
| no_raw_body_weight | fold_2022 | reuse    | ok       |
| no_raw_body_weight | fold_2023 | reuse    | ok       |
| no_raw_body_weight | fold_2024 | reuse    | ok       |

Elapsed seconds: `72.0`
