# C1R0 Feature Cleanup Phase2 Results

## Existing Five Ablations
| model_key                                            |   combined_logloss |   combined_brier |   combined_ece |   calibration_slope |   calibration_intercept |   worst_year_logloss |   worst_year_brier |   residual_mean |   residual_std |   residual_std_cv |   abs_residual_p90 |   abs_residual_p95 |   abs_residual_p99 |   abs_residual_p95_cv |   abs_residual_p99_cv |   ev_ge_1_count_sum |   ev_ge_1_count_cv |   market_lt1_to_final_ge1_sum |   market_ge1_to_final_lt1_sum |   ev_roi_spearman |   ev_ge_1_roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p500 |   bootstrap_roi_p975 |
|:-----------------------------------------------------|-------------------:|-----------------:|---------------:|--------------------:|------------------------:|---------------------:|-------------------:|----------------:|---------------:|------------------:|-------------------:|-------------------:|-------------------:|----------------------:|----------------------:|--------------------:|-------------------:|------------------------------:|------------------------------:|------------------:|--------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| C1R0_fixed300_ablation_drop_person_codes             |           0.405716 |         0.130356 |     0.00336398 |            0.999845 |            -0.000184642 |             0.410444 |           0.13171  |      -0.0116503 |       0.160968 |         0.0754061 |           0.247503 |           0.316674 |           0.506951 |             0.080014  |             0.0746082 |                1376 |           0.516279 |                          1180 |                           117 |          0.495238 |       111.485 |            104.961 |            94.0579 |            85.1832 |             66.5603 |              78.4426 |              110.817 |              146.415 |
| C1R0_fixed300_ablation_drop_meeting_admin            |           0.405987 |         0.130437 |     0.00392139 |            0.999782 |            -0.000180528 |             0.410734 |           0.131752 |      -0.0731557 |       0.175753 |         0.0469649 |           0.303227 |           0.384142 |           0.577185 |             0.0541771 |             0.0550009 |                1498 |           0.360694 |                          1315 |                           130 |          0.442857 |       107.665 |            101.456 |            92.5671 |            84.5028 |             67.7363 |              76.3427 |              107.269 |              141.445 |
| C1R0_pure_market_offset_fixed300_base                |           0.406029 |         0.130457 |     0.00410332 |            0.99977  |            -0.000249473 |             0.410603 |           0.131714 |      -0.0697407 |       0.178185 |         0.0488705 |           0.302687 |           0.384876 |           0.582732 |             0.0572863 |             0.0713355 |                1621 |           0.398592 |                          1420 |                           112 |          0.6      |       111.96  |            106.589 |            97.246  |            88.9379 |             72.4454 |              81.0888 |              111.209 |              145.798 |
| C1R0_fixed300_ablation_drop_unadjusted_raw_time      |           0.406179 |         0.130496 |     0.00392249 |            0.999826 |            -0.000232801 |             0.411222 |           0.131964 |      -0.0767862 |       0.178936 |         0.0430328 |           0.310997 |           0.391961 |           0.587485 |             0.0417644 |             0.0611777 |                1586 |           0.35533  |                          1395 |                           122 |          0.319048 |       111.858 |            106.288 |            96.7792 |            88.3252 |             71.7487 |              79.8312 |              111.669 |              146.057 |
| C1R0_fixed300_ablation_drop_global_cumulative_starts |           0.406252 |         0.130549 |     0.00331239 |            0.999928 |            -0.000159393 |             0.411122 |           0.131943 |      -0.10754   |       0.208389 |         0.0232805 |           0.374699 |           0.48452  |           0.76186  |             0.0326881 |             0.048835  |                1823 |           0.320457 |                          1628 |                           118 |          0.628571 |       111.852 |            107.281 |            98.8347 |            91.3594 |             75.2396 |              81.9907 |              111.152 |              144.338 |
| C1R0_fixed300_ablation_drop_raw_body_weight          |           0.406418 |         0.130578 |     0.00391583 |            0.999805 |            -5.98772e-05 |             0.411079 |           0.131844 |      -0.0843121 |       0.18198  |         0.0440899 |           0.322406 |           0.404314 |           0.595306 |             0.0463157 |             0.0733802 |                1800 |           0.367885 |                          1600 |                           113 |          0.47619  |       107.899 |            103.333 |            95.5337 |            88.4026 |             73.4201 |              80.1729 |              107.685 |              139.94  |

## Working Base
{
  "selected_model_key": "C1R0_fixed300_ablation_drop_person_codes",
  "selection_years": [
    2020,
    2021,
    2022,
    2023,
    2024
  ],
  "selection_rule": "2020-2024 only; Logloss, Brier, calibration, residual stability, EV count stability, EV-ROI Spearman; ROI auxiliary.",
  "selected_row": {
    "model_key": "C1R0_fixed300_ablation_drop_person_codes",
    "combined_logloss": 0.40571562844863307,
    "combined_brier": 0.13035585431001873,
    "combined_ece": 0.0033639802296566027,
    "calibration_slope": 0.9998446787085726,
    "calibration_intercept": -0.00018464215427187338,
    "worst_year_logloss": 0.4104435070777986,
    "worst_year_brier": 0.13170983082372525,
    "residual_mean": -0.011650275752976197,
    "residual_std": 0.1609678441732988,
    "residual_std_cv": 0.07540610839467246,
    "abs_residual_p90": 0.24750314543120683,
    "abs_residual_p95": 0.31667447228394563,
    "abs_residual_p99": 0.5069508442752244,
    "abs_residual_p95_cv": 0.08001398781237062,
    "abs_residual_p99_cv": 0.07460823676946822,
    "ev_ge_1_count_sum": 1376,
    "ev_ge_1_count_cv": 0.5162787321750154,
    "market_lt1_to_final_ge1_sum": 1180,
    "market_ge1_to_final_lt1_sum": 117,
    "ev_roi_spearman": 0.49523809523809526,
    "ev_ge_1_roi": 111.48526590221279,
    "top1_removed_roi": 104.96121419929281,
    "top3_removed_roi": 94.05790606982893,
    "top5_removed_roi": 85.18321185639763,
    "top10_removed_roi": 66.56031760791916,
    "bootstrap_roi_p025": 78.44259534772384,
    "bootstrap_roi_p500": 110.8165256145091,
    "bootstrap_roi_p975": 146.4152102892279
  }
}

## MonthDay Audit
| feature   | encoding     | numeric_or_categorical   |   missing_rate |   unique_count |   spearman_with_year |   spearman_with_month |   spearman_with_kaiji |   spearman_with_nichiji | december_january_discontinuity   | interpretation                                                                                          |
|:----------|:-------------|:-------------------------|---------------:|---------------:|---------------------:|----------------------:|----------------------:|------------------------:|:---------------------------------|:--------------------------------------------------------------------------------------------------------|
| MonthDay  | MMDD integer | numeric                  |              0 |            359 |           -0.0097932 |               0.99649 |              0.792736 |               0.0319386 | True                             | season/date-order signal, not direct result leakage; numeric MMDD has artificial year-end discontinuity |

## MonthDay Ablation
| model_key                                   |   combined_logloss |   combined_brier |   combined_ece |   calibration_slope |   calibration_intercept |   worst_year_logloss |   worst_year_brier |   residual_mean |   residual_std |   residual_std_cv |   abs_residual_p90 |   abs_residual_p95 |   abs_residual_p99 |   abs_residual_p95_cv |   abs_residual_p99_cv |   ev_ge_1_count_sum |   ev_ge_1_count_cv |   market_lt1_to_final_ge1_sum |   market_ge1_to_final_lt1_sum |   ev_roi_spearman |   ev_ge_1_roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p500 |   bootstrap_roi_p975 |
|:--------------------------------------------|-------------------:|-----------------:|---------------:|--------------------:|------------------------:|---------------------:|-------------------:|----------------:|---------------:|------------------:|-------------------:|-------------------:|-------------------:|----------------------:|----------------------:|--------------------:|-------------------:|------------------------------:|------------------------------:|------------------:|--------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| C1R0_fixed300_ablation_drop_person_codes    |           0.405716 |         0.130356 |     0.00336398 |            0.999845 |            -0.000184642 |             0.410444 |           0.13171  |      -0.0116503 |       0.160968 |         0.0754061 |           0.247503 |           0.316674 |           0.506951 |             0.080014  |             0.0746082 |                1376 |           0.516279 |                          1180 |                           117 |          0.495238 |       111.485 |           104.961  |            94.0579 |            85.1832 |             66.5603 |              78.4426 |              110.817 |              146.415 |
| C1R0_300_feature_cleanup_phase2_no_monthday |           0.405788 |         0.130372 |     0.00336333 |            0.999831 |            -0.000235419 |             0.410631 |           0.131717 |      -0.0101642 |       0.159528 |         0.0628831 |           0.24481  |           0.312659 |           0.503366 |             0.0652293 |             0.0817774 |                1460 |           0.423953 |                          1267 |                           120 |          0.3      |       103.947 |            97.7523 |            87.9564 |            80.0381 |             63.5493 |              73.4498 |              103.056 |              136.47  |

## Cumulative Starts Comparison
| model_key                                             |   combined_logloss |   combined_brier |   combined_ece |   calibration_slope |   calibration_intercept |   worst_year_logloss |   worst_year_brier |   residual_mean |   residual_std |   residual_std_cv |   abs_residual_p90 |   abs_residual_p95 |   abs_residual_p99 |   abs_residual_p95_cv |   abs_residual_p99_cv |   ev_ge_1_count_sum |   ev_ge_1_count_cv |   market_lt1_to_final_ge1_sum |   market_ge1_to_final_lt1_sum |   ev_roi_spearman |   ev_ge_1_roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p500 |   bootstrap_roi_p975 |
|:------------------------------------------------------|-------------------:|-----------------:|---------------:|--------------------:|------------------------:|---------------------:|-------------------:|----------------:|---------------:|------------------:|-------------------:|-------------------:|-------------------:|----------------------:|----------------------:|--------------------:|-------------------:|------------------------------:|------------------------------:|------------------:|--------------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p |           0.405658 |         0.130335 |     0.0036852  |            0.999791 |            -0.000220857 |             0.410325 |           0.13163  |      -0.0122272 |       0.162519 |         0.0736238 |           0.249942 |           0.319148 |           0.512545 |             0.0802991 |             0.081962  |                1390 |           0.428828 |                          1188 |                           111 |          0.371429 |       107.345 |           101.417  |            92.2604 |            84.3905 |             66.8884 |              75.0139 |              106.999 |              140.993 |
| C1R0_300_feature_cleanup_phase2_starts_clip_p99       |           0.405662 |         0.130336 |     0.00374495 |            0.999789 |            -0.000220929 |             0.410309 |           0.131628 |      -0.0124215 |       0.162569 |         0.0737528 |           0.25022  |           0.319376 |           0.513087 |             0.0805735 |             0.0827027 |                1389 |           0.435144 |                          1183 |                           107 |          0.457143 |       104.709 |            98.7366 |            89.2867 |            81.5196 |             63.7059 |              74.1191 |              104.301 |              138.174 |
| C1R0_300_feature_cleanup_phase2_starts_drop           |           0.405697 |         0.130349 |     0.00381148 |            0.99997  |            -0.000208113 |             0.410787 |           0.131831 |      -0.0385002 |       0.185413 |         0.053667  |           0.28677  |           0.384801 |           0.664589 |             0.0679674 |             0.0639218 |                1367 |           0.406794 |                          1183 |                           129 |          0.261905 |       104.435 |            98.4733 |            87.8983 |            78.9903 |             60.7608 |              71.0927 |              103.993 |              140.531 |
| C1R0_300_feature_cleanup_phase2_starts_log1p          |           0.405712 |         0.130358 |     0.00389757 |            0.999834 |            -0.0001957   |             0.410435 |           0.131708 |      -0.0118197 |       0.161547 |         0.0731588 |           0.248786 |           0.317562 |           0.507529 |             0.0777406 |             0.0752238 |                1438 |           0.523874 |                          1241 |                           116 |          0.533333 |       110.983 |           103.781  |            93.9556 |            85.901  |             68.1133 |              77.8248 |              111.111 |              147.121 |
| C1R0_300_feature_cleanup_phase2_starts_raw            |           0.405716 |         0.130356 |     0.00336398 |            0.999845 |            -0.000184642 |             0.410444 |           0.13171  |      -0.0116503 |       0.160968 |         0.0754061 |           0.247503 |           0.316674 |           0.506951 |             0.080014  |             0.0746082 |                1376 |           0.516279 |                          1180 |                           117 |          0.495238 |       111.485 |           104.961  |            94.0579 |            85.1832 |             66.5603 |              78.4426 |              110.817 |              146.415 |

## Selected Preprocessing
{
  "selected_model_key": "C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p",
  "selection_years": [
    2020,
    2021,
    2022,
    2023,
    2024
  ],
  "selection_rule": "2020-2024 only; Logloss/Brier near-tie tier, then calibration, residual tails, EV stability, EV-ROI Spearman, and simplicity. ROI auxiliary.",
  "selected_row": {
    "model_key": "C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p",
    "combined_logloss": 0.40565800665718144,
    "combined_brier": 0.1303346636126485,
    "combined_ece": 0.0036852049403601123,
    "calibration_slope": 0.9997911969359503,
    "calibration_intercept": -0.00022085681620573962,
    "worst_year_logloss": 0.41032489304665015,
    "worst_year_brier": 0.1316301975192163,
    "residual_mean": -0.012227215439998116,
    "residual_std": 0.16251912918345507,
    "residual_std_cv": 0.07362380784735557,
    "abs_residual_p90": 0.24994238151622197,
    "abs_residual_p95": 0.31914820988295134,
    "abs_residual_p99": 0.5125454024612454,
    "abs_residual_p95_cv": 0.08029912934390399,
    "abs_residual_p99_cv": 0.08196201017038683,
    "ev_ge_1_count_sum": 1390,
    "ev_ge_1_count_cv": 0.4288276731173015,
    "market_lt1_to_final_ge1_sum": 1188,
    "market_ge1_to_final_lt1_sum": 111,
    "ev_roi_spearman": 0.3714285714285715,
    "ev_ge_1_roi": 107.34539539078864,
    "top1_removed_roi": 101.41723919581962,
    "top3_removed_roi": 92.26036375666808,
    "top5_removed_roi": 84.39049099225953,
    "top10_removed_roi": 66.88838384700773,
    "bootstrap_roi_p025": 75.01394863874961,
    "bootstrap_roi_p500": 106.99898057731657,
    "bootstrap_roi_p975": 140.99294073113396,
    "simplicity_rank": 3
  }
}

## 2025/2026 Diagnostic Metrics
| model_key                                             | period              |   Year |   rows |   positives |   logloss |    brier |      auc |        ece |       mce |   calibration_slope |   calibration_intercept |
|:------------------------------------------------------|:--------------------|-------:|-------:|------------:|----------:|---------:|---------:|-----------:|----------:|--------------------:|------------------------:|
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | latest_holdout_2026 |   2026 |  21276 |        4500 |  0.383657 | 0.122473 | 0.835759 | 0.00878751 | 0.0345077 |             1.05636 |             0.0357246   |
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | test_2025           |   2025 |  47497 |       10276 |  0.401232 | 0.12884  | 0.819942 | 0.00411582 | 0.0252486 |             1.00285 |            -0.00230486  |
| C1R0_fixed300_ablation_drop_person_codes              | latest_holdout_2026 |   2026 |  21276 |        4500 |  0.383668 | 0.122487 | 0.835785 | 0.00799261 | 0.0364075 |             1.05757 |             0.0380643   |
| C1R0_fixed300_ablation_drop_person_codes              | test_2025           |   2025 |  47497 |       10276 |  0.401109 | 0.1288   | 0.820061 | 0.0049654  | 0.020754  |             1.0043  |            -0.000244553 |

## 2025/2026 EV
| model_key                                             | period              |   Year |   rows |   ev_ge_1_count |   ev_ge_1_rate |   market_only_ev_ge_1_count |   market_lt1_to_final_ge1 |   market_ge1_to_final_lt1 |   ev_roi_spearman |
|:------------------------------------------------------|:--------------------|-------:|-------:|----------------:|---------------:|----------------------------:|--------------------------:|--------------------------:|------------------:|
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | latest_holdout_2026 |   2026 |  21276 |             131 |     0.00615717 |                          46 |                       105 |                        20 |          0.452381 |
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | test_2025           |   2025 |  47497 |             200 |     0.00421079 |                          64 |                       166 |                        30 |         -0.452381 |
| C1R0_fixed300_ablation_drop_person_codes              | latest_holdout_2026 |   2026 |  21276 |             132 |     0.00620417 |                          46 |                       105 |                        19 |          0.571429 |
| C1R0_fixed300_ablation_drop_person_codes              | test_2025           |   2025 |  47497 |             197 |     0.00414763 |                          64 |                       165 |                        32 |         -0.190476 |

## 2025/2026 ROI
| model_key                                             | period              |   Year |   bets |      roi |   top1_removed_roi |   top3_removed_roi |   top5_removed_roi |   top10_removed_roi |   bootstrap_roi_p025 |   bootstrap_roi_p500 |   bootstrap_roi_p975 |
|:------------------------------------------------------|:--------------------|-------:|-------:|---------:|-------------------:|-------------------:|-------------------:|--------------------:|---------------------:|---------------------:|---------------------:|
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | latest_holdout_2026 |   2026 |    131 | 105.115  |            93.2308 |            76.0938 |            63.8889 |             40.7438 |              63.7351 |             105.188  |             150.773  |
| C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p | test_2025           |   2025 |    200 |  67.4    |            61.407  |            52.1827 |            44      |             26.1053 |              40.85   |              66.7673 |              97.9152 |
| C1R0_fixed300_ablation_drop_person_codes              | latest_holdout_2026 |   2026 |    132 | 104.015  |            92.2137 |            75.1938 |            62.2835 |             37.9508 |              63.6043 |             102.554  |             150.411  |
| C1R0_fixed300_ablation_drop_person_codes              | test_2025           |   2025 |    197 |  73.3503 |            65.051  |            54.1753 |            45.2083 |             26.738  |              44.4524 |              72.6864 |             109.088  |

## Reuse Training Log
| model                 | fold      | action   | reason   |
|:----------------------|:----------|:---------|:---------|
| no_monthday           | fold_2020 | reuse    | ok       |
| no_monthday           | fold_2021 | reuse    | ok       |
| no_monthday           | fold_2022 | reuse    | ok       |
| no_monthday           | fold_2023 | reuse    | ok       |
| no_monthday           | fold_2024 | reuse    | ok       |
| starts_drop           | fold_2020 | reuse    | ok       |
| starts_drop           | fold_2021 | reuse    | ok       |
| starts_drop           | fold_2022 | reuse    | ok       |
| starts_drop           | fold_2023 | reuse    | ok       |
| starts_drop           | fold_2024 | reuse    | ok       |
| starts_log1p          | fold_2020 | reuse    | ok       |
| starts_log1p          | fold_2021 | reuse    | ok       |
| starts_log1p          | fold_2022 | reuse    | ok       |
| starts_log1p          | fold_2023 | reuse    | ok       |
| starts_log1p          | fold_2024 | reuse    | ok       |
| starts_clip_p99       | fold_2020 | reuse    | ok       |
| starts_clip_p99       | fold_2021 | reuse    | ok       |
| starts_clip_p99       | fold_2022 | reuse    | ok       |
| starts_clip_p99       | fold_2023 | reuse    | ok       |
| starts_clip_p99       | fold_2024 | reuse    | ok       |
| starts_clip_p99_log1p | fold_2020 | reuse    | ok       |
| starts_clip_p99_log1p | fold_2021 | reuse    | ok       |
| starts_clip_p99_log1p | fold_2022 | reuse    | ok       |
| starts_clip_p99_log1p | fold_2023 | reuse    | ok       |
| starts_clip_p99_log1p | fold_2024 | reuse    | ok       |

Elapsed seconds: `53.4`
