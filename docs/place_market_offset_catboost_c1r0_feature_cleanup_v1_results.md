# C1R0 Feature Cleanup Results

- Fixed reference model: `C1R0_pure_market_offset_fixed300`
- Tree count: `300` fixed
- Selection years: `2020-2024 only`
- 2025/2026: `fixed diagnostic only`
- DB read: `not performed`

## Feature Cleanup Decisions
| ablation_name                 | drop_features                             | ablation_required   | decision_reason                                                                                                               |   best_shap_rank_in_group |   best_pvc_rank_in_group |
|:------------------------------|:------------------------------------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------|--------------------------:|-------------------------:|
| drop_person_codes             | KisyuCode,ChokyosiCode                    | True                | unknown category max=0.0803; high-cardinality identity categorical; KisyuCode/ChokyosiCode are top SHAP/PVC                   |                         2 |                        3 |
| drop_global_cumulative_starts | jockey_past_starts,trainer_past_starts    | True                | raw cumulative starts with no window/smoothing; trainer_past_starts is top SHAP/PVC; year proxy signal max_abs_spearman=0.533 |                         1 |                        1 |
| drop_raw_body_weight          | BaTaijyu                                  | True                | raw current-race body weight; production availability timing is operationally sensitive                                       |                         8 |                        2 |
| drop_unadjusted_raw_time      | horse_last3_avg_time,horse_last5_avg_time | True                | horse_last*_avg_time are simple raw averages, not distance/venue/going adjusted                                               |                        13 |                       17 |
| drop_meeting_admin            | Kaiji,Nichiji,RaceNum                     | True                | Kaiji/Nichiji/RaceNum are schedule/admin fields and can encode meeting/order effects                                          |                        73 |                       73 |

## 2020-2024 Comparison
| model_key                                            |   mean_logloss |   mean_brier |   mean_ece |   mean_calibration_slope |   residual_std_mean |   residual_std_cv |   abs_residual_p95_mean |   abs_residual_p99_mean |   ev_ge_1_count_sum |   ev_ge_1_count_cv |   ev_roi_spearman_mean |   mean_roi |
|:-----------------------------------------------------|---------------:|-------------:|-----------:|-------------------------:|--------------------:|------------------:|------------------------:|------------------------:|--------------------:|-------------------:|-----------------------:|-----------:|
| C1R0_fixed300_ablation_drop_person_codes             |       0.405716 |     0.130356 | 0.00336398 |                 0.999845 |            0.160968 |         0.0754061 |                0.316674 |                0.506951 |                1376 |           0.516279 |               0.495238 |    111.485 |
| C1R0_fixed300_ablation_drop_meeting_admin            |       0.405987 |     0.130437 | 0.00392139 |                 0.999782 |            0.175753 |         0.0469649 |                0.384142 |                0.577185 |                1498 |           0.360694 |               0.442857 |    107.665 |
| C1R0_pure_market_offset_fixed300_base                |       0.406029 |     0.130457 | 0.00410332 |                 0.99977  |            0.178185 |         0.0488705 |                0.384876 |                0.582732 |                1621 |           0.398592 |               0.6      |    111.96  |
| C1R0_fixed300_ablation_drop_unadjusted_raw_time      |       0.406179 |     0.130496 | 0.00392249 |                 0.999826 |            0.178936 |         0.0430328 |                0.391961 |                0.587485 |                1586 |           0.35533  |               0.319048 |    111.858 |
| C1R0_fixed300_ablation_drop_global_cumulative_starts |       0.406252 |     0.130549 | 0.00331239 |                 0.999928 |            0.208389 |         0.0232805 |                0.48452  |                0.76186  |                1823 |           0.320457 |               0.628571 |    111.852 |
| C1R0_fixed300_ablation_drop_raw_body_weight          |       0.406418 |     0.130578 | 0.00391583 |                 0.999805 |            0.18198  |         0.0440899 |                0.404314 |                0.595306 |                1800 |           0.367885 |               0.47619  |    107.899 |

## Selected Model
- `C1R0_fixed300_ablation_drop_person_codes`
- Reason: Selected using 2020-2024 probability metrics first, then residual and EV count stability; ROI auxiliary only.

## 2025/2026 Diagnostic
| model_key                                | period              |   Year |   rows |   positives |   logloss |    brier |      auc |        ece |       mce |   calibration_slope |   calibration_intercept |
|:-----------------------------------------|:--------------------|-------:|-------:|------------:|----------:|---------:|---------:|-----------:|----------:|--------------------:|------------------------:|
| C1R0_fixed300_ablation_drop_person_codes | latest_holdout_2026 |   2026 |  21276 |        4500 |  0.383668 | 0.122487 | 0.835785 | 0.00799261 | 0.0364075 |            1.05757  |             0.0380643   |
| C1R0_fixed300_ablation_drop_person_codes | test_2025           |   2025 |  47497 |       10276 |  0.401109 | 0.1288   | 0.820061 | 0.0049654  | 0.020754  |            1.0043   |            -0.000244553 |
| C1R0_pure_market_offset_fixed300_base    | latest_holdout_2026 |   2026 |  21276 |        4500 |  0.383807 | 0.122618 | 0.83531  | 0.00629057 | 0.0476059 |            1.053    |             0.0678943   |
| C1R0_pure_market_offset_fixed300_base    | test_2025           |   2025 |  47497 |       10276 |  0.402056 | 0.129081 | 0.818914 | 0.00354165 | 0.031852  |            0.995368 |             0.00874076  |

## Reuse And Training Log
| ablation_name                 | fold            | action   | reason   |   rows | model_path                                                                                                          |
|:------------------------------|:----------------|:---------|:---------|-------:|:--------------------------------------------------------------------------------------------------------------------|
| drop_person_codes             | fold_2020       | reuse    | ok       |  47876 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\folds\fold_2020\model.cbm             |
| drop_person_codes             | fold_2021       | reuse    | ok       |  47476 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\folds\fold_2021\model.cbm             |
| drop_person_codes             | fold_2022       | reuse    | ok       |  46840 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\folds\fold_2022\model.cbm             |
| drop_person_codes             | fold_2023       | reuse    | ok       |  47273 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\folds\fold_2023\model.cbm             |
| drop_person_codes             | fold_2024       | reuse    | ok       |  46752 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\folds\fold_2024\model.cbm             |
| drop_global_cumulative_starts | fold_2020       | reuse    | ok       |  47876 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_global_cumulative_starts\folds\fold_2020\model.cbm |
| drop_global_cumulative_starts | fold_2021       | reuse    | ok       |  47476 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_global_cumulative_starts\folds\fold_2021\model.cbm |
| drop_global_cumulative_starts | fold_2022       | reuse    | ok       |  46840 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_global_cumulative_starts\folds\fold_2022\model.cbm |
| drop_global_cumulative_starts | fold_2023       | reuse    | ok       |  47273 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_global_cumulative_starts\folds\fold_2023\model.cbm |
| drop_global_cumulative_starts | fold_2024       | reuse    | ok       |  46752 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_global_cumulative_starts\folds\fold_2024\model.cbm |
| drop_raw_body_weight          | fold_2020       | reuse    | ok       |  47876 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_raw_body_weight\folds\fold_2020\model.cbm          |
| drop_raw_body_weight          | fold_2021       | reuse    | ok       |  47476 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_raw_body_weight\folds\fold_2021\model.cbm          |
| drop_raw_body_weight          | fold_2022       | reuse    | ok       |  46840 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_raw_body_weight\folds\fold_2022\model.cbm          |
| drop_raw_body_weight          | fold_2023       | reuse    | ok       |  47273 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_raw_body_weight\folds\fold_2023\model.cbm          |
| drop_raw_body_weight          | fold_2024       | reuse    | ok       |  46752 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_raw_body_weight\folds\fold_2024\model.cbm          |
| drop_unadjusted_raw_time      | fold_2020       | reuse    | ok       |  47876 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_unadjusted_raw_time\folds\fold_2020\model.cbm      |
| drop_unadjusted_raw_time      | fold_2021       | reuse    | ok       |  47476 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_unadjusted_raw_time\folds\fold_2021\model.cbm      |
| drop_unadjusted_raw_time      | fold_2022       | reuse    | ok       |  46840 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_unadjusted_raw_time\folds\fold_2022\model.cbm      |
| drop_unadjusted_raw_time      | fold_2023       | reuse    | ok       |  47273 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_unadjusted_raw_time\folds\fold_2023\model.cbm      |
| drop_unadjusted_raw_time      | fold_2024       | reuse    | ok       |  46752 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_unadjusted_raw_time\folds\fold_2024\model.cbm      |
| drop_meeting_admin            | fold_2020       | reuse    | ok       |  47876 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_meeting_admin\folds\fold_2020\model.cbm            |
| drop_meeting_admin            | fold_2021       | reuse    | ok       |  47476 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_meeting_admin\folds\fold_2021\model.cbm            |
| drop_meeting_admin            | fold_2022       | reuse    | ok       |  46840 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_meeting_admin\folds\fold_2022\model.cbm            |
| drop_meeting_admin            | fold_2023       | reuse    | ok       |  47273 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_meeting_admin\folds\fold_2023\model.cbm            |
| drop_meeting_admin            | fold_2024       | reuse    | ok       |  46752 | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_meeting_admin\folds\fold_2024\model.cbm            |
| drop_person_codes             | final_2025_2026 | reuse    | ok       |    nan | models\place_market_offset_catboost_c1r0_feature_cleanup_v1\drop_person_codes\final\model.cbm                       |

Elapsed seconds: `46.3`
