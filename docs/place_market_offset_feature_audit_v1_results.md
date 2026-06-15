# Place Market Offset Feature Audit V1

## Scope
- Target model: `C1_market_offset_fundamental`
- No retraining, no DB read, no feature dataset rebuild, no 2025/2026 adjustment.
- Missing required handover: `['keiba_ai_handover_market_offset_v1.md']`

## Top PredictionValuesChange
| feature                   | group                    |   weighted_mean |   unweighted_mean |   median |       min |      max |      std |   fold_count |   rank_weighted_mean |   rank_median |
|:--------------------------|:-------------------------|----------------:|------------------:|---------:|----------:|---------:|---------:|-------------:|---------------------:|--------------:|
| Year                      | race_metadata            |        26.852   |          26.852   | 25.359   | 22.6812   | 32.4655  | 4.40939  |            5 |                    1 |             1 |
| p_market                  | market_baseline          |        10.7086  |          10.7086  | 10.4952  |  8.97253  | 12.6152  | 1.35013  |            5 |                    2 |             2 |
| market_logit              | market_baseline          |         7.70524 |           7.70524 |  7.08273 |  6.37417  | 10.7541  | 1.80156  |            5 |                    3 |             3 |
| BaTaijyu                  | weight_and_gate          |         2.66054 |           2.66054 |  2.76136 |  1.89705  |  3.06668 | 0.452494 |            5 |                    4 |             4 |
| horse_distance_diff_last  | horse_recent_form        |         1.97219 |           1.97219 |  1.96236 |  1.72062  |  2.17975 | 0.189673 |            5 |                    5 |             5 |
| SyussoTosu                | race_metadata            |         1.77424 |           1.77424 |  1.63957 |  1.51031  |  2.13845 | 0.278508 |            5 |                    6 |             8 |
| horse_days_since_last     | horse_recent_form        |         1.73952 |           1.73952 |  1.95941 |  0.92036  |  2.26761 | 0.524779 |            5 |                    7 |             6 |
| trainer_past_starts       | trainer                  |         1.69361 |           1.69361 |  1.80424 |  1.22341  |  2.26164 | 0.445879 |            5 |                    8 |             7 |
| horse_surface_past_starts | horse_course_suitability |         1.55924 |           1.55924 |  1.34944 |  1.34525  |  1.90627 | 0.291427 |            5 |                    9 |            10 |
| KisyuCode                 | other                    |         1.29044 |           1.29044 |  1.2156  |  1.00093  |  1.95942 | 0.386692 |            5 |                   10 |            13 |
| horse_last3_avg_time      | horse_recent_form        |         1.27332 |           1.27332 |  1.2508  |  0.960859 |  1.51276 | 0.207705 |            5 |                   11 |            11 |
| TorokuTosu                | race_metadata            |         1.27135 |           1.27135 |  1.25014 |  0.954207 |  1.57931 | 0.234804 |            5 |                   12 |            12 |
| horse_last5_avg_haron_l3  | horse_recent_form        |         1.26782 |           1.26782 |  1.42214 |  0.989247 |  1.47345 | 0.244767 |            5 |                   13 |             9 |
| JyoCD                     | venue_identity           |         1.1698  |           1.1698  |  1.0804  |  0.910249 |  1.67072 | 0.291964 |            5 |                   14 |            18 |
| ChokyosiCode              | other                    |         1.13633 |           1.13633 |  1.11176 |  0.880287 |  1.49053 | 0.246097 |            5 |                   15 |            16 |

## Top LossFunctionChange
| feature                   | group                    |   weighted_mean |   unweighted_mean |      median |          min |         max |         std |   fold_count |   rank_weighted_mean |   rank_median |
|:--------------------------|:-------------------------|----------------:|------------------:|------------:|-------------:|------------:|------------:|-------------:|---------------------:|--------------:|
| Year                      | race_metadata            |     0.000637633 |       0.000637633 | 0.000562739 |  0.000402194 | 0.00112944  | 0.000285039 |            5 |                    1 |             1 |
| p_market                  | market_baseline          |     0.0002209   |       0.0002209   | 0.000202274 |  7.26082e-05 | 0.000342584 | 0.000102178 |            5 |                    2 |             2 |
| market_logit              | market_baseline          |     0.000124249 |       0.000124249 | 0.000133936 |  1.73233e-05 | 0.000198923 | 6.61285e-05 |            5 |                    3 |             3 |
| horse_surface_past_starts | horse_course_suitability |     9.38486e-05 |       9.38486e-05 | 8.43449e-05 |  5.47505e-05 | 0.000129805 | 3.30231e-05 |            5 |                    4 |             4 |
| BaTaijyu                  | weight_and_gate          |     5.06092e-05 |       5.06092e-05 | 4.94719e-05 |  3.41894e-05 | 7.21715e-05 | 1.52716e-05 |            5 |                    5 |             6 |
| horse_surface_top3_rate   | horse_course_suitability |     4.39127e-05 |       4.39127e-05 | 5.98048e-05 | -2.64247e-05 | 8.14164e-05 | 4.30678e-05 |            5 |                    6 |             5 |
| horse_days_since_last     | horse_recent_form        |     3.55678e-05 |       3.55678e-05 | 4.91328e-05 | -1.14977e-05 | 7.6388e-05  | 3.83974e-05 |            5 |                    7 |             7 |
| horse_surface_win_rate    | horse_course_suitability |     3.37169e-05 |       3.37169e-05 | 2.38715e-05 |  4.86707e-06 | 6.52986e-05 | 2.78669e-05 |            5 |                    8 |             8 |
| horse_last3_avg_time      | horse_recent_form        |     2.82349e-05 |       2.82349e-05 | 1.75992e-05 |  1.25538e-05 | 4.82295e-05 | 1.75888e-05 |            5 |                    9 |            13 |
| JyokenCD2                 | race_metadata            |     2.5871e-05  |       2.5871e-05  | 2.02791e-05 |  1.04241e-05 | 4.26982e-05 | 1.49385e-05 |            5 |                   10 |            10 |
| MonthDay                  | race_metadata            |     1.91433e-05 |       1.91433e-05 | 1.6563e-05  |  3.6486e-06  | 3.20776e-05 | 1.21408e-05 |            5 |                   11 |            15 |
| horse_past_starts         | horse_recent_form        |     1.83385e-05 |       1.83385e-05 | 2.24941e-05 | -3.53422e-06 | 3.35196e-05 | 1.37202e-05 |            5 |                   12 |             9 |
| horse_last3_ren_rate      | horse_recent_form        |     1.74537e-05 |       1.74537e-05 | 1.09418e-05 |  4.47499e-06 | 4.2331e-05  | 1.48902e-05 |            5 |                   13 |            21 |
| SexCD                     | weight_and_gate          |     1.51857e-05 |       1.51857e-05 | 1.48012e-05 |  1.01565e-05 | 2.2342e-05  | 4.66318e-06 |            5 |                   14 |            17 |
| horse_baba_past_starts    | horse_course_suitability |     1.51533e-05 |       1.51533e-05 | 1.32914e-05 |  7.51055e-06 | 2.72829e-05 | 8.31939e-06 |            5 |                   15 |            19 |

## Top SHAP
| feature                     | group                    |   mean_abs_shap |   mean_signed_shap |   median_abs_shap |   p90_abs_shap |   p99_abs_shap |   positive_share |   sample_rows |
|:----------------------------|:-------------------------|----------------:|-------------------:|------------------:|---------------:|---------------:|-----------------:|--------------:|
| Year                        | race_metadata            |      0.0918813  |        0.09152     |        0.104385   |      0.139792  |      0.156754  |          0.9656  |         12500 |
| p_market                    | market_baseline          |      0.0329264  |       -0.00351726  |        0.0315142  |      0.050813  |      0.201452  |          0.41648 |         12500 |
| market_logit                | market_baseline          |      0.0258739  |       -0.00260373  |        0.0238145  |      0.0445285 |      0.125917  |          0.40856 |         12500 |
| horse_surface_past_starts   | horse_course_suitability |      0.0210046  |        0.00174188  |        0.0184239  |      0.0364005 |      0.0465869 |          0.70808 |         12500 |
| horse_distance_diff_last    | horse_recent_form        |      0.0128502  |       -0.000710347 |        0.0102825  |      0.0276484 |      0.0423817 |          0.46696 |         12500 |
| horse_last3_place_paid_rate | horse_recent_form        |      0.0104442  |       -0.00137583  |        0.00704662 |      0.0279477 |      0.0403067 |          0.70968 |         12500 |
| BaTaijyu                    | weight_and_gate          |      0.0100864  |       -9.26601e-05 |        0.00725099 |      0.0178051 |      0.0785686 |          0.6808  |         12500 |
| trainer_past_starts         | trainer                  |      0.00997145 |        0.00712654  |        0.00935048 |      0.0176992 |      0.0348689 |          0.824   |         12500 |
| horse_last3_avg_time        | horse_recent_form        |      0.00961094 |       -0.000394537 |        0.00726107 |      0.0212988 |      0.0355392 |          0.6892  |         12500 |
| horse_past_starts           | horse_recent_form        |      0.00918013 |        0.00129047  |        0.00819013 |      0.0161983 |      0.0277083 |          0.70736 |         12500 |
| KisyuCode                   | other                    |      0.00917408 |       -0.00390157  |        0.00642108 |      0.0216599 |      0.031163  |          0.4268  |         12500 |
| ChokyosiCode                | other                    |      0.00860796 |       -0.00413583  |        0.00672701 |      0.0177843 |      0.0331715 |          0.4308  |         12500 |
| horse_days_since_last       | horse_recent_form        |      0.00848089 |       -0.000161191 |        0.005684   |      0.0154231 |      0.0564012 |          0.668   |         12500 |
| horse_surface_top3_rate     | horse_course_suitability |      0.0083294  |        0.00226753  |        0.0051874  |      0.0193014 |      0.0341834 |          0.66112 |         12500 |
| horse_surface_win_rate      | horse_course_suitability |      0.00776484 |        0.000864935 |        0.00472123 |      0.0177402 |      0.0298547 |          0.4308  |         12500 |

## Group Permutation
| group                     |   logloss_delta_mean |   brier_delta_mean |   ece_delta_mean |   ev_roi_spearman_delta_mean |   mean_abs_p_change |   ev_ge_1_count_delta_mean |   feature_count |
|:--------------------------|---------------------:|-------------------:|-----------------:|-----------------------------:|--------------------:|---------------------------:|----------------:|
| market_baseline           |          0.300132    |        0.0818292   |      0.154742    |                   -0.258159  |         0.215849    |                     3003.6 |               2 |
| horse_recent_form         |          0.0013288   |        0.000516367 |      0.00100292  |                   -0.237559  |         0.0129327   |                        0.2 |              20 |
| horse_course_suitability  |          0.000939179 |        0.00033701  |      0.00168015  |                   -0.466378  |         0.00775816  |                       -2   |              12 |
| race_metadata             |          0.000380633 |        0.000138531 |     -0.000593185 |                   -0.256083  |         0.00613949  |                        2   |              16 |
| weight_and_gate           |          0.000354796 |        0.000127668 |     -0.000610271 |                    0.0769017 |         0.00522767  |                       -0.4 |               8 |
| trainer                   |          0.000329031 |        0.000129723 |      0.000314017 |                   -0.0218857 |         0.00354259  |                        1   |               4 |
| jockey_overall            |          0.00012379  |        5.22259e-05 |     -0.00059371  |                   -0.0291129 |         0.00363884  |                       -0.4 |               4 |
| course_context            |          0.000116514 |        3.79637e-05 |     -0.00037956  |                   -0.147975  |         0.00255525  |                       -0.8 |               5 |
| other                     |          7.18999e-05 |        2.56878e-05 |      0.000552299 |                   -0.165149  |         0.00418537  |                        0.2 |               3 |
| jockey_course_suitability |          6.38851e-05 |        1.70581e-05 |      0.000152646 |                    0.0339559 |         0.00332491  |                       -0.2 |               6 |
| distance                  |          3.6768e-05  |        1.47726e-05 |      0.000156813 |                   -0.0980952 |         0.000646658 |                        0.2 |               1 |
| venue_identity            |         -2.84673e-06 |        4.70582e-07 |      0.000244125 |                   -0.0617311 |         0.00260386  |                        0.2 |               1 |
| horse_jockey_pair         |         -1.15628e-05 |       -7.67329e-06 |     -1.09108e-05 |                   -0.0206273 |         0.00119987  |                       -0.2 |               3 |

## EV Count Shift
|   Year |   eligible_rows |   races |   ev_ge_1_count |   ev_ge_1_rate |   market_only_ev_ge_1_count |
|-------:|----------------:|--------:|----------------:|---------------:|----------------------------:|
|   2020 |           47876 |    3456 |             106 |    0.00221405  |                          93 |
|   2021 |           47476 |    3456 |             142 |    0.00299098  |                         107 |
|   2022 |           46840 |    3456 |              49 |    0.00104611  |                          40 |
|   2023 |           47273 |    3456 |              55 |    0.00116345  |                          48 |
|   2024 |           46752 |    3454 |              22 |    0.000470568 |                          25 |
|   2025 |           47497 |    3455 |             655 |    0.0137903   |                          64 |
|   2026 |           21276 |    1506 |             253 |    0.0118913   |                          46 |

## Additional 2025/2026 Diagnostic
| period          |   Year | scenario                              | group                                |   rows |   logloss_delta |   brier_delta |   ece_delta |   mean_abs_p_final_change |   ev_ge_1_count_delta |   base_ev_ge_1_count |   permuted_ev_ge_1_count |
|:----------------|-------:|:--------------------------------------|:-------------------------------------|-------:|----------------:|--------------:|------------:|--------------------------:|----------------------:|---------------------:|-------------------------:|
| diagnostic_2025 |   2025 | market_baseline_baseline_only         | market_baseline                      |   8000 |     0.286445    |   0.0741974   |  0.136068   |                 0.199483  |                  2675 |                  113 |                     2788 |
| diagnostic_2025 |   2025 | residual_market_features_only         | residual_p_market_market_logit       |   8000 |     0.00280957  |   0.00118906  |  0.00447033 |                 0.0234987 |                    46 |                  113 |                      159 |
| diagnostic_2025 |   2025 | baseline_and_residual_market_features | market_baseline_plus_residual_market |   8000 |     0.322523    |   0.0825047   |  0.148763   |                 0.209717  |                  2729 |                  113 |                     2842 |
| diagnostic_2025 |   2025 | feature_group                         | horse_recent_form                    |   8000 |     0.00564438  |   0.00150352  |  0.0102583  |                 0.0334587 |                    25 |                  113 |                      138 |
| diagnostic_2025 |   2025 | feature_group                         | horse_course_suitability             |   8000 |     0.00441209  |   0.00152642  |  0.00791258 |                 0.0286919 |                    21 |                  113 |                      134 |
| diagnostic_2025 |   2025 | feature_group                         | race_metadata                        |   8000 |     0.00113529  |   0.00033075  |  0.00398201 |                 0.0209796 |                     5 |                  113 |                      118 |
| diagnostic_2025 |   2025 | feature_group                         | weight_and_gate                      |   8000 |    -0.000165017 |  -0.000129447 |  0.00473387 |                 0.0219166 |                   -16 |                  113 |                       97 |
| diagnostic_2026 |   2026 | market_baseline_baseline_only         | market_baseline                      |   8000 |     0.303524    |   0.0756661   |  0.126337   |                 0.191611  |                  2596 |                   94 |                     2690 |
| diagnostic_2026 |   2026 | residual_market_features_only         | residual_p_market_market_logit       |   8000 |     0.00587072  |   0.00245271  |  0.00709006 |                 0.0243733 |                    65 |                   94 |                      159 |
| diagnostic_2026 |   2026 | baseline_and_residual_market_features | market_baseline_plus_residual_market |   8000 |     0.348891    |   0.0861942   |  0.140867   |                 0.203685  |                  2643 |                   94 |                     2737 |
| diagnostic_2026 |   2026 | feature_group                         | horse_recent_form                    |   8000 |     0.00517612  |   0.00124055  |  0.00248969 |                 0.0314033 |                    15 |                   94 |                      109 |
| diagnostic_2026 |   2026 | feature_group                         | horse_course_suitability             |   8000 |     0.0047418   |   0.00177108  |  0.00979733 |                 0.0256628 |                    -6 |                   94 |                       88 |
| diagnostic_2026 |   2026 | feature_group                         | race_metadata                        |   8000 |     0.00186158  |   0.000524917 |  0.00131764 |                 0.0175504 |                    15 |                   94 |                      109 |
| diagnostic_2026 |   2026 | feature_group                         | weight_and_gate                      |   8000 |     0.00300523  |   0.00111378  |  0.00368476 |                 0.0204599 |                   -23 |                   94 |                       71 |

## Previous Fold Model On 2025
| comparison_scope                              | constraint                                                                                                      |   rows |   residual_raw_mean |   residual_raw_median |   residual_raw_p10 |   residual_raw_p50 |   residual_raw_p90 |   residual_raw_p95 |   residual_raw_p99 |   residual_raw_abs_p90 |   residual_raw_abs_p95 |   residual_raw_abs_p99 |   ev_ge_1_count |   ev_lt_1_to_ge_1_by_residual |   ev_ge_1_to_lt_1_by_residual |
|:----------------------------------------------|:----------------------------------------------------------------------------------------------------------------|-------:|--------------------:|----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------------:|-----------------------:|-----------------------:|----------------:|------------------------------:|------------------------------:|
| official_2025_final_model                     | 2025 p_market/market_logit fixed from saved official predictions; market baseline model update is not isolated. |  47497 |          -0.184663  |            -0.167412  |          -0.697939 |         -0.167412  |           0.299923 |           0.423913 |           0.686344 |               0.718081 |               0.878482 |               1.24062  |             655 |                           642 |                            51 |
| previous_fold_2024_model_on_2025_fixed_market | 2025 p_market/market_logit fixed from saved official predictions; market baseline model update is not isolated. |  47497 |          -0.0319784 |            -0.0225946 |          -0.188042 |         -0.0225946 |           0.105045 |           0.139652 |           0.247378 |               0.202193 |               0.257839 |               0.392479 |              70 |                            42 |                            36 |
| previous_minus_official_residual              | diagnostic difference only                                                                                      |  47497 |           0.152685  |             0.134377  |          -0.261137 |          0.134377  |           0.589251 |           0.733423 |           1.03015  |               0.607618 |               0.748361 |               1.04299  |             nan |                           nan |                           nan |

## Year / Market SHAP 2024 vs 2025
|   Year | feature      |   mean_abs_shap |   mean_signed_shap |   median_abs_shap |   p90_abs_shap |   p99_abs_shap |   positive_share |   sample_rows |
|-------:|:-------------|----------------:|-------------------:|------------------:|---------------:|---------------:|-----------------:|--------------:|
|   2024 | Year         |       0.0679507 |        0.0678268   |         0.0765308 |      0.105882  |       0.121986 |           0.9792 |          2500 |
|   2024 | p_market     |       0.0352546 |       -0.00239889  |         0.0318288 |      0.0552629 |       0.261718 |           0.4472 |          2500 |
|   2024 | market_logit |       0.0189336 |       -0.000337447 |         0.0159989 |      0.0303437 |       0.161381 |           0.4076 |          2500 |
|   2025 | Year         |       0.0917937 |        0.0917073   |         0.094424  |      0.139592  |       0.169741 |           0.9956 |          2500 |
|   2025 | p_market     |       0.0798018 |       -0.0179775   |         0.0690723 |      0.148301  |       0.396997 |           0.346  |          2500 |
|   2025 | market_logit |       0.0408389 |       -0.006566    |         0.0355466 |      0.0721153 |       0.200098 |           0.4276 |          2500 |

## Updated Cause Conclusion
2024->2025 EV>=1急増は複数要因。市場単体のEV>=1は25->64で増加したが、C1は22->655まで増えており、主因はCatBoost残差によるEV<1からEV>=1への上抜け(14->642)。前年fold CatBoostを2025の保存済みmarket_logitへ適用した診断、Year/市場特徴SHAP、baseline-only/residual-market shuffleの結果から、CatBoost fold更新、Year特徴、baselineと残差側市場特徴の二重利用、2025市場分布変化が重なった可能性が高い。DB/再学習なしの監査ではリークとは断定しない。

## Course Structure
| concept               | status        | matched_columns     | included_in_c1   | notes   |
|:----------------------|:--------------|:--------------------|:-----------------|:--------|
| turn_direction        | indirect_only | JyoCD               | True             |         |
| right_turn            | absent        |                     | False            |         |
| left_turn             | absent        |                     | False            |         |
| inner                 | ambiguous     | CourseKubunCD       | True             |         |
| outer                 | ambiguous     | CourseKubunCD       | True             |         |
| inner_outer           | ambiguous     | CourseKubunCD       | True             |         |
| straight              | absent        |                     | False            |         |
| straight_length       | absent        |                     | False            |         |
| elevation             | absent        |                     | False            |         |
| height_difference     | absent        |                     | False            |         |
| slope                 | absent        |                     | False            |         |
| gradient              | absent        |                     | False            |         |
| final_slope           | absent        |                     | False            |         |
| steep_slope           | absent        |                     | False            |         |
| corner_count          | absent        |                     | False            |         |
| corner_radius         | absent        |                     | False            |         |
| first_corner_distance | absent        |                     | False            |         |
| course_width          | absent        |                     | False            |         |
| small_turn            | indirect_only | JyoCD               | True             |         |
| small_course          | indirect_only | JyoCD               | True             |         |
| course_id             | encoded       | JyoCD,TrackCD,Kyori | True             |         |

## Pedigree
| term           | status   | matched_columns   | included_features   | notes                                                                      |
|:---------------|:---------|:------------------|:--------------------|:---------------------------------------------------------------------------|
| sire           | absent   |                   |                     |                                                                            |
| dam            | absent   |                   |                     |                                                                            |
| damsire        | absent   |                   |                     |                                                                            |
| broodmare_sire | absent   |                   |                     |                                                                            |
| father         | absent   |                   |                     |                                                                            |
| mother         | absent   |                   |                     |                                                                            |
| pedigree       | absent   |                   |                     |                                                                            |
| bloodline      | absent   |                   |                     |                                                                            |
| lineage        | absent   |                   |                     |                                                                            |
| Ketto          | unknown  |                   |                     | KettoNum/Bamei are identity/raw fields, not pedigree-performance features. |
| Hansyoku       | absent   |                   |                     |                                                                            |
| Bamei          | unknown  |                   |                     | KettoNum/Bamei are identity/raw fields, not pedigree-performance features. |
| 父              | absent   |                   |                     |                                                                            |
| 母              | absent   |                   |                     |                                                                            |
| 血統             | absent   |                   |                     |                                                                            |
| 種牡馬            | absent   |                   |                     |                                                                            |
