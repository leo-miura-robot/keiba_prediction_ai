# C1R0 Pure Market Offset Results

- Model: `C1R0_pure_market_offset`
- DB read: `not_read; existing parquet feature dataset only`
- Feature dataset rebuild: `not_performed`
- Selection years: `2020-2024 only`
- 2025/2026: `fixed diagnostic only`
- C1R0 feature count: `81`
- C1 feature count: `85`
- Baseline raw consistency max abs: `4.767243293457568e-07`

## 2020-2024 Model Comparison
| model_key                    | period               |   validation logloss |   validation Brier |   validation ECE |   validation calibration slope |   high-odds calibration gap |   EV-ROI Spearman |   EV>=1 count |   EV>=1 ROI |   EV>=1.05 count |   EV>=1.05 ROI |   2025 ROI |   2026 ROI |   combined ROI |   top5 removed ROI | bootstrap CI         |
|:-----------------------------|:---------------------|---------------------:|-------------------:|-----------------:|-------------------------------:|----------------------------:|------------------:|--------------:|------------:|-----------------:|---------------:|-----------:|-----------:|---------------:|-------------------:|:---------------------|
| B_market_baseline            | validation_2020_2024 |             0.406278 |           0.130581 |      5.53728e-09 |                       0.99974  |                 0.000598115 |          0.857143 |           459 |     129.521 |              198 |        122.475 |        nan |        nan |            nan |            114.537 | 100.64,129.90,158.50 |
| C1R0_pure_market_offset      | validation_2020_2024 |             0.405605 |           0.130302 |      4.71177e-09 |                       0.999741 |                 0.000539523 |          0.952381 |           648 |     129.707 |              278 |        132.734 |        nan |        nan |            nan |            119.114 | 105.97,129.71,155.09 |
| C1_market_offset_fundamental | validation_2020_2024 |             0.405918 |           0.130382 |      0.00302126  |                       0.997545 |                -0.000210662 |          0.928571 |           374 |     132.112 |              159 |        158.931 |        nan |        nan |            nan |            113.117 | 98.93,131.73,167.23  |

## 2025/2026 Fixed Diagnostic
| model_key                    | period              |   validation logloss |   validation Brier |   validation ECE |   validation calibration slope |   high-odds calibration gap |   EV-ROI Spearman |   EV>=1 count |   EV>=1 ROI |   EV>=1.05 count |   EV>=1.05 ROI |   2025 ROI |   2026 ROI |   combined ROI |   top5 removed ROI | bootstrap CI        |
|:-----------------------------|:--------------------|---------------------:|-------------------:|-----------------:|-------------------------------:|----------------------------:|------------------:|--------------:|------------:|-----------------:|---------------:|-----------:|-----------:|---------------:|-------------------:|:--------------------|
| B_market_baseline            | latest_holdout_2026 |             0.384858 |           0.122898 |       0.00802953 |                       1.0576   |                 0.00644482  |         0.52381   |            75 |    126.933  |               33 |       180.909  |   nan      |   126.933  |       126.933  |            54.1429 | 62.09,123.40,205.86 |
| B_market_baseline            | test_2025           |             0.4024   |           0.129094 |       0.00369608 |                       0.998637 |                 0.000752001 |         0.5       |           109 |     99.7248 |               42 |       111.667  |    99.7248 |   nan      |        99.7248 |            44.6154 | 44.87,97.46,165.74  |
| C1R0_pure_market_offset      | latest_holdout_2026 |             0.391514 |           0.125193 |       0.0231545  |                       0.964485 |                -0.00176184  |        -0.0952381 |           330 |     82.5152 |              220 |        84.2273 |   nan      |    82.5152 |        82.5152 |            67.7538 | 64.20,82.03,102.28  |
| C1R0_pure_market_offset      | test_2025           |             0.409889 |           0.131218 |       0.0184495  |                       0.910023 |                -0.00473631  |         0.261905  |           879 |    105.677  |              566 |       109.541  |   105.677  |   nan      |       105.677  |            95.9954 | 90.17,106.22,123.77 |
| C1_market_offset_fundamental | latest_holdout_2026 |             0.389038 |           0.124523 |       0.0243146  |                       0.969618 |                -0.00389187  |         0.785714  |           253 |    105.85   |              165 |       112.061  |   nan      |   105.85   |       105.85   |            85.8468 | 80.52,104.81,135.26 |
| C1_market_offset_fundamental | test_2025           |             0.408437 |           0.131178 |       0.0204331  |                       0.912848 |                -0.00651666  |         0.261905  |           655 |     93.6183 |              399 |        98.4962 |    93.6183 |   nan      |        93.6183 |            82.6462 | 76.47,93.62,110.50  |

## C1R0 Top PVC
| feature                   | group                    |   weighted_mean |   unweighted_mean |   median |      min |      max |      std |   fold_count |
|:--------------------------|:-------------------------|----------------:|------------------:|---------:|---------:|---------:|---------:|-------------:|
| trainer_past_starts       | trainer                  |        16.9227  |          16.9227  | 19.7121  | 11.8552  | 20.921   | 4.51272  |            5 |
| BaTaijyu                  | weight_and_gate          |         4.50536 |           4.50536 |  4.57375 |  4.12321 |  4.72511 | 0.226048 |            5 |
| KisyuCode                 | other                    |         3.59895 |           3.59895 |  3.70481 |  3.07371 |  4.19649 | 0.487226 |            5 |
| jockey_past_starts        | jockey_overall           |         3.35995 |           3.35995 |  3.26109 |  2.92547 |  4.04051 | 0.420527 |            5 |
| horse_last1_avg_finish    | horse_recent_form        |         3.2294  |           3.2294  |  2.95163 |  2.85215 |  3.9161  | 0.484976 |            5 |
| horse_surface_top3_rate   | horse_course_suitability |         2.8259  |           2.8259  |  2.72983 |  2.47766 |  3.32242 | 0.322817 |            5 |
| horse_last5_avg_finish    | horse_recent_form        |         2.7292  |           2.7292  |  2.96177 |  1.74934 |  3.70942 | 0.854603 |            5 |
| horse_distance_diff_last  | horse_recent_form        |         2.61271 |           2.61271 |  2.71482 |  2.27746 |  2.8552  | 0.233982 |            5 |
| horse_days_since_last     | horse_recent_form        |         2.43427 |           2.43427 |  2.62627 |  1.78492 |  3.07753 | 0.536949 |            5 |
| MonthDay                  | race_metadata            |         2.34357 |           2.34357 |  2.3064  |  1.95963 |  2.93736 | 0.406518 |            5 |
| ChokyosiCode              | other                    |         2.2948  |           2.2948  |  2.4375  |  1.22986 |  3.0106  | 0.661416 |            5 |
| horse_surface_past_starts | horse_course_suitability |         2.26403 |           2.26403 |  2.3817  |  1.87234 |  2.59543 | 0.289878 |            5 |
| horse_baba_past_starts    | horse_course_suitability |         2.16188 |           2.16188 |  2.10179 |  1.94743 |  2.54351 | 0.225846 |            5 |
| horse_last3_win_rate      | horse_recent_form        |         2.13429 |           2.13429 |  2.06467 |  1.87937 |  2.61443 | 0.27972  |            5 |
| horse_last3_avg_finish    | horse_recent_form        |         2.01452 |           2.01452 |  1.85612 |  1.27494 |  2.86052 | 0.620538 |            5 |

## C1R0 Top SHAP
| feature                      | group                     |   mean_abs_shap |   mean_signed_shap |   median_abs_shap |   p90_abs_shap |   p99_abs_shap |   positive_share |   sample_rows |
|:-----------------------------|:--------------------------|----------------:|-------------------:|------------------:|---------------:|---------------:|-----------------:|--------------:|
| trainer_past_starts          | trainer                   |      0.055995   |        0.0455248   |        0.0505673  |      0.100077  |      0.167759  |         0.949    |          6000 |
| horse_surface_past_starts    | horse_course_suitability  |      0.0175278  |        0.00197921  |        0.0152296  |      0.0295365 |      0.0434278 |         0.716667 |          6000 |
| horse_surface_top3_rate      | horse_course_suitability  |      0.0165037  |        0.00235779  |        0.012734   |      0.0325792 |      0.052144  |         0.556833 |          6000 |
| jockey_past_starts           | jockey_overall            |      0.0157564  |        0.00722649  |        0.0142192  |      0.0272729 |      0.050433  |         0.776667 |          6000 |
| horse_distance_diff_last     | horse_recent_form         |      0.0138683  |       -0.000553275 |        0.0109312  |      0.0321994 |      0.04343   |         0.617667 |          6000 |
| horse_last3_win_rate         | horse_recent_form         |      0.0121273  |       -0.000347278 |        0.00763669 |      0.0294696 |      0.0649984 |         0.813833 |          6000 |
| KisyuCode                    | other                     |      0.012037   |       -0.00411379  |        0.0086655  |      0.0275201 |      0.0436379 |         0.402167 |          6000 |
| horse_baba_past_starts       | horse_course_suitability  |      0.0118003  |        0.00236594  |        0.0110911  |      0.0207895 |      0.0307921 |         0.688833 |          6000 |
| BaTaijyu                     | weight_and_gate           |      0.0104875  |        0.000175918 |        0.0073764  |      0.0195203 |      0.0750397 |         0.707167 |          6000 |
| ChokyosiCode                 | other                     |      0.00965708 |       -0.00404885  |        0.00726619 |      0.0205744 |      0.036449  |         0.478    |          6000 |
| horse_last3_avg_time         | horse_recent_form         |      0.00907726 |        0.000411012 |        0.00686487 |      0.0199882 |      0.0330872 |         0.721167 |          6000 |
| jockey_dist_band_past_starts | jockey_course_suitability |      0.00885559 |        0.00457746  |        0.00851165 |      0.0132745 |      0.0306054 |         0.769333 |          6000 |
| horse_days_since_last        | horse_recent_form         |      0.00823593 |       -5.72161e-05 |        0.00553533 |      0.0159739 |      0.0524766 |         0.731667 |          6000 |
| Barei                        | weight_and_gate           |      0.00798479 |        0.000620776 |        0.00500475 |      0.0170092 |      0.0473219 |         0.872    |          6000 |
| horse_last3_top3_rate        | horse_recent_form         |      0.00785046 |       -5.05303e-05 |        0.00545584 |      0.0196513 |      0.0299679 |         0.769333 |          6000 |

## EV Crossing
| model_key                    | period               |   Year |   rows |   market_only_ev_ge_1 |   final_ev_ge_1 |   market_lt1_to_final_ge1 |   market_ge1_to_final_lt1 |   ev_ge_1_rate |   ev_ge_1_year_over_year_ratio |
|:-----------------------------|:---------------------|-------:|-------:|----------------------:|----------------:|--------------------------:|--------------------------:|---------------:|-------------------------------:|
| C1R0_pure_market_offset      | latest_holdout_2026  |   2026 |  21276 |                    46 |             330 |                       321 |                        37 |    0.0155104   |                       0.375427 |
| C1R0_pure_market_offset      | test_2025            |   2025 |  47497 |                    64 |             879 |                       862 |                        47 |    0.0185064   |                      11.5658   |
| C1R0_pure_market_offset      | validation_2020_2024 |   2020 |  47876 |                    93 |             171 |                       105 |                        27 |    0.00357173  |                     nan        |
| C1R0_pure_market_offset      | validation_2020_2024 |   2021 |  47476 |                   107 |             211 |                       129 |                        25 |    0.00444435  |                       1.23392  |
| C1R0_pure_market_offset      | validation_2020_2024 |   2022 |  46840 |                    40 |              83 |                        53 |                        10 |    0.00177199  |                       0.393365 |
| C1R0_pure_market_offset      | validation_2020_2024 |   2023 |  47273 |                    48 |             107 |                        85 |                        26 |    0.00226345  |                       1.28916  |
| C1R0_pure_market_offset      | validation_2020_2024 |   2024 |  46752 |                    25 |              76 |                        63 |                        12 |    0.0016256   |                       0.71028  |
| C1_market_offset_fundamental | latest_holdout_2026  |   2026 |  21276 |                    46 |             253 |                       245 |                        38 |    0.0118913   |                       0.38626  |
| C1_market_offset_fundamental | test_2025            |   2025 |  47497 |                    64 |             655 |                       642 |                        51 |    0.0137903   |                      29.7727   |
| C1_market_offset_fundamental | validation_2020_2024 |   2020 |  47876 |                    93 |             106 |                        44 |                        31 |    0.00221405  |                     nan        |
| C1_market_offset_fundamental | validation_2020_2024 |   2021 |  47476 |                   107 |             142 |                        66 |                        31 |    0.00299098  |                       1.33962  |
| C1_market_offset_fundamental | validation_2020_2024 |   2022 |  46840 |                    40 |              49 |                        28 |                        19 |    0.00104611  |                       0.34507  |
| C1_market_offset_fundamental | validation_2020_2024 |   2023 |  47273 |                    48 |              55 |                        33 |                        26 |    0.00116345  |                       1.12245  |
| C1_market_offset_fundamental | validation_2020_2024 |   2024 |  46752 |                    25 |              22 |                        14 |                        17 |    0.000470568 |                       0.4      |

## Adoption Note
C1R0 adoption judgment is based only on 2020-2024 metrics. ROI is not used as the sole criterion.
