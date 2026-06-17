# Phase 5B Year Strategy Selection Report

???????? `outputs/place_market_offset_year_strategy_phase5b_v2/` ?????CatBoost???DB???calibration fit?2025/2026??????EV????????????

## Recommendation

**ROLLING_10Y?????**???Logloss???Brier???3????????????????????EXPANDING_FULL_2006?????????10?rolling?????????????????????????????????????

## Top3 Probability Summary

| strategy            |   mean_logloss |   mean_brier |   mean_ece |   worst_year_logloss |   worst_year_brier |   logloss_std |   brier_std |
|:--------------------|---------------:|-------------:|-----------:|---------------------:|-------------------:|--------------:|------------:|
| ROLLING_10Y         |       0.405659 |     0.130333 | 0.00578208 |             0.41001  |           0.131608 |    0.00276225 | 0.000894513 |
| ROLLING_15Y         |       0.405697 |     0.130341 | 0.00579677 |             0.409928 |           0.131559 |    0.00268847 | 0.000855499 |
| EXPANDING_FULL_2006 |       0.405748 |     0.130365 | 0.00558872 |             0.409929 |           0.131559 |    0.00267135 | 0.00085607  |

## Pairwise Race Bootstrap

??? candidate - baseline????candidate???

| candidate   | baseline            | metric   |   point_estimate_candidate_minus_baseline |   bootstrap_mean |   ci95_lower |   ci95_upper |   candidate_better_probability |   races |   rows |
|:------------|:--------------------|:---------|------------------------------------------:|-----------------:|-------------:|-------------:|-------------------------------:|--------:|-------:|
| ROLLING_10Y | ROLLING_15Y         | logloss  |                              -3.68311e-05 |     -3.56311e-05 | -0.000163911 |  9.1761e-05  |                         0.7052 |   17278 | 236217 |
| ROLLING_10Y | ROLLING_15Y         | brier    |                              -7.82161e-06 |     -7.31751e-06 | -5.88479e-05 |  4.50474e-05 |                         0.613  |   17278 | 236217 |
| ROLLING_10Y | EXPANDING_FULL_2006 | logloss  |                              -8.76182e-05 |     -8.85079e-05 | -0.000220369 |  4.40894e-05 |                         0.9028 |   17278 | 236217 |
| ROLLING_10Y | EXPANDING_FULL_2006 | brier    |                              -3.11795e-05 |     -3.15347e-05 | -8.39927e-05 |  2.25805e-05 |                         0.8748 |   17278 | 236217 |
| ROLLING_15Y | EXPANDING_FULL_2006 | logloss  |                              -5.07872e-05 |     -5.06724e-05 | -0.000134909 |  3.3427e-05  |                         0.8786 |   17278 | 236217 |
| ROLLING_15Y | EXPANDING_FULL_2006 | brier    |                              -2.33579e-05 |     -2.33457e-05 | -5.71924e-05 |  1.05467e-05 |                         0.9096 |   17278 | 236217 |

## Yearly Wins

| metric   | winner      |   wins |
|:---------|:------------|-------:|
| brier    | ROLLING_10Y |      3 |
| brier    | ROLLING_15Y |      2 |
| logloss  | ROLLING_10Y |      3 |
| logloss  | ROLLING_15Y |      2 |

| metric   |   Year | winner      |   winner_value |
|:---------|-------:|:------------|---------------:|
| logloss  |   2020 | ROLLING_15Y |       0.409928 |
| logloss  |   2021 | ROLLING_10Y |       0.405836 |
| logloss  |   2022 | ROLLING_10Y |       0.405428 |
| logloss  |   2023 | ROLLING_15Y |       0.404522 |
| logloss  |   2024 | ROLLING_10Y |       0.402444 |
| brier    |   2020 | ROLLING_15Y |       0.131559 |
| brier    |   2021 | ROLLING_10Y |       0.130397 |
| brier    |   2022 | ROLLING_10Y |       0.13032  |
| brier    |   2023 | ROLLING_15Y |       0.130235 |
| brier    |   2024 | ROLLING_10Y |       0.129082 |

## Worst-Year / CV

| strategy            |   worst_logloss_year |   worst_logloss |   worst_brier_year |   worst_brier |   logloss_cv |   brier_cv |
|:--------------------|---------------------:|----------------:|-------------------:|--------------:|-------------:|-----------:|
| EXPANDING_FULL_2006 |                 2020 |        0.409929 |               2020 |      0.131559 |   0.00658378 | 0.00656674 |
| ROLLING_10Y         |                 2020 |        0.41001  |               2020 |      0.131608 |   0.00680927 | 0.00686329 |
| ROLLING_15Y         |                 2020 |        0.409928 |               2020 |      0.131559 |   0.0066268  | 0.00656354 |

## Residual Tail

| strategy            |   residual_p95_mean |   residual_p99_mean |   residual_p95_max |   residual_p99_max |
|:--------------------|--------------------:|--------------------:|-------------------:|-------------------:|
| EXPANDING_FULL_2006 |            0.290318 |            0.424875 |           0.297202 |           0.431914 |
| ROLLING_10Y         |            0.287082 |            0.431183 |           0.30051  |           0.446661 |
| ROLLING_15Y         |            0.294065 |            0.432543 |           0.297635 |           0.437508 |

## Calibration

| strategy            |   ece_mean |   calibration_slope_mean |   calibration_intercept_mean |
|:--------------------|-----------:|-------------------------:|-----------------------------:|
| EXPANDING_FULL_2006 | 0.00558872 |                  1.01357 |                   0.0106795  |
| ROLLING_10Y         | 0.00578208 |                  1.0128  |                   0.0167181  |
| ROLLING_15Y         | 0.00579677 |                  1.01263 |                   0.00998683 |

## ROI Diagnostics

ROI????????ROI???ROI?????? total payout / total stake?

| strategy            |   bet_count_total |   stake_total |   payout_total |   combined_roi_total_payout_over_total_stake |   yearly_roi_min |   yearly_roi_max |   max_single_payout |   top1_payout_share |   top3_payout_share |
|:--------------------|------------------:|--------------:|---------------:|---------------------------------------------:|-----------------:|-----------------:|--------------------:|--------------------:|--------------------:|
| EXPANDING_FULL_2006 |               324 |         32400 |          35880 |                                      110.741 |          25.5932 |          229.091 |                1640 |           0.0457079 |           0.128484  |
| ROLLING_10Y         |               448 |         44800 |          50950 |                                      113.728 |          72.043  |          152.424 |                1640 |           0.0321884 |           0.0900883 |
| ROLLING_15Y         |               362 |         36200 |          37950 |                                      104.834 |          33.5616 |          165.522 |                1640 |           0.0432148 |           0.120949  |

`payout_zeroed_stress_roi <= normal_roi`: True

| strategy            |   limit |   row_removed_roi_min |   row_removed_roi_mean |   payout_zeroed_roi_min |   payout_zeroed_roi_mean |
|:--------------------|--------:|----------------------:|-----------------------:|------------------------:|-------------------------:|
| ROLLING_10Y         |       1 |               57.5    |               97.4789  |                56.6667  |                 96.3532  |
| ROLLING_10Y         |       3 |               32.2727 |               74.2269  |                30.8696  |                 71.6939  |
| ROLLING_10Y         |       5 |               12.8125 |               55.464   |                11.8841  |                 52.3732  |
| ROLLING_10Y         |      10 |                0      |               23.573   |                 0       |                 21.0821  |
| ROLLING_15Y         |       1 |               19.7222 |               85.229   |                19.4521  |                 84.0186  |
| ROLLING_15Y         |       3 |                9      |               58.571   |                 8.63014 |                 56.1671  |
| ROLLING_15Y         |       5 |                0      |               38.9686  |                 0       |                 36.416   |
| ROLLING_15Y         |      10 |                0      |               11.2372  |                 0       |                  9.66007 |
| EXPANDING_FULL_2006 |       1 |               15      |               94.6528  |                14.7458  |                 92.9804  |
| EXPANDING_FULL_2006 |       3 |                0      |               62.5162  |                 0       |                 59.2564  |
| EXPANDING_FULL_2006 |       5 |                0      |               41.244   |                 0       |                 37.7294  |
| EXPANDING_FULL_2006 |      10 |                0      |                5.46117 |                 0       |                  4.56712 |

## Output Files

- `outputs\place_market_offset_year_strategy_phase5b_v2\top3_pairwise_bootstrap.csv`
- `outputs\place_market_offset_year_strategy_phase5b_v2\roi_combined_2020_2024.csv`
- `outputs\place_market_offset_year_strategy_phase5b_v2\final_strategy_recommendation.json`
