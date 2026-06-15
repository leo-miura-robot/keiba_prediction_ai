# Place Market Offset CatBoost V1 Results

- Selected model: `C1_market_offset_fundamental`
- Selection: 2020-2024 probability metrics, high-odds calibration gap, EV-ROI Spearman; ROI not sole criterion
- DB cache status: `cache_manifest_reused_without_db_read`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- CatBoost: `1.2.10`
- Elapsed seconds: `633.7`

## Model Comparison
| model_key                       | period               |   validation logloss |   validation Brier |   validation ECE |   validation calibration slope |   high-odds calibration gap |   EV-ROI Spearman |   EV>=1 count |   EV>=1 ROI |   EV>=1.05 count |   EV>=1.05 ROI |   2025 ROI |   2026 ROI |   combined ROI |   top5 removed ROI | bootstrap CI         |
|:--------------------------------|:---------------------|---------------------:|-------------------:|-----------------:|-------------------------------:|----------------------------:|------------------:|--------------:|------------:|-----------------:|---------------:|-----------:|-----------:|---------------:|-------------------:|:---------------------|
| A_current_market_aware          | latest_holdout_2026  |             0.388504 |           0.124294 |      0.0231294   |                       0.969103 |                -0.0042531   |          0.309524 |           203 |     88.1773 |              104 |      103.846   |   nan      |    88.1773 |        88.1773 |           74.4444  | 68.77,87.99,109.38   |
| A_current_market_aware          | test_2025            |             0.406583 |           0.130562 |      0.0181116   |                       0.928588 |                -0.00680477  |          0.904762 |           475 |    110.253  |              270 |      114.926   |   110.253  |   nan      |       110.253  |           97.1915  | 91.46,110.21,132.43  |
| A_current_market_aware          | validation_2020_2024 |             0.406566 |           0.130549 |      0.00410394  |                       1.00922  |                -0.000240728 |         -0.357143 |           460 |     56.4783 |              330 |        9.75758 |   nan      |   nan      |       nan      |            7.27473 | 9.74,54.52,133.52    |
| B_market_baseline               | latest_holdout_2026  |             0.384858 |           0.122898 |      0.00802953  |                       1.0576   |                 0.00644482  |          0.52381  |            75 |    126.933  |               33 |      180.909   |   nan      |   126.933  |       126.933  |           54.1429  | 62.09,123.40,205.86  |
| B_market_baseline               | test_2025            |             0.4024   |           0.129094 |      0.00369608  |                       0.998637 |                 0.000752001 |          0.5      |           109 |     99.7248 |               42 |      111.667   |    99.7248 |   nan      |        99.7248 |           44.6154  | 44.87,97.46,165.74   |
| B_market_baseline               | validation_2020_2024 |             0.406278 |           0.130581 |      5.53728e-09 |                       0.99974  |                 0.000598115 |          0.857143 |           459 |    129.521  |              198 |      122.475   |   nan      |   nan      |       nan      |          114.537   | 100.64,129.90,158.50 |
| C1_market_offset_fundamental    | latest_holdout_2026  |             0.389038 |           0.124523 |      0.0243146   |                       0.969618 |                -0.00389187  |          0.785714 |           253 |    105.85   |              165 |      112.061   |   nan      |   105.85   |       105.85   |           85.8468  | 80.52,104.81,135.26  |
| C1_market_offset_fundamental    | test_2025            |             0.408437 |           0.131178 |      0.0204331   |                       0.912848 |                -0.00651666  |          0.261905 |           655 |     93.6183 |              399 |       98.4962  |    93.6183 |   nan      |        93.6183 |           82.6462  | 76.47,93.62,110.50   |
| C1_market_offset_fundamental    | validation_2020_2024 |             0.405918 |           0.130382 |      0.00302126  |                       0.997545 |                -0.000210662 |          0.928571 |           374 |    132.112  |              159 |      158.931   |   nan      |   nan      |       nan      |          113.117   | 98.93,131.73,167.23  |
| C2_market_offset_limited_market | latest_holdout_2026  |             0.392308 |           0.125357 |      0.0168863   |                       0.925385 |                -0.00158705  |          0.214286 |           703 |     88.9474 |              507 |       91.8738  |   nan      |    88.9474 |        88.9474 |           76.3754  | 71.51,88.89,106.68   |
| C2_market_offset_limited_market | test_2025            |             0.412027 |           0.132271 |      0.0196828   |                       0.872096 |                -0.00460261  |          0.452381 |          1746 |     88.992  |             1207 |       89.8509  |    88.992  |   nan      |        88.992  |           83.4118  | 78.35,88.88,100.39   |
| C2_market_offset_limited_market | validation_2020_2024 |             0.405978 |           0.130409 |      0.00283241  |                       1.00187  |                -1.81139e-05 |          0.452381 |           313 |     97.0607 |              138 |      123.478   |   nan      |   nan      |       nan      |           73.0519  | 64.54,96.74,133.86   |

## ROI Comparison
| model_key                       | period               | strategy                        |   bets |   return |       roi |   hit_rate |   max_losing_streak |   max_drawdown |
|:--------------------------------|:---------------------|:--------------------------------|-------:|---------:|----------:|-----------:|--------------------:|---------------:|
| A_current_market_aware          | latest_holdout_2026  | strategy1_odds_1_2_2_5_ev_0_85  |    370 |    39920 | 107.892   |  0.57027   |                   5 |           1210 |
| A_current_market_aware          | latest_holdout_2026  | strategy2_ev_1_00               |    203 |    17900 |  88.1773  |  0.315271  |                   8 |           3120 |
| A_current_market_aware          | latest_holdout_2026  | strategy3_ev_1_05               |    104 |    10800 | 103.846   |  0.346154  |                  10 |           1000 |
| A_current_market_aware          | latest_holdout_2026  | strategy4_no_odds_limit_ev_1_00 |    203 |    17900 |  88.1773  |  0.315271  |                   8 |           3120 |
| A_current_market_aware          | test_2025            | strategy1_odds_1_2_2_5_ev_0_85  |    961 |    96440 | 100.354   |  0.568158  |                   8 |           3470 |
| A_current_market_aware          | test_2025            | strategy2_ev_1_00               |    475 |    52370 | 110.253   |  0.324211  |                  18 |           3480 |
| A_current_market_aware          | test_2025            | strategy3_ev_1_05               |    270 |    31030 | 114.926   |  0.274074  |                  18 |           2200 |
| A_current_market_aware          | test_2025            | strategy4_no_odds_limit_ev_1_00 |    475 |    52370 | 110.253   |  0.324211  |                  18 |           3480 |
| A_current_market_aware          | validation_2020_2024 | strategy1_odds_1_2_2_5_ev_0_85  |    586 |    58370 |  99.6075  |  0.571672  |                   8 |           2400 |
| A_current_market_aware          | validation_2020_2024 | strategy2_ev_1_00               |    460 |    25980 |  56.4783  |  0.0282609 |                 171 |          26030 |
| A_current_market_aware          | validation_2020_2024 | strategy3_ev_1_05               |    330 |     3220 |   9.75758 |  0.0121212 |                 131 |          29780 |
| A_current_market_aware          | validation_2020_2024 | strategy4_no_odds_limit_ev_1_00 |    460 |    25980 |  56.4783  |  0.0282609 |                 171 |          26030 |
| B_market_baseline               | latest_holdout_2026  | strategy1_odds_1_2_2_5_ev_0_85  |     51 |     5410 | 106.078   |  0.627451  |                   3 |            370 |
| B_market_baseline               | latest_holdout_2026  | strategy2_ev_1_00               |     75 |     9520 | 126.933   |  0.213333  |                  18 |           1800 |
| B_market_baseline               | latest_holdout_2026  | strategy3_ev_1_05               |     33 |     5970 | 180.909   |  0.212121  |                  10 |           1000 |
| B_market_baseline               | latest_holdout_2026  | strategy4_no_odds_limit_ev_1_00 |     75 |     9520 | 126.933   |  0.213333  |                  18 |           1800 |
| B_market_baseline               | test_2025            | strategy1_odds_1_2_2_5_ev_0_85  |    100 |    10520 | 105.2     |  0.67      |                   5 |            950 |
| B_market_baseline               | test_2025            | strategy2_ev_1_00               |    109 |    10870 |  99.7248  |  0.137615  |                  20 |           2000 |
| B_market_baseline               | test_2025            | strategy3_ev_1_05               |     42 |     4690 | 111.667   |  0.0952381 |                  14 |           1400 |
| B_market_baseline               | test_2025            | strategy4_no_odds_limit_ev_1_00 |    109 |    10870 |  99.7248  |  0.137615  |                  20 |           2000 |
| B_market_baseline               | validation_2020_2024 | strategy1_odds_1_2_2_5_ev_0_85  |    336 |    34140 | 101.607   |  0.675595  |                   3 |           1800 |
| B_market_baseline               | validation_2020_2024 | strategy2_ev_1_00               |    459 |    59450 | 129.521   |  0.176471  |                  23 |           3300 |
| B_market_baseline               | validation_2020_2024 | strategy3_ev_1_05               |    198 |    24250 | 122.475   |  0.136364  |                  39 |           4170 |
| B_market_baseline               | validation_2020_2024 | strategy4_no_odds_limit_ev_1_00 |    459 |    59450 | 129.521   |  0.176471  |                  23 |           3300 |
| C1_market_offset_fundamental    | latest_holdout_2026  | strategy1_odds_1_2_2_5_ev_0_85  |    351 |    36930 | 105.214   |  0.581197  |                   7 |           1680 |
| C1_market_offset_fundamental    | latest_holdout_2026  | strategy2_ev_1_00               |    253 |    26780 | 105.85    |  0.300395  |                  11 |           3980 |
| C1_market_offset_fundamental    | latest_holdout_2026  | strategy3_ev_1_05               |    165 |    18490 | 112.061   |  0.254545  |                  11 |           2090 |
| C1_market_offset_fundamental    | latest_holdout_2026  | strategy4_no_odds_limit_ev_1_00 |    253 |    26780 | 105.85    |  0.300395  |                  11 |           3980 |
| C1_market_offset_fundamental    | test_2025            | strategy1_odds_1_2_2_5_ev_0_85  |    969 |    92000 |  94.9432  |  0.558308  |                   9 |           6770 |
| C1_market_offset_fundamental    | test_2025            | strategy2_ev_1_00               |    655 |    61320 |  93.6183  |  0.232061  |                  24 |           6840 |
| C1_market_offset_fundamental    | test_2025            | strategy3_ev_1_05               |    399 |    39300 |  98.4962  |  0.180451  |                  18 |           4530 |
| C1_market_offset_fundamental    | test_2025            | strategy4_no_odds_limit_ev_1_00 |    655 |    61320 |  93.6183  |  0.232061  |                  24 |           6840 |
| C1_market_offset_fundamental    | validation_2020_2024 | strategy1_odds_1_2_2_5_ev_0_85  |    671 |    64930 |  96.766   |  0.600596  |                   5 |           3400 |
| C1_market_offset_fundamental    | validation_2020_2024 | strategy2_ev_1_00               |    374 |    49410 | 132.112   |  0.147059  |                  26 |           3580 |
| C1_market_offset_fundamental    | validation_2020_2024 | strategy3_ev_1_05               |    159 |    25270 | 158.931   |  0.169811  |                  18 |           1800 |
| C1_market_offset_fundamental    | validation_2020_2024 | strategy4_no_odds_limit_ev_1_00 |    374 |    49410 | 132.112   |  0.147059  |                  26 |           3580 |
| C2_market_offset_limited_market | latest_holdout_2026  | strategy1_odds_1_2_2_5_ev_0_85  |    624 |    59200 |  94.8718  |  0.517628  |                  11 |           4560 |
| C2_market_offset_limited_market | latest_holdout_2026  | strategy2_ev_1_00               |    703 |    62530 |  88.9474  |  0.227596  |                  26 |          12080 |
| C2_market_offset_limited_market | latest_holdout_2026  | strategy3_ev_1_05               |    507 |    46580 |  91.8738  |  0.201183  |                  28 |           9230 |
| C2_market_offset_limited_market | latest_holdout_2026  | strategy4_no_odds_limit_ev_1_00 |    703 |    62530 |  88.9474  |  0.227596  |                  26 |          12080 |
| C2_market_offset_limited_market | test_2025            | strategy1_odds_1_2_2_5_ev_0_85  |   1671 |   151670 |  90.766   |  0.514063  |                  11 |          15990 |
| C2_market_offset_limited_market | test_2025            | strategy2_ev_1_00               |   1746 |   155380 |  88.992   |  0.225659  |                  24 |          22210 |
| C2_market_offset_limited_market | test_2025            | strategy3_ev_1_05               |   1207 |   108450 |  89.8509  |  0.1715    |                  28 |          14560 |
| C2_market_offset_limited_market | test_2025            | strategy4_no_odds_limit_ev_1_00 |   1746 |   155380 |  88.992   |  0.225659  |                  24 |          22210 |
| C2_market_offset_limited_market | validation_2020_2024 | strategy1_odds_1_2_2_5_ev_0_85  |    655 |    61840 |  94.4122  |  0.566412  |                   5 |           4510 |
| C2_market_offset_limited_market | validation_2020_2024 | strategy2_ev_1_00               |    313 |    30380 |  97.0607  |  0.111821  |                  33 |           5700 |
| C2_market_offset_limited_market | validation_2020_2024 | strategy3_ev_1_05               |    138 |    17040 | 123.478   |  0.137681  |                  20 |           2920 |
| C2_market_offset_limited_market | validation_2020_2024 | strategy4_no_odds_limit_ev_1_00 |    313 |    30380 |  97.0607  |  0.111821  |                  33 |           5700 |
