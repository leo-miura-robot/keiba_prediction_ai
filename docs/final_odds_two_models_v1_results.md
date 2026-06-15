# Final Odds Two Models V1 Results

- Ideal condition: `This is a final-odds ideal-condition model. It uses final current-race odds as model inputs and is not a pre-race live operation model.`
- Feature set: `market_aware`
- Calibration: `{'win': 'none', 'place': 'none'}`
- Alpha: `{'win': 0.5, 'place': 1.0}`
- Elapsed seconds: `686.2`

## Selected Rules
| target   | strategy_type   |   model_rank_max |   ev_min |   edge_min |   margin_min |   odds_min |   odds_max |   rank_gap_min |   bets |   races |      stake |      return |   profit |     roi |   hit_count |   hit_rate |   average_odds |   median_odds |   average_payout |   max_payout |   max_losing_streak |   max_drawdown |   year_roi_mean |   year_roi_min |   year_roi_std |   roi_remove_top1 |   roi_remove_top5 |   top1_profit_share |   top5_profit_share |   relaxed_min_validation_bets |   score |
|:---------|:----------------|-----------------:|---------:|-----------:|-------------:|-----------:|-----------:|---------------:|-------:|--------:|-----------:|------------:|---------:|--------:|------------:|-----------:|---------------:|--------------:|-----------------:|-------------:|--------------------:|---------------:|----------------:|---------------:|---------------:|------------------:|------------------:|--------------------:|--------------------:|------------------------------:|--------:|
| place    | core            |                1 |        0 |       -999 |            0 |          1 |        999 |           -999 |  17278 |   17278 | 1.7278e+06 | 1.46961e+06 |  -258190 | 85.0567 |       11161 |   0.645966 |        1.25904 |           1.2 |          85.0567 |          500 |                   9 |         258380 |         85.057  |        83.4259 |        1.24422 |           85.0327 |           84.9789 |           0.0578484 |            0.214591 |                           300 |  930160 |
| win      | core            |                1 |        0 |       -999 |            0 |          1 |        999 |           -999 |  17278 |   17278 | 1.7278e+06 | 1.40818e+06 |  -319620 | 81.5013 |        5801 |   0.335745 |        2.70637 |           2.6 |          81.5013 |          730 |                  19 |         320020 |         81.5014 |        79.9653 |        0.96261 |           81.4638 |           81.3275 |           0.0798474 |            0.302024 |                           300 |  890930 |

## ROI Summary
| rule_id       | target   | strategy_type   | eval_period          |   bets |   races |           stake |           return |   profit |     roi |   hit_count |   hit_rate |   average_odds |   median_odds |   average_payout |   max_payout |   max_losing_streak |   max_drawdown |
|:--------------|:---------|:----------------|:---------------------|-------:|--------:|----------------:|-----------------:|---------:|--------:|------------:|-----------:|---------------:|--------------:|-----------------:|-------------:|--------------------:|---------------:|
| place_core_01 | place    | core            | validation_2020_2024 |  17278 |   17278 |      1.7278e+06 |      1.46961e+06 |  -258190 | 85.0567 |       11161 |   0.645966 |        1.25904 |           1.2 |          85.0567 |          500 |                   9 |         258380 |
| win_core_02   | win      | core            | validation_2020_2024 |  17278 |   17278 |      1.7278e+06 |      1.40818e+06 |  -319620 | 81.5013 |        5801 |   0.335745 |        2.70637 |           2.6 |          81.5013 |          730 |                  19 |         320020 |
| place_core_01 | place    | core            | test_2025            |   3455 |    3455 | 345500          | 304820           |   -40680 | 88.2258 |        2257 |   0.653256 |        1.28689 |           1.2 |          88.2258 |          400 |                   8 |          40900 |
| win_core_02   | win      | core            | test_2025            |   3455 |    3455 | 345500          | 270290           |   -75210 | 78.2315 |        1089 |   0.315195 |        2.86981 |           2.6 |          78.2315 |          900 |                  21 |          76120 |
| place_core_01 | place    | core            | latest_holdout_2026  |   1506 |    1506 | 150600          | 134350           |   -16250 | 89.2098 |        1009 |   0.669987 |        1.27968 |           1.2 |          89.2098 |          360 |                   7 |          16320 |
| win_core_02   | win      | core            | latest_holdout_2026  |   1506 |    1506 | 150600          | 124570           |   -26030 | 82.7158 |         482 |   0.320053 |        2.96049 |           2.7 |          82.7158 |          850 |                  15 |          26790 |
| place_core_01 | place    | core            | test_latest_combined |   4961 |    4961 | 496100          | 439170           |   -56930 | 88.5245 |        3266 |   0.658335 |        1.2847  |           1.2 |          88.5245 |          400 |                   8 |          56970 |
| win_core_02   | win      | core            | test_latest_combined |   4961 |    4961 | 496100          | 394860           |  -101240 | 79.5928 |        1571 |   0.31667  |        2.89734 |           2.6 |          79.5928 |          900 |                  21 |         101930 |

## Dependency
| rule_id       | target   | eval_period          | removed_top_payouts   |      roi |   top1_profit_share |   top5_profit_share |
|:--------------|:---------|:---------------------|:----------------------|---------:|--------------------:|--------------------:|
| place_core_01 | place    | validation_2020_2024 | 0                     |  85.0567 |         nan         |          nan        |
| place_core_01 | place    | validation_2020_2024 | 1                     |  85.0327 |         nan         |          nan        |
| place_core_01 | place    | validation_2020_2024 | 3                     |  85.0026 |         nan         |          nan        |
| place_core_01 | place    | validation_2020_2024 | 5                     |  84.9789 |         nan         |          nan        |
| place_core_01 | place    | validation_2020_2024 | 10                    |  84.927  |         nan         |          nan        |
| place_core_01 | place    | validation_2020_2024 | dependency            | nan      |           0.0578484 |            0.214591 |
| win_core_02   | win      | validation_2020_2024 | 0                     |  81.5013 |         nan         |          nan        |
| win_core_02   | win      | validation_2020_2024 | 1                     |  81.4638 |         nan         |          nan        |
| win_core_02   | win      | validation_2020_2024 | 3                     |  81.3928 |         nan         |          nan        |
| win_core_02   | win      | validation_2020_2024 | 5                     |  81.3275 |         nan         |          nan        |
| win_core_02   | win      | validation_2020_2024 | 10                    |  81.1837 |         nan         |          nan        |
| win_core_02   | win      | validation_2020_2024 | dependency            | nan      |           0.0798474 |            0.302024 |
| place_core_01 | place    | test_2025            | 0                     |  88.2258 |         nan         |          nan        |
| place_core_01 | place    | test_2025            | 1                     |  88.1355 |         nan         |          nan        |
| place_core_01 | place    | test_2025            | 3                     |  87.978  |         nan         |          nan        |
| place_core_01 | place    | test_2025            | 5                     |  87.8377 |         nan         |          nan        |
| place_core_01 | place    | test_2025            | 10                    |  87.5414 |         nan         |          nan        |
| place_core_01 | place    | test_2025            | dependency            | nan      |           0.0734328 |            0.243933 |
| win_core_02   | win      | test_2025            | 0                     |  78.2315 |         nan         |          nan        |
| win_core_02   | win      | test_2025            | 1                     |  77.9936 |         nan         |          nan        |
| win_core_02   | win      | test_2025            | 3                     |  77.6043 |         nan         |          nan        |
| win_core_02   | win      | test_2025            | 5                     |  77.2464 |         nan         |          nan        |
| win_core_02   | win      | test_2025            | 10                    |  76.4499 |         nan         |          nan        |
| win_core_02   | win      | test_2025            | dependency            | nan      |           0.103352  |            0.350022 |
| place_core_01 | place    | latest_holdout_2026  | 0                     |  89.2098 |         nan         |          nan        |
| place_core_01 | place    | latest_holdout_2026  | 1                     |  89.0299 |         nan         |          nan        |
| place_core_01 | place    | latest_holdout_2026  | 3                     |  88.7292 |         nan         |          nan        |
| place_core_01 | place    | latest_holdout_2026  | 5                     |  88.481  |         nan         |          nan        |
| place_core_01 | place    | latest_holdout_2026  | 10                    |  87.9011 |         nan         |          nan        |
| place_core_01 | place    | latest_holdout_2026  | dependency            | nan      |           0.0816143 |            0.264275 |

## Bootstrap
| rule_id       | target   | eval_period          |   roi_p025 |   roi_p500 |   roi_p975 |
|:--------------|:---------|:---------------------|-----------:|-----------:|-----------:|
| place_core_01 | place    | validation_2020_2024 |    84.0229 |    85.0695 |    86.069  |
| win_core_02   | win      | validation_2020_2024 |    79.5547 |    81.5436 |    83.3957 |
| place_core_01 | place    | test_2025            |    86.0112 |    88.2315 |    90.6787 |
| win_core_02   | win      | test_2025            |    74.2274 |    78.2706 |    82.4101 |
| place_core_01 | place    | latest_holdout_2026  |    85.5045 |    89.1102 |    92.637  |
| win_core_02   | win      | latest_holdout_2026  |    75.6306 |    82.6162 |    89.2243 |
| place_core_01 | place    | test_latest_combined |    86.6115 |    88.5285 |    90.4075 |
| win_core_02   | win      | test_latest_combined |    76.172  |    79.6059 |    83.3709 |
