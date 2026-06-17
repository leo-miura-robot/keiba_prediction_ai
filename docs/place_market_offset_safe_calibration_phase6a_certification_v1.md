# Phase 6A Calibration Certification

- checks: 35
- failed: 0
- 10Y Platt: CERTIFIED as operational candidate, not activated
- 15Y Isotonic: CERTIFIED as challenger shadow, not activated

## Pooled Selection
| strategy    | calibration_method   |   pooled_rows |   pooled_races |   pooled_logloss |   pooled_brier |   pooled_ece_10 |   pooled_ece_20 |   pooled_calibration_slope |   pooled_calibration_intercept |   worst_year_logloss |   worst_year_brier |   logloss_cv |   brier_cv |
|:------------|:---------------------|--------------:|---------------:|-----------------:|---------------:|----------------:|----------------:|---------------------------:|-------------------------------:|---------------------:|-------------------:|-------------:|-----------:|
| ROLLING_10Y | PLATT_SCALING        |        236217 |          17278 |         0.40567  |       0.130338 |      0.0050572  |      0.00533005 |                   0.995912 |                     0.00233853 |             0.410061 |           0.131618 |   0.00687441 | 0.0069143  |
| ROLLING_10Y | RAW_IDENTITY         |        236217 |          17278 |         0.405678 |       0.130339 |      0.00471859 |      0.00528196 |                   1.01276  |                     0.0167784  |             0.41001  |           0.131608 |   0.00680927 | 0.00686329 |
| ROLLING_10Y | TEMPERATURE_SCALING  |        236217 |          17278 |         0.405681 |       0.130339 |      0.00463701 |      0.00543593 |                   1.00227  |                     0.0167553  |             0.410054 |           0.131613 |   0.00685055 | 0.00688637 |
| ROLLING_10Y | ISOTONIC             |        236217 |          17278 |         0.40569  |       0.130301 |      0.00126984 |      0.00210089 |                   0.990279 |                    -0.00324586 |             0.410093 |           0.131599 |   0.0069037  | 0.0070768  |
| ROLLING_15Y | ISOTONIC             |        236217 |          17278 |         0.405679 |       0.130295 |      0.00117369 |      0.00173034 |                   0.988375 |                    -0.0100243  |             0.409938 |           0.13152  |   0.00650445 | 0.00665098 |
| ROLLING_15Y | PLATT_SCALING        |        236217 |          17278 |         0.405709 |       0.130346 |      0.00525077 |      0.00551087 |                   0.994162 |                    -0.00449809 |             0.409986 |           0.131569 |   0.0066941  | 0.00660664 |
| ROLLING_15Y | TEMPERATURE_SCALING  |        236217 |          17278 |         0.405711 |       0.130344 |      0.00503777 |      0.00519821 |                   1.00055  |                     0.0100171  |             0.409974 |           0.131562 |   0.00667792 | 0.00658908 |
| ROLLING_15Y | RAW_IDENTITY         |        236217 |          17278 |         0.405715 |       0.130347 |      0.00502649 |      0.00509084 |                   1.01256  |                     0.0100477  |             0.409928 |           0.131559 |   0.0066268  | 0.00656354 |

## Bootstrap
| strategy    | candidate     | baseline            | years                    | metric   |   delta_candidate_minus_baseline |   bootstrap_mean |   ci95_lower |   ci95_upper |   candidate_better_probability |   races |   rows |   n_bootstrap |
|:------------|:--------------|:--------------------|:-------------------------|:---------|---------------------------------:|-----------------:|-------------:|-------------:|-------------------------------:|--------:|-------:|--------------:|
| ROLLING_10Y | PLATT_SCALING | RAW_IDENTITY        | 2020,2021,2022,2023,2024 | logloss  |                     -8.03424e-06 |     -7.9975e-06  | -3.5391e-05  |  1.99651e-05 |                         0.7184 |   17278 | 236217 |          5000 |
| ROLLING_10Y | PLATT_SCALING | RAW_IDENTITY        | 2020,2021,2022,2023,2024 | brier    |                     -1.64713e-06 |     -1.64512e-06 | -9.62482e-06 |  6.05942e-06 |                         0.6596 |   17278 | 236217 |          5000 |
| ROLLING_10Y | PLATT_SCALING | TEMPERATURE_SCALING | 2020,2021,2022,2023,2024 | logloss  |                     -1.06598e-05 |     -1.06503e-05 | -2.13671e-05 |  5.30454e-08 |                         0.9748 |   17278 | 236217 |          5000 |
| ROLLING_10Y | PLATT_SCALING | TEMPERATURE_SCALING | 2020,2021,2022,2023,2024 | brier    |                     -1.4732e-06  |     -1.47676e-06 | -5.90356e-06 |  2.96706e-06 |                         0.738  |   17278 | 236217 |          5000 |
| ROLLING_10Y | PLATT_SCALING | RAW_IDENTITY        | 2025,2026                | logloss  |                     -6.00938e-05 |     -6.07607e-05 | -0.000108254 | -1.42084e-05 |                         0.9942 |    4961 |  68773 |          5000 |
| ROLLING_10Y | PLATT_SCALING | RAW_IDENTITY        | 2025,2026                | brier    |                     -1.11792e-05 |     -1.11563e-05 | -2.58013e-05 |  3.64774e-06 |                         0.9216 |    4961 |  68773 |          5000 |
| ROLLING_15Y | ISOTONIC      | RAW_IDENTITY        | 2020,2021,2022,2023,2024 | logloss  |                     -3.57508e-05 |     -3.9049e-05  | -0.000214719 |  0.000161688 |                         0.677  |   17278 | 236217 |          5000 |
| ROLLING_15Y | ISOTONIC      | RAW_IDENTITY        | 2020,2021,2022,2023,2024 | brier    |                     -5.15784e-05 |     -5.1321e-05  | -8.9444e-05  | -1.33475e-05 |                         0.996  |   17278 | 236217 |          5000 |
| ROLLING_15Y | ISOTONIC      | PLATT_SCALING       | 2020,2021,2022,2023,2024 | logloss  |                     -2.9286e-05  |     -3.26034e-05 | -0.000202194 |  0.000166283 |                         0.6584 |   17278 | 236217 |          5000 |
| ROLLING_15Y | ISOTONIC      | PLATT_SCALING       | 2020,2021,2022,2023,2024 | brier    |                     -5.0566e-05  |     -5.03034e-05 | -8.93909e-05 | -1.26685e-05 |                         0.995  |   17278 | 236217 |          5000 |
| ROLLING_15Y | ISOTONIC      | RAW_IDENTITY        | 2025,2026                | logloss  |                     -0.00025162  |     -0.000249472 | -0.00048889  | -6.22079e-06 |                         0.9774 |    4961 |  68773 |          5000 |
| ROLLING_15Y | ISOTONIC      | RAW_IDENTITY        | 2025,2026                | brier    |                     -9.73546e-05 |     -9.74189e-05 | -0.000165494 | -2.93442e-05 |                         0.9974 |    4961 |  68773 |          5000 |

## Requirement Coverage
| requirement_id   | requirement                                  | existing_check                                  | status   | evidence   | new_check_added   |
|:-----------------|:---------------------------------------------|:------------------------------------------------|:---------|:-----------|:------------------|
| R01              | Selection uses pooled 2020-2024 Logloss      | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R02              | Year mean not primary                        | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R03              | 2025/2026 excluded from selection            | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R04              | Each calibrator fit uses prior years only    | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R05              | 2020 fit window 2016-2019                    | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R06              | 2026 fit window 2016-2025                    | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R07              | target_place_paid configured                 | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R08              | target is binary                             | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R09              | actual_place matches target_place_paid       | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R10              | target_place_paid matches fuku_pay>0         | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R11              | No .le(3) rank transform                     | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R12              | Raw predictions have no duplicate keys       | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R13              | Raw predictions have no missing keys         | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R14              | Raw prediction key hashes recorded           | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R15              | Prediction hashes recorded                   | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R16              | 10Y Platt vs RAW bootstrap                   | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R17              | 10Y Platt vs second-best bootstrap           | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R18              | 15Y Isotonic vs RAW bootstrap                | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R19              | 15Y Isotonic vs second-best bootstrap        | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R20              | 2025/2026 bootstrap diagnostic only          | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R21              | Isotonic unique counts recorded              | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R22              | Isotonic plateau counts recorded             | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R23              | Isotonic extreme probability counts recorded | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R24              | Platt coefficients recorded                  | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R25              | Platt intercepts recorded                    | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R26              | Platt fit positive rate recorded             | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R27              | ROI threshold fixed EV>=1.00                 | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R28              | ROI recalculation matches source             | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R29              | payout_zeroed <= normal ROI                  | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R30              | operationally_activated=false                | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R31              | Champion not changed                         | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R32              | No new CatBoost training in certification    | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R33              | No DB connection in certification            | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R34              | No new calibrator family                     | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
| R35              | Audit report generated                       | Phase6A 24-check audit plus certification audit | PASS     | True       | True              |
