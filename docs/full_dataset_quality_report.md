# Full Dataset Quality Report

Total rows: `505881`
Total races: `36269`
Win payout rows: `36111`
Place payout rows: `107588`

## Year Summary

| year | rows | races | RA join | O1 join | tan_pay | fuku_pay | win mismatch | place mismatch | elapsed sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 50076 | 3454 | 1.0 | 1.0 | 3459 | 10355 | 0 | 17 | 2.182 |
| 2017 | 49838 | 3491 | 1.0 | 0.99255588 | 3458 | 10328 | 0 | 45 | 2.698 |
| 2018 | 49124 | 3491 | 1.0 | 0.98969954 | 3460 | 10322 | 0 | 47 | 2.87 |
| 2019 | 47933 | 3491 | 1.0 | 0.99340746 | 3457 | 10300 | 0 | 62 | 2.66 |
| 2020 | 48427 | 3466 | 1.0 | 1.0 | 3465 | 10326 | 1 | 57 | 2.715 |
| 2021 | 47821 | 3456 | 1.0 | 1.0 | 3465 | 10312 | 0 | 64 | 2.675 |
| 2022 | 47220 | 3456 | 1.0 | 1.0 | 3460 | 10292 | 0 | 78 | 2.677 |
| 2023 | 47672 | 3456 | 1.0 | 1.0 | 3459 | 10303 | 0 | 73 | 2.637 |
| 2024 | 47212 | 3456 | 1.0 | 1.0 | 3460 | 10274 | 0 | 97 | 2.704 |
| 2025 | 48058 | 3468 | 1.0 | 0.99671231 | 3461 | 10276 | 0 | 100 | 2.444 |
| 2026 | 22500 | 1584 | 1.0 | 0.97097778 | 1507 | 4500 | 0 | 23 | 1.163 |

See `outputs/column_quality_summary.csv` for null rates and unique counts by column.
See `outputs/special_result_cases.csv` for abnormal and payout/target mismatch rows.

## Final Read-Back Validation

- Parquet files: `11`
- Total rows read back from Parquet: `505881`
- Total unique `race_id`: `36269`
- `entry_id` duplicates: `0`
- Overall `NL_RA` join success rate: `1.0`
- Overall `NL_O1` row join success rate: `0.996038594056705`
- `target_win` count: `36111`
- `target_ren` count: `72168`
- `target_place` count: `108235`
- `target_win` vs `is_win_paid` mismatch among normal rows: `1`
- `target_place` vs `is_place_paid` mismatch among normal rows: `663`
- Special-case output rows: `7223`

`target_place` mismatch is expected in part because `target_place` is defined as 3着以内, while JRA place payout can be top-2 only in small fields. Other mismatches are retained in `outputs/special_result_cases.csv`.

## Highest Missing Rates

| column | nulls | missing_rate |
| --- | ---: | ---: |
| `JyokenName` | 505881 | 100.0000% |
| `tan_pay_ninki` | 469770 | 92.8618% |
| `tan_slot` | 469770 | 92.8618% |
| `fuku_pay_ninki` | 398293 | 78.7325% |
| `fuku_slot` | 398293 | 78.7325% |
| `GradeCD` | 374573 | 74.0437% |
| `TanOdds` / `tan_odds` | 362956 | 71.7473% |
| `FukuOddsLow` / `fuku_odds_low` | 362956 | 71.7473% |
| `TanVote` | 362189 | 71.5957% |
| `CourseKubunCD` | 256782 | 50.7594% |

## IJyoCD Counts

| IJyoCD | rows |
| --- | ---: |
| `0` | 501744 |
| `1` | 778 |
| `3` | 1044 |
| `4` | 2294 |
| `5` | 1 |
| `7` | 20 |
