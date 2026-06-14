# Odds Missingness Audit V1

DB: `D:\keiba\new_jra_2016-2026\keiba.db`

This audit is read-only. It does not modify the source DB, V2.1.1 feature Parquet, CatBoost models, or existing predictions.

## Main Finding

For 2016-2026 JRA runner rows, `NL_O1` joins for most rows, but odds values are often NULL:

| item | rows |
|---|---:|
| SE runner rows | 505,881 |
| O1 missing rows | 2,004 |
| O1 row exists but TanOdds NULL | 360,952 |
| TanOdds invalid/zero/sentinel | 0 |
| valid TanOdds | 142,925 |
| O1 row exists but place odds NULL | 360,952 |
| valid place odds | 142,925 |
| valid SE.Odds | 501,241 |

The dominant issue is not a broad join failure. It is that `NL_O1` rows exist for many runners with `TanOdds`, `FukuOddsLow`, and `FukuOddsHigh` left NULL.

## Data Lineage

The production base dataset maps:

- `tan_odds` from `NL_O1.TanOdds`
- `tan_ninki` from `NL_O1.TanNinki`
- `fuku_odds_low` from `NL_O1.FukuOddsLow`
- `fuku_odds_high` from `NL_O1.FukuOddsHigh`
- `fuku_ninki` from `NL_O1.FukuNinki`
- `TanVote` / `FukuVote` from `NL_O1`

`NL_SE.Odds` and `NL_SE.Ninki` are retained separately. They are not used as fallback for `tan_odds`.

## SE vs O1

Where both `NL_SE.Odds` and `NL_O1.TanOdds` are valid, they match exactly:

- compared rows: 142,925
- exact match rate: 1.0
- ranking match rate by year: 1.0
- scale: same unit, no `/10` or `/100` conversion needed

This supports that both columns encode the same final single-win odds for overlapping rows. It does not prove that `SE.Odds` is safe as a pre-race model input.

## Cause Assessment

- `O1` odds values not recorded: confirmed
- `NL_O1` chosen as formal market feature source: confirmed
- join key mismatch: possible but not the main explanation
- cancellation/exclusion only: disproved
- DataKubun/flag dependence: possible
- import overwrite: possible externally, not confirmed from repository code
- type conversion or scale problem: disproved
- odds acquisition range/record type shortage: highly likely

## Use of SE.Odds

`SE.Odds` has high coverage and exact overlap with `O1.TanOdds` where both exist. It may be usable for historical final-market comparison after timing is documented. It should not be unconditionally used as `market_aware` model input until the availability time is confirmed.

## Outputs

Audit CSVs are under `outputs/odds_missingness_audit_v1/`.

Key files:

- `odds_join_coverage_by_year.csv`
- `odds_join_coverage_by_status.csv`
- `se_o1_tan_odds_comparison_summary.csv`
- `odds_missing_race_samples.csv`
- `root_cause_assessment.csv`
- `recommended_fix_plan.csv`

## Next Fix Direction

Do not apply `COALESCE`, switch to `SE.Odds`, regenerate V2.1.1, retrain models, or compute ROI yet.

Recommended next step is to inspect the external JV/Odds ingestion process and confirm why `NL_O1` contains rows with NULL odds values. If the ingestion source/timing is fixed, rebuild from base runner dataset through features and retrain market-aware models.
