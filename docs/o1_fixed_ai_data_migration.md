# O1 Fixed AI Data Migration

Source DB: `D:\keiba\new_jra_2016-2026_fixed\keiba.db`

This migration reads the fixed O1 DB in read-only mode and writes a separate base runner dataset under `outputs/base_runner_dataset_o1_fixed`.

Base table is `NL_SE`, one row per runner. Race metadata joins from `NL_RA` by race key. Odds joins from `NL_O1` by race key plus `Umaban`. Win and place payouts are expanded from `NL_HR` payout slots and joined by race key plus `Umaban`.

`tan_odds`, `tan_ninki`, `fuku_odds_low`, `fuku_odds_high`, and `fuku_ninki` come from `NL_O1`. `COALESCE(O1, SE)` is not used.

`market_aware` downstream is an ideal-condition final-odds dataset. It is not a pre-race live operation dataset.

The dataset is limited to JRA central racecourse codes `01` through `10`. Other `JyoCD` values in `NL_SE` do not have matching JRA odds/payout records in `NL_O1/NL_HR` and are excluded from this base JRA runner dataset.

`race_id` is `YYYYMMDDJyoCDKaijiNichijiRaceNum` with zero padding. `entry_id` appends zero-padded `Umaban`. `race_date` is derived from `Year` and `MonthDay`.

Outputs are partitioned by year under `outputs/base_runner_dataset_o1_fixed/year=YYYY/data.parquet`.

## Split

Fixed split is read from YAML: train 2016-2023, validation 2024, test 2025, latest_holdout 2026.
