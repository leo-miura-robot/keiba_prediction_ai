# Full Runner Dataset Design

Source DB: `D:\keiba\new_jra_2016-2026\keiba.db`

Base table is `NL_SE`, one row per runner. Race metadata joins from `NL_RA` by race key. Odds joins from `NL_O1` by race key plus `Umaban`. Win and place payouts are expanded from `NL_HR` payout slots and joined by race key plus `Umaban`.

The dataset is limited to JRA central racecourse codes `01` through `10`. Other `JyoCD` values in `NL_SE` do not have matching JRA odds/payout records in `NL_O1/NL_HR` and are excluded from this base JRA runner dataset.

`race_id` is `YYYYMMDDJyoCDKaijiNichijiRaceNum` with zero padding. `entry_id` appends zero-padded `Umaban`. `race_date` is derived from `Year` and `MonthDay`.

Outputs are partitioned by year under `outputs/base_runner_dataset/year=YYYY/data.parquet`.
