# Dataset Design

## One Row Per Runner

Use `NL_SE` as the base table. It is already one row per runner, keyed by `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban`.

Join race metadata from `NL_RA` on `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum`.

Join win/place odds from `NL_O1` on `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban`.

Join payouts from `NL_HR` on race key only. `TanUmaban/TanPay` identify the winning horse and win return. `FukuUmaban/FukuPay` are race-level place payout fields, but the limited sample did not expose all top-3 place payouts, so they require parser/schema confirmation before being used for full place-return evaluation.

Optional master joins: `NL_UM` by `KettoNum`, `NL_KS` by `KisyuCode`, and `NL_CH` by `ChokyosiCode`. These should be treated carefully because master tables may contain cumulative or latest records rather than strictly pre-race state.

## IDs

- `race_id`: concatenate zero-padded `Year`, `MonthDay`, `JyoCD`, `Kaiji`, `Nichiji`, `RaceNum`, for example `2016-0105-06-01-01-01`.
- `entry_id`: `race_id` plus zero-padded `Umaban`.
- `horse_id`: `NL_SE.KettoNum`.

## Odds And Payout Columns

- Win odds: prefer `NL_O1.TanOdds` for the canonical odds table. `NL_SE.Odds` appears to be final win odds in runner results and should be treated as post-result/evaluation unless the timing is confirmed.
- Place odds: use `NL_O1.FukuOddsLow` and `NL_O1.FukuOddsHigh`.
- Win payout: use `NL_HR.TanPay`, matched by race key and `NL_HR.TanUmaban`.
- Place payout: `NL_HR.FukuPay` is only safe for rows where `NL_HR.FukuUmaban` matches the target `Umaban`. In the limited sample it did not cover every top-3 horse, so full place payout extraction is unresolved.

## Targets

- `target_win`: `1` when `KakuteiJyuni = 1` and normal starter, else `0`.
- `target_ren`: `1` when `KakuteiJyuni <= 2` and normal starter, else `0`.
- `target_place`: `1` when `KakuteiJyuni <= 3` and normal starter, else `0`. For races with fewer than standard starters, place payout rules may differ and should be handled separately in evaluation.
- Exclude or separately flag abnormal rows using `IJyoCD`. Limited observed values: `0`, `1`, `3`, `4`, `7`.

## Predictable Features

`NL_RA`: `Year`, `MonthDay`, `JyoCD`, `YoubiCD`, `GradeCD`, `SyubetuCD`, `JyokenCD1-5`, `JyokenName`, `Kyori`, `TrackCD`, `CourseKubunCD`, `HassoTime`, `TorokuTosu`, `SyussoTosu`, and pre-race weather/going if confirmed available before prediction.

`NL_SE`: `Wakuban`, `Umaban`, `KettoNum`, `SexCD`, `Barei`, `ChokyosiCode`, `KisyuCode`, `Futan`, `BaTaijyu`, `ZogenFugo`, `ZogenSa`, `Blinker`, `MinaraiCD` if known before the bet.

`NL_O1`: `TanOdds`, `TanNinki`, `FukuOddsLow`, `FukuOddsHigh`, `FukuNinki`, and vote columns only when using a clearly defined pre-deadline snapshot. For final odds modeling, keep them as market features but document the prediction timing.

## Leakage Columns

Do not use `KakuteiJyuni`, `NyusenJyuni`, `Time`, `ChakusaCD`, `Jyuni1c-4c`, `HaronTimeL3/L4`, `TimeDiff`, `DMTime`, `DMJyuni`, `KyakusituKubun`, prize columns from the completed race, or any `NL_HR` payout fields as model features.

Avoid using `NL_RA` race-result fields such as `LapTime`, `Haron3F`, `Haron4F`, `Haron3L`, `Haron4L`, `Corner`, and `TsukaJyuni` as features.

## Ambiguous Features

`NL_SE.Odds/Ninki` and `NL_O1.TanOdds/TanNinki` may be final odds. They are acceptable only if the intended prediction point is immediately before betting close and the same timing can be reproduced. For stricter no-leakage prediction, use `TS_O1` or `TS_SOKUHO_O1` snapshots filtered by time.

`BaTaijyu` and `ZogenSa` are usually announced before the race, but the operational timing should be confirmed. `TenkoCD`, `SibaBabaCD`, and `DirtBabaCD` can change during the day, so use only the latest value available before prediction.

`NL_UM`, `NL_KS`, and `NL_CH` contain master and cumulative fields. Use stable identity/profile fields directly; cumulative performance fields need as-of-date reconstruction to avoid using future records.

## Recommended Next Design

Build a thin dataset extraction script with explicit selected columns, an as-of date filter, and no broad aggregation. Start from `NL_SE`, join `NL_RA` and `NL_O1`, create targets from `KakuteiJyuni`, and create win evaluation returns from `NL_HR`. Before modeling place ROI, resolve the complete source for all place payouts. After that, add historical rolling features using only races before each target race.
