# New DB Fuku Payout Validation

Validated `D:\keiba\new_jra_2016-2026\keiba.db` in read-only mode.

## Result

### KakuteiJyuni_normal_only

- Eligible place rows checked: `107544`
- Matched place payout rows: `107543`
- Missing place payout rows: `1`
- Match rate: `99.999070%`
- Races with missing place payout: `1`
- Max fuku slots used in a race: `5`
- Elapsed seconds: `1.325`

### NyusenJyuni_all

- Eligible place rows checked: `107559`
- Matched place payout rows: `107557`
- Missing place payout rows: `2`
- Match rate: `99.998141%`
- Races with missing place payout: `2`
- Max fuku slots used in a race: `5`
- Elapsed seconds: `0.996`

## Interpretation

The new DB schema includes `FukuUmaban2-5`, `FukuPay2-5`, and `FukuNinki2-5` in `NL_HR`/`RT_HR`.

The vertical payout export contains `107588` fuku payout rows across `36054` races.

`KakuteiJyuni_normal_only` validates normal runners by final placing. It has one apparent mismatch caused by an abnormal case where a horse has `NyusenJyuni=1`, `KakuteiJyuni=0`, and `IJyoCD=5`, while payout slots follow the arrival-order payout result.

`NyusenJyuni_all` validates payout slots against arrival order. It is useful for parser preservation checks, but abnormal cases still require special handling because some payout records follow final placing after disqualification/demotion while one sampled case follows arrival-order payout.

The expected place cutoff used here is JRA's practical rule: 8+ starters pay top 3, 7 or fewer starters pay top 2.

## Vertical Export Integrity

`outputs\new_db_fuku_payouts_vertical.csv` was checked after export:

- Rows: `107588`
- Rows with blank `Bamei`: `0`
- Rows with blank `NyusenJyuni`: `0`
- Slot counts: slot1 `36054`, slot2 `36054`, slot3 `35392`, slot4 `87`, slot5 `1`
- `IJyoCD` counts among payout rows: `0` = `107572`, `7` = `15`, `5` = `1`

This means every exported fuku payout row can be joined back to `NL_SE` by race key plus `Umaban`.

## Outputs

- `outputs\new_db_fuku_payout_full_summary.csv`
- `outputs\new_db_fuku_payout_mismatches.csv`
- `outputs\new_db_fuku_payout_sample.csv`
- `outputs\new_db_fuku_payouts_vertical.csv`
