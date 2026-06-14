# Odds Missingness Audit V1.1

This is a read-only cause audit. It does not modify the source DB, V2.1.1 feature data, CatBoost models, or predictions.

DB: `D:\keiba\new_jra_2016-2026\keiba.db`

## Summary

`NL_O1` has the expected state columns: `RecordSpec`, `DataKubun`, `MakeDate`, `TanFlag`, `FukuFlag`, `WakurenFlag`, `HassoTime`, `TorokuTosu`, `SyussoTosu`.

For 2016-2026, the dominant pattern is race-level missingness:

| pattern | races | runner rows |
|---|---:|---:|
| all_null | 25,666 | 358,205 |
| all_valid | 9,609 | 133,519 |
| partially_valid | 837 | 12,153 |
| missing_o1_rows | 157 | 2,004 |

Single-win and place odds patterns are identical at the race-pattern level in the audit output.

## Flags And DataKubun

`TanFlag=7` and `FukuFlag=7` contain both valid and NULL odds. Therefore, the flag value alone does not identify odds availability without considering the actual odds fields.

`DataKubun=5` also contains both valid and NULL odds:

- `DataKubun=5`, `TanFlag=7`, `FukuFlag=7`, valid: 142,925 rows
- `DataKubun=5`, `TanFlag=7`, `FukuFlag=7`, NULL: 360,137 rows
- `DataKubun=9`: 815 rows, all NULL
- missing O1 rows: 2,004

Official value meanings for these codes were not confirmed from accessible public HTML, so this audit does not infer their semantic labels.

## MakeDate Timing

`MakeDate` is stored as `YYYYMMDD`. Relative to race date:

| bucket | rows | valid TanOdds |
|---|---:|---:|
| day_after | 244,270 | 69,619 |
| 2_to_7_days_after | 258,303 | 72,973 |
| same_day | 1,077 | 237 |
| more_than_7_days_after | 227 | 96 |
| unknown | 2,004 | 0 |

Valid odds exist in same-day and post-race buckets. `MakeDate` alone is not sufficient to prove the live availability timestamp.

## Runner Count

Race-count consistency output contains 36,269 races. Count anomaly flag is set for 1,928 races, including the 157 races with missing O1 rows. Most odds NULL cases are not caused by O1 rows being absent.

## Import Code

The V1.1 external import scan excludes `tasks/`, `tests/`, `docs/`, `outputs/`, `src/audit/`, and `scripts/audit_*`. It found no actual external O1 import/upsert code in this repository. Empty-record overwrite remains possible externally but is not confirmed here.

## Source Tool Follow-Up

The DB was later identified as being built with `miyamamoto/jrvltsql` quickstart:

- Documentation: `https://miyamamoto.github.io/jrvltsql/`
- Repository: `https://github.com/miyamamoto/jrvltsql`

Relevant findings from the external source tool:

- `quickstart.bat` runs `scripts/quickstart.py` and creates SQLite `data\keiba.db`.
- The documented normal-data path stores `NL_RA`, `NL_SE`, `NL_HR`, and `NL_O1`-`NL_O6`.
- `NL_O1`-`NL_O6` are documented as final odds, not investment-decision-time odds.
- Official long-retention time-series odds are stored separately in `TS_O1` and `TS_O2`.
- `TS_O1` contains single-win/place/frame odds time-series and keeps `HassoTime` plus `CollectedAt`.
- In jrvltsql schema, `NL_O1` has primary key `(Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban)`.
- In jrvltsql schema, `TS_O1` has primary key `(Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban, Kumi, HassoTime)` and also has `CollectedAt`.
- The importer uses `INSERT OR REPLACE` for batch insertion. This confirms an upsert/replace write path exists in the source tool, but it still does not prove that valid odds were overwritten by empty odds in this DB without raw fetch history or logs.

This strengthens the distinction between:

- historical final-market comparison: `NL_O1` or `SE.Odds` may be usable after missingness handling;
- model input at prediction time: `NL_O1` should not be treated as live pre-race odds;
- live or cutoff-time odds modeling: prefer `TS_O1` where available, with an explicit cutoff using `HassoTime` / `CollectedAt`.

## Official Specs

JRA-VAN official DataLab specification pages/files were found, but this audit could not verify detailed `TanFlag`, `FukuFlag`, or `DataKubun` value meanings from accessible HTML. The official JV-Data workbook/PDF/manual should be checked directly before assigning semantic labels to code values.

## Updated Cause Assessment

- O1 odds values unrecorded: confirmed
- DataKubun dependence: highly likely
- TanFlag/FukuFlag dependence: highly likely as a distribution correlate, but value meanings remain unconfirmed
- Race-level acquisition/state issue: highly likely
- Join key mismatch: disproved as main cause
- Cancellation/exclusion: disproved
- Type conversion/scale problem: disproved
- External import overwrite: possible, not confirmed. jrvltsql has an `INSERT OR REPLACE` path, but overwrite loss is not proven without raw fetch records/logs.

## Recommended Fix Direction

Primary fix direction is to inspect jrvltsql quickstart options/logs and decide whether the DB needs `quickstart_timeseries.bat --db sqlite --from <FROM> --to <TO>` or `quickstart.bat --yes --include-timeseries` for `TS_O1`/`TS_O2` coverage. Because JRA-VAN time-series retention is documented as about one year, this can only backfill recent years from the service; older years likely require already-collected local data.

`SE.Odds` may be considered only for historical final-market comparison. `COALESCE(O1, SE)` is not recommended before timing semantics are confirmed.

No fallback, COALESCE, DB re-import, feature regeneration, model retraining, ROI, or EV calculation was performed.

Outputs: `outputs/odds_missingness_audit_v1_1/`
