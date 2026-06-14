# DB Schema Summary

This report was generated in lightweight schema-only mode. It does not run COUNT(*), COUNT(DISTINCT), NULL counts, or full-table scans.

## SQLite files

| path                            | size_gb | last_write_time     | table_count |
| ------------------------------- | ------- | ------------------- | ----------- |
| D:\keiba\new_jra_2016-2026\keiba.db | 17.55   | 2026-06-01T12:20:23 | 74          |

## Tables

| table        | columns | primary_key                                                                         | race_key | entry_key | horse_id | date  |
| ------------ | ------- | ----------------------------------------------------------------------------------- | -------- | --------- | -------- | ----- |
| NL_AV        | 7       | KettoNum, SaleHostName, SaleName                                                    | False    | False     | True     | False |
| NL_BN        | 14      | BanusiCode                                                                          | False    | False     | False    | True  |
| NL_BR        | 14      | BreederCode                                                                         | False    | False     | False    | True  |
| NL_BT        | 8       | HansyokuNum                                                                         | False    | False     | True     | True  |
| NL_CC        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| NL_CH        | 58      | ChokyosiCode                                                                        | False    | False     | False    | True  |
| NL_CK        | 103     | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, KettoNum                            | True     | False     | True     | True  |
| NL_CS        | 9       | JyoCD, Kyori, TrackCD                                                               | False    | False     | False    | True  |
| NL_DM        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| NL_H1        | 40      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, BetType, Kumi                       | True     | False     | False    | True  |
| NL_H6        | 19      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, SanrentanKumi                       | True     | False     | False    | True  |
| NL_HA        | 20      | KaisaiDate, JyoCD, Kaiji, Nichiji, RaceNum                                          | False    | False     | False    | True  |
| NL_HC        | 11      | ChokyosiCode, Num, SetYear                                                          | False    | False     | False    | True  |
| NL_HN        | 21      | HansyokuNum                                                                         | False    | False     | True     | True  |
| NL_HR        | 110     | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| NL_HS        | 15      | KettoNum, SaleCode, FromDate                                                        | False    | False     | True     | True  |
| NL_HY        | 7       | Bamei                                                                               | False    | False     | False    | True  |
| NL_JC        | 20      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| NL_JG        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, KettoNum                            | True     | False     | True     | True  |
| NL_KS        | 68      | KisyuCode                                                                           | False    | False     | False    | True  |
| NL_NC        | 10      | JyoCD                                                                               | False    | False     | False    | True  |
| NL_NU        | 6       | UmaID                                                                               | False    | False     | True     | True  |
| NL_O1        | 29      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| NL_O2        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| NL_O3        | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| NL_O4        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| NL_O5        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| NL_O6        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| NL_OA        | 12      | KaisaiDate, JyoCD, Kaiji, Nichiji, RaceNum, OddsType, Kumi                          | False    | False     | False    | True  |
| NL_RA        | 62      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| NL_RC        | 32      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, RecInfoKubun                        | True     | False     | False    | True  |
| NL_SE        | 70      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | True     | True  |
| NL_SK        | 14      | KettoNum                                                                            | False    | False     | True     | True  |
| NL_TC        | 14      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| NL_TK        | 47      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, KettoNum                            | True     | False     | True     | True  |
| NL_TM        | 13      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| NL_UM        | 61      | KettoNum                                                                            | False    | False     | True     | True  |
| NL_WC        | 30      | ChokyoDate, ChokyoTime, KettoNum, Course                                            | False    | False     | True     | True  |
| NL_WE        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, HenkoID                                      | False    | False     | False    | True  |
| NL_WF        | 24      | Year, MonthDay                                                                      | False    | False     | False    | True  |
| NL_WH        | 16      | Year, MonthDay, JyoCD, Kaiji, Nichiji, HappyoTime, HenkoID                          | False    | False     | False    | True  |
| NL_YS        | 22      | Year, MonthDay, JyoCD, Kaiji, Nichiji                                               | False    | False     | False    | True  |
| RT_AV        | 7       | KettoNum, SaleHostName, SaleName                                                    | False    | False     | True     | False |
| RT_CC        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| RT_DM        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| RT_H1        | 40      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, BetType, Kumi                       | True     | False     | False    | True  |
| RT_H6        | 19      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, SanrentanKumi                       | True     | False     | False    | True  |
| RT_HR        | 110     | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| RT_JC        | 20      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| RT_O1        | 29      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| RT_O2        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| RT_O3        | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| RT_O4        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| RT_O5        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| RT_O6        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi                                | True     | False     | False    | True  |
| RT_RA        | 62      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| RT_RC        | 15      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| RT_SE        | 70      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | True     | True  |
| RT_TC        | 14      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum                                      | True     | False     | False    | True  |
| RT_TM        | 13      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban                              | True     | True      | False    | True  |
| RT_WE        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, HenkoID                                      | False    | False     | False    | True  |
| RT_WH        | 16      | Year, MonthDay, JyoCD, Kaiji, Nichiji, HappyoTime, HenkoID                          | False    | False     | False    | True  |
| TS_O1        | 29      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban, Kumi, HassoTime             | True     | True      | False    | True  |
| TS_O2        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime                     | True     | False     | False    | True  |
| TS_O3        | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime                     | True     | False     | False    | True  |
| TS_O4        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime                     | True     | False     | False    | True  |
| TS_O5        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime                     | True     | False     | False    | True  |
| TS_O6        | 17      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime                     | True     | False     | False    | True  |
| TS_SOKUHO_O1 | 30      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban, Kumi, HassoTime, SourceSpec | True     | True      | False    | True  |
| TS_SOKUHO_O2 | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime, SourceSpec         | True     | False     | False    | True  |
| TS_SOKUHO_O3 | 19      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime, SourceSpec         | True     | False     | False    | True  |
| TS_SOKUHO_O4 | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime, SourceSpec         | True     | False     | False    | True  |
| TS_SOKUHO_O5 | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime, SourceSpec         | True     | False     | False    | True  |
| TS_SOKUHO_O6 | 18      | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Kumi, HassoTime, SourceSpec         | True     | False     | False    | True  |

## Main data structure

- Race key is generally `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum`.
- Entry key is generally race key plus `Umaban`; horse identifier is mainly `KettoNum`.
- Race-level tables: `NL_RA` / `RT_RA` contain race metadata such as course, distance, class and race name.
- Entry-level tables: `NL_SE` / `RT_SE` contain runners, horse, jockey, trainer, weight, body weight and result columns.
- Odds tables: `NL_O1` / `RT_O1` / `TS_O1` / `TS_SOKUHO_O1` contain win/place odds fields such as `TanOdds`, `FukuOddsLow`, `FukuOddsHigh`.
- Payout tables: `NL_HR` / `RT_HR` and `NL_HA` / `RT_HA` appear to contain payout/refund data. Detailed payout decoding should be done in a separate lightweight step.

## Candidate join for one row per runner

Use `RT_SE` or `NL_SE` as the base runner table. Join `RT_RA`/`NL_RA` by race key, and join `RT_O1`/`NL_O1` or time-series `TS_O1` by race key plus `Umaban`. Payout tables join by race key and then require bet-type/combination decoding for win/place returns.

## Sample rows from main tables

### NL_HA

No rows returned by `LIMIT 5`.

### NL_HR

First 12 columns shown for readability; query used was `SELECT * FROM table LIMIT 5`.

| column         | row1     | row2     | row3     | row4     | row5     |
| -------------- | -------- | -------- | -------- | -------- | -------- |
| RecordSpec     | HR       | HR       | HR       | HR       | HR       |
| DataKubun      | 2        | 2        | 2        | 2        | 2        |
| MakeDate       | 20160106 | 20160106 | 20160106 | 20160106 | 20160106 |
| Year           | 2016     | 2016     | 2016     | 2016     | 2016     |
| MonthDay       | 105      | 105      | 105      | 105      | 105      |
| JyoCD          | 06       | 06       | 06       | 06       | 06       |
| Kaiji          | 1        | 1        | 1        | 1        | 1        |
| Nichiji        | 1        | 1        | 1        | 1        | 1        |
| RaceNum        | 1        | 2        | 3        | 4        | 5        |
| TorokuTosu     | 16       | 16       | 16       | 16       | 15       |
| SyussoTosu     | 16       | 16       | 16       | 16       | 15       |
| FuseirituFlag1 | 0        | 0        | 0        | 0        | 0        |

### NL_O1

First 12 columns shown for readability; query used was `SELECT * FROM table LIMIT 5`.

| column     | row1     | row2     | row3     | row4     | row5     |
| ---------- | -------- | -------- | -------- | -------- | -------- |
| RecordSpec | O1       | O1       | O1       | O1       | O1       |
| DataKubun  | 5        | 5        | 5        | 5        | 5        |
| MakeDate   | 20160106 | 20160106 | 20160106 | 20160106 | 20160106 |
| Year       | 2016     | 2016     | 2016     | 2016     | 2016     |
| MonthDay   | 105      | 105      | 105      | 105      | 105      |
| JyoCD      | 06       | 06       | 06       | 06       | 06       |
| Kaiji      | 1        | 1        | 1        | 1        | 1        |
| Nichiji    | 1        | 1        | 1        | 1        | 1        |
| RaceNum    | 1        | 1        | 1        | 1        | 1        |
| HassoTime  | 00000000 | 00000000 | 00000000 | 00000000 | 00000000 |
| TorokuTosu | 16       | 16       | 16       | 16       | 16       |
| SyussoTosu | 16       | 16       | 16       | 16       | 16       |

### NL_RA

First 12 columns shown for readability; query used was `SELECT * FROM table LIMIT 5`.

| column     | row1     | row2     | row3     | row4     | row5     |
| ---------- | -------- | -------- | -------- | -------- | -------- |
| RecordSpec | RA       | RA       | RA       | RA       | RA       |
| DataKubun  | 7        | 7        | 7        | 7        | 7        |
| MakeDate   | 20160106 | 20160106 | 20160106 | 20160106 | 20160106 |
| Year       | 2016     | 2016     | 2016     | 2016     | 2016     |
| MonthDay   | 105      | 105      | 105      | 105      | 105      |
| JyoCD      | 06       | 06       | 06       | 06       | 06       |
| Kaiji      | 1        | 1        | 1        | 1        | 1        |
| Nichiji    | 1        | 1        | 1        | 1        | 1        |
| RaceNum    | 1        | 2        | 3        | 4        | 5        |
| YoubiCD    | 5        | 5        | 5        | 5        | 5        |
| TokuNum    | 0000     | 0000     | 0000     | 0000     | 0000     |
| Hondai     |          |          |          |          |          |

### NL_SE

First 12 columns shown for readability; query used was `SELECT * FROM table LIMIT 5`.

| column     | row1       | row2       | row3       | row4       | row5       |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| RecordSpec | SE         | SE         | SE         | SE         | SE         |
| DataKubun  | 7          | 7          | 7          | 7          | 7          |
| MakeDate   | 20160106   | 20160106   | 20160106   | 20160106   | 20160106   |
| Year       | 2016       | 2016       | 2016       | 2016       | 2016       |
| MonthDay   | 105        | 105        | 105        | 105        | 105        |
| JyoCD      | 06         | 06         | 06         | 06         | 06         |
| Kaiji      | 1          | 1          | 1          | 1          | 1          |
| Nichiji    | 1          | 1          | 1          | 1          | 1          |
| RaceNum    | 1          | 1          | 1          | 1          | 1          |
| Wakuban    | 1          | 1          | 2          | 2          | 3          |
| Umaban     | 1          | 2          | 3          | 4          | 5          |
| KettoNum   | 2013105621 | 2013104555 | 2013104573 | 2013102283 | 2013100107 |

### RT_HR

No rows returned by `LIMIT 5`.

### RT_O1

No rows returned by `LIMIT 5`.

### RT_RA

No rows returned by `LIMIT 5`.

### RT_SE

No rows returned by `LIMIT 5`.

### TS_O1

No rows returned by `LIMIT 5`.

### TS_SOKUHO_O1

No rows returned by `LIMIT 5`.

## Notes

- `outputs/table_columns.csv` contains all table/column definitions obtained from `PRAGMA table_info`.
- No row counts or distinct counts are included in this lightweight pass.
- Feature inventory and leakage classification should be generated after confirming table meanings and payout encoding.
