# Main Table Detail

This detail check only used targeted `SELECT ... LIMIT 20`, indexed joins from limited base rows, and a limited `IJyoCD` check over the first 10000 `NL_SE` rows. It does not run full-table counts or all-column aggregation.

## Target tables

- `NL_SE`: runner-level base table.
- `NL_RA`: race-level metadata.
- `NL_O1`: win/place odds by race key plus `Umaban`.
- `NL_HR`: race-level payouts for win/place and other bet types.
- `NL_UM`: horse master, joinable by `KettoNum`.
- `NL_KS`: jockey master, joinable by `KisyuCode`.
- `NL_CH`: trainer master, joinable by `ChokyosiCode`.

## Limited samples

### NL_SE

| column   | row1       | row2       | row3       | row4       | row5       |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Year     | 2016       | 2016       | 2016       | 2016       | 2016       |
| MonthDay | 105        | 105        | 105        | 105        | 105        |
| JyoCD    | 06         | 06         | 06         | 06         | 06         |
| Kaiji    | 1          | 1          | 1          | 1          | 1          |
| Nichiji  | 1          | 1          | 1          | 1          | 1          |
| RaceNum  | 1          | 1          | 1          | 1          | 1          |
| Wakuban  | 1          | 1          | 2          | 2          | 3          |
| Umaban   | 1          | 2          | 3          | 4          | 5          |
| KettoNum | 2013105621 | 2013104555 | 2013104573 | 2013102283 | 2013100107 |
| Bamei    | プロジェクション   | トウショウシェル   | トウショウアパッチ  | レオベローナ     | ブライトピスケス   |
| SexCD    | 2          | 1          | 1          | 2          | 1          |
| Barei    | 3          | 3          | 3          | 3          | 3          |

### NL_RA

| column    | row1 | row2 | row3 | row4 | row5 |
| --------- | ---- | ---- | ---- | ---- | ---- |
| Year      | 2016 | 2016 | 2016 | 2016 | 2016 |
| MonthDay  | 105  | 105  | 105  | 105  | 105  |
| JyoCD     | 06   | 06   | 06   | 06   | 06   |
| Kaiji     | 1    | 1    | 1    | 1    | 1    |
| Nichiji   | 1    | 1    | 1    | 1    | 1    |
| RaceNum   | 1    | 2    | 3    | 4    | 5    |
| YoubiCD   | 5    | 5    | 5    | 5    | 5    |
| GradeCD   |      |      |      |      |      |
| SyubetuCD | 12   | 12   | 14   | 14   | 12   |
| JyokenCD1 | 000  | 000  | 000  | 000  | 000  |
| JyokenCD2 | 703  | 703  | 000  | 000  | 703  |
| JyokenCD3 | 000  | 000  | 005  | 005  | 000  |

### NL_O1

| column      | row1     | row2     | row3     | row4     | row5     |
| ----------- | -------- | -------- | -------- | -------- | -------- |
| Year        | 2016     | 2016     | 2016     | 2016     | 2016     |
| MonthDay    | 105      | 105      | 105      | 105      | 105      |
| JyoCD       | 06       | 06       | 06       | 06       | 06       |
| Kaiji       | 1        | 1        | 1        | 1        | 1        |
| Nichiji     | 1        | 1        | 1        | 1        | 1        |
| RaceNum     | 1        | 1        | 1        | 1        | 1        |
| HassoTime   | 00000000 | 00000000 | 00000000 | 00000000 | 00000000 |
| Umaban      | 1        | 2        | 3        | 4        | 5        |
| TanOdds     | 190.5    | 84.2     | 14.7     | 292.7    | 5.4      |
| TanNinki    | 14       | 11       | 6        | 16       | 3        |
| FukuUmaban  | 1        | 2        | 3        | 4        | 5        |
| FukuOddsLow | 36.0     | 18.8     | 3.1      | 48.3     | 1.5      |

### NL_HR

| column     | row1 | row2 | row3 | row4 | row5 |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| Year       | 2016 | 2016 | 2016 | 2016 | 2016 |
| MonthDay   | 105  | 105  | 105  | 105  | 105  |
| JyoCD      | 06   | 06   | 06   | 06   | 06   |
| Kaiji      | 1    | 1    | 1    | 1    | 1    |
| Nichiji    | 1    | 1    | 1    | 1    | 1    |
| RaceNum    | 1    | 2    | 3    | 4    | 5    |
| TanUmaban  | 12   | 05   | 12   | 10   | 02   |
| TanPay     | 350  | 1580 | 360  | 240  | 400  |
| TanNinki   | 1    | 7    | 1    | 1    | 2    |
| FukuUmaban | 12   | 05   | 12   | 10   | 02   |
| FukuPay    | 120  | 440  | 200  | 110  | 170  |
| FukuNinki  | 1    | 5    | 1    | 1    | 2    |

## Join check sample

| race_id               | Umaban | Bamei     | KakuteiJyuni | joined_ra | joined_o1 | joined_hr | se_odds | o1_tan_odds | odds_relation | TanUmaban | TanPay | FukuUmaban | FukuPay | place_in_fuku_umaban |
| --------------------- | ------ | --------- | ------------ | --------- | --------- | --------- | ------- | ----------- | ------------- | --------- | ------ | ---------- | ------- | -------------------- |
| 2016-0105-06-01-01-01 | 1      | プロジェクション  | 16           | 1         | 1         | 1         | 190.5   | 190.5       | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 2      | トウショウシェル  | 10           | 1         | 1         | 1         | 84.2    | 84.2        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 3      | トウショウアパッチ | 14           | 1         | 1         | 1         | 14.7    | 14.7        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 4      | レオベローナ    | 8            | 1         | 1         | 1         | 292.7   | 292.7       | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 5      | ブライトピスケス  | 5            | 1         | 1         | 1         | 5.4     | 5.4         | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 6      | ワンダフルブルー  | 12           | 1         | 1         | 1         | 115.0   | 115.0       | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 7      | マイティジャック  | 6            | 1         | 1         | 1         | 14.9    | 14.9        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 8      | モンマルトル    | 15           | 1         | 1         | 1         | 23.4    | 23.4        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 9      | ビバラビダ     | 4            | 1         | 1         | 1         | 65.6    | 65.6        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 10     | カリアティード   | 9            | 1         | 1         | 1         | 6.9     | 6.9         | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 11     | タイトルリーフ   | 2            | 1         | 1         | 1         | 7.9     | 7.9         | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 12     | シゲルヒラマサ   | 1            | 1         | 1         | 1         | 3.5     | 3.5         | same          | 12        | 350    | 12         | 120     | 1                    |
| 2016-0105-06-01-01-01 | 13     | ノボホウセイ    | 3            | 1         | 1         | 1         | 3.8     | 3.8         | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 14     | アキツシマ     | 7            | 1         | 1         | 1         | 227.7   | 227.7       | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 15     | グットトキメク   | 11           | 1         | 1         | 1         | 112.0   | 112.0       | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-01 | 16     | リリーウェントス  | 13           | 1         | 1         | 1         | 52.1    | 52.1        | same          | 12        | 350    | 12         | 120     | 0                    |
| 2016-0105-06-01-01-02 | 1      | サンマルトゥーレ  | 15           | 1         | 1         | 1         | 182.5   | 182.5       | same          | 05        | 1580   | 05         | 440     | 0                    |
| 2016-0105-06-01-01-02 | 2      | ダイメイキング   | 10           | 1         | 1         | 1         | 313.1   | 313.1       | same          | 05        | 1580   | 05         | 440     | 0                    |
| 2016-0105-06-01-01-02 | 3      | ニシノタイタン   | 8            | 1         | 1         | 1         | 8.8     | 8.8         | same          | 05        | 1580   | 05         | 440     | 0                    |
| 2016-0105-06-01-01-02 | 4      | ノーブルポセイドン | 14           | 1         | 1         | 1         | 252.2   | 252.2       | same          | 05        | 1580   | 05         | 440     | 0                    |

## Limited IJyoCD values

`IJyoCD` values observed in the first 10000 `NL_SE` rows: `0`, `1`, `3`, `4`, `7`

## Odds relation

In the limited join sample, `NL_SE.Odds` and `NL_O1.TanOdds` are compared in `outputs/join_check_sample.csv` with `odds_relation`. Use this as a structural check only; a later validation should sample multiple race dates intentionally without full scans.

## Payout relation

In the limited sample, `NL_HR.TanUmaban` matches the `KakuteiJyuni = 1` runner, so win payout linkage is straightforward by race key plus winning `Umaban`.

`NL_HR.FukuUmaban/FukuPay` did not expose all top-3 place payouts in the first sampled race. For example, the first race has top-3 `Umaban` 12, 11, and 13 in `NL_SE`, but `NL_HR.FukuUmaban` is only `12`. This is enough to confirm a winner-place payout, but not enough to safely evaluate every `target_place` row. The payout parser/schema or another payout source must be confirmed before a full place-return backtest.
