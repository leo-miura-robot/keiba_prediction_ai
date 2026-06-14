# Fuku Payout Design

## Conclusion

複勝払戻の全対象馬を `NL_HR` だけから確定することは、今回の少数サンプルではできませんでした。`NL_HR.FukuUmaban/FukuPay` は存在しますが、最初のサンプルレースでは3着以内の全馬を表していません。

`NL_H1` には `BetType = Fukusyo` と `Kumi` があり、馬番別の行はあります。ただし `Hyo` と `Ninki` は票数・人気であり、払戻額ではありません。したがって `NL_H1` は複勝の馬番候補や投票情報の確認には使えますが、複勝払戻額の取得元としては不足しています。

現時点で単勝払戻は `NL_HR.TanUmaban/TanPay` で扱えますが、複勝回収率を正しく計算するには、全複勝払戻を持つ別テーブル、未展開カラム、または元データパーサの確認が必要です。

列名上の別候補として `NL_HA.PayKumi1-3/PayAmount1-3` がありますが、このDBでは `LIMIT 5` で行が返らず、利用できる実データは確認できませんでした。`NL_WF.PayJyushosiki` はWIN5系の集計に見えるため、出走馬単位の複勝払戻ソースではないと判断しています。

## Race Samples

### 2016010506010101

- 1着馬番: 12
- 2着馬番: 11
- 3着馬番: 13
- NL_HR上の複勝馬番候補: `12`
- NL_HR上の複勝払戻候補: `120`
- NL_H1上の複勝候補: 01:1671票(ninki=15), 02:3249票(ninki=12), 03:23099票(ninki=6), 04:1242票(ninki=16), 05:57716票(ninki=3), 06:2656票(ninki=13), 07:21887票(ninki=7), 08:14291票(ninki=8), 09:5151票(ninki=10), 10:38559票(ninki=5), 11:39928票(ninki=4), 12:98569票(ninki=1), 13:71859票(ninki=2), 14:1802票(ninki=14), 15:4693票(ninki=11), 16:6671票(ninki=9)
- 結論: NL_HR covers only part of the top-3 place horses in this sample.

### 2016010506010102

- 1着馬番: 5
- 2着馬番: 15
- 3着馬番: 7
- NL_HR上の複勝馬番候補: `05`
- NL_HR上の複勝払戻候補: `440`
- NL_H1上の複勝候補: 01:1959票(ninki=13), 02:1114票(ninki=14), 03:38682票(ninki=4), 04:2072票(ninki=12), 05:26298票(ninki=5), 06:88637票(ninki=2), 07:7075票(ninki=9), 08:10900票(ninki=8), 09:20480票(ninki=7), 10:45148票(ninki=3), 11:141166票(ninki=1), 12:4984票(ninki=11), 13:6343票(ninki=10), 14:1046票(ninki=15), 15:21043票(ninki=6), 16:981票(ninki=16)
- 結論: NL_HR covers only part of the top-3 place horses in this sample.

### 2016010506010103

- 1着馬番: 12
- 2着馬番: 8
- 3着馬番: 13
- NL_HR上の複勝馬番候補: `12`
- NL_HR上の複勝払戻候補: `200`
- NL_H1上の複勝候補: 01:11622票(ninki=9), 02:31698票(ninki=8), 03:44396票(ninki=3), 04:7315票(ninki=13), 05:33921票(ninki=6), 06:3059票(ninki=14), 07:33149票(ninki=7), 08:10151票(ninki=11), 09:2109票(ninki=16), 10:7712票(ninki=12), 11:45635票(ninki=2), 12:65223票(ninki=1), 13:10202票(ninki=10), 14:2412票(ninki=15), 15:42058票(ninki=4), 16:41695票(ninki=5)
- 結論: NL_HR covers only part of the top-3 place horses in this sample.

### 2016010506010104

- 1着馬番: 10
- 2着馬番: 7
- 3着馬番: 2
- NL_HR上の複勝馬番候補: `10`
- NL_HR上の複勝払戻候補: `110`
- NL_H1上の複勝候補: 01:33875票(ninki=3), 02:29254票(ninki=4), 03:7029票(ninki=11), 04:3551票(ninki=14), 05:2035票(ninki=15), 06:14812票(ninki=7), 07:117915票(ninki=2), 08:1731票(ninki=16), 09:7071票(ninki=9), 10:180926票(ninki=1), 11:27237票(ninki=6), 12:10383票(ninki=8), 13:7037票(ninki=10), 14:28270票(ninki=5), 15:6507票(ninki=13), 16:6756票(ninki=12)
- 結論: NL_HR covers only part of the top-3 place horses in this sample.

### 2016010506010105

- 1着馬番: 2
- 2着馬番: 13
- 3着馬番: 10
- NL_HR上の複勝馬番候補: `02`
- NL_HR上の複勝払戻候補: `170`
- NL_H1上の複勝候補: 01:22864票(ninki=7), 02:76932票(ninki=2), 03:18769票(ninki=9), 04:4459票(ninki=13), 05:105280票(ninki=1), 06:7320票(ninki=10), 07:1848票(ninki=14), 08:21193票(ninki=8), 09:23864票(ninki=6), 10:60763票(ninki=4), 11:6961票(ninki=11), 12:1843票(ninki=15), 13:65257票(ninki=3), 14:6705票(ninki=12), 15:57117票(ninki=5)
- 結論: NL_HR covers only part of the top-3 place horses in this sample.

## Vertical Format Feasibility

Target format is:

```text
race_id,Umaban,fuku_pay
2016010506010112,12,120
2016010506010111,11,
2016010506010113,13,
```

The format itself can be produced, and `race_id + Umaban` can be joined back to `NL_SE`. However, using only `NL_HR.FukuUmaban/FukuPay`, only rows whose `Umaban` equals `FukuUmaban` can receive a payout. Other placed horses remain unresolved in the current sample.

## Candidate Vertical Rows From Sample

| race_id          | Umaban | KakuteiJyuni | fuku_pay | matched | notes                                       |
| ---------------- | ------ | ------------ | -------- | ------- | ------------------------------------------- |
| 2016010506010101 | 12     | 1            | 120      | 1       | matched exact FukuUmaban                    |
| 2016010506010101 | 11     | 2            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010101 | 13     | 3            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010102 | 5      | 1            | 440      | 1       | matched exact FukuUmaban                    |
| 2016010506010102 | 15     | 2            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010102 | 7      | 3            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010103 | 12     | 1            | 200      | 1       | matched exact FukuUmaban                    |
| 2016010506010103 | 8      | 2            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010103 | 13     | 3            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010104 | 10     | 1            | 110      | 1       | matched exact FukuUmaban                    |
| 2016010506010104 | 7      | 2            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010104 | 2      | 3            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010105 | 2      | 1            | 170      | 1       | matched exact FukuUmaban                    |
| 2016010506010105 | 13     | 2            |          | 0       | not represented by sampled NL_HR FukuUmaban |
| 2016010506010105 | 10     | 3            |          | 0       | not represented by sampled NL_HR FukuUmaban |

## Recommended Source

- Use `NL_HR.TanUmaban/TanPay` for win payout.
- Do not treat `NL_H1.Hyo` as place payout; it is vote count.
- Treat `NL_HR.FukuUmaban/FukuPay` as an incomplete place payout source until parser/schema confirmation.
- `NL_HA.PayKumi1-3/PayAmount1-3` would be a natural source if populated, but it appears empty in this DB.
- Check whether the data loader lost repeated JRA-VAN HR payout slots for place payouts, or whether another table/file contains the full repeated place payouts.

## Edge Cases

- Small fields: races with low starter counts can have fewer place payout targets; target/evaluation rules must follow JRA place-payout rules for the race size.
- Scratches/exclusions: use `IJyoCD`, refund flags, and `HenkanUma*` fields so canceled runners do not receive normal losing outcomes.
- Dead heats: multiple horses can share the same placing; payout rows may exceed the usual 2 or 3 targets and must be decoded from official payout slots.
- Coupled or special refund cases: `TokubaraiFlag*`, `FuseirituFlag*`, and refund fields should remain evaluation-only metadata.

## Implementation Notes

Keep payout extraction as a separate decoder that outputs `race_id, Umaban, fuku_pay, source_table, source_columns`. Do not merge it into feature extraction. Add assertions on sampled races: every normal top-3 horse should have a place payout unless race-size rules say otherwise.
