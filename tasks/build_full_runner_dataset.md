新DBを使って、2016〜2026年の「1行=1出走馬」の基礎データセットを全件作成してください。

## 使用DB

```text
D:\keiba\new_jra_2016-2026\keiba.db
```

旧DBは使用しないでください。

## 事前確認

まず既存の調査資料とスクリプトを確認し、すでに確定している結合方法や複勝払戻仕様を再調査しないでください。

確認対象:

```text
docs/db_schema_summary.md
docs/main_table_detail.md
docs/dataset_design.md
docs/new_db_fuku_payout_validation.md
scripts/validate_new_db_fuku_payout_full.py
```

確認済み事項:

* `NL_SE` を1行=1出走馬のベースにできる
* `NL_RA` はrace keyで結合できる
* `NL_O1` はrace key + Umabanで結合できる
* 単勝払戻は `NL_HR.TanUmaban / TanPay`
* 複勝払戻は `FukuUmaban / FukuPay` と2〜5番目のスロットを縦持ち展開して取得できる
* 新DBでは通常ケースの複勝払戻欠損は解消済み
* `IJyoCD=5/7` など、失格・降着系の特殊ケースが少数存在する

## 今回の目的

今回はモデル学習やバックテストは行いません。

以下を作成してください。

1. 2016〜2026年の全出走馬基礎データセット
2. 単勝・連対・複勝の目的変数
3. 単勝・複勝の実払戻額
4. 特徴量候補一覧
5. 未来情報リーク一覧
6. 欠損率・カテゴリ数・結合率などの品質レポート
7. 特殊結果の一覧

## 使用テーブル

* 出走馬ベース: `NL_SE`
* レース情報: `NL_RA`
* 単勝・複勝オッズ: `NL_O1`
* 単勝・複勝払戻: `NL_HR`

結合キーは既存の設計資料に従ってください。

race key:

```text
Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum
```

entry key:

```text
Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban
```

以下のIDを作成してください。

```text
race_id
entry_id
race_date
```

年をまたいでも重複しない、固定のゼロ埋め規則を使用してください。

## 目的変数

確定着順を基準として以下を作成してください。

```text
target_win   = 1着
target_ren   = 2着以内
target_place = 3着以内
```

ただし、取消、除外、競走中止、失格、降着などを無条件で通常データとして扱わないでください。

以下の状態列も作成してください。

```text
is_normal_runner
is_abnormal_result
is_win_paid
is_place_paid
```

## 払戻データ

### 単勝

```text
TanUmaban
TanPay
```

購入対象馬の `Umaban` と `TanUmaban` が一致する場合に `tan_pay` を設定し、それ以外は0にしてください。

### 複勝

以下をすべて縦持ち展開してください。

```text
FukuUmaban / FukuPay / FukuNinki
FukuUmaban2 / FukuPay2 / FukuNinki2
FukuUmaban3 / FukuPay3 / FukuNinki3
FukuUmaban4 / FukuPay4 / FukuNinki4
FukuUmaban5 / FukuPay5 / FukuNinki5
```

race key + Umabanで `NL_SE` に結合し、`fuku_pay` を作成してください。

以下の列を持たせてください。

```text
tan_pay
fuku_pay
tan_payout_record_found
fuku_payout_record_found
is_win_paid
is_place_paid
```

一般の非的中馬は払戻0としてください。ただし、払戻レコード未取得と非的中を区別できるようにしてください。

## 使用する主な列

### NL_SE

少なくとも以下を含めてください。

```text
Year
MonthDay
JyoCD
Kaiji
Nichiji
RaceNum
Wakuban
Umaban
KettoNum
Bamei
SexCD
Barei
ChokyosiCode
ChokyosiRyakusyo
KisyuCode
KisyuRyakusyo
Futan
BaTaijyu
ZogenFugo
ZogenSa
IJyoCD
NyusenJyuni
KakuteiJyuni
Odds
Ninki
Time
ChakusaCD
HaronTimeL3
Jyuni1c
Jyuni2c
Jyuni3c
Jyuni4c
```

### NL_RA

少なくとも以下を含めてください。

```text
YoubiCD
GradeCD
SyubetuCD
JyokenCD1
JyokenCD2
JyokenCD3
JyokenCD4
JyokenCD5
JyokenName
Kyori
TrackCD
CourseKubunCD
HassoTime
TorokuTosu
SyussoTosu
TenkoCD
SibaBabaCD
DirtBabaCD
```

### NL_O1

少なくとも以下を含めてください。

```text
TanOdds
TanNinki
TanVote
FukuOddsLow
FukuOddsHigh
FukuNinki
FukuVote
```

単勝オッズは原則として `NL_O1.TanOdds` を正式列として使用してください。

## 出力形式

全件データを巨大な単一CSVにしないでください。

年別Parquetとして保存してください。

```text
outputs/base_runner_dataset/
  year=2016/data.parquet
  year=2017/data.parquet
  ...
  year=2026/data.parquet
```

圧縮形式は `zstd` または `snappy` を使用してください。

追加出力:

```text
outputs/base_runner_dataset_sample.csv
outputs/full_dataset_summary.csv
outputs/column_quality_summary.csv
outputs/special_result_cases.csv
docs/full_dataset_design.md
docs/feature_inventory_full.md
docs/leakage_check_full.md
docs/full_dataset_quality_report.md
logs/build_full_dataset.log
```

## 特殊ケース

`IJyoCD=5/7` など、確定着順と払戻対象が単純に一致しないケースは削除せず、以下へ出力してください。

```text
outputs/special_result_cases.csv
```

含める列:

```text
race_id
entry_id
Umaban
Bamei
IJyoCD
NyusenJyuni
KakuteiJyuni
target_win
target_ren
target_place
tan_pay
fuku_pay
is_win_paid
is_place_paid
notes
```

今回は特殊ケースを勝手に補正せず、通常データと分けて記録してください。

## 特徴量分類

全カラムを以下に分類してください。

### 直接使用可能

例:

* 競馬場
* 距離
* 芝・ダート
* コース
* グレード
* レース条件
* 頭数
* 枠番
* 馬番
* 性別
* 年齢
* 斤量
* 騎手
* 調教師

### 使用時点に注意が必要

例:

* 単勝オッズ
* 複勝オッズ
* 人気
* 票数
* 馬体重
* 馬体重増減
* 天候
* 馬場状態

これらは予測を行う時点によって利用可否が変わるため、`conditionally_usable` として分類してください。

### 過去走から加工して利用可能

例:

* 過去着順
* 過去タイム
* 過去上がり3F
* 過去通過順位
* 過去オッズ
* 過去人気
* 騎手過去成績
* 調教師過去成績
* 同競馬場成績
* 同距離成績

当該レースの値はリークですが、過去レースの値は特徴量の材料にできます。

### 当該レースでは未来情報リーク

例:

```text
KakuteiJyuni
NyusenJyuni
Time
ChakusaCD
HaronTimeL3
Jyuni1c〜4c
TanPay
FukuPay
```

### 目的変数・評価専用

```text
target_win
target_ren
target_place
tan_pay
fuku_pay
is_win_paid
is_place_paid
```

## 品質確認

年別および全体で以下を集計してください。

* 出走馬行数
* レース数
* entry_id重複数
* `NL_RA` 結合成功率
* `NL_O1` 結合成功率
* 単勝オッズ取得率
* 複勝オッズ取得率
* 単勝払戻件数
* 複勝払戻件数
* target_win件数
* target_ren件数
* target_place件数
* target_winとis_win_paidの不一致数
* target_placeとis_place_paidの不一致数
* IJyoCD別件数
* 各カラムの欠損率
* カテゴリ列のユニーク数
* 単勝オッズの0以下・異常値件数
* 複勝オッズ上下限の異常件数
* entry_idが重複するケース

## 長時間処理への対応

18GB超のSQLite DBなので、全件を一度にDataFrameへ読み込まないでください。

必ず以下を実装してください。

* SQLiteは読み取り専用で接続する
* 年単位またはチャンク単位で処理する
* 年ごとにParquetを保存する
* 完了した年にチェックポイントを残す
* 再実行時に完了済み年をスキップする
* `--resume`、`--force`、`--years` を実装する
* 一時ファイルへ保存し、正常終了時に正式ファイルへ変更する
* 各年の開始・終了・行数・経過時間をログへ出す
* `print(..., flush=True)` を使用する
* エラー時は対象年とスタックトレースを記録する

作成する実行スクリプト:

```text
scripts/build_full_runner_dataset.py
```

実行例:

```bash
python scripts/build_full_runner_dataset.py --years 2016
python scripts/build_full_runner_dataset.py --resume
python scripts/build_full_runner_dataset.py --force
```

## 実行順序

最初に2016年だけ処理してください。

```bash
python scripts/build_full_runner_dataset.py --years 2016
```

以下を確認してください。

* 年別Parquetが正常に作成された
* entry_id重複がない
* `NL_RA` と `NL_O1` が結合できた
* 単勝払戻が結合できた
* 複勝払戻が複数頭分結合できた
* 特殊ケースが別出力された
* resume機能が動作する

2016年で問題がなければ、2016〜2026年を全件処理してください。

実行開始後は放置せず、定期的にログを確認してください。

エラーや長時間無出力が発生した場合は、原因を調査して修正し、完了済み年を再処理せずに再開してください。

## 今回行わないこと

* CatBoost学習
* LightGBM学習
* Transformer学習
* ランキング学習
* Optuna
* バックテスト
* 購入条件の最適化

## 最後に報告すること

1. 作成・変更ファイル一覧
2. 年別処理行数
3. 年別処理時間
4. 全出走馬行数
5. 全レース数
6. 結合成功率
7. 単勝・複勝払戻件数
8. targetと実払戻の不一致件数
9. 特殊ケースの内容
10. 欠損率が高い列
11. 直接使用可能な特徴量
12. 過去走加工に使える列
13. 未来情報リーク列
14. 処理が停止した場合は停止年・原因・再開方法
