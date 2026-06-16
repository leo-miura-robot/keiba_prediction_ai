# Task: 単勝・複勝オッズ欠損の追加監査 V1.1

## 目的

V1監査で、以下は確認済みである。

- `NL_O1`行なしは少数
- 大半は`NL_O1`行あり・`TanOdds`/`FukuOdds`がNULL
- `NL_SE.Odds`は約99%有効
- `NL_SE.Odds`と`NL_O1.TanOdds`は重複範囲で100%一致
- JOINキー不一致、取消・除外限定、型変換・倍率違いは主因ではない

一方で、以下は未確定である。

- `TanFlag` / `FukuFlag` / `DataKubun`とNULLの関係
- `MakeDate`や取得時点との関係
- NULLがレース単位か馬単位か
- O1のどのレコード状態でオッズが入るのか
- 外部取込処理・取得設定・更新順序
- 空レコード上書きの可能性
- `NL_SE.Odds`を過去市場比較へ安全に使えるか

今回の目的は、**O1オッズが大量にNULLになる上流原因をさらに絞り込み、修正方針を確定できる状態にすること**である。

今回も原因監査のみとし、DB、V2.1.1、CatBoostモデル、既存予測、ROI/EV処理は変更しない。

---

# 1. 最初に確認するもの

以下を確認する。

```text
tasks/audit_odds_missingness_v1_task.md
docs/odds_missingness_audit_v1.md
outputs/odds_missingness_audit_v1/

scripts/audit_odds_missingness.py
src/audit/odds_schema.py
src/audit/odds_missingness.py
src/audit/odds_source_comparison.py
src/audit/odds_import_audit.py

tests/test_odds_*.py
```

最初に実行する。

```bash
git status
git diff
python -m pytest -q
```

元DBはread-onlyで開く。

---

# 2. V1.1を別監査として追加する

推奨構成:

```text
scripts/audit_odds_missingness_v1_1.py

src/audit/odds_flag_audit.py
src/audit/odds_timing_audit.py
src/audit/odds_race_pattern_audit.py
src/audit/odds_external_import_audit.py

tests/test_odds_flag_audit.py
tests/test_odds_timing_audit.py
tests/test_odds_race_pattern_audit.py
tests/test_odds_external_import_audit.py

docs/odds_missingness_audit_v1_1.md

outputs/odds_missingness_audit_v1_1/
```

既存V1監査結果を上書きしない。

---

# 3. `TanFlag`・`FukuFlag`・`DataKubun`監査

## 3.1 実schema確認

最初に`PRAGMA table_info(NL_O1)`等で、実在する列と型を確認する。

以下が存在する場合は必ず監査する。

```text
TanFlag
FukuFlag
WakurenFlag
DataKubun
MakeDate
RecordSpec
HassoTime
TorokuTosu
SyussoTosu
```

存在しない列を推測で参照しない。

## 3.2 クロス集計

最低限、次の組み合わせを集計する。

```text
DataKubun
TanFlag
FukuFlag
TanOdds status
FukuOdds status
```

statusは次の4分類。

```text
valid
null
zero_or_invalid
missing_row
```

出力項目:

- rows
- valid_tan_rows
- null_tan_rows
- valid_fuku_rows
- null_fuku_rows
- tan_valid_rate
- fuku_valid_rate
- race_count
- complete_race_count

出力:

```text
o1_flag_datakubun_cross_summary.csv
o1_flag_datakubun_by_year.csv
```

## 3.3 条件分離

以下を確認する。

- 特定`TanFlag`でのみ`TanOdds`が有効か
- 特定`FukuFlag`でのみ複勝オッズが有効か
- `DataKubun=5`の中でもflagで有効/NULLが分かれるか
- `DataKubun=9`が常にNULLか
- flag値と発売有無・確定状態が対応しているか
- flagがNULLでもオッズ有効な例があるか
- flagが有効に見えてもオッズNULLな例があるか

値の意味は、公式仕様・既存コード・データ分布の3点で確認する。

公式仕様が確認できない場合は、値の意味を断定しない。

---

# 4. `MakeDate`・取得時点監査

## 4.1 日時形式を確定する

`MakeDate`、更新日時、取得日時に相当する列について、実際の形式を確認する。

- 文字列
- 整数
- 日付
- 日時
- 年月日のみ
- レース後更新日

不正値やNULLも集計する。

## 4.2 レース日との差

`race_date`と`MakeDate`の差を計算する。

```text
make_date_minus_race_date_days
```

最低限、次へ分類する。

```text
before_race
same_day
day_after
2_to_7_days_after
more_than_7_days_after
unknown
```

各分類で単勝・複勝オッズ有効率を出す。

出力:

```text
o1_make_date_timing_summary.csv
o1_make_date_timing_by_year.csv
```

## 4.3 仮説検証

次の仮説を検証する。

```text
H1: レース前または当日初期レコードはNULL
H2: レース後更新レコードだけオッズが入る
H3: 月曜成績更新後のみSE.Oddsが埋まる
H4: O1は一部の取得時点しか保存されていない
```

各仮説について、支持件数・反例件数を出す。

---

# 5. NULLがレース単位か馬単位かを調べる

各race_idについて、次を計算する。

```text
runner_count
tan_valid_count
tan_null_count
fuku_valid_count
fuku_null_count
```

レースを分類する。

```text
all_valid
all_null
partially_valid
missing_o1_rows
```

年別・競馬場別・月別・DataKubun別に集計する。

出力:

```text
o1_race_level_missingness_summary.csv
o1_race_level_missingness_by_year.csv
o1_partial_missing_race_samples.csv
```

重点確認:

- 大半が`all_null`なら取得単位の問題が有力
- `partially_valid`が多ければ馬単位取込・更新・join問題の可能性
- 同一レースで単勝と複勝の有効パターンが完全一致するか
- O1行なしがレース全体か特定馬だけか

---

# 6. レコード状態と頭数の整合性

以下を比較する。

```text
SE.TorokuTosu
SE.SyussoTosu
O1.TorokuTosu
O1.SyussoTosu
race内SE行数
race内O1行数
```

確認すること:

- 登録頭数と出走頭数が不一致のレース
- O1行数が出走頭数より少ないレース
- O1行数は揃っているがオッズが全NULLのレース
- 取消・除外馬を含む場合の期待行数
- `Umaban`空欄・0・特殊値の有無

出力:

```text
o1_runner_count_consistency.csv
o1_runner_count_anomaly_samples.csv
```

---

# 7. `RecordSpec`・`HassoTime`・その他状態列の監査

実在する場合、以下を集計する。

```text
RecordSpec
HassoTime
DataKubun
MakeDate
TanFlag
FukuFlag
```

目的:

- 特定`RecordSpec`だけ有効オッズを持つか
- 発走時刻未設定レコードでNULLが多いか
- 発走後のレコードで有効率が高いか
- 同じレースで複数状態が保存されているか

出力:

```text
o1_record_state_summary.csv
o1_record_state_examples.csv
```

---

# 8. 外部取込処理の追加監査

## 8.1 リポジトリ内検索のノイズ除去

検索対象から以下を除外する。

```text
tasks/
tests/
docs/
outputs/
src/audit/
scripts/audit_*
```

実際の取込処理候補だけを検索する。

検索語:

```text
NL_O1
TanOdds
FukuOddsLow
INSERT OR REPLACE
REPLACE INTO
ON CONFLICT
executemany
JVOpen
JVGets
JVRead
JVLink
DataLab
```

出力:

```text
external_import_code_candidates.csv
```

各候補に以下を付ける。

```text
file
line
matched_text
is_actual_import_code
reason
```

## 8.2 外部コード・設定の探索

リポジトリ外パスが設定ファイルやREADMEに記録されている場合のみ確認する。

勝手に広範囲のディスク検索をしない。

確認対象候補:

- DB作成バッチ
- JV-Link取込ツール
- DataLab設定
- 定期実行スクリプト
- Windowsタスク
- SQL dump生成コード

見つからない場合は、外部取込処理が未提供であると明記する。

## 8.3 上書き仮説

実取込コードが見つかった場合、次を確認する。

- 主キー
- UPSERT方式
- NULLで既存有効値を上書きするか
- 更新条件
- MakeDate比較
- DataKubun比較
- 後着レコード優先か
- transaction順序

実コードが見つからない場合、上書きをconfirmedにしない。

---

# 9. SE.Oddsの提供時点監査

## 9.1 DB内で確認可能なこと

`NL_SE`の`MakeDate`、`DataKubun`等があれば、`SE.Odds`有効行と作成日時の関係を調べる。

## 9.2 用途別判定

以下を別々に評価する。

```text
A. 過去レースの最終市場ベンチマーク
B. 過去バックテストの最終オッズ参考
C. 学習時のmarket_aware入力
D. 発走前リアルタイム推論入力
```

判定候補:

```text
safe
conditionally_safe
unsafe
unknown
```

根拠を必ず付ける。

`SE.Odds`が最終結果更新後に埋まる場合、A/Bには使えてもC/Dにはそのまま使わない。

---

# 10. 公式仕様の確認

可能であれば、JRA-VAN Data Lab.または対象データ仕様の公式資料を確認する。

確認対象:

- O1レコードの意味
- `DataKubun`
- `TanFlag`
- `FukuFlag`
- `MakeDate`
- O1の更新タイミング
- SE.Oddsの更新タイミング
- 蓄積系と速報系の違い

ルール:

- 公式資料を優先
- 非公式記事だけで断定しない
- 引用元・資料名・確認日をdocsへ記録
- 公式資料へアクセスできない場合は、その旨を明記する
- 長い転載はしない

出力:

```text
official_spec_findings.csv
```

---

# 11. 原因判定を更新する

V1の原因候補をV1.1の結果で更新する。

形式:

```text
cause
previous_confidence
updated_confidence
supporting_evidence
counter_evidence
remaining_question
```

最低限:

- O1値未収録
- O1正式列採用
- DataKubun依存
- TanFlag/FukuFlag依存
- 取得時点依存
- レース単位取得不足
- 馬単位欠損
- JOINキー不一致
- 取消・除外
- 取込上書き
- 外部取得設定不足
- 型変換・倍率問題

出力:

```text
root_cause_assessment_v1_1.csv
```

---

# 12. 修正案を3段階で提示する

原因に応じて、以下の3案を比較する。

## 案A: 外部O1取込を修正・再取得

評価:

- 根本解決度
- 単勝coverage
- 複勝coverage
- 発走前利用可能性
- 再取得コスト
- 再生成範囲

## 案B: 単勝の過去市場比較だけSE.Oddsを使う

評価:

- 用途限定
- リークリスク
- coverage
- model inputへ使わない条件
- ROI評価への影響

## 案C: `COALESCE(O1, SE)`を使用

評価:

- coverage
- 値の時点混在
- 再現性
- market_aware入力としての危険
- 比較用に限定できるか

推奨案を1つに絞る必要はない。用途別に推奨してよい。

出力:

```text
recommended_fix_plan_v1_1.csv
```

---

# 13. 今回は禁止すること

- DB書込
- O1再取込
- SE fallback適用
- `COALESCE`適用
- base dataset再生成
- V2.1.1再生成
- CatBoost再学習
- 既存予測更新
- ROI/EV計算
- 買い目生成
- キャリブレーション適用

---

# 14. 必須テスト

最低限以下を追加する。

1. `TanFlag`別集計
2. `FukuFlag`別集計
3. `DataKubun`とのクロス集計
4. flag列が存在しない場合の明確な処理
5. MakeDate形式判定
6. race dateとの差分分類
7. all_valid/all_null/partial分類
8. race内行数整合
9. 単勝・複勝パターン一致判定
10. 検索ノイズ除外
11. 監査コード自身をimport候補に含めない
12. read-only DB動作
13. 同じ監査を2回行って同じ結果
14. 原因確度更新
15. 公式仕様未取得時に断定しない

---

# 15. 実行手順

## 構文・テスト

```bash
python -m py_compile scripts/audit_odds_missingness_v1_1.py
python -m pytest -q
```

## schema / flag確認

```bash
python scripts/audit_odds_missingness_v1_1.py \
  --db-path <DB_PATH> \
  --output-dir outputs/odds_missingness_audit_v1_1 \
  --schema-and-flags-only
```

## 完全監査

```bash
python scripts/audit_odds_missingness_v1_1.py \
  --db-path <DB_PATH> \
  --output-dir outputs/odds_missingness_audit_v1_1
```

---

# 16. 出力

```text
outputs/odds_missingness_audit_v1_1/
  audit_manifest.json

  o1_flag_datakubun_cross_summary.csv
  o1_flag_datakubun_by_year.csv

  o1_make_date_timing_summary.csv
  o1_make_date_timing_by_year.csv

  o1_race_level_missingness_summary.csv
  o1_race_level_missingness_by_year.csv
  o1_partial_missing_race_samples.csv

  o1_runner_count_consistency.csv
  o1_runner_count_anomaly_samples.csv

  o1_record_state_summary.csv
  o1_record_state_examples.csv

  external_import_code_candidates.csv
  odds_overwrite_risk_report_v1_1.csv

  official_spec_findings.csv
  se_odds_usage_assessment.csv

  root_cause_assessment_v1_1.csv
  recommended_fix_plan_v1_1.csv

logs/audit_odds_missingness_v1_1.log

docs/odds_missingness_audit_v1_1.md
```

---

# 17. 完了条件

- `TanFlag`・`FukuFlag`・`DataKubun`のクロス監査完了
- MakeDateとレース日の関係を監査
- NULLがレース単位か馬単位か判定
- runner count整合性を確認
- RecordSpec/HassoTime等を監査
- 実取込コード候補だけを抽出
- 上書き仮説の確度を更新
- SE.Oddsを用途別に評価
- 公式仕様の確認結果を記録
- 原因候補の確度を更新
- 用途別修正案を提示
- DB・V2.1.1・モデル・予測を変更していない
- ROI/EV計算を行っていない

1つでも満たさない場合、V1.1監査完了とは判定しない。

---

# 18. 最終報告

1. git status / git diff
2. 追加・変更ファイル
3. 使用DBとread-only確認
4. O1実schema
5. TanFlagの値一覧と有効率
6. FukuFlagの値一覧と有効率
7. DataKubun×flag×odds状態
8. MakeDateの形式
9. MakeDateとレース日の関係
10. O1有効オッズの取得時点傾向
11. all_valid / all_null / partialレース数
12. 単勝と複勝の欠損パターン一致率
13. runner count不整合件数
14. RecordSpec別有効率
15. HassoTime別有効率
16. 外部取込コード候補
17. 取込上書きの確度
18. O1取得設定不足の確度
19. DataKubun依存の確度
20. flag依存の確度
21. 取得時点依存の確度
22. SE.Oddsの提供時点評価
23. SE.Oddsを過去市場比較に使えるか
24. SE.Oddsをmarket_aware入力に使えるか
25. 公式仕様確認結果
26. 更新後の最有力原因
27. 原因候補ごとの確度
28. 推奨修正案A/B/C
29. 修正時の再生成範囲
30. モデル再学習が必要になる条件
31. 未解決事項

完了後は原因監査の報告だけを行い、データ修正には進まず停止する。
