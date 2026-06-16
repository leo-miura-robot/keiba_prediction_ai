# Task: 単勝・複勝オッズ欠損の原因監査 V1

## 目的

`keiba_prediction_ai` で、単勝オッズが完全にそろうレースが全体の約4分の1しか残らない原因を、元DB・取込処理・特徴量生成処理までさかのぼって特定する。

今回の目的は**原因監査と修正案の提示**である。  
原因が確定する前に、元DB、V2.1.1特徴量Parquet、CatBoostモデル、既存予測、既存バックテスト用成果物を変更しない。

Phase 1の最終目標は引き続き次のとおり。

```text
単勝回収率 >= 90%
複勝回収率 >= 90%
```

ただし、オッズの取得・定義が誤っている状態ではROIを計算しない。

---

# 1. 監査方針

以下を区別して調査する。

1. `NL_O1`の行そのものが存在しない
2. `NL_O1`行は存在するが`TanOdds`がNULL
3. 空文字、0、負値、特殊なsentinel値
4. 文字列→数値変換失敗
5. JOINキー不一致
6. 重複行・主キー衝突
7. 後から来た空レコードによる上書き
8. 取消・除外・競走中止馬に限った欠損
9. `DataKubun`やflag、提供時点による違い
10. `NL_SE.Odds`と`NL_O1.TanOdds`の用途・値・更新時点の違い
11. 単勝と複勝で欠損原因が異なる可能性
12. オッズの単位・スケーリング違い

推測で結論を出さず、実データ件数とサンプルキーで根拠を示す。

---

# 2. 最初に確認するファイル

現在の`main`を確認する。

```text
scripts/build_full_runner_dataset.py
scripts/build_model_features_v2_1_1.py
scripts/train_catboost_baseline_v1_0_2.py
scripts/analyze_catboost_baseline_v1_0_2.py

src/models/catboost_market_comparison.py

outputs/full_dataset_summary.csv
outputs/column_quality_summary.csv
outputs/model_training/catboost_baseline_v1_0_2/
```

さらに、SQLiteまたは元データ取込に関係するコードをリポジトリ全体から検索する。

検索語例:

```text
NL_SE
NL_O1
TanOdds
FukuOdds
INSERT OR REPLACE
REPLACE INTO
ON CONFLICT
executemany
MakeDate
DataKubun
TanFlag
FukuFlag
```

DB schema、DDL、主キー、index、取込順序を確認する。

最初に実行:

```bash
git status
git diff
python -m pytest -q
```

---

# 3. 新規監査スクリプト

推奨:

```text
scripts/audit_odds_missingness.py

src/audit/odds_schema.py
src/audit/odds_lineage.py
src/audit/odds_missingness.py
src/audit/odds_source_comparison.py
src/audit/odds_import_audit.py

tests/test_odds_missingness.py
tests/test_odds_source_comparison.py
tests/test_odds_import_audit.py

docs/odds_missingness_audit_v1.md
```

CLI例:

```bash
python scripts/audit_odds_missingness.py \
  --db-path <SQLite DB path> \
  --output-dir outputs/odds_missingness_audit_v1
```

元DBはread-onlyで開く。

SQLiteの場合の推奨:

```python
sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
```

監査中にDBへCREATE、UPDATE、DELETE、INSERT、VACUUMを行わない。

---

# 4. データリネージを確定する

`tan_odds`、`tan_ninki`、`fuku_odds_low`、`fuku_odds_high`が、どの元テーブル・元列からどの変換を経て作られているかを明示する。

出力例:

```text
final_column
source_table
source_column
join_keys
conversion
fallback
filter
null_handling
```

最低限確認する列:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
TanVote
FukuVote
```

`NL_SE.Odds`、`NL_SE.Ninki`が取得されている場合、最終列に採用されているか、単に補助列として残っているかを確認する。

出力:

```text
outputs/odds_missingness_audit_v1/odds_data_lineage.csv
```

---

# 5. Schemaと値表現を確認する

対象テーブル:

```text
NL_SE
NL_O1
```

実際のschemaから列名を取得し、存在しない列名を推測で使用しない。

確認項目:

- column name
- declared type
- nullableか
- primary key
- unique index
- default value
- row count
- duplicate key count

オッズ値について確認する。

- 整数か小数か文字列か
- 10倍、100倍などのスケーリングがあるか
- `"0000"`、空文字、スペース、`0`、`9999`などの特殊値
- Python/Pandas/Polars変換時の扱い
- `NULL`と0を同じ欠損として扱っていないか

出力:

```text
odds_schema_summary.csv
odds_value_encoding_summary.csv
```

---

# 6. JOIN結果を4種類へ分類する

`NL_SE`を基準母集団として、`NL_O1`とのJOIN結果を次へ分類する。

```text
A: O1行なし
B: O1行あり・TanOdds有効
C: O1行あり・TanOdds NULL/空
D: O1行あり・TanOdds 0/無効/sentinel
```

年別に次を出す。

- SE rows
- O1 matched rows
- O1 missing rows
- valid TanOdds rows
- NULL TanOdds rows
- zero/invalid TanOdds rows
- coverage rate

同様に複勝オッズも分類する。

出力:

```text
odds_join_coverage_by_year.csv
odds_join_coverage_by_status.csv
```

JOINキーは実コードとschemaから確定する。少なくとも以下候補を確認する。

```text
Year
MonthDay
JyoCD
Kaiji
Nichiji
RaceNum
Umaban
```

型、ゼロ埋め、空白、文字列/整数差も検証する。

---

# 7. 欠損パターンを条件別に集計する

最低限、次の軸で単勝・複勝の有効率を集計する。

- Year
- Month
- JyoCD
- RaceNum
- DataKubun
- TanFlag
- FukuFlag
- TorokuTosu
- SyussoTosu
- runner status
- 取消
- 除外
- 競走中止
- 確定・未確定
- MakeDate
- データ取得日または更新日時
- weekday
- race grade
- 芝/ダート

実際に存在する列だけを使用する。

特定の`DataKubun`、flag、年度、レース種別に欠損が集中しているか確認する。

出力:

```text
tan_odds_missingness_by_dimension.csv
place_odds_missingness_by_dimension.csv
```

---

# 8. SE.OddsとO1.TanOddsを比較する

両方有効な行だけで比較する。

## 事前正規化

値の単位を確認し、必要なら次の候補を比較する。

```text
SE.Odds
SE.Odds / 10
SE.Odds / 100

O1.TanOdds
O1.TanOdds / 10
O1.TanOdds / 100
```

分布と既知の現実的なオッズ範囲から、正しいスケールを特定する。

## 比較指標

- compared rows
- exact match count/rate
- tolerance match count/rate
- mean absolute difference
- median absolute difference
- p95 difference
- p99 difference
- max difference
- ranking agreement by race
- favorite agreement rate
- year別一致率

同じレース内で人気順位も比較する。

```text
SE.Ninki
O1.TanNinki
オッズから再計算した順位
```

出力:

```text
se_o1_tan_odds_comparison_summary.csv
se_o1_tan_odds_comparison_by_year.csv
se_o1_tan_odds_mismatch_samples.csv
```

不一致サンプルにはレースキー、馬番、両値、flag、DataKubun、MakeDateを含める。

---

# 9. SE.Oddsの利用可能時点を確認する

`NL_SE.Odds`が最終確定単勝オッズとして使用可能かを、コード・schema・データ更新情報から確認する。

次を区別する。

```text
A. 過去レースの市場ベンチマーク・最終オッズ
B. バックテストの払戻評価
C. 発走前予測時のモデル入力
D. リアルタイム運用時の取得可能オッズ
```

`NL_SE.Odds`がレース後更新値の場合:

- 過去レースの最終市場ベンチマークには使える可能性
- 払戻評価の補助には使える可能性
- 発走前の利用可能時点を再現する入力としては、そのまま使えない可能性

を明記する。

「値が埋まっているから無条件にmarket_aware特徴量へ採用する」という結論にしない。

---

# 10. O1取込処理を監査する

リポジトリ内に取込コードがある場合、次を確認する。

- INSERT方式
- `INSERT OR REPLACE`
- `REPLACE INTO`
- `ON CONFLICT DO UPDATE`
- 主キー
- 更新順序
- 同一キーの複数レコード
- NULL/空値で既存有効値を上書きする可能性
- MakeDateや取得時刻を主キーへ含めているか
- transaction境界
- エラー時のrollback
- 型変換
- trim処理

特に次のケースを検証する。

```text
先に有効オッズを保存
↓
同一キーの後続空レコードを受信
↓
REPLACEで有効値が消える
```

取込コードがリポジトリにない場合は、断定せず「外部取込処理の確認が必要」と報告する。

## DB内で可能な監査

同一レース・同一馬について、履歴テーブルや元ログが存在する場合:

- 複数更新件数
- 最初の値
- 最後の値
- 有効→欠損への遷移

を調べる。

履歴が残っていない場合は、現DBだけでは上書きを証明できないことを明記する。

出力:

```text
odds_import_code_audit.csv
odds_overwrite_risk_report.csv
```

---

# 11. 欠損レースのサンプルを保存する

年・競馬場・欠損理由を分散させて、最低50レースを抽出する。

各レースで:

- 全出走馬
- SE.Odds
- SE.Ninki
- O1.TanOdds
- O1.TanNinki
- O1複勝オッズ
- DataKubun
- flags
- runner status
- JOIN成功有無

を保存する。

出力:

```text
odds_missing_race_samples.csv
```

個人情報や不要な巨大列は出力しない。

---

# 12. 単勝と複勝を別々に結論づける

## 単勝

次の候補を比較する。

```text
1. O1.TanOddsを継続
2. SE.Oddsへ切替
3. COALESCE(O1.TanOdds, SE.Odds)
4. DB再取込
```

各案について:

- coverage
- 値の意味
- 利用可能時点
- リークリスク
- 実運用再現性
- バックテスト用途
- market_aware入力用途

を評価する。

## 複勝

SEに代替列がない場合は、次を調査する。

- O1内の別列
- 別オッズテーブル
- 元データ再取得
- 払戻テーブルはROI計算だけに利用
- EV用の発走前複勝オッズ不足

複勝払戻額と複勝オッズを混同しない。

---

# 13. 原因の確度を分類する

最終報告では、原因候補ごとに次の形式で出す。

```text
原因候補
確度: confirmed / highly likely / possible / disproved
根拠
反証
追加確認事項
```

最低限評価する候補:

- O1の値自体が未収録
- SEではなくO1を正式列に選んだ
- JOINキー不一致
- 取消・除外に限定
- DataKubun/flag依存
- 取込上書き
- 型変換・スケール問題
- 取得期間不足

---

# 14. 修正はまだ適用しない

今回禁止:

- `tan_odds`をSEへ切り替える
- `COALESCE`を本番データへ適用
- V2.1.1を再生成
- CatBoostを再学習
- 既存予測を更新
- ROIを計算
- DBを再取込
- 元DBを書き換える

代わりに、原因確定後の修正案を設計資料へ記載する。

```text
recommended_fix
affected_files
required_rebuild_range
leakage_consideration
model_retraining_required
expected_coverage
```

---

# 15. 必須テスト

synthetic SQLite fixtureなどを使い、最低限以下を確認する。

1. O1行なしを分類できる
2. O1行ありNULLを分類できる
3. 0/空文字/sentinelを分類できる
4. 有効値を分類できる
5. JOINキー型違いを検出できる
6. 重複キーを検出できる
7. 単位スケール候補を比較できる
8. SE/O1一致率を計算できる
9. 条件別欠損率を計算できる
10. read-only DBで動作する
11. DBへの書込を行わない
12. 元テーブルが不足した場合に明確に停止する
13. 存在しない列名を推測で参照しない
14. 同じ監査を2回実行して出力が再現する

---

# 16. 実行手順

## 構文・テスト

```bash
python -m py_compile scripts/audit_odds_missingness.py
python -m pytest -q
```

## Schema確認のみ

```bash
python scripts/audit_odds_missingness.py \
  --db-path <DB_PATH> \
  --output-dir outputs/odds_missingness_audit_v1 \
  --schema-only
```

## 完全監査

```bash
python scripts/audit_odds_missingness.py \
  --db-path <DB_PATH> \
  --output-dir outputs/odds_missingness_audit_v1
```

長時間処理では進捗と処理時間をログへ出す。

---

# 17. 出力

```text
outputs/odds_missingness_audit_v1/
  audit_manifest.json
  odds_data_lineage.csv
  odds_schema_summary.csv
  odds_value_encoding_summary.csv
  odds_join_coverage_by_year.csv
  odds_join_coverage_by_status.csv
  tan_odds_missingness_by_dimension.csv
  place_odds_missingness_by_dimension.csv
  se_o1_tan_odds_comparison_summary.csv
  se_o1_tan_odds_comparison_by_year.csv
  se_o1_tan_odds_mismatch_samples.csv
  odds_import_code_audit.csv
  odds_overwrite_risk_report.csv
  odds_missing_race_samples.csv
  root_cause_assessment.csv
  recommended_fix_plan.csv

logs/audit_odds_missingness_v1.log

docs/odds_missingness_audit_v1.md
```

manifestへ保存:

- DB pathの匿名化した識別情報
- DB file size
- DB mtime
- DB SHA-256または現実的なfingerprint
- schema hash
- audit code hash
- Git SHA
- Git dirty状態
- 実行日時
- Python/pandas/SQLite version

---

# 18. 完了条件

- 元DBをread-onlyで監査
- 単勝欠損を「行なし」「NULL」「0/無効」へ分類
- JOINキー不一致率を算出
- SE/O1の値と人気順位を比較
- オッズ単位・スケールを確認
- DataKubun/flag別の欠損率を算出
- 取消・除外との関連を確認
- 取込上書きリスクを調査
- 単勝と複勝を別々に結論づける
- 原因候補へ確度を付ける
- 修正案と影響範囲を提示
- 元DB・V2.1.1・モデル・予測を変更していない
- ROI/EV計算を行っていない

1つでも満たさない場合、原因監査完了とは判定しない。

---

# 19. 最終報告

1. git status / git diff
2. 追加・変更ファイル
3. 使用したDBとschema
4. `tan_odds`・複勝オッズのデータリネージ
5. NL_SEとNL_O1の主キー・JOINキー
6. O1行なし件数
7. O1行ありTanOdds NULL件数
8. TanOdds 0/無効/sentinel件数
9. 年別単勝オッズ有効率
10. 年別複勝オッズ有効率
11. DataKubun別欠損率
12. flag別欠損率
13. 取消・除外との関連
14. SE.Oddsの有効率
15. SE.OddsとO1.TanOddsの単位
16. SE/O1一致率と差分
17. 人気順位一致率
18. 欠損レースの代表例
19. JOINキー不一致が原因か
20. 取込上書きが原因か
21. O1取得範囲不足が原因か
22. 単勝欠損の最有力原因
23. 複勝欠損の最有力原因
24. 原因候補ごとの確度
25. `SE.Odds`を過去市場比較へ使えるか
26. `SE.Odds`をモデル入力へ使えるか
27. 推奨修正案
28. 修正時に再生成が必要な成果物
29. モデル再学習が必要か
30. 未解決事項

完了後は原因監査の報告だけを行い、データ修正・再生成には進まない。
