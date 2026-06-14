# Task: 修正版O1 DBへの切替・基礎データ再生成・V2.1.1再構築

## 目的

修正済みの `jrvltsql` で再取得した新DBを、競馬予想AIリポジトリへ安全に接続し直す。

新DB:

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

このDBでは、2016-01-01〜2026-06-13のJRA中央競馬について、`NL_O1` の単勝・複勝確定オッズがほぼ完全に保存されている。

確認済み品質:

```text
SE runner rows: 505,881
valid TanOdds: 501,241
valid FukuOddsLow: 501,241
valid FukuOddsHigh: 501,241
coverage: 約99.08%

SE.Odds / O1.TanOdds:
501,241 / 501,241 完全一致
exact match rate: 100.0%

old_null_new_valid: 358,316
old_valid_new_null: 0

旧 all_null → 新 all_valid:
24,436 races

新レース単位:
all_valid: 34,341
partially_valid: 1,713
all_null: 58
missing_o1_rows: 157
```

今回の目的:

1. 新DBの存在・read-only接続・O1品質をAI側で再確認する
2. DB参照先を新DBへ安全に切り替える
3. `base_runner_dataset` を新DBから再生成する
4. V2.1.1相当の特徴量データを新DBから再生成する
5. 新旧データを比較し、O1改善が反映されたか確認する
6. 時系列リーク監査・再現性監査を再実行する
7. CatBoost再学習へ進める状態か判定する

今回はデータ再生成と品質確認までとし、以下には進まない。

```text
CatBoost再学習
確率キャリブレーション
EV計算
ROIバックテスト
買い目生成
Ability/Ranker導入
```

---

## 1. 重要方針

### 1.1 使用DB

新DB:

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

旧DB:

```text
D:\keiba\new_jra_2016-2026\keiba.db
```

旧DBは比較用read-onlyとして扱い、変更しない。

### 1.2 新DBもread-onlyで利用する

SQLite接続例:

```python
sqlite3.connect(
    "file:D:/keiba/new_jra_2016-2026_fixed/keiba.db?mode=ro",
    uri=True,
)
```

DBへ以下を行わない。

```text
INSERT
UPDATE
DELETE
REPLACE
VACUUM
CREATE TABLE
DROP TABLE
ALTER TABLE
```

### 1.3 旧成果物を上書きしない

既存の以下を直接上書きしない。

```text
outputs/base_runner_dataset/
outputs/model_features_v2_1_1/
outputs/model_training/
models/
```

新規成果物はversioned outputへ出す。

推奨:

```text
outputs/base_runner_dataset_o1_fixed/
outputs/model_features_v2_1_2/
```

既存V2.1.1を削除・変更しない。

---

## 2. 最初に行うこと

```bash
git status
git diff
git rev-parse HEAD
```

確認:

- 現在branch
- dirty状態
- DB pathを定義しているconfig・script・環境変数
- 旧DB hard-codeの有無
- 既存成果物
- 自動commitされていないこと

---

## 3. 新DB存在確認

確認対象:

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

確認項目:

- file exists
- file size
- last modified
- read-only接続成功
- `PRAGMA integrity_check`
- 主要テーブル存在
- `NL_SE`行数
- `NL_O1`行数
- DB fingerprint

主要テーブル:

```text
NL_RA
NL_SE
NL_HR
NL_H1
NL_H6
NL_O1
NL_O2
NL_O3
NL_O4
NL_O5
NL_O6
```

異常があれば停止する。

---

## 4. AI側でO1品質を再確認

新DBに対して以下を再集計する。

### 行単位

```text
SE runner rows
O1 matched rows
valid TanOdds rows
valid FukuOddsLow rows
valid FukuOddsHigh rows
coverage
```

### レース単位

```text
all_valid
all_null
partially_valid
missing_o1_rows
```

### SE/O1一致

両方有効な行で:

```text
compared rows
exact match count
exact match rate
max difference
```

期待値:

```text
coverage 約99.08%
SE/O1 exact match rate 100%
```

出力:

```text
outputs/o1_fixed_preflight/
  db_summary.json
  o1_coverage_summary.csv
  o1_race_completeness.csv
  se_o1_comparison.csv
```

---

## 5. DB pathの切替設計

DB pathをコードへ直書きしない。

既存configまたは環境変数方式を優先する。

例:

```yaml
database:
  path: "D:/keiba/new_jra_2016-2026_fixed/keiba.db"
  mode: "read_only"
```

または:

```text
KEIBA_DB_PATH=D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

新DB用configをversionedにする。

例:

```text
config/base_runner_dataset_o1_fixed.yaml
config/model_features_v2_1_2.yaml
```

manifestへ実際のDB pathとfingerprintを保存する。

---

## 6. base_runner_dataset再生成

既存の正式なbase dataset生成処理を確認し、新DBから再生成する。

必須要件:

- 新DBをread-onlyで読む
- 2016〜2026-06-13を対象
- `NL_SE`を基準とする
- `NL_O1`を正しいキーでLEFT JOINする
- `tan_odds = NL_O1.TanOdds`
- `tan_ninki = NL_O1.TanNinki`
- `fuku_odds_low = NL_O1.FukuOddsLow`
- `fuku_odds_high = NL_O1.FukuOddsHigh`
- `fuku_ninki = NL_O1.FukuNinki`
- `NL_SE.Odds/Ninki`は照合用列として保持してよい
- `COALESCE(O1, SE)`は適用しない
- 旧DB由来成果物を再利用しない
- DB fingerprint不一致ならresumeしない

推奨出力:

```text
outputs/base_runner_dataset_o1_fixed/
  base_runner_dataset.parquet
  summary.csv
  column_quality_summary.csv
  o1_quality_summary.csv
  manifest.json
```

---

## 7. base dataset品質確認

最低限:

```text
row count
race count
entry_id unique
race_id valid
year range
target columns
tan_odds coverage
fuku_odds_low coverage
fuku_odds_high coverage
tan_ninki coverage
fuku_ninki coverage
SE/O1一致率
```

期待runner rows:

```text
約505,881
```

旧データとの比較:

```text
old tan_odds valid rows
new tan_odds valid rows
old fuku valid rows
new fuku valid rows
old_null_new_valid
old_valid_new_null
```

出力:

```text
base_dataset_old_new_comparison.csv
```

---

## 8. V2.1.1相当の特徴量再生成

既存V2.1.1の設計を維持する。

新DB・新base dataset向けにV2.1.2として派生させる。

推奨:

```text
config/model_features_v2_1_2.yaml
scripts/build_model_features_v2_1_2.py
outputs/model_features_v2_1_2/
docs/model_features_v2_1_2_results.md
```

V2.1.2は、V2.1.1の特徴量ロジックを維持し、入力DBとO1 coverageだけを改善した版とする。

新特徴量を勝手に追加しない。

Ability/ANA/Rankerを導入しない。

---

## 9. 特徴量セット

V2.1.1と同じ3系統を維持する。

```text
market_free
market_history
market_aware
```

### market_free

現在レースの市場情報を使わない。

### market_history

過去レースの人気・市場履歴だけを使う。必ず過去レースのみ。

### market_aware

今回レースの確定オッズ・人気を使う。

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
```

Phase 1では、確定オッズを利用した理想条件バックテストとして扱う。

---

## 10. 複勝オッズ

下限・上限を別列で保持する。

将来のEV計算では安全側として原則:

```text
fuku_ev = predicted_place_probability × fuku_odds_low
```

を使う予定だが、今回はEV計算を実装しない。

---

## 11. 時系列リーク監査

V2.1.1で実施した監査を再実行する。

最低限:

```text
same race history references = 0
same day future references = 0
future race references = 0
cutoff violations = 0
```

成果物とdocsへ必ず:

```text
確定オッズを利用した理想条件モデル
発走前実運用モデルではない
```

と明記する。

---

## 12. split

固定splitを維持する。

```text
train: 2016-2023
validation: 2024
test: 2025
latest_holdout: 2026
```

YAMLを正本とし、hard-codeしない。

split hashを保存する。

今回はwalk-forward、race-count balancing、time decayを導入しない。

---

## 13. 再現性・resume

manifestへ保存する。

```text
source DB path
source DB fingerprint
source DB size
source DB mtime
base dataset hash
feature config hash
feature code hash
split hash
output row count
output column count
Git SHA
Git dirty
Python version
Polars/Pandas version
start/end time
```

DB fingerprintやconfigが変わった場合、旧成果物を再利用しない。

---

## 14. 新旧比較

旧V2.1.1と新V2.1.2を比較する。

```text
rows
races
columns
feature set columns
target counts
split counts
market column coverage
missingness
entry_id agreement
actual agreement
```

特に:

```text
tan_odds non-null
fuku_odds_low non-null
fuku_odds_high non-null
tan_ninki non-null
fuku_ninki non-null
```

の改善を示す。

出力:

```text
outputs/model_features_v2_1_2/
  old_new_dataset_comparison.csv
  market_feature_coverage_comparison.csv
```

---

## 15. テスト

最低限:

```text
DB path config test
read-only DB test
DB fingerprint test
O1 coverage preflight test
base dataset generation test
resume invalidation test
V2.1.2 feature generation test
market_free column exclusion test
market_history leakage test
market_aware required column test
split YAML test
old/new comparison test
```

既存テストも実行する。

```bash
python -m pytest -q
```

---

## 16. 禁止事項

- 旧DB変更
- 新DB変更
- `COALESCE(O1, SE)`適用
- V2.1.1既存成果物上書き
- CatBoost再学習
- calibration
- ROI/EV計算
- 買い目生成
- 資金配分
- Abilityモデル
- Ranker
- 自動commit
- 自動push

---

## 17. 完了条件

- 新DB存在とread-only接続確認
- O1 coverage約99%をAI側でも再確認
- SE/O1一致率100%を確認
- DB pathをconfig化
- 新DBからbase dataset再生成
- base datasetのO1 coverage改善確認
- V2.1.2新規生成
- V2.1.1ロジック維持
- 時系列リーク監査通過
- split維持
- 旧V2.1.1未変更
- CatBoost再学習へ進める状態を判定
- DB・モデル・予測・ROI未変更

---

## 18. 推奨出力

```text
config/base_runner_dataset_o1_fixed.yaml
config/model_features_v2_1_2.yaml

scripts/build_full_runner_dataset_o1_fixed.py
scripts/build_model_features_v2_1_2.py

outputs/o1_fixed_preflight/
outputs/base_runner_dataset_o1_fixed/
outputs/model_features_v2_1_2/

docs/o1_fixed_ai_data_migration.md
docs/model_features_v2_1_2_results.md
```

既存構造に自然に統合できる場合は調整してよいが、versioned outputを維持する。

---

## 19. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 新DB絶対パス
5. 新DB read-only確認
6. integrity check
7. DB fingerprint
8. NL_SE行数
9. NL_O1行数
10. AI側でのTanOdds coverage
11. AI側でのFukuOddsLow/High coverage
12. AI側でのSE/O1一致率
13. DB pathのconfig接続方法
14. 旧DB hard-code残存有無
15. base dataset出力先
16. base dataset rows/races/columns
17. base datasetの単勝・複勝coverage
18. 旧base datasetとの改善件数
19. V2.1.2出力先
20. V2.1.2 rows/races/columns
21. market_free列数
22. market_history列数
23. market_aware列数
24. split件数
25. split hash
26. leakage audit結果
27. same race/day/future違反件数
28. resume/fingerprint動作
29. old/new feature comparison
30. pytest結果
31. 旧V2.1.1未変更確認
32. DB未変更確認
33. CatBoost未実行確認
34. ROI/EV未実行確認
35. CatBoost再学習へ進めるか
36. 次に再学習すべきモデル
37. 未解決事項
38. 次の推奨手順

完了後はデータ再生成と品質確認の報告まで行い、CatBoost再学習には進まず停止する。
