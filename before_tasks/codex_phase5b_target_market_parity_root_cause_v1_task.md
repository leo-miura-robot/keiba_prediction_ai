# Codex Phase 5B Parity Root-Cause Audit v1
## Resolve target and market baseline mismatch before any multi-strategy run

## 0. 現在の状況

Phase 5Bの`LEGACY_2016 / validation 2024` parity smokeで、
キー型不一致は修正され、全46,752行が1対1で対応した。

確認済み:

```text
old_rows = 46752
new_rows = 46752
both_count = 46752
old_only_count = 0
new_only_count = 0
row_key_match_rate = 1.0
```

しかしparityは失敗した。

```text
target mismatch = 97 rows
market_logit p99 abs diff = 0.8662293819202378
probability_raw p99 abs diff = 0.12531130253969672
Logloss abs diff = 0.0009602458506162703
Brier abs diff = 0.0003333121807478634
```

この差は許容できる数値誤差ではない。

本タスクでは、97件のtarget不一致とmarket baseline差の根本原因を特定し、
正しいLEGACY基準を確定してからparity smokeを再実行する。

全7戦略の学習、および2020～2024本実行はまだ禁止する。

---

# 1. 最重要原則

旧BASEを無条件に「正解」と仮定してはいけない。

次の2つを分離する。

```text
compatibility:
旧BASE成果物と新runnerが同じ処理を再現するか

semantic correctness:
複勝target、市場baseline、学習期間の定義が競馬ルールと
プロジェクト仕様に照らして正しいか
```

旧BASE側のtargetまたはmarket baselineが誤っている場合、
誤りを新runnerへコピーしてparityを通してはいけない。

最終的に以下のどちらかを明確に判定する。

```text
A. 旧BASEが正しく、新runnerを旧BASEへ合わせる

B. 旧BASE成果物に誤りまたは古い仕様があり、
   corrected LEGACY_2016を新しい比較基準として固定する
```

---

# 2. 絶対条件

- 全7戦略実行禁止
- 2020～2024本実行禁止
- CatBoost hyperparameter変更禁止
- tree count 300維持
- feature allowlist変更禁止
- parity許容差を広げない
- calibration追加禁止
- DB更新禁止
- Parquet上書き禁止
- 既存成果物削除・上書き禁止
- git add / commit / push / reset / clean禁止

許可:

- コード・成果物の読み取り
- 97行の詳細監査
- market modelの短時間再現
- LEGACY_2016 / 2024の短時間再学習
- py_compile / pytest
- 新規監査成果物の作成

---

# 3. Stage A: target不一致97行の完全監査

## 3.1 必須出力

新規出力先:

```text
outputs/place_market_offset_year_strategy_phase5b_v2_parity_root_cause_v1/
```

作成:

```text
target_mismatch_97_rows.csv
target_mismatch_reason_summary.csv
target_definition_provenance.json
```

`target_mismatch_97_rows.csv`には最低限以下を含める。

```text
entry_id
race_id
race_date
Year
old_actual_place
new_actual_place

finish_position source
finish_position parsed
field size source
field size parsed
starter count
place_rank_limit
target calculated from canonical rule

cancel flag
exclude flag
scratch flag
disqualification flag
did_not_finish flag
dead_heat related columns if present

old source prediction path
new source prediction path
old target source column
new target source column
reason_category
```

実在しない列は推測で作らず、利用可能な元列名を記録する。

## 3.2 重点的に調べる候補

97行を以下の観点で分類する。

```text
7頭以下など複勝対象が2頭のレース
出走取消・除外後に頭数が変わったレース
登録頭数と実出走頭数の違い
着順文字列・同着・失格・中止・除外
finish position欠損
place_rank_limit定義差
actual_placeが既に0/1なのに再変換
旧成果物と新成果物のtarget生成時点差
```

これらは候補であり、コードとデータで証明する。

## 3.3 canonical target

以下をコードから特定する。

```text
target生成関数
利用した着順列
利用した頭数列
取消・除外の扱い
複勝対象2頭/3頭のルール
欠損・失格・中止の扱い
```

canonical ruleは一箇所のhelperへ集約する。

例示だけで実装を決めない。
JRAルール、既存DB定義、既存targetコードの一致を確認する。

## 3.4 target decision

次を明記する。

```text
old target correct count
new target correct count
old incorrect count
new incorrect count
ambiguous count
```

ambiguousが1件でも残る場合は、parity再学習へ進まず停止する。

---

# 4. Stage B: market baseline差の根本原因

`market_logit p99 abs diff = 0.866`は大きすぎる。
target差の影響だけか、学習条件差もあるかを分離する。

## 4.1 旧・新market model provenance

以下を旧BASE・新runnerで並べる。

```text
model type
implementation file
function
training start/end date
training row count
training race count
target column and positive rate
input feature columns
feature order
odds source column
odds transformation
missing handling
clipping
normalization / scaling
solver
penalty
C
class_weight
max_iter
random_state
fit_intercept
coefficient
intercept
serialized model path
source hash
```

出力:

```text
market_model_provenance_old_vs_new.csv
market_model_input_parity.csv
market_model_coefficient_comparison.csv
```

## 4.2 同一入力での比較

同じ2024 validation rowsへ、以下を保存する。

```text
old market probability
new market probability
old market_logit
new market_logit
absolute difference
```

出力:

```text
market_logit_row_comparison_2024.csv
```

分布:

```text
mean
std
min
p1
p50
p95
p99
max
NaN count
inf count
clip count
```

## 4.3 差の分解

最低限、次を順に比較する。

```text
1. 旧target + 旧training rows + 旧market config
2. canonical target + 旧training rows + 旧market config
3. canonical target + 新training rows + 旧market config
4. canonical target + 新training rows + 新market config
```

可能な範囲でmarket modelだけを短時間fitし、
どの変更で`market_logit`差が発生するかを特定する。

CatBoost全戦略学習は不要。

## 4.4 training key parity

旧・新のmarket model training rowsについて:

```text
row count
race count
key intersection
old_only
new_only
target positive rate
odds distribution
```

を出す。

学習期間表記が同じでも行集合が違う可能性を確認する。

---

# 5. Stage C: residual model入力のparity

targetとmarket baselineを直した後、
CatBoost入力について旧・新を比較する。

必須:

```text
feature names exact match
feature order exact match
categorical feature list exact match
numeric dtype summary
missing count by feature
train row key match
validation row key match
sample weight use/no-use
seed
task_type
loss_function
iterations
depth
learning_rate
l2_leaf_reg
random_seed
```

出力:

```text
legacy_feature_parity.csv
legacy_train_key_parity.csv
legacy_catboost_config_parity.json
```

---

# 6. Corrected referenceの決定

## Case A: 旧BASEが正しい

新runnerを旧BASEのcanonical target・market baseline・行集合へ合わせる。

その後、LEGACY 2024 parity gateを再実行する。

## Case B: 旧BASEが誤りまたは旧仕様

旧BASEへの完全一致を目的にしない。

以下を新規に定義する。

```text
CORRECTED_LEGACY_2016_V1
```

条件:

```text
canonical target
正しいmarket baseline
outer validation leakageなし
iterations=300
eval_setなし
early stoppingなし
正式feature allowlist
probability_raw
```

旧BASEとの差を監査成果物として残し、
Phase 5Bの全年度戦略はこのcorrected LEGACYと同じ定義で比較する。

旧BASEとcorrected LEGACYを混同しない。

---

# 7. コード修正

修正は根本原因特定後に最小限だけ行う。

対象候補:

```text
scripts/run_place_market_offset_year_strategy_phase5b_v2.py
tests/test_place_market_offset_year_strategy_phase5b_v2.py
```

target helperを共有化する必要がある場合は、
新規helperをPhase 5Bファイル内または新規utilityとして追加する。

既存本番コードを広範囲に変更しない。

---

# 8. 必須テスト

## Target tests

1. 0/1 actual_placeを再度`<=3`変換しない
2. 複勝対象2頭の境界
3. 複勝対象3頭の境界
4. 取消・除外後の実出走頭数
5. 失格・中止・欠損着順
6. target helperが単一のcanonical ruleを使う
7. 97件の原因分類が再現可能

## Market tests

8. market training window一致
9. market training key一致
10. target positive rate記録
11. input column order一致
12. coefficient/intercept保存
13. NaN/infなし
14. same config + same rows + same targetで同じmarket_logit

## Parity tests

15. target exact match
16. market_logit許容差
17. feature list/order exact
18. train key parity
19. validation key parity
20. probability_raw許容差
21. Logloss/Brier許容差

既存安全テストも維持する。

---

# 9. Codexが実行する範囲

Codexが行う:

```text
静的監査
97行の原因分類
market model短時間再現
コード最小修正
py_compile
pytest
LEGACY_2016 / 2024 parity smoke再実行
```

Codexは以下を行わない。

```text
2024全7戦略
2020～2024全7戦略
2025/2026診断
EV閾値探索
calibration
```

---

# 10. Parity再実行

新しい出力先を使う。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016 `
  --years 2024 `
  --parity-check `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_parity_2024_v3 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_parity_2024_v3
```

Case Bの場合は、
corrected LEGACY referenceとのparity gateを実行するオプションを明示的に追加する。

---

# 11. 完了条件

以下がすべて満たされた場合のみPhase 5B全戦略へ進める。

```text
97 target mismatchesの原因が全件説明済み
canonical targetが確定
market_logit差の原因が説明済み
market model training rows/configが記録済み
feature parity確認済み
LEGACYまたはcorrected LEGACY parity gate通過
py_compile通過
pytest通過
```

---

# 12. 最終報告

簡潔に以下を報告する。

1. 97件の原因分類
2. canonical target定義
3. 旧targetと新targetのどちらが正しかったか
4. market_logit差の原因
5. 旧・新market modelのtraining row差
6. 旧・新market config差
7. corrected referenceが必要か
8. 修正ファイル
9. 追加テスト
10. py_compile結果
11. pytest結果
12. parity再実行結果
13. target match
14. market_logit p99差
15. probability_raw p99差
16. Logloss/Brier差
17. Phase 5B全戦略へ進めるか
18. git status --short
19. git diff --stat

全戦略実行、commit/pushは行わない。
