# Codex Phase 5B Corrected Legacy Certification Task v1
## Certify the safe baseline and run the short 2024 seven-strategy smoke

## 0. 背景

Phase 5B LEGACY parity監査で、以下が確定した。

### Target差

Phase 5B configが誤って:

```text
target_column = target_place
```

を使用していた。

旧BASEは:

```text
target_column = target_place_paid
```

を使用していた。

不一致97行はすべて:

```text
SyussoTosu = 5～7
place_rank_limit = 2
3着
target_place = 1
target_place_paid = 0
fuku_pay = 0
```

であり、複勝払戻と整合するcanonical targetは:

```text
target_place_paid
```

である。

### Market baseline

targetを`target_place_paid`へ修正後:

```text
market training rows old/new = 383401 / 383401
training key old_only/new_only = 0 / 0
positive rate old/new = exact match
market_logit p99 abs diff = 0.0
```

となった。

### 残る差

```text
target match = True
market_logit p99 abs diff = 0.0
probability_raw p99 abs diff = 0.06268139750902087
Logloss abs diff = 0.0001955070291937977
Brier abs diff = 0.00004195334546022722
```

残差CatBoostの学習経路が旧BASEと新runnerで異なる。

旧BASEの完全再現には、outer validationをeval_setへ渡すことや、
use_best_model / overfitting detector等の危険な経路をコピーする可能性がある。

これらは再導入しない。

---

# 1. 目的

1. 旧BASEと新runnerの残差学習経路差を完全に記録する
2. 安全な`CORRECTED_LEGACY_2016_V1`をPhase 5B比較基準として認証する
3. 旧BASEとの予測値完全一致をblocking条件から外す
4. 代わりに構造・データ・意味のparityをblocking条件とする
5. corrected baselineの再現性を短時間で確認する
6. 2024年の全7戦略smokeをCodex側で実行する
7. 2020～2024全fold本実行は行わない

---

# 2. 絶対条件

- canonical targetは`target_place_paid`
- tree count設定は300
- outer validationをeval_setへ渡さない
- `use_best_model=False`
- early stopping無効
- overfitting detector無効
- feature allowlist変更禁止
- market baseline定義変更禁止
- calibration禁止
- 2025/2026使用禁止
- 既存成果物上書き禁止
- DB接続禁止
- git add / commit / push / reset / clean禁止

旧BASEの危険な学習経路をコピーしてはいけない。

---

# 3. 旧BASEとの比較を2種類に分ける

## 3.1 Structural / semantic parity

以下はblocking条件とする。

```text
validation row keys exact match
training row keys exact match
target exact match
target column = target_place_paid
market training keys exact match
market target exact match
market_logit exact match
feature names exact match
feature order exact match
categorical feature list exact match
training feature values一致またはhash一致
validation feature values一致またはhash一致
CatBoost hyperparameter差が説明済み
```

## 3.2 Historical prediction parity

以下は診断値として保存するが、blocking条件にしない。

```text
old probability_raw vs corrected probability_raw
old Logloss vs corrected Logloss
old Brier vs corrected Brier
old ROI vs corrected ROI
```

理由:

旧BASEの学習経路にouter validation利用などの安全上の問題がある場合、
完全一致は望ましい目標ではない。

---

# 4. 残差学習経路差の完全監査

旧BASEと新runnerについて以下を並べる。

```text
training start/end
training row count
training race count
training key hash
validation key hash
target column
target positive rate

feature names
feature order
categorical feature indices/names
feature matrix hash
missing counts

iterations configured
actual tree_count_
learning_rate
depth
loss_function
eval_metric
random_seed
task_type
thread_count
bootstrap_type
random_strength
l2_leaf_reg
rsm
subsample
class_weights
sample_weight
has_time

eval_set used
eval_set year
use_best_model
best_iteration
od_type
od_wait
early_stopping_rounds
```

必須成果物:

```text
legacy_residual_training_path_diff.csv
legacy_catboost_parameter_diff.csv
legacy_feature_matrix_parity.csv
legacy_train_validation_key_parity.csv
```

判定:

旧・新の差が以下だけなら続行可能。

```text
eval_setの有無
use_best_model
best_iteration
od_type / od_wait
early stopping / overfitting detector
それらに伴うactual tree_count
```

上記以外の差がある場合は、2024全戦略smokeへ進まず停止する。

---

# 5. CORRECTED_LEGACY_2016_V1

以下を新しい安全な基準として定義する。

```text
name = CORRECTED_LEGACY_2016_V1
history start = 2016
training rows = 2016 ～ validation前年
target = target_place_paid
market model rows = residual model rows
market model = Pipeline(StandardScaler, LogisticRegression)
iterations = 300
outer eval_set = none
use_best_model = false
early stopping = none
overfitting detector = none
feature allowlist = official C1R0 feature set
probability = probability_raw
```

保存:

```text
corrected_legacy_reference.json
corrected_legacy_manifest.json
```

内容:

```text
source hashes
config hash
feature hash
train key hash
validation key hash
market model provenance
CatBoost params
target provenance
model path
prediction path
metrics
```

---

# 6. Corrected baselineの再現性

同一設定・同一データ・同一seedで、
2024 corrected LEGACYを2回短時間実行する。

異なる出力先を使用する。

比較:

```text
row keys
market_logit
probability_raw
Logloss
Brier
tree_count
```

CPU実行で決定的なら予測hash完全一致を期待する。

完全一致しない場合は以下を記録する。

```text
max abs diff
p99 abs diff
mean abs diff
metric differences
task_type
thread_count
```

許容値を勝手に設定・拡張せず、
差の原因を報告する。

成果物:

```text
corrected_legacy_repeatability.csv
```

repeatabilityに重大な問題がある場合は全戦略へ進まない。

---

# 7. Parity gateの更新

既存parity gateを以下へ変更する。

## Blocking

```text
key parity
target parity
market parity
feature parity
training row parity
validation row parity
safe CatBoost settings
corrected baseline repeatability
```

## Diagnostic only

```text
historical old probability difference
historical old metric difference
```

出力には必ず:

```text
reference_type = historical_old_base
comparison_type = diagnostic_non_blocking
```

または:

```text
reference_type = corrected_legacy
comparison_type = blocking
```

を入れる。

旧BASEとcorrected baselineを同じ「BASE」と表記しない。

---

# 8. 必須テスト

1. config target_columnが`target_place_paid`
2. `target_place`へ戻らない
3. 5～7頭立て3着・払戻0はtarget 0
4. market_logit exact parity
5. feature names/order exact parity
6. training key parity
7. validation key parity
8. outer validationをeval_setへ渡さない
9. use_best_model=False
10. early stopping無効
11. overfitting detector無効
12. iterations=300
13. corrected reference manifest必須
14. historical comparisonがnon-blocking
15. corrected comparisonがblocking
16. repeatability成果物作成
17. calibrationを使わない
18. StressROI不変条件
19. 既存成果物を上書きしない

---

# 9. Codexが実行する範囲

Codexが行う:

```text
残差学習経路差の監査
parity gate更新
corrected reference作成
py_compile
pytest
corrected LEGACY 2024 repeatability実行
2024全7戦略smoke
smoke監査
```

Codexが行わない:

```text
2020～2024全7戦略本実行
2016～2019補助評価
2025/2026診断
calibration
EV閾値探索
```

---

# 10. 2024全7戦略smoke

corrected baseline認証後のみ実行する。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2024 `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_smoke_2024_v3 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_smoke_2024_v3
```

監査:

```powershell
python scripts\audit_place_market_offset_year_strategy_phase5b_v2.py `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_smoke_2024_v3
```

smoke確認:

```text
7戦略完走
validation row数一致
target一致
market/residual window正しい
probability_raw NaN/infなし
TIME_DECAY mean weight=1
ESS妥当
StressROI不変条件
戦略別成果物分離
```

---

# 11. 長時間本実行

Codexは実行しない。

smoke通過後にユーザーがローカルで実行する。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2020,2021,2022,2023,2024 `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2
```

旧BASEとのhistorical prediction parityはblockingではないため、
旧`--parity-check`が完全一致を要求する場合は、
corrected reference用の明示的オプションへ変更する。

例:

```text
--reference-mode corrected
```

実際のCLI名は実装後のrunbookへ記載する。

---

# 12. 最終報告

簡潔に以下を報告する。

1. 旧・新residual training path差
2. 旧BASEのeval_set/use_best_model/actual tree count
3. corrected baselineの安全設定
4. feature matrix parity
5. train/validation key parity
6. corrected reference manifest
7. repeatability結果
8. py_compile結果
9. pytest結果
10. 2024全7戦略smoke結果
11. smoke監査結果
12. 全fold本実行へ進めるか
13. 正式な本実行コマンド
14. resumeコマンド
15. git status --short
16. git diff --stat

2020～2024本実行、commit/pushは行わない。
