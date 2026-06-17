# Codex Task: Phase 6C Official Market Baseline Artifact Recovery v1

## 0. 背景

Phase 6Cのraw pre-race一気通貫パイプラインは、以下で停止した。

```text
BLOCKED_MARKET_ARTIFACT
```

理由:

```text
学習時のmarket baselineを構成する
StandardScaler / LogisticRegression artifactが保存先不明または未保存
```

現行C1R0の構造:

```text
market_logit
+ CatBoost residual_raw
= final_logit

probability_raw = sigmoid(final_logit)
```

保存済み:

```text
Official CatBoost:
models/place_market_offset_champion_challenger_phase5c_v1/
ROLLING_10Y/validation_2026/model.cbm

CatBoost SHA256:
4c6f1b9e236391bd84b9d75a14f7ea8ea3fe44761737bb645b8f21d74ed38256

Official Platt:
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json

Platt SHA256:
ffee1efc19c38f3a76a1efa93488153429e8463bc65f693b331617274e208e98
```

不足している可能性があるもの:

```text
market feature names
market feature order
market feature transforms
missing-value rules
clip rules
StandardScaler mean_
StandardScaler scale_
StandardScaler var_
LogisticRegression coef_
LogisticRegression intercept_
classes_
solver / penalty等のprovenance
```

本タスクでは、2026 forward Championに対応するmarket baselineの
既存parameter・serialized artifactを探索し、
再fitなしでofficial artifactとして復元できるか監査する。

---

# 1. 目的

```text
既存market artifact / parameterを探索
→ 2026 Champion foldとの対応を確認
→ feature orderと変換規則を確定
→ 既存値だけでimmutable artifact化
→ 保存済みmarket_logitとの再現性を検証
→ raw-to-Phase6C bridgeへ接続
```

最終目標:

```text
raw pre-race CSV
→ market_logit
→ CatBoost residual
→ official Platt
→ EV
→ Phase 6C登録
```

---

# 2. 絶対条件

禁止:

```text
StandardScalerの再fit
LogisticRegressionの再fit
CatBoost再学習
Platt / Isotonic再fit
OOFや学習データからmarket parameterを再推定
market feature定義変更
Champion変更
EV閾値変更
保存値の推測
raw probabilityへのfallback
commit/push
```

許可:

```text
既存artifactの探索
既存serialized objectのread-only load
既存CSV / JSON / parquetからparameter抽出
既存parameterのimmutable artifact化
hash計算
保存済みmarket_logitとの再現性比較
py_compile
pytest
artifact audit
fixture smoke
```

重要:

```text
既存値を保存形式へ移すことは可
データから係数を再計算することは不可
```

---

# 3. 対象となるmarket model

対象は2026 forward Championに対応するfold。

```text
strategy:
ROLLING_10Y

validation year:
2026

expected training period:
2016-2025
```

実際のtraining periodは既存provenanceから確認し、
推測で固定しない。

CatBoost modelとの対応を必ず確認する。

```text
models/place_market_offset_champion_challenger_phase5c_v1/
ROLLING_10Y/validation_2026/model.cbm
```

別年・別strategyのmarket modelを流用しない。

---

# 4. 探索対象

以下を再帰検索する。

```text
models/
outputs/
config/
scripts/
src/
docs/
```

検索語:

```text
market_logit
p_market
market_model
market baseline
StandardScaler
LogisticRegression
scaler
coef_
intercept_
mean_
scale_
var_
n_features_in_
market_features
feature_order
validation_2026
ROLLING_10Y
```

対象拡張子:

```text
.joblib
.pkl
.pickle
.json
.yaml
.yml
.csv
.parquet
.npz
.npy
.txt
.md
```

---

# 5. 最優先で探す値

## 5.1 Market feature contract

```text
feature_names
feature_order
feature_count
numeric transforms
log transforms
inverse transforms
race-relative transforms
missing-value rules
clip rules
dtype
```

候補特徴:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
SyussoTosu
place_rank_limit

market_rank
tan_rank
fuku_odds_width
log_tan_odds
log_fuku_low
log_fuku_high
fuku_low_inverse
fuku_mid_inverse
fuku_low_to_race_min
fuku_low_to_race_mean
```

実際の正式featureは既存コード・provenanceに従う。

## 5.2 StandardScaler

```text
mean_
scale_
var_
n_features_in_
feature_names_in_
with_mean
with_std
```

## 5.3 LogisticRegression

```text
coef_
intercept_
classes_
n_features_in_
feature_names_in_
solver
penalty
C
max_iter
class_weight
fit_intercept
multi_class
```

運用再現に不要な学習設定はmetadata扱いでよいが、
係数と入力契約は必須。

---

# 6. 探索結果の分類

次のいずれかへ分類する。

```text
SERIALIZED_ARTIFACT_FOUND
FULL_PARAMETERS_FOUND
PARTIAL_PARAMETERS_FOUND
PARAMETERS_NOT_FOUND
MULTIPLE_CONFLICTING_ARTIFACTS
```

判定:

### SERIALIZED_ARTIFACT_FOUND

```text
scalerとLogisticRegressionをread-only load可能
2026 ROLLING_10Y foldとの対応が確認可能
```

### FULL_PARAMETERS_FOUND

```text
serialized objectはないが、
scaler・LogisticRegression・feature orderの全値が保存済み
```

### PARTIAL_PARAMETERS_FOUND

例:

```text
coefはあるがscaler mean/scaleがない
feature orderがない
interceptがない
2026 foldとの対応が不明
```

この場合は復元しない。

---

# 7. Official market artifact仕様

完全な既存値が見つかった場合のみ作成する。

保存先:

```text
outputs/place_market_offset_official_market_phase5c_v1/
```

候補:

```text
rolling_10y_validation_2026_market_scaler.json
rolling_10y_validation_2026_market_logistic.json
rolling_10y_validation_2026_market_contract.json
official_market_manifest.json
certification_report.md
```

joblibを利用する場合も、
人間が監査できるJSON metadataを必ず保存する。

共通metadata:

```text
artifact_version
strategy
validation_year
training_period_start
training_period_end
market_feature_names
market_feature_order
market_feature_count
source_artifact_paths
source_sha256
materialized_sha256
created_from_existing_parameters
refit_performed
parameter_generation_performed
```

必須:

```text
created_from_existing_parameters = true
refit_performed = false
parameter_generation_performed = false
```

---

# 8. Read-only loader

候補:

```text
src/market/official_market_baseline_loader.py
```

API例:

```python
artifact = load_official_market_baseline(
    artifact_root,
    expected_strategy="ROLLING_10Y",
    expected_validation_year=2026,
)

market_logit = apply_official_market_baseline(
    artifact,
    market_feature_frame,
)
```

必須チェック:

```text
strategy一致
validation_year一致
feature count一致
feature names一致
feature order一致
hash一致
parameter completeness
NaN / inf拒否
出力行数一致
```

勝手な列補完・並べ替え・0埋めは禁止。
必要な並べ替えは正式contractに従い、監査記録を残す。

---

# 9. 再現性認証

既存の保存済み予測成果物から、
同じ行のmarket inputとmarket_logitを探す。

候補:

```text
Phase 5C predictions
Phase 6A OOF predictions
Phase 6B prediction inputs
2026 validation predictions
latest model validation outputs
```

再現テスト:

```text
official market artifactでmarket_logitを再計算
vs
既存保存済みmarket_logit
```

出力:

```text
rows_compared
mean_absolute_error
max_absolute_error
p99_absolute_error
allclose_at_1e-12
allclose_at_1e-9
```

認証基準:

```text
PASS:
max_absolute_error <= 1e-12

CONDITIONAL_PASS:
max_absolute_error <= 1e-9
かつ差の理由を説明可能

FAIL:
max_absolute_error > 1e-9
```

比較対象行・feature inputが保存されていない場合:

```text
BLOCKED_NO_MARKET_LOGIT_REFERENCE
```

---

# 10. Conflicting artifact対応

複数候補が見つかった場合、
以下で対応関係を判定する。

```text
strategy
validation_year
training period
feature order
source run id
model directory
prediction parity
```

最良一致を勝手に選ばない。

一意に確定できなければ:

```text
MULTIPLE_CONFLICTING_ARTIFACTS
```

で停止する。

---

# 11. Raw bridgeへの接続

market artifactが認証PASSした場合のみ、
以下へ接続する。

```text
scripts/run_phase6c_raw_to_official_champion.py
scripts/run_phase6c_raw_to_official_champion.ps1
```

処理:

```text
raw input audit
→ history feature generation
→ 79特徴生成
→ official market feature生成
→ official market_logit
→ CatBoost residual
→ probability_raw
→ official Platt
→ EV / tier
→ Phase 6C immutable登録
```

market artifact認証前に接続しない。

---

# 12. Fixture smoke

認証PASS時のみ実施する。

確認:

```text
raw fixture読込
market feature生成
official scaler適用
official LogisticRegression適用
market_logit生成
79特徴parity
CatBoost推論
official Platt
EV / tier
Phase 6C登録
duplicate拒否
immutable確認
```

fixture:

```text
fixture = true
forward実績reportから除外
```

---

# 13. Fail-closed条件

```text
market artifact不存在
scaler parameter不足
LogisticRegression parameter不足
feature names不足
feature order不足
2026 fold対応不明
strategy不一致
training period不一致
hash不一致
複数候補の競合
market_logit再現性不一致
fit/refit検出
parameter再生成検出
NaN / inf
raw fallback検出
```

---

# 14. 出力先

```text
outputs/place_market_offset_official_market_phase5c_v1/
```

必須成果物:

```text
market_artifact_search_report.json
market_parameter_inventory.json
market_feature_contract.json
market_reproduction_comparison.csv
official_market_manifest.json
artifact_audit.json
certification_report.md
```

認証PASS時:

```text
rolling_10y_validation_2026_market_scaler.json
rolling_10y_validation_2026_market_logistic.json
```

BLOCKED時:

```text
blocked_market_artifact_recovery.json
```

---

# 15. 追加・変更ファイル候補

```text
scripts/audit_and_recover_official_market_baseline_phase5c_v1.py
src/market/official_market_baseline_loader.py
tests/test_official_market_baseline_loader.py
tests/test_official_market_baseline_reproduction.py
docs/official_market_baseline_recovery_phase5c_v1.md
```

認証PASS時のみ:

```text
scripts/run_phase6c_raw_to_official_champion.py
config/place_market_offset_forward_paper_phase6c_v2.yaml
```

---

# 16. 必須テスト

1. fit呼び出しなし
2. fit_transform呼び出しなし
3. LogisticRegression.fit呼び出しなし
4. StandardScaler.fit呼び出しなし
5. parameter再生成なし
6. strategy一致
7. validation_year一致
8. feature count一致
9. feature names一致
10. feature order一致
11. scaler mean/scale完全性
12. logistic coef/intercept完全性
13. hash一致
14. hash不一致拒否
15. parameter不足拒否
16. conflicting artifact拒否
17. NaN / inf拒否
18. market_logit再現性比較
19. 1e-12判定
20. 1e-9判定
21. raw fallbackなし
22. CatBoost再学習なし
23. Platt再fitなし
24. Champion変更なし
25. commit/pushなし

---

# 17. 最終判定

候補:

```text
OFFICIAL_MARKET_ARTIFACT_RECOVERED
OFFICIAL_MARKET_ARTIFACT_RECOVERED_CONDITIONAL
BLOCKED_MISSING_MARKET_PARAMETERS
BLOCKED_MISSING_MARKET_FEATURE_CONTRACT
BLOCKED_NO_MARKET_LOGIT_REFERENCE
BLOCKED_MARKET_LOGIT_REPRODUCTION_MISMATCH
MULTIPLE_CONFLICTING_ARTIFACTS
BLOCKED_REFIT_DETECTED
MULTIPLE_BLOCKERS
```

boolean:

```text
serialized_artifact_found
full_parameters_found
official_market_artifact_created
refit_performed
parameter_generation_performed
market_logit_reproduction_passed
raw_bridge_connected
fixture_smoke_passed
ready_for_real_forward_prediction
```

---

# 18. Codex側実行範囲

実施:

```text
既存artifact探索
parameter inventory作成
feature contract特定
2026 fold対応確認
hash計算
完全値があればartifact化
market_logit再現性比較
loader実装
py_compile
pytest
artifact audit
認証PASS時のみraw bridge fixture smoke
report作成
```

再fitは禁止。

不足時は推測せずBLOCKEDで終了する。

---

# 19. 最終報告

簡潔に以下を報告する。

1. search status
2. serialized artifactの有無
3. scaler parameterの有無
4. LogisticRegression parameterの有無
5. market feature count
6. market feature names / orderの保存有無
7. training period
8. 2026 Champion foldとの対応
9. source paths / hashes
10. official artifact path / hash
11. market_logit比較行数
12. mean / max absolute error
13. 1e-12 / 1e-9 parity
14. refitなしの証拠
15. parameter再生成なしの証拠
16. raw bridge接続結果
17. fixture smoke
18. ready_for_real_forward_prediction
19. final status
20. py_compile
21. pytest
22. artifact checks / failed
23. 作成・変更ファイル
24. git status --short

commit/pushは行わない。
