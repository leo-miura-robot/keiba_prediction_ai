# Codex Task: Phase 6C Official Champion Platt Integration v1

## 0. 背景

Phase 6AのChampion用Platt calibratorは、以下の正式artifactとして固定済み。

```text
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json
```

正式値:

```text
strategy: ROLLING_10Y
calibrator_type: PLATT_SCALING
coef: 1.0162527329694642
intercept: 0.016713944459665484
clip_min: 1e-6
clip_max: 0.999999
input_space: logit_probability_raw
refit_performed: false
```

Phase 6C v2は、外部で作成した`pre_race_predictions.csv`を登録する構成であり、そのCSVがofficial calibratorを使ったかを保証できていない。

本タスクでは、Phase 6Cの予測入力生成からSQLite登録までを、official 10Y Platt artifactへ接続する。

---

## 1. 目的

```text
pre-race feature input
→ ROLLING_10Y CatBoost raw prediction
→ official Platt artifactをread-only load
→ calibrated probability
→ EV計算
→ 4 tier判定
→ Phase 6C SQLiteへimmutable登録
```

正式Champion:

```text
ROLLING_10Y + official PLATT_SCALING
```

15Y Isotonic:

```text
BLOCKED_MISSING_ISOTONIC_THRESHOLDS
```

---

## 2. 絶対条件

禁止:

```text
CatBoost再学習
Platt再fit
Isotonic再fit
OOFからparameter再生成
Champion変更
EV閾値変更
ROI最適化
raw probabilityへの黙ったfallback
15Y Isotonicの推測復元
実購入
commit/push
```

許可:

```text
既存CatBoost artifactのread-only load
official Platt artifactのread-only load
pre-race推論
EV計算
Phase 6C paper trade登録
py_compile
pytest
fixture smoke
artifact audit
```

---

## 3. Official artifact固定

使用artifact:

```text
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json
```

Phase 6C設定へ以下を追加する。

```yaml
champion:
  strategy: ROLLING_10Y
  calibration:
    type: PLATT_SCALING
    artifact_path: outputs/place_market_offset_official_calibrators_phase6a_v1/rolling_10y_platt_phase6a_v1.json
    required_artifact_sha256: "<actual artifact file sha256>"
    fail_closed: true
    allow_refit: false
    allow_raw_fallback: false
```

実際のartifact file SHA256を取得して固定する。

---

## 4. 予測入力生成

作成候補:

```text
scripts/prepare_place_forward_predictions_phase6c_v2.py
```

入力:

```text
race_date
pre-race feature CSV
official CatBoost model artifact
official Platt artifact
odds columns
```

出力:

```text
outputs/place_market_offset_forward_paper_phase6c_v2/input/
pre_race_predictions_YYYYMMDD.csv
```

最低限の列:

```text
prediction_run_id
race_id
entry_id
race_date
Umaban
strategy
model_artifact_path
model_artifact_sha256
calibrator_artifact_path
calibrator_artifact_sha256
calibrator_type
probability_market
probability_raw
probability_calibrated
fuku_odds_low
expected_value
odds_snapshot_type
odds_observed_at
prediction_created_at
retrospective_only
```

可能なら追加:

```text
horse_name
JyoCD
RaceNum
market_logit
residual_raw
```

---

## 5. 確率計算

Raw:

```text
probability_raw = sigmoid(market_logit + residual_raw)
```

Official calibrated:

```text
src/calibration/official_calibrator_loader.py
```

を必ず使う。係数を別実装へ複製しない。

EV:

```text
expected_value = probability_calibrated * fuku_odds_low
```

Phase 6B/6Cの既存定義と完全一致させる。

---

## 6. Tier判定

```text
CORE:      EV >= 1.00
MARGIN:    EV >= 1.05
HIGH:      EV >= 1.10
VERY_HIGH: EV >= 1.15
```

包含関係:

```text
VERY_HIGH ⊆ HIGH ⊆ MARGIN ⊆ CORE
```

COREのみofficial candidate。他3つはshadow monitoring。

---

## 7. SQLite provenance

必要なら破壊的でないmigrationを追加する。

`prediction_runs`へ最低限:

```text
champion_strategy
model_artifact_path
model_artifact_sha256
calibrator_artifact_path
calibrator_artifact_sha256
calibrator_type
calibrator_input_space
calibrator_refit_performed
ev_definition
odds_snapshot_type
retrospective_only
```

`predictions`へ最低限:

```text
probability_market
probability_raw
probability_calibrated
expected_value
```

必須:

```text
calibrator_refit_performed = false
```

それ以外は登録拒否。

---

## 8. Fail-closed条件

以下では停止する。

```text
official artifact不存在
artifact hash不一致
strategy不一致
calibrator type不一致
input_space不一致
refit_performed != false
model artifact不存在
model hash不一致
required feature不足
feature order不一致
NaN / inf
probability範囲外
odds欠損
odds <= 0
odds snapshot type不明
duplicate prediction
prediction timestampがrace start後
raw fallback検出
```

過去の最終オッズ利用時:

```text
odds_snapshot_type = FINAL_ODDS
retrospective_only = true
```

---

## 9. コマンド

入力CSV生成:

```powershell
python scripts\prepare_place_forward_predictions_phase6c_v2.py `
  --race-date 2026-06-20 `
  --pre-race-feature-csv inputs\forward\pre_race_20260620.csv `
  --output-csv outputs\place_market_offset_forward_paper_phase6c_v2\input\pre_race_predictions_20260620.csv
```

Phase 6C登録:

```powershell
python scripts\run_place_market_offset_forward_paper_phase6c_v2.py predict `
  --race-date 2026-06-20 `
  --input-csv outputs\place_market_offset_forward_paper_phase6c_v2\input\pre_race_predictions_20260620.csv `
  --output-root outputs\place_market_offset_forward_paper_phase6c_v2
```

一気通貫wrapper:

```text
scripts/run_forward_predict_official_champion_phase6c_v2.ps1
```

例:

```powershell
.\scripts\run_forward_predict_official_champion_phase6c_v2.ps1 `
  -RaceDate 2026-06-20 `
  -PreRaceFeatureCsv "inputs\forward\pre_race_20260620.csv" `
  -OutputRoot "outputs\place_market_offset_forward_paper_phase6c_v2"
```

内部:

```text
artifact audit
→ model inference
→ official Platt
→ EV
→ tier
→ immutable registration
```

---

## 10. 対象ファイル

```text
config/place_market_offset_forward_paper_phase6c_v2.yaml
scripts/prepare_place_forward_predictions_phase6c_v2.py
scripts/run_forward_predict_official_champion_phase6c_v2.ps1
scripts/run_place_market_offset_forward_paper_phase6c_v2.py
scripts/audit_place_market_offset_forward_paper_phase6c_v2.py
src/calibration/official_calibrator_loader.py
tests/test_phase6c_official_champion_integration.py
docs/place_market_offset_phase6c_official_champion_integration_v1.md
```

既存Phase 6C fixtureやSQLite schemaを壊さない。

---

## 11. Run manifest

各runで保存:

```text
prediction_run_id
race_date
created_at
strategy
model_artifact_path
model_artifact_sha256
calibrator_artifact_path
calibrator_artifact_sha256
calibrator_type
calibrator_input_space
calibrator_refit_performed
ev_definition
tier_thresholds
odds_snapshot_type
retrospective_only
row_count
core_count
margin_count
high_count
very_high_count
```

保存先例:

```text
outputs/place_market_offset_forward_paper_phase6c_v2/manifests/
prediction_run_YYYYMMDD_<id>.json
```

---

## 12. 監査追加

既存監査へ追加:

```text
official model artifact存在
official model hash一致
official calibrator artifact存在
official calibrator hash一致
strategy一致
type一致
input_space一致
refit_performed=false
raw fallbackなし
expected_value再計算一致
tier包含関係
prediction immutable
duplicateなし
timestamp整合
```

---

## 13. fixture smoke

実未来入力がなければfixtureで実施。

fixtureは:

```text
fixture = true
```

として保存し、forward実績reportから除外する。

確認:

```text
10Y CatBoost raw prediction生成
official Platt適用
expected_value生成
4 tier生成
SQLite登録
duplicate拒否
hash不一致拒否
fit/refitなし
report除外
```

---

## 14. 必須テスト

1. official Platt artifact read-only load
2. artifact hash一致
3. hash不一致拒否
4. strategy不一致拒否
5. type不一致拒否
6. input_space不一致拒否
7. refit_performed=false
8. fit呼び出しなし
9. fit_transform呼び出しなし
10. raw fallbackなし
11. CatBoost再学習なし
12. feature allowlist一致
13. feature order一致
14. probability_raw範囲
15. probability_calibrated範囲
16. EV再計算一致
17. CORE threshold一致
18. 4 tier包含関係
19. duplicate予測拒否
20. prediction immutable
21. timestamp violation拒否
22. FINAL_ODDSはretrospective_only=true
23. fixture report除外
24. 15Y calibrated登録なし
25. Champion変更なし
26. commit/pushなし

---

## 15. 最終判定

候補:

```text
PHASE6C_OFFICIAL_CHAMPION_INTEGRATION_PASSED
BLOCKED_MODEL_ARTIFACT
BLOCKED_CALIBRATOR_ARTIFACT
BLOCKED_ARTIFACT_HASH_MISMATCH
BLOCKED_FEATURE_SCHEMA
BLOCKED_ODDS_INPUT
BLOCKED_REFIT_DETECTED
MULTIPLE_BLOCKERS
```

boolean:

```text
official_champion_model_loaded
official_platt_loaded
refit_performed
raw_fallback_used
phase6c_registration_performed
fixture_only
ready_for_real_forward_prediction
```

---

## 16. Codex側実行範囲

実施:

```text
既存Phase 6C調査
official Platt接続
入力CSV生成処理
run manifest追加
SQLite provenance追加
wrapper作成
py_compile
pytest
fixture smoke
artifact audit
runbook更新
```

安全な実未来pre-race入力がなければ実レース登録は行わずfixture smokeまで。

---

## 17. 最終報告

1. official model artifact path / hash
2. official Platt artifact path / hash
3. loader利用箇所
4. fit/refitなしの証拠
5. raw fallbackなしの証拠
6. probability_raw生成確認
7. probability_calibrated生成確認
8. EV定義
9. tier件数
10. SQLite provenance列
11. run manifest
12. duplicate / immutable監査
13. 15Y status
14. fixture smoke
15. 実未来予測の実施有無
16. ready_for_real_forward_prediction
17. final status
18. py_compile
19. pytest
20. artifact checks / failed
21. 作成・変更ファイル
22. git status --short

commit/pushは行わない。
