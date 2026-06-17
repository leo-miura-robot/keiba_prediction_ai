# Codex Task: Phase 6A Champion Platt Artifact Certification and Revalidation v1

## 0. 背景

Phase 6Aのcalibrator保存状況を調査した結果、以下が確認された。

```text
ROLLING_10Y + PLATT_SCALING
coef: 1.0162527329694642
intercept: 0.016713944459665484
clip_min: 1e-6
clip_max: 0.999999
input_space: logit_probability_raw
fit_rows: 477650
fit_period: 2016-2025
```

認証元:

```text
outputs/place_market_offset_safe_calibration_phase6a_v1/calibrator_fit_provenance.csv
config/place_market_offset_safe_calibration_phase6a_v1.yaml
scripts/run_place_market_offset_safe_calibration_phase6a_v1.py
```

一方、ROLLING_15Y + ISOTONICは以下が欠落しており、旧認証artifactを完全復元できない。

```text
x_thresholds: NOT_FOUND
y_thresholds: NOT_FOUND
threshold_count: NOT_FOUND
```

したがって本タスクでは、完全な認証済みparameterが残っている
ChampionのROLLING_10Y + PLATT_SCALINGだけを正式artifact化し、
2026-06-13 / 2026-06-14を再検証する。

15Y IsotonicはBLOCKEDのまま維持し、再fitしない。

---

# 1. 目的

以下を行う。

```text
10Y Platt認証値をimmutable artifact化
→ source hashとprovenanceを固定
→ validation時はread-only load
→ fit/refitを完全禁止
→ 2026-06-13/14をChampion正式構成で再評価
```

正式Champion:

```text
ROLLING_10Y + PLATT_SCALING
```

15Y Shadow:

```text
BLOCKED_MISSING_ISOTONIC_THRESHOLDS
```

---

# 2. 絶対条件

禁止:

```text
CatBoost再学習
Platt再fit
Isotonic再fit
OOFからparameter再生成
2026-06-13/14をparameter調整に使用
Champion変更
EV閾値変更
ROI最適化
15Y Isotonicの推測復元
rawへの黙ったfallback
commit/push
```

許可:

```text
既存認証値のimmutable artifact化
hash計算
read-only loader実装
既存モデルへの軽量推論
py_compile
pytest
artifact audit
```

---

# 3. 10Y Platt正式artifact

保存先:

```text
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json
```

内容:

```json
{
  "artifact_version": "phase6a_v1",
  "strategy": "ROLLING_10Y",
  "calibrator_type": "PLATT_SCALING",
  "input_space": "logit_probability_raw",
  "output_space": "probability",
  "coef": 1.0162527329694642,
  "intercept": 0.016713944459665484,
  "clip_min": 0.000001,
  "clip_max": 0.999999,
  "fit_rows": 477650,
  "fit_period_start": 2016,
  "fit_period_end": 2025,
  "created_from_existing_certified_parameters": true,
  "refit_performed": false
}
```

加えて以下を保存する。

```text
source_artifacts
source_sha256
artifact_sha256
created_at
certification_status
```

source候補:

```text
outputs/place_market_offset_safe_calibration_phase6a_v1/calibrator_fit_provenance.csv
config/place_market_offset_safe_calibration_phase6a_v1.yaml
scripts/run_place_market_offset_safe_calibration_phase6a_v1.py
```

---

# 4. 適用式の確認

既存Phase 6A実装を読み、入力空間と式を確定する。

期待候補:

```text
p_clipped = clip(probability_raw, 1e-6, 1 - 1e-6)
x = log(p_clipped / (1 - p_clipped))
z = coef * x + intercept
p_calibrated = sigmoid(z)
```

ただし既存実装と完全一致させること。

推測実装は禁止。

既存実装と一致しない場合はBLOCKED。

---

# 5. Loader

候補:

```text
src/calibration/official_calibrator_loader.py
```

必要API:

```python
artifact = load_official_platt_calibrator(path)

p_calibrated = apply_official_platt_calibrator(
    artifact,
    probability_raw=probability_raw,
)
```

必須チェック:

```text
strategy == ROLLING_10Y
calibrator_type == PLATT_SCALING
input_space == logit_probability_raw
hash一致
coef存在
intercept存在
clip値存在
NaN / infなし
出力0〜1
```

---

# 6. 既存validation修正

対象:

```text
scripts/validate_latest_model_on_jrvltsql_db.py
```

変更:

```text
10Y Plattはofficial artifactをread-only load
fit / fit_transform / refitを禁止
artifact pathとhashを出力へ記録
15Y Isotonicは正式calibrated評価から除外
15YはBLOCKED理由を明示
```

15Yをraw-only参考として残す場合は、正式Shadow calibrated結果と混同しない。

ラベル例:

```text
ROLLING_15Y_RAW_DIAGNOSTIC_ONLY
```

---

# 7. 再検証対象

DB:

```text
C:\Users\leole\jrvltsql\data\quickstart_20260608_20260617_20260617_100814\keiba.db
```

対象日:

```text
2026-06-13
2026-06-14
```

履歴DB:

```text
D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

履歴cutoff:

```text
2026-06-13予測:
2026-06-12まで

2026-06-14予測:
2026-06-13まで
```

オッズ:

```text
NL_O1
FINAL_ODDS_RETROSPECTIVE
```

---

# 8. 比較対象

10Yについて同一行で保存する。

```text
probability_market
probability_raw
probability_official_platt
probability_invalid_refit_platt
```

`probability_invalid_refit_platt`は過去結果との比較用のみ。

正式評価へ混ぜない。

---

# 9. 指標

10Yについて:

```text
Logloss
Brier
ECE
mean predicted probability
actual positive rate
calibration gap
calibration slope
calibration intercept
race-wise Spearman
```

比較:

```text
raw - market
official Platt - raw
official Platt - market
official Platt - invalid refit Platt
```

ROIは補助のみ。

```text
usable_for_roi_judgement = false
```

72レースかつbet数極小のため、
ROIでモデル変更しない。

---

# 10. 出力先

```text
outputs/latest_model_validation_on_jrvltsql_20260608_official_10y_platt_v1/
```

必須成果物:

```text
validation_report.md
predictions.parquet
metrics_market_raw_official_platt.csv
comparison_with_invalid_refit.csv
calibration_diagnostics.csv
roi_ev_ge_1_auxiliary.csv
official_calibrator_manifest_snapshot.json
artifact_audit.json
run_manifest.json
```

---

# 11. 15Yの扱い

明示的に以下を記録する。

```text
strategy: ROLLING_15Y
calibrator_type: ISOTONIC
status: BLOCKED_MISSING_ISOTONIC_THRESHOLDS
official_calibrated_evaluation_performed: false
refit_performed: false
```

15Yの旧Isotonicを推測して作らない。

---

# 12. 必須テスト

1. 10Y artifactをread-only load
2. coef一致
3. intercept一致
4. clip値一致
5. input_space一致
6. hash一致
7. strategy不一致拒否
8. type不一致拒否
9. NaN / inf拒否
10. 出力0〜1
11. fit呼び出しなし
12. fit_transform呼び出しなし
13. OOF再生成なし
14. raw fallbackなし
15. 15Y Isotonic再fitなし
16. 15Y正式calibrated評価なし
17. market/raw/official Platt同一行比較
18. invalid refit版を正式指標へ混ぜない
19. ROI判断不可
20. Champion変更なし
21. commit/pushなし

fit系メソッドをmonkeypatchし、
呼び出されたらテスト失敗させる。

---

# 13. 最終判定

候補:

```text
OFFICIAL_10Y_PLATT_VALIDATION_PASSED
BLOCKED_10Y_ARTIFACT_MISMATCH
BLOCKED_10Y_INPUT_SPACE_MISMATCH
BLOCKED_REFIT_DETECTED
```

追加boolean:

```text
official_10y_platt_loaded
refit_performed
usable_for_probability_diagnostic
usable_for_model_limit_judgement
usable_for_roi_judgement
shadow_15y_official_calibration_available
```

期待:

```text
official_10y_platt_loaded = true
refit_performed = false
usable_for_probability_diagnostic = true
usable_for_model_limit_judgement = false
usable_for_roi_judgement = false
shadow_15y_official_calibration_available = false
```

---

# 14. Codex側実行範囲

実施:

```text
10Y artifact作成
source/artifact hash計算
loader実装
validation修正
py_compile
pytest
artifact audit
2026-06-13/14軽量再検証
report作成
```

長時間学習は行わない。

---

# 15. 最終報告

簡潔に以下を報告する。

1. artifact path
2. source hash / artifact hash
3. coef / intercept
4. input_space
5. 適用式
6. refitなしの証拠
7. market-only指標
8. raw C1R0指標
9. official Platt指標
10. invalid refit版との差
11. official Plattがrawを改善したか
12. モデル限界判断に使えるか
13. ROI判断に使えるか
14. 15Y status
15. final status
16. py_compile
17. pytest
18. artifact checks / failed
19. 作成・変更ファイル
20. git status --short

commit/pushは行わない。
