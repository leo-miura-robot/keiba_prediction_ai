# Codex Task: Phase 6A Official Calibrator Artifact Fix v1

## 0. 背景

2026-06-13 / 2026-06-14の追加検証を監査した結果、判定は以下だった。

```text
CALIBRATION_ISSUE
```

履歴特徴量、feature schema、market入力、targetは大きく壊れていなかった。

一方、既存検証スクリプト:

```text
scripts/validate_latest_model_on_jrvltsql_db.py
```

が、Phase 6Aで認証済みのcalibrator artifactをロードせず、
OOF予測からcalibratorを再構成・再fitしていた。

今回のタスクでは、Phase 6Aの正式な認証結果を
再fitなしでimmutable artifactとして読み込み、
同じ検証を再実行できるようにする。

---

# 1. 目的

以下を実現する。

```text
Phase 6A認証成果物を特定
→ 正式calibratorパラメータをimmutable artifact化
→ hash / provenanceを固定
→ validation時はread-only loadのみ
→ fit / refitを完全禁止
→ 2026-06-13/14を正式artifactで再評価
```

正式構成:

```text
Champion:
ROLLING_10Y + PLATT_SCALING

Shadow:
ROLLING_15Y + ISOTONIC
```

Champion変更は禁止。

---

# 2. 絶対条件

禁止:

```text
CatBoost再学習
calibrator再fit
OOF予測からの再推定
2026-06-13/14を使ったparameter調整
calibrator type変更
Champion変更
EV閾値変更
ROI最適化
commit/push
```

許可:

```text
Phase 6A既存認証成果物の読み取り
認証済み係数・knots・thresholdsのartifact化
hash計算
metadata追加
既存モデルへの軽量再推論
py_compile
pytest
artifact audit
```

重要:

```text
「既存認証値を保存形式へ移す」ことは可
「データから係数を再計算する」ことは不可
```

---

# 3. 最初に行う調査

Phase 6A関連の保存済み成果物を探索する。

候補:

```text
outputs/
docs/
config/
scripts/
```

検索対象:

```text
PLATT_SCALING
ISOTONIC
temperature
calibration
calibrator
coefficient
intercept
threshold
knot
x_thresholds
y_thresholds
phase6a
certification
```

次を明示する。

```text
strategy
calibrator_type
source_artifact_path
source_artifact_format
source_hash
certification_period
selection_period
input_space
output_space
clip_rule
parameters_available
```

特に入力空間を必ず確認する。

候補:

```text
raw_probability
raw_logit
final_logit
```

推測は禁止。

---

# 4. Artifact化ルール

## 4.1 既存serialized artifactがある場合

そのartifactを正式artifactとして利用する。

必要ならversioned copyを作るが、
中身を再学習・再fitしない。

## 4.2 既存serialized artifactがない場合

Phase 6A認証成果物に以下が完全保存されている場合のみ、
そこからimmutable artifactを作る。

### Platt

最低限:

```text
coefficient / slope
intercept
input_space
clip_rule
```

### Isotonic

最低限:

```text
x_thresholds
y_thresholds
increasing
out_of_bounds rule
input_space
clip_rule
```

Phase 6Aの既存出力に完全な値がない場合は、
OOFから再fitせず次で停止する。

```text
BLOCKED_MISSING_CERTIFIED_CALIBRATOR_PARAMETERS
```

---

# 5. 正式artifact仕様

保存先:

```text
outputs/place_market_offset_official_calibrators_phase6a_v1/
```

推奨ファイル:

```text
rolling_10y_platt_phase6a_v1.json
rolling_15y_isotonic_phase6a_v1.json
official_calibrator_manifest.json
certification_report.md
```

必要ならread-only joblib/pickleも併用可。
ただしJSON metadataを必ず用意する。

共通metadata:

```text
artifact_version
strategy
calibrator_type
source_artifact_path
source_artifact_sha256
materialized_artifact_sha256
certification_status
certification_period
selection_metric
input_space
output_space
clip_min
clip_max
created_from_existing_certified_parameters
refit_performed
created_at
code_version_or_git_status
```

必須値:

```text
created_from_existing_certified_parameters = true
refit_performed = false
```

---

# 6. 適用関数

共通のread-only loader / apply関数を実装する。

候補:

```text
src/calibration/official_calibrator_loader.py
```

API例:

```python
artifact = load_official_calibrator(
    artifact_path,
    expected_strategy="ROLLING_10Y",
    expected_type="PLATT_SCALING",
)

p_calibrated = apply_official_calibrator(
    artifact,
    probability_raw=...,
    final_logit=...,
)
```

要件:

```text
strategy一致
type一致
input_space一致
hash一致
required parameter存在
NaN / inf拒否
出力0〜1
勝手なfallback禁止
```

artifactがない場合にrawへ黙ってfallbackしない。

---

# 7. 既存validationスクリプト修正

対象:

```text
scripts/validate_latest_model_on_jrvltsql_db.py
```

修正内容:

```text
OOFからcalibratorをfitする処理を削除または無効化
official artifact pathを明示的に受け取る
read-only loadのみ許可
artifact hashを出力へ記録
calibrator sourceを出力へ記録
```

CLI例:

```powershell
python scripts\validate_latest_model_on_jrvltsql_db.py `
  --db-path "C:\Users\leole\jrvltsql\data\quickstart_20260608_20260617_20260617_100814\keiba.db" `
  --calibrator-root outputs\place_market_offset_official_calibrators_phase6a_v1 `
  --output-root outputs\latest_model_validation_on_jrvltsql_20260608_official_calibrator_v1
```

必要なら既存引数へ追加する。

---

# 8. fail-closed条件

以下では即時BLOCKED。

```text
artifact不存在
hash不一致
strategy不一致
calibrator type不一致
input_space不明
required parameter不足
Phase 6A認証元が不明
fit/refit呼び出し検出
出力にNaN / inf
出力確率が0未満または1超
```

raw probabilityへ自動fallbackしない。

---

# 9. 再検証対象

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

今回も実forwardではなくretrospective診断である。

---

# 10. 比較する確率

同一行について必ず3種類を保存する。

```text
probability_market
probability_raw
probability_official_calibrated
```

既存refit版も比較用に残す場合は、
明確に無効ラベルを付ける。

```text
probability_invalid_refit_calibrated
```

正式指標へ混ぜない。

---

# 11. 指標

strategy × probability_typeごとに:

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
official calibrated - raw
official calibrated - market
official calibrated - invalid refit calibrated
```

ROIは補助表示のみ。

```text
usable_for_roi_judgement = false
```

対象が72レースかつbet数極小のため、
ROIでモデル変更しない。

---

# 12. 出力先

```text
outputs/latest_model_validation_on_jrvltsql_20260608_official_calibrator_v1/
```

必須成果物:

```text
validation_report.md
official_calibrator_manifest_snapshot.json
predictions.parquet
metrics_market_raw_official_calibrated.csv
comparison_with_invalid_refit.csv
calibration_diagnostics.csv
roi_ev_ge_1_auxiliary.csv
run_manifest.json
artifact_audit.json
```

---

# 13. 追加・変更ファイル

候補:

```text
src/calibration/official_calibrator_loader.py
scripts/materialize_official_calibrator_artifacts_phase6a_v1.py
scripts/audit_official_calibrator_artifacts_phase6a_v1.py
scripts/validate_latest_model_on_jrvltsql_db.py
tests/test_official_calibrator_loader.py
tests/test_latest_model_validation_uses_official_calibrator.py
docs/phase6a_official_calibrator_artifact_v1.md
```

既存構成へ合わせて最小変更でよい。

---

# 14. 必須テスト

1. official artifactをread-onlyでload
2. Platt係数・intercept一致
3. Isotonic thresholds一致
4. input_space一致
5. strategy不一致を拒否
6. type不一致を拒否
7. hash不一致を拒否
8. parameter不足を拒否
9. artifact不存在を拒否
10. NaN / inf入力を拒否または明示処理
11. 出力0〜1
12. rawへの黙ったfallbackなし
13. fit呼び出しなし
14. fit_transform呼び出しなし
15. OOFからの再推定なし
16. Champion変更なし
17. calibrator type変更なし
18. 2026-06-13/14をparameter生成に使わない
19. market/raw/official calibratedを同一行で比較
20. invalid refit結果を正式指標へ混ぜない
21. ROI判断不可フラグ
22. commit/pushなし

可能ならfit系メソッドをmonkeypatchして、
呼ばれたらテスト失敗させる。

---

# 15. Artifact audit

最低限:

```text
source exists
source hash recorded
materialized hash recorded
strategy match
type match
input_space documented
parameter completeness
refit_performed=false
prediction output finite
probability range valid
```

全check数とfailed数を出す。

---

# 16. 最終判定

以下のいずれか。

```text
OFFICIAL_CALIBRATOR_VALIDATION_PASSED
BLOCKED_MISSING_CERTIFIED_CALIBRATOR_PARAMETERS
BLOCKED_ARTIFACT_HASH_MISMATCH
BLOCKED_INPUT_SPACE_UNKNOWN
BLOCKED_REFIT_DETECTED
MULTIPLE_BLOCKERS
```

さらにboolean:

```text
official_calibrator_loaded
refit_performed
usable_for_probability_diagnostic
usable_for_model_limit_judgement
usable_for_roi_judgement
```

期待:

```text
refit_performed = false
usable_for_roi_judgement = false
```

`usable_for_model_limit_judgement`は72レースのため原則false。

---

# 17. Codex側の実行範囲

実施:

```text
Phase 6A成果物探索
正式parameter特定
artifact materialization
loader実装
既存validation修正
py_compile
pytest
artifact audit
2026-06-13/14軽量再検証
report作成
```

長時間学習は行わない。

認証parameterが不足している場合は、
再fitせずBLOCKEDで終了する。

---

# 18. 最終報告

簡潔に報告する。

1. 10Y official calibratorの認証元
2. 15Y official calibratorの認証元
3. artifact path
4. source hash / artifact hash
5. input_space
6. Platt parameter概要
7. Isotonic parameter概要
8. refitが行われていない証拠
9. market-only指標
10. raw C1R0指標
11. official calibrated指標
12. invalid refit版との差
13. calibrationが改善したか悪化したか
14. モデル限界判断に使えるか
15. ROI判断に使えるか
16. final status
17. py_compile
18. pytest
19. artifact checks / failed
20. 作成・変更ファイル
21. git status --short

commit/pushは行わない。
