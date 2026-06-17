# Codex Phase 6A.1 Task v1
## Safe Calibration Certification Audit

## 0. 目的

Phase 6Aで次のcalibrator候補が生成された。

```text
Champion model:
ROLLING_10Y + PLATT_SCALING

Challenger shadow:
ROLLING_15Y + ISOTONIC
```

Phase 6A実行は完了しているが、元タスクMarkdownがリポジトリ内に存在せず、
プロンプト本文を仕様として実装された。

また、報告では以下となっている。

```text
pytest: 28 passed, 1 warning
artifact audit: 24 checks, failed 0
operationally_activated = false
```

元仕様は31項目の監査を要求していたため、
新規学習を行わず、コード・設定・既存成果物だけで
Phase 6Aを正式認証できるか監査する。

本タスクはcalibratorの再探索ではない。

---

# 1. 絶対条件

禁止:

```text
CatBoost再学習
DB接続
新しいcalibrator family追加
parameter search
Optuna
EV threshold sweep
2025/2026を使ったcalibrator選択
Champion変更
commit/push
```

使用可能:

```text
既存コード
既存設定
既存Phase 5B/5C/6A成果物
既存parquet/csv/json
軽量な再集計
paired bootstrap再計算
pytest
py_compile
```

---

# 2. 最重要監査

## 2.1 Calibrator選択期間

calibrator familyの選択が
2020～2024だけで行われていることを確認する。

2025/2026の列・指標・結果が
選択処理へ入っていないことをコード上と成果物上で証明する。

## 2.2 Primary metric

元仕様のPrimaryは次である。

```text
2020～2024の全行を結合して再計算したLogloss
```

年度別Loglossの単純平均をPrimaryにしてはいけない。

以下を確認する。

```text
selection code
calibrator_comparison_2020_2024.csv
calibrator_selection.csv
```

もし現実装が年度平均で選択している場合:

1. 新規学習なしで結合指標を再計算
2. 選択結果が変わるか確認
3. 選択ロジックを結合指標へ修正
4. 修正前後をレポート
5. 2025/2026を使わず再認証

## 2.3 Walk-forward leakage

各評価年Yについて、calibrator fit dataが
Yより前のOOF predictionだけであることを確認する。

必須:

```text
2020 fit = 2016～2019
2021 fit = 2016～2020
2022 fit = 2016～2021
2023 fit = 2016～2022
2024 fit = 2016～2023
2025 fit = 2016～2024
2026 fit = 2016～2025
```

strategy別・method別に
fit_start_year / fit_end_year / fit_rows / fit_racesを検証する。

## 2.4 Target integrity

必須:

```text
target = target_place_paid
unique subset {0,1}
.le(3)や順位変換なし
target_place_paidとactual_place一致
target_place_paidと(fuku_pay > 0)の整合性
```

不一致があれば件数、率、サンプルキーを出す。

## 2.5 Existing raw prediction reuse

2020～2026のraw predictionが
Phase 5B/5C既存成果物から再利用され、
再学習・重複・欠落・key mismatchがないことを確認する。

必須:

```text
strategy
Year
rows
races
key hash
prediction hash
duplicate key count
missing key count
```

---

# 3. Calibrator別の正式比較

対象:

```text
RAW_IDENTITY
TEMPERATURE_SCALING
PLATT_SCALING
ISOTONIC
```

strategy別に2020～2024結合データから再計算する。

```text
rows
races
Logloss
Brier
ECE 10-bin
ECE 20-bin
calibration slope
calibration intercept
```

追加:

```text
worst-year Logloss
worst-year Brier
year-to-year CV
```

年度平均値と結合値を分けて保存する。

---

# 4. Bootstrap認証

2020～2024だけでrace単位5000回。

## ROLLING_10Y

```text
PLATT_SCALING vs RAW_IDENTITY
PLATT_SCALING vs second-best non-raw method
```

## ROLLING_15Y

```text
ISOTONIC vs RAW_IDENTITY
ISOTONIC vs second-best non-raw method
```

対象:

```text
delta Logloss
delta Brier
```

出力:

```text
point estimate
bootstrap mean
95% CI lower
95% CI upper
candidate better probability
```

2025/2026 bootstrapは診断として別表に置く。
認証判定には使用しない。

---

# 5. Isotonic安全性監査

ROLLING_15YのIsotonicについて:

```text
fit unique probability count
output unique probability count
largest step
minimum plateau rows
maximum plateau rows
p<0.01 / p>0.99 counts
clip count
NaN / inf count
```

2020～2024および2025/2026を分ける。

tailで極端な段差・少数plateauがある場合は、
shadow-onlyの注意事項に残す。

---

# 6. Platt安全性監査

ROLLING_10YのPlattについて各年:

```text
coefficient a
intercept b
fit rows
fit positive rate
convergence status
n_iter
```

異常な係数変動や未収束がないか確認する。

---

# 7. ROI診断の位置づけ

ROIは認証基準に使わない。

既存結果から以下だけ再確認する。

```text
EV threshold = 1.00固定
combined ROI = total payout / total stake
payout_zeroed_stress_roi <= normal_roi
betなしはN/A
```

2025+2026で報告された:

```text
10Y calibrated ROI = 96.69%
15Y calibrated ROI = 109.17%
```

について、元行から再計算して一致確認する。

top10 zeroed後の大幅低下を、
「高配当依存あり」と明示する。

このROIだけで運用有効化・Champion変更を行わない。

---

# 8. 認証判定

strategyごとに次を出す。

```text
recommended_method
certification_status
operational_activation_recommended
operationally_activated
reason
```

`operationally_activated`は本タスクでもfalseのまま。

## ROLLING_10Yの推奨基準

Plattを正式な運用候補として認証できるのは:

```text
selectionが2020～2024結合Loglossベース
walk-forward leakageなし
target監査通過
2020～2024結合LoglossがRAWより改善
race bootstrapが改善方向
Brierの重大悪化なし
Platt fitが全fold収束
```

## ROLLING_15Y

IsotonicはChallenger shadow用として認証する。

Champion変更・本番採用はしない。

---

# 9. 不足していた監査項目

元Phase 6A仕様の31項目を一覧化し、
現在の24 audit checksと対応表を作る。

出力列:

```text
requirement_id
requirement
existing_check
status
evidence
new_check_added
```

不足分はaudit scriptとtestへ追加する。

最終的に可能なら:

```text
31 checks, failed 0
```

または31項目以上へする。

満たせない項目はPASSにせず、
NOT_IMPLEMENTED / BLOCKEDとして明示する。

---

# 10. 出力

既存Phase 6A成果物は上書きしない。

出力先:

```text
outputs/place_market_offset_safe_calibration_phase6a_certification_v1/
```

必須成果物:

```text
selection_logic_audit.json
selection_metrics_pooled_2020_2024.csv
selection_metrics_year_mean_2020_2024.csv
walk_forward_fit_window_audit.csv
raw_prediction_reuse_audit.csv
target_integrity_audit.json

calibrator_certification_bootstrap.csv
isotonic_safety_audit.csv
platt_parameter_stability.csv

roi_recalculation_check.csv
phase6a_requirement_coverage.csv
calibration_certification.json
manifest.json
audit_report.md
```

文書:

```text
docs/place_market_offset_safe_calibration_phase6a_certification_v1.md
```

必要な場合のみ変更:

```text
scripts/run_place_market_offset_safe_calibration_phase6a_v1.py
scripts/audit_place_market_offset_safe_calibration_phase6a_v1.py
tests/test_place_market_offset_safe_calibration_phase6a_v1.py
```

---

# 11. 検証

実行:

```text
py_compile
Phase 6A関連pytest
Phase 5B/5C回帰pytest
certification audit
```

新規学習は行わない。

---

# 12. 最終報告

簡潔に以下を報告する。

1. Selectionがpooledかyear meanか
2. 修正の有無
3. 2025/2026が選択から完全除外されているか
4. Walk-forward fit window
5. Target integrity
6. Raw prediction再利用監査
7. 10Y Platt vs RAWの2020～2024 bootstrap
8. 10Y Platt vs second-best
9. 15Y Isotonic vs RAW
10. Isotonic safety
11. Platt parameter stability
12. ROI再計算一致
13. 高配当依存
14. 31要件coverage
15. 10Y Platt認証可否
16. 15Y Isotonic shadow認証可否
17. operationally_activated=false
18. py_compile
19. pytest
20. audit checks / failed
21. 作成・変更ファイル
22. git status --short

commit/pushは行わない。
