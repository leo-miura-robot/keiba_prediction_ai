# Codex Task: Latest Model Validation Pipeline Audit v1
## 2026-06-13 / 2026-06-14 retrospective validation root-cause audit

## 0. 背景

対象SQLite DB:

```text
C:\Users\leole\jrvltsql\data\quickstart_20260608_20260617_20260617_100814\keiba.db
```

評価対象:

```text
2026-06-13
2026-06-14
```

最新構成:

```text
Champion:
ROLLING_10Y + PLATT_SCALING

Shadow:
ROLLING_15Y + ISOTONIC
```

既存検証結果:

```text
72 races
974 rows / strategy

ROLLING_10Y + PLATT:
Logloss = 0.420532
Brier   = 0.136917

ROLLING_15Y + ISOTONIC:
Logloss = 0.422238
Brier   = 0.137434

EV >= 1.00:
10Y = 1 bet / ROI 0.0%
15Y = 4 bets / ROI 55.0%
```

既存成果物:

```text
outputs/latest_model_validation_on_jrvltsql_20260608/
```

既存検証スクリプト:

```text
scripts/validate_latest_model_on_jrvltsql_db.py
```

今回の新DBは2026-06-08以降を中心に作成されているため、
6月13日・14日の予測に必要な過去履歴特徴が不足している可能性がある。

目的は、性能悪化が次のどれによるか特定すること。

```text
A. 現行モデルの限界
B. 履歴特徴量不足
C. 学習時と推論時のschema / dtype / 単位不一致
D. market入力不一致
E. calibration劣化
F. 最終オッズ利用による条件差
G. 小標本
```

---

# 1. 絶対条件

禁止:

```text
CatBoost再学習
calibrator再fit
特徴量追加
EV閾値変更
Champion変更
2026-06-13/14を使ったモデル選択
ROI最大化
commit/push
```

使用可能:

```text
既存モデルartifact
既存calibrator artifact
既存Phase 5B/5C/6A/6B成果物
既存SQLite DB
既存長期履歴DB
軽量な再推論
軽量な再集計
pytest
py_compile
```

---

# 2. 履歴特徴量の供給元監査

`validate_latest_model_on_jrvltsql_db.py`が、
6月13日・14日の履歴特徴量をどのデータ源から生成したか確認する。

分類:

```text
新規短期DBのみ
既存長期履歴DBのみ
両者を結合
既存保存済み特徴量を再利用
不明
```

必須出力:

```text
history_source
history_db_path
history_start_date
history_end_date
history_rows
history_races
validation_dates
```

正しい期待:

```text
2026-06-13:
2026-06-12までの履歴のみ使用

2026-06-14:
2026-06-13までの履歴のみ使用
```

評価日自身の結果や未来データを履歴へ含めない。

---

# 3. 履歴特徴量充足率

最低限、次を監査する。

```text
horse_past_starts
horse_days_since_last

horse_last1_avg_finish
horse_last3_avg_finish
horse_last5_avg_finish

horse_last3_win_rate
horse_last5_win_rate
horse_last3_ren_rate
horse_last5_ren_rate
horse_last3_top3_rate
horse_last5_top3_rate
horse_last3_place_paid_rate
horse_last5_place_paid_rate

horse_jyo_past_starts
horse_surface_past_starts
horse_dist_band_past_starts
horse_baba_past_starts

jockey_past_starts
jockey_win_rate
jockey_top3_rate

trainer_past_starts
trainer_win_rate
trainer_top3_rate

horse_jockey_past_starts
horse_jockey_win_rate
horse_jockey_top3_rate
```

各特徴で出力:

```text
feature
rows
non_null_count
null_rate
zero_rate
mean
std
min
p10
p25
p50
p75
p90
p95
p99
max
```

特に以下を明示する。

```text
horse_past_starts == 0
jockey_past_starts == 0
trainer_past_starts == 0
```

---

# 4. 学習時分布との比較

比較対象の優先順位:

```text
1. Phase 5B/5Cの2024または2025 validation features
2. Phase 5B/5C保存済みprediction input
3. 同じfeature builderで長期DBから再生成した参照期間
```

出力列:

```text
feature
reference_mean
current_mean
mean_ratio
reference_null_rate
current_null_rate
reference_zero_rate
current_zero_rate
reference_p50
current_p50
reference_p95
current_p95
PSI
distribution_status
```

status:

```text
OK
WARNING
SEVERE_SHIFT
MISSING_REFERENCE
```

---

# 5. Feature schema parity

正式feature allowlistと、今回モデルへ渡した実列を比較する。

確認:

```text
column name
column order
dtype
categorical / numeric role
missing column
extra column
duplicate column
all-null column
constant column
```

必須:

```text
official feature allowlistと一致
列順一致
KisyuCode / ChokyosiCodeの除外状態
Year / odds / popularity / result / payout / IDの除外状態
```

1列でも重大不一致があれば、
`MODEL_VALIDATION_INVALID`候補とする。

---

# 6. Raw列・単位・コード体系監査

以下を学習時参照と比較する。

```text
JyoCD
TrackCD
CourseKubunCD
Kyori
SyussoTosu
Wakuban
Umaban
Barei
SexCD
Futan
BaTaijyu
ZogenSa
TenkoCD
SibaBabaCD
DirtBabaCD
tan_odds
fuku_odds_low
fuku_odds_high
tan_ninki
fuku_ninki
```

確認:

```text
dtype
min / max
単位
コード体系
leading zero
string padding
欠損表現
0埋め
異常値
```

特に:

```text
斤量
馬体重
距離
オッズ倍率
人気の基準
コード列の文字列/整数差
```

---

# 7. Market入力監査

今回は`NL_O1`の確定オッズを使ったretrospective validationである。

確認:

```text
使用テーブル
使用列
日付範囲
odds snapshot type
欠損行
除外行
異常値
```

market入力:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
SyussoTosu
place_rank_limit
```

派生列:

```text
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

学習時と同じ変換・clip・欠損処理か確認する。

---

# 8. Target / place rule監査

対象列:

```text
target_place_paid
fuku_pay
KakuteiJyuni
IJyoCD
SyussoTosu
place_rank_limit
```

必須:

```text
target_place_paid == (fuku_pay > 0)
```

禁止:

```python
KakuteiJyuni <= 3
target.le(3)
```

5～7頭立ての3着が正しく0になるか確認する。

---

# 9. Market-only / Raw / Calibrated比較

同じ974行で次を評価する。

## Market-only

```text
probability_market = sigmoid(market_logit)
```

## Raw C1R0

```text
probability_raw = sigmoid(market_logit + residual_raw)
```

## Calibrated

```text
10Y: PLATT_SCALING(probability_raw)
15Y: ISOTONIC(probability_raw)
```

各構成で:

```text
Logloss
Brier
ECE
calibration slope
calibration intercept
race-wise Spearman
```

差分:

```text
raw - market
calibrated - raw
calibrated - market
```

---

# 10. Residual診断

strategy別:

```text
residual_mean
residual_std
abs_residual_p50
abs_residual_p90
abs_residual_p95
abs_residual_p99
residual_min
residual_max
```

Phase 5B/5C参照分布と比較する。

確認:

```text
residualがほぼ0
極端に大きい
一方向へ偏る
履歴不足行でのみ極端
```

異常行を最大20件出す。

---

# 11. Prediction分布

strategy別・market/raw/calibrated別:

```text
mean
std
min
p01
p05
p10
p25
p50
p75
p90
p95
p99
max
actual positive rate
mean predicted probability
calibration gap
```

---

# 12. Segment診断

最低限:

```text
race_date
JyoCD
surface
distance_bucket
field_size_bucket
odds_bucket
```

出力:

```text
rows
races
actual positive rate
market Logloss
raw Logloss
calibrated Logloss
delta raw-market
delta calibrated-raw
small_sample
```

`rows < 100`は`small_sample=true`。

---

# 13. 正しい履歴を結合した軽量再検証

監査で短期DBのみを使っていた場合、
既存長期履歴DBを使って再検証する。

履歴:

```text
既存長期DB:
2006～2026-06-07

新規短期DB:
2026-06-08～2026-06-14
```

評価:

```text
2026-06-13:
6/12までを履歴に使用

2026-06-14:
6/13までを履歴に使用
```

重複時の優先規則を記録する。

禁止:

```text
評価日自身の結果を履歴へ使用
新規CatBoost学習
calibrator再fit
```

既存モデルartifactへの再推論だけ行う。

---

# 14. 最終判定

以下から選ぶ。

```text
VALID_MODEL_EVALUATION
INVALID_HISTORY_FEATURES
INVALID_SCHEMA_PARITY
INVALID_MARKET_INPUT
INVALID_TARGET_LOGIC
CALIBRATION_ISSUE
SMALL_SAMPLE_ONLY
MULTIPLE_ROOT_CAUSES
BLOCKED
```

今回結果についてbooleanを出す。

```text
usable_for_model_limit_judgement
usable_for_probability_diagnostic
usable_for_roi_judgement
```

ROIはbet数1件/4件のため、原則false。

---

# 15. 出力先

```text
outputs/latest_model_validation_on_jrvltsql_20260608_audit_v1/
```

必須成果物:

```text
audit_summary.json
history_source_audit.json
history_feature_completeness.csv
history_feature_distribution_comparison.csv
feature_schema_parity.csv
raw_column_unit_audit.csv
market_input_audit.csv
target_integrity_audit.json

probability_metrics_market_raw_calibrated.csv
residual_distribution_audit.csv
prediction_distribution_audit.csv
segment_metrics.csv
abnormal_rows.csv

rerun_comparison.csv
final_assessment.json
manifest.json
audit_report.md
```

コード:

```text
scripts/audit_latest_model_validation_on_jrvltsql_db.py
tests/test_latest_model_validation_on_jrvltsql_db_audit.py
docs/latest_model_validation_on_jrvltsql_db_audit_v1.md
```

---

# 16. 必須テスト

1. 履歴開始日・終了日を記録
2. 評価日自身の結果を履歴へ入れない
3. 6/13は6/12まで
4. 6/14は6/13まで
5. feature allowlist一致
6. feature order一致
7. dtype重大不一致検出
8. missing feature検出
9. extra feature検出
10. target_place_paid整合
11. 5～7頭立て3着誤判定なし
12. market-only計算
13. raw計算
14. calibrated計算
15. 同一行で3確率比較
16. final oddsをpre-raceと表示しない
17. 履歴不足率を計算
18. 分布差計算
19. 小標本フラグ
20. ROI判断不可フラグ
21. 新規学習なし
22. calibrator再fitなし
23. Champion変更なし
24. commit/pushなし

---

# 17. Codex側の実行範囲

実施:

```text
コード監査
成果物監査
DB監査
軽量再集計
必要なら長期履歴DB結合で再推論
py_compile
pytest
artifact audit
結果レポート
```

長時間学習は行わない。

長期履歴DBが見つからない場合はBLOCKEDとして必要パスを報告する。

---

# 18. 最終報告

簡潔に以下を報告する。

1. 履歴特徴量の供給元
2. 履歴期間
3. 履歴不足率
4. 代表特徴量のゼロ率・欠損率
5. 学習時分布との乖離
6. feature schema parity
7. raw列・単位・コード不一致
8. market入力監査
9. target監査
10. market-only指標
11. raw C1R0指標
12. calibrated指標
13. residual分布
14. 再検証の有無
15. 再検証後の指標
16. 根本原因
17. 今回結果がモデル限界判断に使えるか
18. ROI判断に使えるか
19. 次に必要な修正
20. py_compile
21. pytest
22. audit checks / failed
23. 作成・変更ファイル
24. git status --short

commit/pushは行わない。
