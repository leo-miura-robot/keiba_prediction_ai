# Antigravity Phase 5 Task v1
## 2006–2015 Historical Database Integration, Warm-up Extension, and Time-Safe Re-evaluation

## 0. 目的

2006～2015年の競馬DBを新たに取得できたため、現在の競馬予想AIへ安全に統合し、履歴特徴の初期不足を改善するとともに、古い学習データを追加する価値を検証する。

ただし、2006～2015年を単純に既存学習データへ追加すると、以下が混ざる。

- 履歴ウォームアップが長くなった効果
- 学習行数が増えた効果
- 古い競馬環境によるconcept driftの影響

そのため、本タスクでは以下を分離して比較する。

```text
BASE_2016
既存正式BASE。履歴生成・モデル学習とも2016年以降。

WARMUP_2006_TRAIN_2016
履歴状態は2006年から生成するが、モデル学習行は2016年以降だけ。
2006～2015年は履歴ウォームアップ専用。

FULL_2006
履歴生成・モデル学習とも2006年以降。
```

必要条件を満たした場合だけ、追加候補として以下を最大1種類検討してよい。

```text
ROLLING_10Y
各評価年直前の10年間だけを学習行に使用。
履歴生成自体は2006年から継続。
```

最終目的は、現在の正式BASEよりも、以下を改善できるか確認することである。

- probability_rawのLogloss / Brier
- calibrationの時系列安全性
- worst-year性能
- 2025/2026固定診断
- ROIの再現性

---

# 1. 現在の正式BASE

```text
C1R0_fixed300_ablation_drop_person_codes
```

基本式:

```text
final_logit = market_logit + residual_raw
probability_raw = sigmoid(final_logit)
```

市場情報はbaselineだけに使う。CatBoostには市場以外の競馬情報を入力する。

固定条件:

```text
tree count = 300
KisyuCode除外
ChokyosiCode除外
Year除外
p_market除外
market_logit除外
raw odds / 人気 / 市場順位除外
結果・払戻・管理ID除外
rate smoothing不採用
trainer_past_starts raw
jockey_past_starts raw
勝率系rate raw
```

モデル選択は未補正の`probability_raw`で行う。

---

# 2. 重要な既知問題

## 2.1 キャリブレーション

過去に以下の問題が発生した。

- calibrated / uncalibratedの混在
- fold内学習予測を使ったIsotonic Regression
- 過学習したキャリブレーターによる確率の異常化
- 全頭に近いEV>=1判定

新しい評価では、キャリブレーターは必ず評価対象年の正解を見ないこと。

正しい例:

```text
評価年2024
↓
2016～2023または2006～2023の学習期間内だけでOOF予測を作成
↓
その学習期間内OOF予測と正解でIsotonicをfit
↓
2024予測へ適用
```

禁止:

```text
2024予測と2024正解でIsotonicをfit
↓
同じ2024を評価
```

## 2.2 ROI

ROIはモデル選択の主指標ではない。ただし、以下を明示して診断する。

```text
probability_rawを使ったROI
probability_calibratedを使ったROI
```

両者を混ぜない。

高配当除外は2種類を分ける。

```text
row_removed_roi:
上位払戻馬券の行自体を除外し、分子と分母の両方を減らす

payout_zeroed_stress_roi:
購入はしたものとして分母を維持し、上位払戻だけを0円扱いにする
```

曖昧な「除外後ROI」という名称だけを使わない。

---

# 3. 公式の評価期間

既存walk-forward:

```text
train 2016–2019 -> validation 2020
train 2016–2020 -> validation 2021
train 2016–2021 -> validation 2022
train 2016–2022 -> validation 2023
train 2016–2023 -> validation 2024
```

Phase 5候補:

## BASE_2016

既存成果物を再利用する。

## WARMUP_2006_TRAIN_2016

```text
history generation starts at 2006
model train rows start at 2016

train rows 2016–2019 -> validation 2020
train rows 2016–2020 -> validation 2021
train rows 2016–2021 -> validation 2022
train rows 2016–2022 -> validation 2023
train rows 2016–2023 -> validation 2024
```

## FULL_2006

```text
train 2006–2019 -> validation 2020
train 2006–2020 -> validation 2021
train 2006–2021 -> validation 2022
train 2006–2022 -> validation 2023
train 2006–2023 -> validation 2024
```

モデル選択・候補選択には2020～2024だけを使う。

```text
2025 = test
2026 = latest holdout / operational diagnostic
```

2025/2026を見て候補や仕様を変更しない。

2026の学習期間は既存プロジェクトの正式ルールをコード・manifestから確認する。2016/2006～2024固定モデルで評価するのか、2025まで学習したrolling operational評価なのかを混在させない。不明なら停止して確認する。

---

# 4. 最初に読むファイル

```text
tasks/antigravity_c1r0_rate_smoothing_phase4_v1_task.md
tasks/antigravity_c1r0_metric_consistency_handover_v2_task.md

docs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1_results.md
docs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md

scripts/run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py
scripts/audit_c1r0_metric_consistency_v2.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py

history_builder_v2_1.py
```

`history_builder_v2_1.py`の実際の場所は検索で特定する。

2006～2015 DBについては、候補ファイルを列挙し、複数ある場合は勝手に選ばない。

---

# 5. 絶対条件

- source DBはread-only
- source DBを書き換えない
- source DBを削除・移動しない
- 既存2016～2026 DB / Parquetを上書きしない
- 新しい統合データは新規パスへ保存
- 2015年以前を無条件に既存BASEへ混ぜない
- random split禁止
- 2025/2026でモデル選択禁止
- tree countは300固定
- baseline方式変更禁止
- CatBoostの他hyperparameter変更禁止
- market特徴をCatBoostへ再混入させない
- KisyuCode / ChokyosiCodeを再混入させない
- rate smoothingを同時変更しない
- タイム特徴を同時変更しない
- コース特徴を同時追加しない
- ROI直接学習禁止
- Ability / ANA禁止
- Ranker禁止
- Kelly禁止
- 大規模Optuna禁止
- 自動購入禁止
- git add / commit / push禁止
- git reset / clean禁止
- 既存成果物を削除・上書きしない

---

# 6. 書き込み許可範囲

原則として以下へ新規作成する。

```text
config/place_market_offset_history_extension_2006_phase5_v1.yaml

scripts/audit_history_db_2006_2015_v1.py
scripts/build_history_extension_2006_phase5_v1.py
scripts/run_place_market_offset_history_extension_phase5_v1.py
scripts/audit_roi_market_baselines_phase5_v1.py

tests/test_history_db_2006_2015_v1.py
tests/test_place_market_offset_history_extension_phase5_v1.py
tests/test_roi_market_baselines_phase5_v1.py

docs/place_market_offset_history_extension_phase5_v1_results.md

outputs/place_market_offset_history_extension_phase5_v1/
models/place_market_offset_history_extension_phase5_v1/
data/derived/history_extension_2006_phase5_v1/
```

既存コード変更が必要でも、まず新規wrapper / utilityで対応する。既存コードを直接変更する必要がある場合は、変更前に理由と対象行を報告して停止する。

---

# 7. Stage 0: 開始時確認

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

過去の未追跡ファイルや手動commitを勝手に修正しない。

source DB候補について以下を出す。

```text
path
size
modified_time
database_type
table_names
date_min
date_max
row_count
sha256
```

複数DBがあり、正しいものを一意に判断できない場合は停止する。

---

# 8. Stage 1: 2006～2015 DB監査

## 8.1 スキーマ比較

2006～2015と2016年以降について、テーブル・列単位で比較する。

```text
table_name
column_name
dtype_old
dtype_new
nullable_old
nullable_new
exists_old
exists_new
```

出力:

```text
db_schema_comparison.csv
db_table_inventory.csv
```

## 8.2 データ品質

年別に確認する。

```text
race_count
runner_count
duplicate_race_count
duplicate_runner_count
missing_rate_by_column
zero_rate_by_column
invalid_date_count
invalid_odds_count
invalid_payout_count
invalid_finish_position_count
```

## 8.3 コード体系

以下のコード体系が2016年前後で変化していないか確認する。

```text
race_id
horse_id
jockey_id
trainer_id
track code
surface code
distance
weather
going
race class
sex
age
finish position
odds
payout
```

人物コードはモデル特徴には使わないが、履歴集計キーとして一貫している必要がある。

## 8.4 オッズ・払戻単位

2006～2015と2016年以降で以下が同じ単位か確認する。

```text
decimal odds
100円払戻
複勝オッズ下限
複勝オッズ上限
確定複勝払戻
```

単位変換を推測しない。

## 8.5 競走制度・分布変化

年別に以下を比較する。

```text
平均頭数
複勝対象頭数
距離分布
芝/ダート比率
競馬場分布
人気分布
オッズ分布
払戻分布
取消・除外・中止率
```

これはconcept drift監査であり、候補選択に2025/2026は使わない。

## 8.6 target_place_paidの定義整合性

2006～2015と2016～2026で`target_place_paid`の定義が一致しているか確認する。

確認項目:

```text
place_rank_limit の年別・頭数別分布
7頭以下レースでの2着払い / 3着払いの処理
取消・除外馬が出走取消前の頭数カウントに含まれているか
target_place_paid = 1 の実際の着順分布
```

出力:

```text
target_definition_consistency.csv
```

---

# 9. Stage 2: 履歴生成の時系列安全性

## 9.1 現在レース除外

各レースの特徴は、そのレースの結果を更新する前に生成する。

```text
generate features
↓
predictable row created
↓
update history with race result
```

これをコードとテストで確認する。

## 9.2 同日順序

同日内の未来レースが過去レース特徴へ混ざらないこと。

並び順を明示する。

```text
race_date
track
race_number
runner key
```

同時刻・順序不明の場合の処理を明示する。

## 9.3 2006年初期

2006年以前の履歴がないため、2006年初期はcold startになる。これは許容するが、以下を年別・月別に記録する。

```text
horse zero-history rate
jockey zero-history rate
trainer zero-history rate
horse_surface zero-history rate
last3 unavailable rate
last5 unavailable rate
```

特に2016年時点で、旧BASEよりどれだけ改善したか比較する。

出力:

```text
history_warmup_quality_by_year.csv
history_warmup_quality_2016_comparison.csv
```

## 9.4 累積履歴特徴の飽和診査

2006年から累積する場合、2024年時点で18年分の成績が蓄積される。以下を年別に記録する。

```text
entity_type (horse, jockey, trainer, etc.)
year
median_past_starts
p90_past_starts
p99_past_starts
median_rate_value
std_rate_value
rate_coefficient_of_variation
new_entity_rate (その年に初めて出現したエンティティの割合)
```

特にjockeyとtrainerは長期間活動するため累積が大きくなる。horseは引退があるため自然にリセットされる。

出力:

```text
history_saturation_by_year.csv
history_saturation_comparison_base_vs_full.csv
```

---

# 10. Stage 3: 新規統合データ作成

source DBをread-onlyで読み込み、新しい派生データとして保存する。

```text
data/derived/history_extension_2006_phase5_v1/
```

最低限保存:

```text
canonical_race_rows_2006_2026.parquet
history_features_2006_2026.parquet
manifest.json
source_hashes.json
schema.json
```

既存Parquetを上書きしない。

manifest必須項目:

```text
source_db_paths
source_db_sha256
source_date_min
source_date_max
row_counts_by_year
race_counts_by_year
schema_hash
feature_builder_version
feature_generation_start_year
model_training_start_year
current_race_excluded
same_day_future_excluded
created_at
```

---

## 10.1 市場ベースラインの扱い

市場ベースラインモデル（LogisticRegression）の学習開始年は全候補で2016年に統一する。

理由:

```text
市場構造（オッズの精度、控除率、馬券購入行動）は
2006年と2016年で大きく異なる可能性がある。
市場ベースラインの変更は本実験の対象外。
市場ベースラインを変えると、交絡因子が増え、
履歴改善・学習行追加の効果を正しく分離できない。
```

FULL_2006の学習行には2006～2015も含まれるが、これらの行の`market_logit`は2016年以降で学習した市場モデルのexpanding OOF予測を使用する。2006～2015行については最初の市場モデルが2016～2019で学習したものであるため、2006～2015の`market_logit`は「モデル外挿」となる。

この外挿による影響を監査するため、以下を出力する。

```text
market_logit_distribution_by_year.csv
market_logit_2006_2015_extrapolation_audit.csv
```

2006～2015のmarket_logitが異常値（p1/p99が2016年以降と大幅に乖離）を取る場合は停止して報告する。

---

# 11. Stage 4: 比較候補

## 11.1 BASE_2016

既存の正式BASEモデル・予測を再利用する。

### 11.1.1 BASE_2016の再利用範囲

BASE_2016は以下を再利用する。

```text
CatBoostモデルファイル: 再学習しない
probability_raw予測値: Phase 4出力parquetから読み込む
market_logit: Phase 1出力から読み込む
```

以下はPhase 5パイプラインで再計算する。

```text
probability_calibrated: Phase 5共通キャリブレーションを新規適用
EV / ROI: Phase 5共通関数で再計算
```

これにより、全候補が同一のキャリブレーション・評価パイプラインを通り、Phase 2/3で発生したcalibrated/uncalibrated混在を防ぐ。

## 11.2 WARMUP_2006_TRAIN_2016

- 履歴は2006年から生成
- 学習行は2016年以降
- モデル構造・特徴allowlist・tree countはBASEと同じ

この候補は、履歴ウォームアップ改善だけを測る。

## 11.3 FULL_2006

- 履歴は2006年から生成
- 学習行も2006年以降
- モデル構造・特徴allowlist・tree countはBASEと同じ

この候補は、古い学習行を追加する効果を測る。

## 11.4 ROLLING_10Y

以下の定量条件の両方を満たす場合だけ実行してよい。

条件A:

```text
FULL_2006 combined_logloss < BASE combined_logloss - 0.0001
```

条件B（いずれか1つ）:

```text
FULL_2006 worst_year_logloss > BASE worst_year_logloss + 0.001
FULL_2006 logloss CV > BASE logloss CV × 1.2
db_concept_drift_summary.csv で feature_distribution_divergence > 0.1 の年が3年以上
```

条件Aを満たさない場合、FULL_2006がBASEに勝っていないため、ROLLING_10Yにも価値がない。

条件Bを満たさない場合、FULL_2006が安定しているため、ROLLING_10Yでさらに改善する動機がない。

最大1候補。無条件実行禁止。

---

# 12. Stage 5: 学習・評価

## 12.1 モデル選択確率

必ず`probability_raw`を使う。

## 12.2 主評価

```text
runner-weighted Logloss
runner-weighted Brier
fixed-bin ECE on probability_raw
calibration slope on probability_raw
calibration intercept on probability_raw
worst-year Logloss
worst-year Brier
```

## 12.3 年度安定性

```text
mean
standard deviation
coefficient of variation
worst year
best year
```

## 12.4 残差

```text
residual mean
residual std
abs residual p90
abs residual p95
abs residual p99
```

## 12.5 予測順位

```text
race-wise Spearman
top probability hit rate
```

---

# 13. Stage 6: 時系列安全なキャリブレーション

## 13.1 foldごとのcalibrator

各validation年について、training期間内だけでOOF予測を作る。

例: validation 2024

```text
outer training = 2006/2016～2023
inner time-based OOF predictions only
fit isotonic on inner OOF
apply calibrator to outer validation 2024
```

inner OOFでもrandom split禁止。

可能なinner分割例:

```text
train through Y-1 -> predict Y
```

training期間内で最低2年以上のOOFを確保する。具体的分割をmanifestへ記録する。

### 13.1.1 inner OOFの具体的な年範囲

全候補で統一する。inner OOFは2020年以降のvalidation fold予測のみを使う。

各outer foldについて:

```text
outer_validation=2020: inner OOFなし。calibratorなし。probability_calibrated = probability_raw, is_calibrated = false
outer_validation=2021: inner OOF = {2020}。fold_2020のvalidation予測を使用
outer_validation=2022: inner OOF = {2020, 2021}。fold_2020, fold_2021のvalidation予測を使用
outer_validation=2023: inner OOF = {2020, 2021, 2022}
outer_validation=2024: inner OOF = {2020, 2021, 2022, 2023}
```

fold_2020のキャリブレーションではinner OOFが不足するため、`probability_calibrated = probability_raw`とし、`is_calibrated = false`とする。

FULL_2006であっても、inner OOFは2020年以降のみ使用する。2006～2019の予測をinner OOFに含めない。理由: 古い時代の確率分布でfitしたIsotonicが2020年以降の評価に悪影響を与えるリスクを避けるため。

## 13.2 保存列

```text
probability_raw
probability_calibrated
probability_used_for_model_selection
probability_used_for_ev
is_calibrated
calibration_method
calibrator_train_start
calibrator_train_end
calibrator_oof_years
outer_validation_year
```

モデル選択:

```text
probability_used_for_model_selection = probability_raw
```

EV診断はrawとcalibratedを別表で出す。

---

# 14. Stage 7: ROIと市場ベンチマーク

## 14.1 必須ベンチマーク

同じ年度・同じ払戻定義で以下を計算する。

```text
ALL_RUNNERS_FLAT
RANDOM_MATCHED_COUNT
MARKET_ONLY
BASE_2016
WARMUP_2006_TRAIN_2016
FULL_2006
ROLLING_10Y if executed
```

## 14.2 ALL_RUNNERS_FLAT

全出走馬を100円ずつ購入した実測ROI。約80%に必ず一致すると決めつけず、実データの券種・控除・払戻・欠損処理を検証する基準とする。

## 14.3 RANDOM_MATCHED_COUNT

モデルのEV>=1件数と同じ件数を、年度・レース構成を保ってランダム抽出する。

```text
n_repeats = 1000
seed fixed
```

平均ROIと95%区間を出す。

## 14.4 MARKET_ONLY

市場確率だけでEVを計算し、以下を監査する。

```text
複勝市場確率の合計
複勝対象2頭/3頭の処理
使用オッズが事前利用可能か
下限/上限/確定払戻の区別
```

## 14.5 モデルROI

```text
EV >= 1.0
100円均等購入
```

出力列:

```text
year
model_key
probability_column
calibration_state
bet_count
stake
payout
roi
ev_roi_spearman
```

## 14.6 高配当依存

2種類を必ず別に出す。

### A. row_removed_roi

```text
remaining_stake = original_stake - removed_bet_stake
remaining_payout = original_payout - removed_payout
roi = remaining_payout / remaining_stake
```

### B. payout_zeroed_stress_roi

```text
stake = original_stake
payout = original_payout - removed_payout
roi = payout / original_stake
```

k:

```text
1
3
5
10
```

必須列:

```text
removed_count
removed_stake
removed_payout
remaining_bet_count
remaining_stake
remaining_payout
row_removed_roi
payout_zeroed_stress_roi
```

---

# 15. Stage 8: 候補選択

優先順位:

1. probability_raw Logloss
2. probability_raw Brier
3. worst-year Logloss/Brier
4. 年度安定性
5. residual tail
6. calibration slope/intercept
7. 複雑さ
8. ROIは補助診断

ROIだけでモデルを選ばない。

最終候補とBASEの差が小さい場合:

```text
sampling unit = race
n_bootstrap = 5000
seed fixed
delta = candidate - BASE
```

対象:

```text
Logloss
Brier
```

CIが0をまたぎ、実質差も極小なら単純なBASEを維持する。

---

# 16. Stage 9: 2025/2026固定診断

2020～2024で候補を完全固定した後だけ実行。

対象:

```text
BASE_2016
最良候補1つ
次点候補は役割が明確な場合だけ最大1つ
```

2025/2026を見て選択変更禁止。

必須:

```text
probability_raw metrics
probability_calibrated metrics
EV>=1 count
stake
payout
ROI
row_removed_roi
payout_zeroed_stress_roi
EV-ROI Spearman
```

2025+2026合算も出す。

```text
combined_stake
combined_payout
combined_roi
```

---

# 17. resume / 再利用

BASEは再学習しない。

候補foldごとに以下を保存する。

```text
candidate_key
outer_validation_year
feature_hash
config_hash
source_hash
training_start_year
training_end_year
tree_count
model_path
prediction_path
status
```

hashと行数が一致する場合だけ再利用。

---

# 18. 必須成果物

```text
db_table_inventory.csv
db_schema_comparison.csv
db_quality_by_year.csv
db_code_distribution_by_year.csv
db_odds_payout_unit_audit.csv
db_concept_drift_summary.csv

history_warmup_quality_by_year.csv
history_warmup_quality_2016_comparison.csv
history_temporal_safety_audit.md

candidate_definition.csv
walk_forward_folds.csv
candidate_metrics_by_fold.csv
candidate_metrics_2020_2024.csv
candidate_residual_stability.csv
candidate_feature_importance.csv
candidate_shap.csv
candidate_shap_additivity.csv

calibration_provenance_by_fold.csv
calibration_metrics_by_fold.csv

market_baseline_roi.csv
random_matched_roi_summary.csv
model_roi_raw.csv
model_roi_calibrated.csv
high_payout_row_removed_roi.csv
high_payout_zeroed_stress_roi.csv

paired_bootstrap_summary.csv
selected_history_strategy.json
phase5_2025_2026_diagnostic.csv
phase5_2025_2026_combined.csv

manifest.json
audit_report.md
```

---

# 19. 必須テスト

1. source DB read-only
2. source DB hashが実行前後で一致
3. 既存Parquet変更なし
4. 新規派生パスだけへ出力
5. random splitなし
6. current race結果を特徴へ含めない
7. same-day futureを含めない
8. 2006年初期cold startを明示
9. 2016年warm-up改善を検証
10. BASE allowlist維持
11. Year / market features再混入なし
12. KisyuCode / ChokyosiCode再混入なし
13. tree count 300
14. model selectionはprobability_raw
15. calibratorはouter validation正解を見ない
16. calibratorはtraining内time-based OOFだけ
17. raw/calibrated列を分離
18. EV計算列を明示
19. ALL_RUNNERS_FLATを算出
20. MARKET_ONLYを算出
21. row_removed_roiで分母を減らす
22. payout_zeroed_stress_roiで分母を維持
23. 2025/2026で候補選択しない
24. fold単位resume
25. source/config/feature hash保存
26. seed固定
27. 自動git操作なし
28. probability_calibratedにNaNが残っていないか（fold_2020の特例を除く）
29. FULL_2006の2006～2015行のmarket_logitが外挿によって異常値を取っていないか
30. 全候補の同一レースでmarket_logitが一致するか（WARMUPとBASEで確認）
31. 累積特徴飽和テスト: jockey_past_startsのp99がBASEの2倍以上になっていないか
32. target_place_paidの正例率が2006～2015と2016～2026で±5pp以内か
33. 各候補のprobability_rawの年別平均が±3pp以内で安定しているか

---

# 20. 停止条件

以下の場合は推測して続けない。

```text
2006～2015 DBを一意に特定できない
schema対応を一意に決められない
race_id / runner keyが衝突する
odds / payout単位が不明
人物・馬コードの連続性が確認できない
履歴builderが現在レース結果を含む
同日順序を安全に決められない
source DBを書き換える必要がある
既存Parquetを上書きする必要がある
calibratorがouter validation正解を見る
2026公式評価規則が不明
```

停止時は、確認できた事実・不明点・必要な追加情報だけを報告する。

---

# 21. 最終報告

日本語で以下を正確に報告する。

1. 使用したDBパス・hash・期間
2. 2006～2015と2016年以降のスキーマ差
3. 欠損・重複・コード体系の問題
4. オッズ・払戻単位
5. concept drift
6. 履歴生成の時系列安全性
7. 2016年時点の履歴不足改善
8. BASE / WARMUP / FULLの正確な定義
9. fold別学習期間
10. probability_raw性能
11. probability_calibrated性能
12. キャリブレーターのOOF構造
13. ALL_RUNNERS_FLAT ROI
14. RANDOM_MATCHED_COUNT ROI
15. MARKET_ONLY ROI
16. 各候補のraw/calibrated ROI
17. row_removed_roi
18. payout_zeroed_stress_roi
19. paired bootstrap
20. 2020～2024で固定した最終候補
21. 2025/2026固定診断
22. 2025+2026合算ROI
23. FI/SHAP
24. 新規学習fold数
25. 再利用fold数
26. 実行時間
27. テスト結果
28. 作成・変更ファイル
29. git status --short
30. git diff --stat

自動commit/pushは行わない。
