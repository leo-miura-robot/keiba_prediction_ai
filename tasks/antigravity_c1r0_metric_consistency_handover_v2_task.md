# Antigravity引き継ぎ・次タスク
## C1R0-300 評価整合性監査 v2

## 0. 目的

競馬予想AIの開発環境をAntigravityへ移行する。

この文書は、新しいエージェントが以下を理解し、安全に作業を再開するための引き継ぎ兼タスク仕様である。

- プロジェクトの目標
- ここまでのモデル改善
- 現在の基準モデル
- Phase 2とPhase 3で起きた評価値の矛盾
- 次に実行する保存済み予測の整合性監査
- Antigravityでの禁止操作と停止条件

今回、モデル再学習や新特徴量追加は行わない。

---

# 1. プロジェクトの目標と固定方針

第一段階の目標:

```text
単勝ROI 90%以上
複勝ROI 90%以上
```

現在は複勝確率モデルを優先している。

役割を分離する。

```text
確率モデル:
複勝圏に入る確率を正しく推定する

購入戦略:
確率とオッズから購入可否を決める
```

禁止方針:

- ROI直接学習禁止
- Ability/ANAモデルはまだ導入しない
- Rankerはまだ導入しない
- Kelly禁止
- 自動購入禁止
- 大規模Optuna禁止
- random split禁止
- 自動commit/push禁止

---

# 2. データと公式時系列分割

対象期間:

```text
2016年以降のみ
```

walk-forward:

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

固定診断:

```text
2025: test
2026: latest holdout
```

採否判断は2020～2024だけで行う。
2025/2026は仕様固定後の診断にのみ使い、結果を見て選択を変更しない。

DBより既存Parquet・保存済み予測・モデル成果物を優先する。

---

# 3. 旧C1からC1R0への流れ

## 3.1 旧C1

基本式:

```text
final_logit = market_logit + residual_raw
p_final = sigmoid(final_logit)
```

旧C1では`market_logit`をbaselineに使いながら、CatBoost入力にも以下が入っていた。

```text
Year
p_market
market_logit
```

問題:

- 市場情報の二重利用
- Yearによる年度の直接記憶
- 2025でresidualとEV>=1候補が急増
- finalモデル3000本が過学習した可能性

## 3.2 C1R0 pure market offset

市場情報をbaselineだけに限定した。

```text
market_logit
+
市場以外の競馬情報を使うCatBoost residual
```

CatBoostから除外:

```text
Year
p_market
market_logit
raw odds
人気
市場順位
結果
払戻
管理ID
```

残した主な情報:

```text
馬の近走
競馬場・芝ダート・距離・馬場適性
騎手・調教師の過去成績
枠
斤量
馬体重
レース条件
開催情報
```

---

# 4. tree count監査

旧finalモデル:

```text
3000 trees
```

比較:

```text
250 / 300 / 350 / 400 / 450
```

2020～2024の確率性能、残差、EV件数、EV-ROI Spearmanのバランスから300本を選択した。

基準:

```text
C1R0_pure_market_offset_fixed300
```

3000本から300本に減らすことで、2025/2026の残差膨張とEV候補急増を大幅に抑えた。

---

# 5. 特徴量監査 Phase 1

明確な時系列リークは未検出。
同日全行の特徴生成後に履歴を更新するため、同日未来レースを混ぜない設計。

## 5.1 人物コード

```text
KisyuCode
ChokyosiCode
```

高cardinalityな人物ID。

未知カテゴリ率:

```text
KisyuCode:
2025 4.63%
2026 8.03%

ChokyosiCode:
2025 3.25%
2026 5.94%
```

除外後、2020～2024のLogloss、Brier、ECE、residual p95が改善。

現在の基準候補:

```text
C1R0_fixed300_ablation_drop_person_codes
```

## 5.2 累積出走数

```text
trainer_past_starts
jockey_past_starts
```

`trainer_past_starts`とYearのSpearmanは約0.533。
年度代理性の可能性がある。

## 5.3 勝率系

```text
*_win_rate
*_ren_rate
*_top3_rate
*_place_paid_rate
```

平滑化なしの単純比率。

## 5.4 生タイム

```text
horse_last3_avg_time
horse_last5_avg_time
```

距離・競馬場・芝ダート・馬場補正なし。

## 5.5 馬体重

```text
BaTaijyu
```

結果リークではないが、運用時の取得時点確認が必要。

---

# 6. Phase 2

人物コード除外モデルを作業基準にした。

`MonthDay`監査:

- MMDD整数
- Yearとの相関は弱い
- Monthとの相関が高い
- 季節・開催時期の信号
- 単独除外でLogloss/Brier悪化

結論:

```text
MonthDayは維持
```

累積数2列を比較:

```text
raw
drop
log1p
train-p99 clip
train-p99 clip + log1p
```

Phase 2結果:

```text
raw Logloss              0.405716
clip_p99_log1p Logloss   0.405658
delta = clip - raw      -0.000058

raw Brier                0.130356
clip_p99_log1p Brier     0.130335
delta = clip - raw      -0.000021
```

単純集計ではclipがわずかに良かった。

---

# 7. Phase 3

差が小さいため、保存済み予測でpaired bootstrapを行った。

差:

```text
delta = clip_p99_log1p - raw
```

報告結果:

```text
race bootstrap Logloss:
point delta = +0.000551
95% CI = [+0.000406, +0.000701]

race bootstrap Brier:
point delta = +0.000155
95% CI = [+0.000104, +0.000208]
```

Phase 3ではrawが明確に良いという結論になった。

追加ablation:

```text
人物コード除外 + Kaiji/Nichiji/RaceNum除外
人物コード除外 + 生タイム2列除外
人物コード除外 + BaTaijyu除外
```

いずれも確率指標またはworst-year性能が悪化し不採用。

暫定基準:

```text
C1R0_fixed300_ablation_drop_person_codes
```

暫定仕様:

```text
除外:
KisyuCode
ChokyosiCode

維持:
raw trainer_past_starts
raw jockey_past_starts
MonthDay
Kaiji
Nichiji
RaceNum
horse_last3_avg_time
horse_last5_avg_time
BaTaijyu
```

---

# 8. 現在の矛盾

Phase 2:

```text
Logloss delta = -0.000058
Brier delta   = -0.000021
```

Phase 3:

```text
Logloss delta = +0.000551
Brier delta   = +0.000155
```

同じ予測、対象行、確率列、指標定義なら、bootstrapのpoint estimateと直接集計差は一致する必要がある。

考えられる原因:

```text
different model artifact
different prediction file
different probability column
different calibration
different row filter
different entry set
merge duplication
runner-weighted vs race-weighted
runner-weighted vs race-date-weighted
metric function difference
ECE bin difference
stale artifact
```

---

# 9. 次のタスク

保存済み予測だけを使い、次を確定する。

1. Phase 2とPhase 3が参照したraw/clip予測
2. entry/race集合
3. target列・確率列
4. calibration
5. row filter
6. runner/race/race-date weighting
7. bootstrap実装
8. 数値逆転の根本原因
9. 統一定義によるraw/clipの正式判定

モデル再学習は禁止。

---

# 10. 最初に読むファイル

```text
tasks/place_market_offset_catboost_c1r0_metric_consistency_audit_v1_task.md

docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md

scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py

config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml
config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml

outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
```

必要時:

```text
outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/
outputs/place_market_offset_catboost_c1r0_tree_count_v1/
```

---

# 11. Antigravity安全ガードレール

## 11.1 禁止操作

```text
DB読込・更新
model.fit
CatBoost学習
feature dataset再作成
元Parquet変更
既存モデル・outputs削除
既存成果物上書き
random split
2025/2026による採否変更
git add
git commit
git push
git reset --hard
git clean
rm -rf
Remove-Item -Recurse
ドライブ直下への操作
```

## 11.2 書き込み許可範囲

```text
config/place_market_offset_catboost_c1r0_metric_consistency_audit_v2.yaml
scripts/audit_c1r0_metric_consistency_v2.py
tests/test_c1r0_metric_consistency_v2.py
docs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2_results.md
outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2/
```

既存ファイル変更ではなく新規v2ファイルを優先する。

## 11.3 停止条件

以下では推測して進まず報告する。

```text
正式entry key不明
正式race key不明
raw/clipの行集合不一致
予測参照先を一意に決められない
calibrated列を判別できない
mergeが1対1にならない
成果物破損
再学習が必要
```

---

# Stage 1. 状態確認

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

監査出力先の途中成果物も確認する。

```text
outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v1/
outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2/
```

---

# Stage 2. 成果物マッピング

raw:

```text
C1R0_fixed300_ablation_drop_person_codes
```

clip:

```text
C1R0_fixed300_drop_person_codes_starts_clip_p99_log1p
```

特定する列:

```text
phase
model_role
model_key
model_path
prediction_path
fold
evaluation_year
tree_count
feature_hash
config_hash
transform_hash
probability_column
target_column
calibration_applied
row_filter
```

出力:

```text
phase2_phase3_artifact_mapping.csv
```

---

# Stage 3. alignment

2020～2024で確認:

```text
row count
unique entry count
duplicate entry count
unique race count
missing/extra entry
missing/extra race
```

正式entry keyで1対1mergeする。

```text
merged_rows
left_only
right_only
many_to_many
duplicate_after_merge
target_mismatch
race_key_mismatch
```

出力:

```text
prediction_alignment_check.csv
prediction_alignment_by_year.csv
```

---

# Stage 4. metric定義監査

Phase 2とPhase 3について特定:

```text
target column
probability column
calibrated or uncalibrated
fold/final calibration
row filter
metric function
ECE bin count
ECE bin strategy
runner/race/race-date weighting
```

出力:

```text
metric_definition_audit.csv
```

---

# Stage 5. 同一DataFrameで直接再計算

## runner-weighted主評価

全runnerを等重みで扱う。

```text
raw Logloss
clip Logloss
delta = clip - raw

raw Brier
clip Brier
delta = clip - raw

raw ECE
clip ECE
delta = clip - raw
```

2020～2024合算と年度別を出す。

## 補助評価

別々に出す。

```text
race-weighted
race-date-weighted
```

出力:

```text
direct_metric_recalculation.csv
direct_metric_recalculation_by_year.csv
```

---

# Stage 6. paired bootstrap監査

正しい主解析:

1. raceを復元抽出
2. 同じrace sampleをraw/clipへ適用
3. 選択raceの全runner行を結合
4. runner-weighted metricを計算
5. `delta = clip - raw`

設定:

```text
n_bootstrap = 5000
seed = fixed
confidence_level = 0.95
sampling_unit = race
metric_weighting = runner-weighted
ECE bins = fixed
```

raceごとのmetricを均等平均する場合は、race-weighted補助解析として分ける。

検証:

```text
direct runner-weighted delta
bootstrap point estimate
```

同一定義なら一致させる。

出力:

```text
bootstrap_implementation_audit.csv
bootstrap_point_estimate_consistency.csv
paired_bootstrap_summary_v2.csv
```

---

# Stage 7. 根本原因と最終判定

原因を分類:

```text
different_model_artifact
different_prediction_file
different_probability_column
different_calibration
different_row_filter
different_entry_set
merge_duplication
runner_weighted_vs_race_weighted
runner_weighted_vs_day_weighted
metric_function_difference
ECE_bin_difference
old_stale_artifact
other
```

出力:

```text
phase2_phase3_inconsistency_root_cause.json
```

正式主評価:

```text
runner-weighted Logloss
runner-weighted Brier
fixed-bin ECE
```

補助:

```text
race-weighted
race-date-weighted
```

採否:

```text
clipが主評価で明確に改善しCIも改善側:
clip採用候補

rawが明確に改善:
raw維持

差が極小またはCIが0をまたぐ:
単純なraw維持

重み付けで逆転:
runner-weightedを正式主評価にする
```

2025/2026は採否に使わない。

出力:

```text
raw_vs_clip_final_consistent_comparison.csv
raw_vs_clip_final_decision.json
```

---

# 12. 出力先と実装ファイル

出力:

```text
outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2/
```

実装:

```text
config/place_market_offset_catboost_c1r0_metric_consistency_audit_v2.yaml
scripts/audit_c1r0_metric_consistency_v2.py
tests/test_c1r0_metric_consistency_v2.py
docs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2_results.md
```

---

# 13. 必須成果物

```text
phase2_phase3_artifact_mapping.csv
prediction_alignment_check.csv
prediction_alignment_by_year.csv
metric_definition_audit.csv
direct_metric_recalculation.csv
direct_metric_recalculation_by_year.csv
bootstrap_implementation_audit.csv
bootstrap_point_estimate_consistency.csv
paired_bootstrap_summary_v2.csv
phase2_phase3_inconsistency_root_cause.json
raw_vs_clip_final_consistent_comparison.csv
raw_vs_clip_final_decision.json
manifest.json
audit_report.md
```

---

# 14. 必須テスト

1. DBへ接続しない
2. `fit` / `train`を呼ばない
3. feature datasetを再作成しない
4. 元Parquetを変更しない
5. entry/race集合を検証
6. mergeが1対1
7. target一致
8. 同一確率列
9. runner-weighted直接差を計算
10. race-weightedを別計算
11. race-date-weightedを別計算
12. bootstrapがpaired
13. 同じrace sampleを両モデルへ適用
14. point estimateと直接差が一致
15. ECE bin固定
16. seed固定
17. 2025/2026を採否に使わない
18. 既存出力を上書きしない
19. git変更操作をしない

---

# 15. 最終報告

日本語で報告:

1. これまでの流れの理解
2. Phase 2/3が参照したraw/clip成果物
3. 同一モデル・予測だったか
4. entry/race集合
5. probability列
6. calibration
7. row filter
8. runner-weighted直接差
9. race-weighted直接差
10. race-date-weighted直接差
11. bootstrap point estimate整合性
12. 数値逆転の根本原因
13. 正式評価定義
14. raw/clip最終判定
15. 勝率平滑化へ進めるか
16. 作成ファイル
17. 再利用成果物
18. テスト結果
19. 実行時間
20. `git status --short`
21. `git diff --stat`

自動commit/pushは行わない。
