# C1R0-300 Metric Consistency Audit for Gemini v1

## 0. 目的

Phase 2とPhase 3で、同じ`raw`モデルと`clip_p99_log1p`モデルを比較しているにもかかわらず、Logloss / Brierの差の符号が逆転している。

このタスクでは、既存予測だけを使って評価定義・参照成果物・対象行・確率列の不一致を監査し、どちらの結果が正しいかを確定する。

再学習や新しい特徴量実験は行わない。

---

## 1. 現在確認されている矛盾

### Phase 2

```text
raw Logloss                0.405716
clip_p99_log1p Logloss     0.405658
delta = clip - raw        -0.000058

raw Brier                  0.130356
clip_p99_log1p Brier       0.130335
delta = clip - raw        -0.000021
```

Phase 2では`clip_p99_log1p`がわずかに良い。

### Phase 3

```text
race paired bootstrap Logloss delta
clip - raw = +0.000551
95% CI = [+0.000406, +0.000701]

race paired bootstrap Brier delta
clip - raw = +0.000155
95% CI = [+0.000104, +0.000208]
```

Phase 3では`raw`が明確に良い。

同一予測・同一対象行・同一確率列・同一指標定義なら、bootstrap前のpoint estimateと直接集計差は一致する必要がある。

---

## 2. 最優先で読むファイル

```text
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md

scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py

config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml
config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml

outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
```

必要に応じて以下も確認する。

```text
outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/
outputs/place_market_offset_catboost_c1r0_tree_count_v1/
```

---

## 3. 絶対条件

- DBを読まない
- feature datasetを再作成しない
- 元Parquetを変更しない
- モデル再学習をしない
- 新特徴量を追加しない
- random splitを使わない
- 2025/2026を選択に使わない
- calibration方式を変更しない
- 既存成果物を上書きしない
- 自動commit/pushをしない
- 不一致原因が解消するまで勝率平滑化へ進まない

今回扱うのは保存済み予測の評価監査だけ。

---

## 4. 作業開始時

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

既存差分を勝手に戻さない。

---

# Stage 1: 参照成果物の特定

## 5. rawとclipモデルを明示する

Phase 2とPhase 3で参照しているモデル・予測ファイルを特定する。

対象:

```text
raw:
C1R0_fixed300_ablation_drop_person_codes

clip:
C1R0_fixed300_drop_person_codes_starts_clip_p99_log1p
```

最低限以下を一覧化する。

```text
phase
model_role
model_key
model_path
prediction_path
feature_hash
config_hash
transform_hash
tree_count
train_period
evaluation_period
probability_column
calibration_applied
row_filter
```

人物コードありモデルや旧selectedモデルが混ざっていないか確認する。

出力:

```text
phase2_phase3_artifact_mapping.csv
```

---

# Stage 2: 対象行とキーの一致確認

## 6. entry集合

rawとclipについて、2020～2024で以下を確認する。

```text
row_count
unique_entry_count
duplicate_entry_count
missing_entry_count_vs_other
extra_entry_count_vs_other
```

entry keyは既存コードで正式に使われている列を使用する。

## 7. race集合

```text
unique_race_count
duplicate_race_entry_pairs
missing_races_vs_other
extra_races_vs_other
```

race keyも既存コード・manifestに従う。

## 8. 年度別行数

2020～2024で年度別にraw/clipの行数・race数を比較する。

## 9. merge後の検証

rawとclipをentry keyで1対1結合し、以下を確認する。

```text
merged_rows
left_only
right_only
many_to_many_detected
duplicate_after_merge
```

出力:

```text
prediction_alignment_check.csv
prediction_alignment_by_year.csv
```

---

# Stage 3: 評価列の一致確認

## 10. probability column

Phase 2とPhase 3が何を使っているか確認する。

候補:

```text
p_final
p_calibrated
p_uncalibrated
prediction
probability
```

各スクリプトで実際に参照される列名とコード位置を記録する。

## 11. target column

同じ目的変数を使っているか確認する。

## 12. calibration

以下を確認する。

```text
Phase 2でcalibration済みか
Phase 3でcalibration済みか
fold別calibrationか
final calibrationか
同じcalibratorか
```

## 13. row filter

以下のどれを評価しているか確認する。

```text
全出走馬
eligible行のみ
odds有効行のみ
EV計算可能行のみ
欠損除外後
特定オッズ帯のみ
```

出力:

```text
metric_definition_audit.csv
```

---

# Stage 4: 同一DataFrameでの直接再計算

## 14. runner-weighted

rawとclipの共通entryだけを使い、全出走馬を等重みとした通常の指標を直接計算する。

```text
raw_logloss
clip_logloss
delta_logloss = clip - raw

raw_brier
clip_brier
delta_brier = clip - raw

raw_ece
clip_ece
delta_ece = clip - raw
```

2020～2024合算と年度別を出す。

## 15. race-weighted

各race内で指標を計算し、raceごとに均等平均する別指標も出す。

```text
race_weighted_logloss
race_weighted_brier
race_weighted_ece
```

runner-weightedとrace-weightedを混同しない。

## 16. race-date-weighted

Phase 3で開催日単位評価を行っている場合、開催日単位均等平均も別に出す。

出力:

```text
direct_metric_recalculation.csv
direct_metric_recalculation_by_year.csv
```

---

# Stage 5: bootstrap実装の監査

## 17. race paired bootstrap

正しい実装は以下を原則とする。

1. race keyを復元抽出
2. 選ばれたraceの全runner行を結合
3. 結合後の全runner行でLogloss/Brierを計算
4. 同じ抽出raceをraw/clipの両方へ適用
5. `delta = clip - raw`

レースごとのLoglossを先に計算して均等平均する場合は、それを`race-weighted metric`として明確に分離する。

## 18. point estimate一致

以下を検証する。

```text
bootstrap point estimate
direct runner-weighted delta
```

同一定義なら浮動小数点誤差程度で一致する必要がある。

許容差例:

```text
abs(diff) <= 1e-12
```

ECEはbinning実装の都合で許容差を別設定してよいが、同じ関数・同じbin定義を使う。

## 19. bootstrap設定

```text
seed
n_bootstrap
sampling_unit
metric_weighting
ECE bins
```

を明示する。

出力:

```text
bootstrap_implementation_audit.csv
bootstrap_point_estimate_consistency.csv
```

---

# Stage 6: 原因分類

## 20. 不一致原因を分類する

以下から該当するものを判定する。

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

複数要因の場合は複数記録する。

出力:

```text
phase2_phase3_inconsistency_root_cause.json
```

---

# Stage 7: 正式結論

## 21. 正式評価定義

これまでのプロジェクトの主評価がrunner単位なら、正式指標を以下に統一する。

```text
runner-weighted Logloss
runner-weighted Brier
固定bin定義のECE
```

race-weighted / race-date-weightedは補助指標とする。

## 22. raw vs clipの再判定

統一定義でrawとclipを再比較する。

出力:

```text
raw_vs_clip_final_consistent_comparison.csv
raw_vs_clip_final_decision.json
```

採否:

- 統一runner-weighted評価でclipが明確に改善 → clip採用候補
- 統一runner-weighted評価でrawが改善 → raw維持
- 差が極小・CIが0をまたぐ → 単純なraw維持
- 評価定義によって逆転 → 主評価をrunner-weighted、補助をrace-weightedとして明記

2025/2026はこの採否に使わない。

---

## 23. 出力先

```text
outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v1/
```

既存成果物を上書きしない。

---

## 24. 実装候補

```text
config/place_market_offset_catboost_c1r0_metric_consistency_audit_v1.yaml
scripts/audit_c1r0_metric_consistency_v1.py
tests/test_c1r0_metric_consistency_v1.py
docs/place_market_offset_catboost_c1r0_metric_consistency_audit_v1_results.md
```

---

## 25. 必須成果物

```text
phase2_phase3_artifact_mapping.csv
prediction_alignment_check.csv
prediction_alignment_by_year.csv
metric_definition_audit.csv
direct_metric_recalculation.csv
direct_metric_recalculation_by_year.csv
bootstrap_implementation_audit.csv
bootstrap_point_estimate_consistency.csv
phase2_phase3_inconsistency_root_cause.json
raw_vs_clip_final_consistent_comparison.csv
raw_vs_clip_final_decision.json
manifest.json
audit_report.md
```

---

## 26. 必須テスト

1. DBへ接続しない
2. model.fitを呼ばない
3. feature datasetを再作成しない
4. 元Parquetを変更しない
5. raw/clipのentry集合一致を検証
6. mergeが1対1
7. 同一targetを使用
8. 同一確率列を使用
9. runner-weighted直接差を再計算
10. race-weightedを別指標として計算
11. bootstrapがpaired
12. 同じrace sampleをraw/clipへ適用
13. bootstrap point estimateと直接差が一致
14. ECE bin定義固定
15. seed固定
16. 2025/2026を採否に使わない
17. 既存出力を上書きしない

---

## 27. 最終報告

日本語で以下を報告する。

1. Phase 2が参照したraw/clip成果物
2. Phase 3が参照したraw/clip成果物
3. モデル・予測ファイルが同一だったか
4. entry/race集合が一致したか
5. probability列が同一だったか
6. calibrationが同一だったか
7. row filterが同一だったか
8. runner-weighted直接差
9. race-weighted直接差
10. race-date-weighted直接差
11. bootstrap point estimateとの一致
12. 不一致の根本原因
13. 正式評価定義
14. rawとclipの最終採否
15. 次に勝率平滑化へ進んでよいか
16. 作成・変更ファイル
17. テスト結果
18. 実行時間
19. `git status --short`
20. `git diff --stat`

自動commit/pushは行わない。
