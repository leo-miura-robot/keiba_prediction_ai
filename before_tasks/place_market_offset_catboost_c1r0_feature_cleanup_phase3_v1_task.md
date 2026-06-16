# C1R0-300 Feature Cleanup Phase 3 Task v1
## Paired Bootstrap Validation and Targeted Combination Ablation

## 0. 目的

現在の固定300本市場残差モデルについて、次の2点を検証する。

1. `trainer_past_starts` / `jockey_past_starts` の
   `raw` と `train-p99 clip + log1p` の差が、統計的に再現性のある改善か確認する
2. 人物コード除外モデルを基準に、追加の既存特徴除外を少数だけ比較する

今回の目的は、新しい特徴量を追加することではなく、
既存特徴量構成を次の勝率平滑化フェーズへ進められる状態まで固定することである。

---

## 1. 現在の候補

作業基準モデル:

```text
C1R0_fixed300_ablation_drop_person_codes
```

累積数変換候補:

```text
C1R0_fixed300_drop_person_codes_starts_clip_p99_log1p
```

基本構造:

```text
final_logit = market_logit + catboost_residual
p_final = sigmoid(final_logit)
```

固定事項:

- tree count = 300
- `market_logit`はPool baseline専用
- `KisyuCode` / `ChokyosiCode`は除外
- `Year` / `p_market` / `market_logit` / 市場特徴はCatBoostへ入れない
- 2020～2024だけで採否判断
- 2025/2026は固定診断のみ

---

## 2. 最初に読むファイル

```text
tasks/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_task.md
tasks/place_market_offset_catboost_c1r0_feature_cleanup_v1_task.md
tasks/place_market_offset_catboost_c1r0_tree_count_v1_task.md

docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_v1_results.md
docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md

config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml
config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml

scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_v1.py

outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/

models/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
models/place_market_offset_catboost_c1r0_feature_cleanup_v1/
```

既存の予測、モデル、manifest、変換パラメータ、比較CSVを最大限再利用する。

---

## 3. 絶対条件

- 2016年以降のみ使用
- random split禁止
- 2020～2024だけでモデル採否を判断
- 2025/2026を選択に使わない
- DB読込禁止
- feature dataset再作成禁止
- 元Parquet変更禁止
- tree countは300固定
- baseline方式変更禁止
- CatBoost hyperparameter変更禁止
- calibration方式変更禁止
- 新特徴量追加禁止
- ROI直接学習禁止
- Ability/ANA、Ranker、Kelly、大規模Optuna禁止
- 自動購入、自動commit/push禁止
- 既存成果物を上書きしない

今回許可する再学習は、指定した追加ablationモデルだけ。

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

# Stage 1: raw vs clip_p99_log1p の統計的検証

## 5. 比較対象

以下の既存予測を再利用する。

```text
RAW:
C1R0_fixed300_ablation_drop_person_codes

TRANSFORMED:
C1R0_fixed300_drop_person_codes_starts_clip_p99_log1p
```

再学習しない。

## 6. 整合性チェック

以下を確認する。

- 2020～2024の全fold予測が存在
- 同一entry集合
- 同一race集合
- 同一baseline
- 同一target
- 同一オッズ・払戻列
- 重複entryなし
- 欠損率一致
- feature/config/model hash
- 変換閾値が学習期間のみから計算されている

出力:

```text
paired_prediction_artifact_check.csv
```

不一致がある場合は、その原因を報告し、勝手に再学習しない。

---

## 7. paired bootstrap単位

主解析はレース単位のpaired bootstrapとする。

理由:

- 同一レース内の出走馬は独立ではない
- rawとtransformedは同じレース・同じ馬に対する予測
- 対応のある差として評価する必要がある

再標本化単位:

```text
race key
```

race keyは既存の一意レース識別列から作る。
推測で不安定なキーを作らず、既存コード・manifestに従う。

補助解析として、可能なら開催日単位bootstrapも実施する。

---

## 8. bootstrap設定

config化する。

推奨:

```text
n_bootstrap = 5000
seed = 固定
confidence_level = 0.95
```

計算コストが高い場合は、最初に1000回で検証し、
正式出力だけ5000回とする。

モデル学習は不要。

---

## 9. 比較指標

各bootstrap標本で、次の差を計算する。

差の定義:

```text
delta = transformed - raw
```

低いほど良い指標では、負ならtransformed改善。

### 主指標

```text
delta_logloss
delta_brier
delta_ece
```

### 補助指標

```text
delta_calibration_slope_abs_error
delta_calibration_intercept_abs_error
delta_ev_ge_1_count
delta_ev_count_cv
delta_ev_roi_spearman
```

ECEは非加法的で不安定になりやすいため、
bin定義を固定し、解釈上の注意を記録する。

ROIはbootstrap主採用基準にしない。

---

## 10. bootstrap出力

```text
paired_bootstrap_summary.csv
paired_bootstrap_samples.parquet
paired_bootstrap_by_year.csv
paired_bootstrap_day_level_summary.csv
```

`paired_bootstrap_summary.csv`の最低限の列:

```text
metric
point_estimate_delta
bootstrap_mean_delta
ci_lower
ci_upper
prob_delta_below_zero
prob_delta_above_zero
better_model
decision
```

年度別にも2020～2024を個別に出す。

---

## 11. 採否ルール

`clip_p99_log1p`を採用する条件:

### 原則採用

- Logloss差の95% CIが0未満
- Brier差の95% CIが0未満
- 5年間の大半で方向が改善
- residual tail / EV件数安定性が悪化しない

### 保留またはraw維持

- Logloss/BrierのCIが0をまたぐ
- 改善方向が年度で不安定
- ECEやEV順位性の悪化が大きい
- 実質差が極めて小さい

差が統計的・実務的に曖昧な場合は、より単純なrawを維持する。

結果を以下へ保存する。

```text
cumulative_starts_transform_decision.json
```

---

# Stage 2: 人物コード除外後の追加ablation

## 12. 作業基準

Stage 1の判断後、次のどちらかを作業基準にする。

```text
drop_person_codes + raw starts
```

または

```text
drop_person_codes + clip_p99_log1p starts
```

2020～2024だけで決定する。

---

## 13. 追加ablation候補

作業基準から次を個別に除外する。

### B1: meeting admin除外

```text
Kaiji
Nichiji
RaceNum
```

モデル名例:

```text
C1R0_300_cleanbase_no_meeting_admin
```

### B2: unadjusted raw time除外

```text
horse_last3_avg_time
horse_last5_avg_time
```

モデル名例:

```text
C1R0_300_cleanbase_no_raw_time
```

### B3: raw body weight除外

```text
BaTaijyu
```

`horse_body_weight_diff_last`は残す。

モデル名例:

```text
C1R0_300_cleanbase_no_raw_body_weight
```

`MonthDay`は前タスクで不採用だったため、今回除外しない。

---

## 14. 計算コスト制御

- 既存成果物が同一基準条件なら再利用
- 基準条件が異なる場合のみ新規学習
- 最大3候補 × 5fold
- fold単位resume
- feature hash / config hash / transform hash確認
- 不足foldだけ再学習
- 2020～2024比較の段階ではFI/SHAP/Bootstrapを全候補へ実施しない
- 選択候補だけ詳細診断する

GPUメモリ制約:

```text
gpu_ram_part = 0.75
```

これは実行環境制約であり、モデル選択hyperparameterではないことをmanifestに記録する。

---

## 15. walk-forward

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

全モデルtree countは300固定。

---

## 16. 評価指標

### 主指標

- Logloss
- Brier
- ECE
- calibration slope/intercept
- worst-year Logloss
- worst-year Brier

### 残差

- residual mean/std
- abs residual p90/p95/p99
- residual std CV
- tailの年度CV

### EV

- EV>=1件数
- EV件数CV
- market-only crossing
- EV-ROI Spearman

### ROI補助

- EV>=1 ROI
- top1/top3/top5/top10除外
- bootstrap CIは選択候補だけ

---

## 17. 採用ルール

優先順位:

1. Logloss
2. Brier
3. calibration
4. residual tail
5. worst-year性能
6. EV件数の過度な膨張を避ける
7. EV件数年度安定性
8. EV-ROI Spearman
9. 特徴構成の単純さ
10. ROIは補助

基準との差が極小なら、より単純で運用時に安定する特徴構成を優先してよい。

ただし、確率指標が明確に悪化する削除は採用しない。

---

# Stage 3: 必要最小限の統合

## 18. 統合条件

追加ablationのうち、2020～2024で個別に有効だった変更だけを統合する。

候補名:

```text
C1R0_300_feature_clean_phase3_v1
```

全組合せ探索は禁止。

次のように最大1つの統合モデルだけ作る。

```text
作業基準
+ 有効だったB1
+ 有効だったB2
+ 有効だったB3
```

個別に悪化した変更は統合しない。

---

# Stage 4: 2025/2026固定診断

## 19. 対象

2020～2024で仕様を固定した後にのみ実施する。

- 作業基準
- 最良単独ablation
- 統合モデルがあれば統合モデル

2025/2026を見て採否を変更しない。

## 20. 診断項目

- Logloss
- Brier
- ECE
- calibration
- residual mean/std
- abs residual p90/p95/p99
- EV>=1件数
- crossing
- EV-ROI Spearman
- ROI
- top1/top3/top5/top10除外

選択モデルだけFI/SHAPを出す。

---

## 21. 出力先

```text
outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
models/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
```

既存成果物を上書きしない。

---

## 22. 実装候補

```text
config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py
tests/test_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md
```

---

## 23. 必須成果物

```text
paired_prediction_artifact_check.csv
paired_bootstrap_summary.csv
paired_bootstrap_samples.parquet
paired_bootstrap_by_year.csv
paired_bootstrap_day_level_summary.csv
cumulative_starts_transform_decision.json

additional_ablation_artifact_check.csv
additional_ablation_by_fold.csv
additional_ablation_2020_2024.csv
additional_ablation_residual_stability.csv
additional_ablation_ev_stability.csv
additional_ablation_roi_diagnostic.csv

selected_feature_set_phase3.json
phase3_2025_2026_diagnostic.csv
selected_model_feature_importance.csv
selected_model_shap.csv
selected_model_shap_additivity.csv
manifest.json
```

---

## 24. 必須テスト

1. DBへ接続しない
2. feature datasetを再作成しない
3. 元Parquetを変更しない
4. random splitを使わない
5. 2015年以前を含めない
6. tree countが300
7. baselineが正しい
8. bootstrapがrace単位pairedである
9. raw/transformedの同一race・entry対応
10. bootstrap seed固定
11. 2025/2026を選択に使わない
12. B1/B2/B3で指定列だけが除外される
13. MonthDayを今回変更しない
14. Year/p_market/market_logit/人物コードが再混入しない
15. clip閾値が学習期間のみから計算される
16. fold単位resumeが機能する
17. feature/config/transform hashを記録
18. `final_logit = market_logit + residual_raw`
19. 既存出力を上書きしない

---

## 25. 最終報告

日本語で以下を報告する。

1. rawとclip_p99_log1pのpoint estimate差
2. race単位paired bootstrapの95% CI
3. 開催日単位bootstrap結果
4. 年度別bootstrap結果
5. clip_p99_log1pを採用したか
6. 採否理由
7. 作業基準モデル
8. B1/B2/B3の比較
9. 有効だった追加削除
10. 悪化した追加削除
11. 統合モデルの有無
12. 2020～2024で固定した最終特徴構成
13. 2025/2026固定診断
14. 選択モデルのFI/SHAP
15. 次に勝率平滑化へ進んでよいか
16. 再利用した成果物
17. 新規学習モデル数
18. 実行時間
19. テスト結果
20. 作成・変更ファイル
21. `git status --short`
22. `git diff --stat`

自動commit/pushは行わない。
