# C1R0-300 Feature Cleanup Phase 2 Task v1
## Existing Ablation Review, MonthDay Audit, and Cumulative-Starts Preprocessing

## 0. 目的

現在の固定300本市場残差モデルについて、既存の特徴量整理を次の段階へ進める。

今回の目的は次の3点である。

1. 既に実行済みの5つのablation結果を、同一基準・同一表で完全比較する
2. `MonthDay`の意味と年度代理性を監査し、単独除外を比較する
3. `trainer_past_starts`と`jockey_past_starts`について、生の累積値・削除・`log1p`・学習期間p99クリップ・`log1p + p99クリップ`を比較する

確率性能と年度外安定性を優先し、ROIは補助指標として扱う。

---

## 1. 現在の前提

基準モデル:

```text
C1R0_pure_market_offset_fixed300
```

現在の次期基準候補:

```text
C1R0_fixed300_ablation_drop_person_codes
```

基本式:

```text
final_logit = market_logit + catboost_residual
p_final = sigmoid(final_logit)
```

固定事項:

- tree count: 300
- `market_logit`はPool baseline専用
- `Year`、`p_market`、`market_logit`、raw odds、人気、市場派生列はCatBoost特徴へ入れない
- 2020～2024だけでモデル選択
- 2025/2026は仕様固定後の診断のみ

---

## 2. 最初に読むファイル

以下を最初から最後まで確認する。

```text
tasks/place_market_offset_catboost_c1r0_feature_cleanup_v1_task.md
tasks/place_market_offset_catboost_c1r0_tree_count_v1_task.md

docs/place_market_offset_catboost_c1r0_feature_cleanup_v1_results.md
docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md
docs/place_market_offset_catboost_c1r0_v1_results.md

config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml
config/place_market_offset_catboost_c1r0_tree_count_v1.yaml

scripts/audit_c1r0_feature_quality_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_v1.py
scripts/run_place_market_offset_catboost_c1r0_tree_count_v1.py

outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/
outputs/place_market_offset_catboost_c1r0_tree_count_v1/

models/place_market_offset_catboost_c1r0_feature_cleanup_v1/
models/place_market_offset_catboost_c1r0_tree_count_v1/
```

既存モデル・予測・比較CSV・FI・SHAP・manifestを最大限再利用する。

---

## 3. 絶対条件

- 2016年以降のみ使用
- random split禁止
- 2020～2024だけで採否・変換方式を決定
- 2025/2026を特徴量選択や変換方式選択に使わない
- DB読込禁止
- feature dataset再作成禁止
- 元Parquetを変更しない
- tree countは300固定
- baseline方式を変更しない
- tree count以外のCatBoost hyperparameterを変更しない
- calibration方式を変更しない
- ROI直接学習禁止
- Ability/ANA、Ranker、Kelly、大規模Optuna禁止
- 自動購入禁止
- 自動commit/push禁止
- 既存成果物を上書きしない

今回許可する前処理は、モデル入力時に行う次の処理のみ。

- `trainer_past_starts`と`jockey_past_starts`の削除
- `log1p`変換
- 学習期間だけで求めたp99による上限クリップ
- p99クリップ後の`log1p`
- `MonthDay`の除外

新しい外部データや新しい競馬特徴は追加しない。

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

# Stage 1: 既存5候補の完全比較

## 5. 比較対象

以下の既存成果物を再利用し、再学習しない。

```text
base_fixed300
drop_person_codes
drop_global_cumulative_starts
drop_raw_body_weight
drop_unadjusted_raw_time
drop_meeting_admin
```

除外内容:

```text
drop_person_codes:
  KisyuCode
  ChokyosiCode

drop_global_cumulative_starts:
  jockey_past_starts
  trainer_past_starts

drop_raw_body_weight:
  BaTaijyu

drop_unadjusted_raw_time:
  horse_last3_avg_time
  horse_last5_avg_time

drop_meeting_admin:
  Kaiji
  Nichiji
  RaceNum
```

## 6. 整合性チェック

再利用前に以下を確認する。

- モデルファイル存在
- tree count = 300
- feature set hash
- config hash
- fold別予測行数
- baseline列整合
- 2020～2024の全foldが存在
- 予測のentry key重複
- 既存manifestとの一致

出力:

```text
existing_ablation_artifact_check.csv
```

不整合がある候補だけ再学習してよい。全候補の無条件再学習は禁止。

## 7. 完全比較表

各候補について以下を同一表にまとめる。

### 確率指標

- combined Logloss
- combined Brier
- combined ECE
- calibration slope
- calibration intercept
- 年度別Logloss/Brier/ECE
- worst-year Logloss
- worst-year Brier

### 残差

- residual mean
- residual std
- residual std CV
- abs residual p90
- abs residual p95
- abs residual p99
- p95の年度CV
- p99の年度CV

### EV

- EV>=1合計件数
- EV>=1年度別件数
- EV件数CV
- market-onlyからの上抜け件数
- market-onlyからの下抜け件数
- EV-ROI Spearman

### ROI補助

- EV>=1 ROI
- top1/top3/top5/top10払戻除外
- bootstrap 95% CI

出力:

```text
existing_five_ablation_full_comparison.csv
existing_five_ablation_by_year.csv
existing_five_ablation_decision.md
```

## 8. 作業基準モデルの確定

2020～2024だけを使い、次の優先順位で作業基準を確定する。

1. Logloss
2. Brier
3. calibration
4. abs residual p95/p99
5. residual年度安定性
6. EV件数の過度な膨張を避ける
7. EV件数年度安定性
8. EV-ROI Spearman
9. ROIは補助
10. 特徴構成の単純さと未知年度への一般化

`drop_person_codes`が引き続き妥当なら、以後の実験の作業基準とする。

ただし既存5候補の完全比較で別候補が明確に優位なら、その理由を記録して作業基準を変更してよい。変更判断は2020～2024だけで行う。

出力:

```text
working_base_model.json
```

---

# Stage 2: MonthDay監査と単独ablation

## 9. MonthDay生成監査

`MonthDay`について、生成コード・型・値域・利用方法を確認する。

最低限確認すること:

- `MMDD`整数か
- day-of-yearか
- 文字列カテゴリか
- 数値特徴かカテゴリ特徴か
- 欠損率
- unique count
- 年度別分布
- `Month`との重複
- `Kaiji`、`Nichiji`との相関
- YearとのSpearman相関
- 2024→2025の分布変化
- SHAP/PVC/LFC順位
- 12月末と1月初めが不連続になる表現か
- 季節情報なのか、開催順序・日付記憶なのか

出力:

```text
monthday_feature_audit.csv
monthday_audit.md
```

## 10. MonthDay単独除外

作業基準モデルから`MonthDay`だけを除外する。

モデル名例:

```text
C1R0_300_working_base_no_monthday
```

比較:

- 作業基準
- MonthDay単独除外

2020～2024だけで評価する。

今回は周期変換や季節カテゴリ追加は行わない。まず生の`MonthDay`を残すべきかだけ確認する。

---

# Stage 3: 累積出走数の前処理比較

## 11. 対象列

今回の変換対象は次の2列だけに限定する。

```text
trainer_past_starts
jockey_past_starts
```

条件別出走数:

```text
horse_surface_past_starts
horse_jyo_past_starts
horse_dist_band_past_starts
horse_baba_past_starts
jockey_jyo_past_starts
jockey_dist_band_past_starts
horse_jockey_past_starts
```

は今回変更しない。

理由:

- 条件別率の信頼度を表す役割がある
- 一度にすべて変更すると原因が分からなくなる

## 12. 比較候補

作業基準モデルを起点に次を比較する。

### S0: raw

```text
生のtrainer_past_starts
生のjockey_past_starts
```

作業基準モデルを再利用する。

### S1: drop

```text
trainer_past_startsを除外
jockey_past_startsを除外
```

既存成果物が作業基準と同一条件なら再利用する。人物コード除外との組合せが異なる場合は再利用しない。

### S2: log1p

モデル入力時にだけ次を適用する。

```text
trainer_past_starts = log1p(trainer_past_starts)
jockey_past_starts = log1p(jockey_past_starts)
```

元Parquetは変更しない。

### S3: train-p99 clip

各foldの学習期間だけで列ごとのp99を算出する。

```text
x_transformed = min(x, train_p99)
```

- validation年の値で閾値を計算しない
- 2025 finalでは2016～2024から閾値を計算
- 2026 finalでは現行の学習期間ルールに従って閾値を計算
- 閾値をmanifestへ保存

### S4: train-p99 clip + log1p

各foldの学習期間だけでp99を求める。

```text
x_clipped = min(x, train_p99)
x_transformed = log1p(x_clipped)
```

`log1p`後のp99クリップではなく、原則としてクリップ後に`log1p`する。

モデル名例:

```text
C1R0_300_working_base_starts_drop
C1R0_300_working_base_starts_log1p
C1R0_300_working_base_starts_clip_p99
C1R0_300_working_base_starts_clip_p99_log1p
```

## 13. 前処理リーク防止

変換パラメータは必ずfold別に保存する。

出力:

```text
cumulative_starts_transform_params_by_fold.csv
```

列:

```text
model_key
fold_eval_year
train_start_year
train_end_year
feature
transform
train_p99
train_min
train_median
train_mean
train_max
```

validation/test/holdoutから閾値を計算しない。

## 14. 比較順序と計算コスト

まず2020～2024の5foldだけを実行する。

- S0は既存予測を再利用
- 再利用可能なS1は再利用
- S2/S3/S4だけ必要に応じて学習
- fold単位resume
- feature/config/transform hashで整合性確認
- 不足foldだけ学習
- 全候補のSHAP・bootstrapは行わない

Stage 3ではまず以下だけを比較する。

- Logloss
- Brier
- ECE
- residual安定性
- EV件数安定性
- EV-ROI Spearman

詳細FI/SHAP、bootstrap、2025/2026 final学習は、選ばれた変換方式だけに限定する。

---

# Stage 4: 選択と必要最小限の統合

## 15. 選択ルール

2020～2024だけで選択する。

優先順位:

1. Logloss / Brier
2. calibration
3. abs residual p95/p99
4. residual年度安定性
5. EV件数の過度な膨張を避ける
6. EV件数年度安定性
7. EV-ROI Spearman
8. ROIは補助
9. 特徴の意味と一般化しやすさ

僅差の場合は、より単純で極端値に強い方式を優先してよい。

例:

- rawと`log1p`がほぼ同等なら`log1p`
- clip単独とclip+log1pがほぼ同等なら、分布と説明可能性を比較
- dropで確率性能が悪化するなら完全削除しない

## 16. 統合候補

MonthDay除外が有効で、累積数変換も有効だった場合のみ、両方を統合したモデルを1つ作る。

モデル名例:

```text
C1R0_300_feature_clean_phase2_v1
```

個別に有効性が確認されていない変更は統合しない。

---

# Stage 5: 2025/2026固定診断

## 17. 対象

2020～2024で完全固定した後、次だけを2025/2026へ適用する。

- 作業基準モデル
- 最良の累積数変換モデル
- 統合モデルがある場合は統合モデル

2025/2026の結果で選択を変更しない。

## 18. 診断指標

- Logloss
- Brier
- ECE
- calibration slope/intercept
- residual mean/std
- abs residual p90/p95/p99
- EV>=1件数
- market-only crossing
- EV-ROI Spearman
- ROI
- top1/top3/top5/top10除外
- 未知カテゴリ率

選択モデルだけFeature Importance、SHAP、bootstrapを実施する。

---

## 19. 出力先

```text
outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
models/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1/
```

既存成果物を上書きしない。

---

## 20. 実装候補

```text
config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml
scripts/audit_c1r0_feature_cleanup_phase2_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
tests/test_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md
```

既存コードを再利用し、大規模リファクタは避ける。

---

## 21. 必須成果物

```text
existing_ablation_artifact_check.csv
existing_five_ablation_full_comparison.csv
existing_five_ablation_by_year.csv
existing_five_ablation_decision.md
working_base_model.json

monthday_feature_audit.csv
monthday_audit.md
monthday_ablation_comparison.csv

cumulative_starts_transform_params_by_fold.csv
cumulative_starts_comparison_by_fold.csv
cumulative_starts_comparison_2020_2024.csv
cumulative_starts_residual_stability.csv
cumulative_starts_ev_stability.csv

selected_preprocessing.json
phase2_model_comparison_2020_2024.csv
phase2_2025_2026_diagnostic.csv
selected_model_feature_importance.csv
selected_model_shap.csv
selected_model_shap_additivity.csv
manifest.json
```

---

## 22. 必須テスト

1. DBへ接続しない
2. feature datasetを再作成しない
3. 元Parquetを変更しない
4. random splitを使わない
5. 2015年以前を含めない
6. tree countが300
7. baselineが正しく設定される
8. 2025/2026を選択に使わない
9. `MonthDay`以外の列がMonthDay ablationで変わらない
10. 累積数変換の対象が2列だけ
11. p99が学習期間だけから計算される
12. validation/testから変換閾値を計算しない
13. raw/drop/log1p/clip/clip+log1pの定義が一致
14. feature/config/transform hashを記録
15. fold単位resumeが機能する
16. `Year`、市場特徴、人物コードが意図せず再混入しない
17. `final_logit = market_logit + residual_raw`
18. seed固定
19. 既存出力を上書きしない

---

## 23. 最終報告

日本語で以下を報告する。

1. 既存5候補の完全比較
2. 各候補の再利用・再学習状況
3. 作業基準モデル
4. `drop_person_codes`を基準にしたか
5. MonthDayの生成方法・型・年度代理性
6. MonthDay単独除外結果
7. `trainer_past_starts`と`jockey_past_starts`の年度代理性
8. raw/drop/log1p/p99 clip/clip+log1p比較
9. fold別p99閾値
10. 2020～2024で選択した処理
11. 統合モデルの有無
12. 2025/2026固定診断
13. 選択モデルのFI/SHAP
14. 次に勝率平滑化へ進むべきか
15. 次に未補正タイム置換へ進むべきか
16. 再利用したモデル・予測
17. 新規学習したモデル数
18. 実行時間
19. テスト結果
20. 作成・変更ファイル
21. `git status --short`
22. `git diff --stat`

自動commit/pushは行わない。
