# C1R0-300 Existing Feature Quality Audit and Targeted Ablation Task v1

## 0. 目的

現在の基準確率モデル候補である `C1R0_pure_market_offset_fixed300` を固定基準として、既存81特徴の意味・生成方法・時系列安全性・年度安定性を監査する。その後、問題の可能性が高い特徴群だけを対象に、小規模なablationを行う。

主な確認対象:

- `KisyuCode`
- `ChokyosiCode`
- `*_past_starts`
- `BaTaijyu`
- 生の走破タイム・上がり系
- 開催管理・時期系
- 高ユニークカテゴリ
- 欠損率や分布が年度間で変化する特徴

目的はROI最大化ではなく、意味が明確で時系列的に安全、かつ年度外で安定する特徴量構成へ整理すること。

## 1. 前提モデル

```text
C1R0_pure_market_offset_fixed300
final_logit = market_logit + catboost_residual
p_final = sigmoid(final_logit)
```

固定条件:

- tree count: 300
- `market_logit`はPool baselineのみ
- `Year`はCatBoost特徴に入れない
- `p_market`と`market_logit`はCatBoost特徴に入れない
- raw odds / ninki / market派生特徴はCatBoost特徴に入れない
- 既存C1R0 allowlistを基準にする

## 2. 最初に読むファイル

```text
tasks/place_market_offset_catboost_c1r0_v1_task.md
tasks/place_market_offset_catboost_c1r0_tree_count_v1_task.md
docs/place_market_offset_catboost_c1r0_v1_results.md
docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md
docs/place_market_offset_feature_audit_v1_results.md
config/place_market_offset_catboost_c1r0_v1.yaml
config/place_market_offset_catboost_c1r0_tree_count_v1.yaml
scripts/run_place_market_offset_catboost_c1r0_v1.py
scripts/run_place_market_offset_catboost_c1r0_tree_count_v1.py
outputs/place_market_offset_catboost_c1r0_v1/
outputs/place_market_offset_catboost_c1r0_tree_count_v1/
models/place_market_offset_catboost_c1r0_v1/
models/place_market_offset_catboost_c1r0_tree_count_v1/
```

既存の300本モデル、予測、allowlist、manifest、Feature Importance、SHAPを再利用する。

## 3. 絶対条件

- 2016年以降のみ使用
- random split禁止
- 2020～2024だけで特徴量採否を判断
- 2025/2026は固定診断のみ
- DB読込禁止
- feature dataset再作成禁止
- 新特徴量追加禁止
- tree countは300固定
- baseline方式を変更しない
- tree count以外のCatBoost hyperparameterを変更しない
- calibration方式を変更しない
- ROI直接学習、Ability/ANA、Ranker、Kelly、大規模Optuna禁止
- 自動購入、自動commit/push禁止
- 既存成果物を上書きしない

今回許可する再学習は指定した特徴量ablationモデルのみ。

## 4. 作業開始時

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

既存差分を勝手に戻さない。

# Stage 1: 再学習なしの特徴量品質監査

## 5. 全特徴量インベントリ

C1R0-300で使用される全特徴量について `feature_quality_inventory.csv` を作る。

列:

```text
feature
dtype
numeric_or_categorical
feature_group
source_file
source_function
generation_summary
uses_only_past_information
same_day_future_risk
current_race_leakage_risk
management_only
identity_feature
market_derived
result_derived
null_rate_2020
null_rate_2021
null_rate_2022
null_rate_2023
null_rate_2024
unique_count_2020_2024
unique_ratio_2020_2024
first_year_non_null
last_year_non_null
distribution_shift_2024_2025
pvc_rank
lfc_rank
shap_rank
recommended_action
reason
```

`recommended_action`:

```text
keep
ablation_candidate
transform_candidate
code_review_required
management_only
forbidden
unknown
```

## 6. 生成コード監査

### 累積出走数系

対象例:

```text
trainer_past_starts
jockey_past_starts
horse_jyo_past_starts
horse_surface_past_starts
horse_dist_band_past_starts
horse_baba_past_starts
jockey_jyo_past_starts
jockey_dist_band_past_starts
horse_jockey_past_starts
```

確認:

- 当該レース以前のみか
- 当該レースを含んでいないか
- 同日の未来レースを含んでいないか
- 全期間累積か
- 過去N年・過去N走制限があるか
- 年度が進むほど機械的に増えるか
- Yearの代理変数になっていないか
- 初出走・初騎乗等の欠損表現

### 勝率・複勝率系

```text
*_win_rate
*_ren_rate
*_top3_rate
*_place_paid_rate
```

確認:

- 分母
- 平滑化の有無
- priorの定義
- 最小サンプル処理
- `past_starts`との組み合わせ
- 0戦時の値

### 生タイム・上がり系

```text
horse_last3_avg_time
horse_last5_avg_time
horse_last3_avg_haron_l3
horse_last5_avg_haron_l3
```

確認:

- 距離補正
- 競馬場補正
- 芝ダート補正
- 馬場状態補正
- 単純秒数平均か
- 欠損時処理

### 馬体重

```text
BaTaijyu
horse_body_weight_diff_last
```

確認:

- 予測時点で利用可能か
- 馬体重発表時刻
- 欠損・異常値処理
- 生体重と変化量の役割重複

### 開催管理・時期系

実際のallowlistに存在する場合:

```text
Month
Kaiji
Nichiji
RaceNum
Day
date-derived columns
```

確認:

- レース条件として意味があるか
- 単なるデータ管理情報か
- クラスや季節の代理変数か
- 年度外汎化を悪化させるか

## 7. 年度代理性の監査

対象:

- `trainer_past_starts`
- `jockey_past_starts`
- その他`*_past_starts`
- `KisyuCode`
- `ChokyosiCode`
- `Kaiji`
- `Nichiji`
- `RaceNum`
- その他高ユニーク特徴

実施:

1. 年度別分布
2. 年度とのSpearman相関
3. 年度別中央値・p90・p99
4. 2024→2025のPSIまたは同等の分布変化
5. 未知カテゴリ率
6. 新規騎手・新規調教師率
7. カテゴリ出現頻度の年度変化

出力:

```text
feature_year_proxy_audit.csv
categorical_novelty_by_year.csv
feature_distribution_shift_by_year.csv
```

年度予測専用の別モデルは原則作らない。

## 8. 既存FI/SHAPの再整理

既存300本モデルの成果物を再利用し、以下を統合する。

- PredictionValuesChange
- LossFunctionChange
- mean absolute SHAP
- signed SHAP
- feature group
- 年度別SHAP変動

出力:

```text
selected300_feature_importance_merged.csv
selected300_feature_group_importance.csv
```

重点確認:

```text
KisyuCode
ChokyosiCode
trainer_past_starts
jockey_past_starts
BaTaijyu
horse_surface_past_starts
生タイム系
開催管理系
```

## 9. Stage 1の判定

各特徴群を次に分類する。

```text
keep
ablation_required
code_fix_required
future_transform_candidate
not_in_current_model
```

出力:

```text
feature_group_decision_stage1.csv
stage1_audit_report.md
```

コードリークや明らかな生成バグが見つかった場合は、ablationを先に進めず報告する。新特徴量の実装や変換は今回行わない。

# Stage 2: 小規模ablation

## 10. 基準モデル

```text
BASE = C1R0_pure_market_offset_fixed300
```

既存BASEモデル・予測を再利用し、再学習しない。

## 11. ablation候補

### A1: 騎手・調教師コード除外

```text
KisyuCode
ChokyosiCode
```

モデル名:

```text
C1R0_300_no_person_codes
```

### A2: 全体累積出走数除外

最低限:

```text
trainer_past_starts
jockey_past_starts
```

Stage 1で年度代理性が高い全体累積数のみ追加してよい。条件別適性の信頼度を表す`horse_surface_past_starts`等は無条件にまとめて除外しない。

モデル名:

```text
C1R0_300_no_global_cumulative_starts
```

### A3: 生馬体重除外

```text
BaTaijyu
```

`horse_body_weight_diff_last`は残す。

モデル名:

```text
C1R0_300_no_raw_body_weight
```

### A4: 生タイム系除外

生成コードが未補正の単純平均である場合のみ実施。

候補:

```text
horse_last3_avg_time
horse_last5_avg_time
```

上がり系はStage 1結果を見て別扱いにする。

モデル名:

```text
C1R0_300_no_raw_time_features
```

### A5: 開催管理系除外

Stage 1で管理情報・代理変数と判定された列だけ除外。

例:

```text
Kaiji
Nichiji
RaceNum
```

`Month`は無条件に除外しない。

モデル名:

```text
C1R0_300_no_management_features
```

## 12. 計算コスト制御

全候補を無条件に実行しない。

- Stage 1で`ablation_required`になった群だけ学習
- 最大でもBASE以外に5候補
- fold単位resume
- 既存モデル・予測再利用
- feature hash確認
- config hash確認
- tree_count確認
- 行数確認
- 不足foldだけ再学習
- 選択されなかった候補のSHAPは実行しない

最初は2020～2024の確率・残差・EV安定性比較まで実施する。Feature Importance、SHAP、bootstrap、2025/2026診断は有望候補とBASEだけに限定する。

## 13. walk-forward

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

tree countは全モデル300固定。

## 14. 評価指標

### 主評価

- Logloss
- Brier
- ECE
- calibration slope
- calibration intercept

### 残差安定性

- residual mean
- residual std
- abs residual p90
- abs residual p95
- abs residual p99
- 年度CV
- 最大年差

### EV安定性

- EV>=1件数
- EV>=1率
- EV件数CV
- market-onlyからのcrossing
- EV-ROI Spearman

### ROI補助

- EV>=1 ROI
- top1/top3/top5/top10除外
- bootstrap CI

ROIだけで採用しない。

## 15. 2020～2024採用ルール

優先順位:

1. Logloss / Brier
2. calibration
3. residual年度安定性
4. EV件数年度安定性
5. EV-ROI Spearman
6. モデルの意味の明確さ
7. ROIは補助

BASEとの差が非常に小さい場合は、より単純で一般化しやすい特徴構成を優先してよい。ただし確率指標が明確に悪化する場合は除外しない。

## 16. 統合モデル

複数ablationが有効だった場合のみ、統合モデルを1つ作る。

```text
C1R0_300_feature_clean_v1
```

2020～2024で個別に妥当性が確認された変更だけを統合する。

## 17. 2025/2026固定診断

2020～2024で候補を固定した後にのみ実施。

対象:

- BASE
- 最良の単独ablation
- 統合モデルがあれば統合モデル

評価:

- Logloss
- Brier
- ECE
- residual分布
- EV>=1件数
- crossing
- ROI
- 高配当除外後ROI
- 未知カテゴリ率

2025/2026の結果で採否を変更しない。

## 18. 出力先

```text
outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/
models/place_market_offset_catboost_c1r0_feature_cleanup_v1/
```

## 19. 実装候補

```text
config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml
scripts/audit_c1r0_feature_quality_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_v1.py
tests/test_place_market_offset_catboost_c1r0_feature_cleanup_v1.py
docs/place_market_offset_catboost_c1r0_feature_cleanup_v1_results.md
```

## 20. 必須成果物

```text
feature_quality_inventory.csv
feature_year_proxy_audit.csv
categorical_novelty_by_year.csv
feature_distribution_shift_by_year.csv
selected300_feature_importance_merged.csv
selected300_feature_group_importance.csv
feature_group_decision_stage1.csv
stage1_audit_report.md
ablation_feature_sets.json
ablation_comparison_by_fold.csv
ablation_comparison_2020_2024.csv
ablation_residual_stability.csv
ablation_ev_stability.csv
ablation_roi_diagnostic.csv
selected_feature_cleanup_model.json
feature_cleanup_2025_2026_diagnostic.csv
manifest.json
```

## 21. 必須テスト

1. DBへ接続しない
2. feature datasetを再作成しない
3. random splitを使わない
4. 2015年以前を含めない
5. tree countが300
6. baselineが正しく設定される
7. BASE allowlistを再現できる
8. 各ablationで指定列だけが除外される
9. Year/p_market/market_logit/市場特徴が再混入しない
10. 2025/2026を選択に使わない
11. feature hashとconfig hashを記録
12. fold単位resumeが機能する
13. 既存出力を上書きしない
14. `final_logit = market_logit + residual_raw`
15. seed固定

## 22. 最終報告

日本語で以下を報告する。

1. C1R0-300の全特徴量数
2. 各特徴の採用・監査・除外候補分類
3. 累積出走数系の生成方法
4. 累積出走数系がYear代理になっているか
5. 騎手・調教師コードの未知カテゴリ率
6. 勝率系の平滑化有無
7. 生タイム系の補正有無
8. 馬体重の利用時点と欠損処理
9. 開催管理系の意味
10. 実行したablation
11. 再利用した成果物
12. 再学習したモデル数
13. 2020～2024比較
14. 有効だった特徴削除
15. 悪化した特徴削除
16. 統合モデルの有無
17. 2025/2026固定診断
18. 次にコース特徴追加へ進んでよいか
19. 作成・変更ファイル
20. テスト結果
21. 実行時間
22. `git status --short`
23. `git diff --stat`

自動commit/pushは行わない。
