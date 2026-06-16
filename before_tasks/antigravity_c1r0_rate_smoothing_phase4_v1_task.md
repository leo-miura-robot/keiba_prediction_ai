# Antigravity C1R0-300 Rate Smoothing Phase 4 Task v1
## Probability Schema Hardening and Targeted Empirical-Bayes Smoothing

## 0. このタスクの目的

現在の正式な基準モデルは次である。

```text
C1R0_fixed300_ablation_drop_person_codes
```

ここまでの監査により、以下が確定した。

- tree countは300本固定
- `KisyuCode` / `ChokyosiCode`は除外
- `trainer_past_starts` / `jockey_past_starts`はrawを維持
- `MonthDay`、`Kaiji`、`Nichiji`、`RaceNum`は維持
- `horse_last3_avg_time`、`horse_last5_avg_time`は当面維持
- `BaTaijyu`は当面維持
- Phase 2 / Phase 3の評価逆転はcalibrated / uncalibrated混在が原因
- モデル選択時の主評価は未補正の`probability_raw`へ統一する

次の課題は、現在単純比率になっている勝率・連対率・複勝率系特徴量を、
少数サンプルの過信を抑える形へ平滑化できるか検証することである。

今回の目的:

1. 新規成果物の確率列スキーマを明確化する
2. rate列と対応する分母・成功定義を監査する
3. trainer / jockey / horse_surfaceの3グループを段階的に平滑化する
4. 2020～2024だけで採否を決める
5. 有効だった変更だけを最大1つの統合モデルへ反映する
6. 仕様固定後に2025/2026を診断する

---

# 1. これまでの重要な経緯

## 1.1 市場残差モデル

基本構造:

```text
final_logit = market_logit + residual_raw
probability_raw = sigmoid(final_logit)
```

市場情報は`market_logit`のbaselineとしてだけ使う。
CatBoostは市場以外の競馬情報による補正を学習する。

CatBoostへ入れないもの:

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
KisyuCode
ChokyosiCode
```

## 1.2 評価期間

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

採否判断:

```text
2020～2024のみ
```

固定診断:

```text
2025
2026
```

2025/2026を見てモデル選択を変更してはいけない。

## 1.3 確率列不一致の教訓

Phase 2とPhase 3では、同名に近い予測列の一方がcalibrated、
他方がuncalibratedであり、不公平な比較が発生した。

今後の新規成果物では、曖昧な`final_probability`だけを保存しない。

---

# 2. 最初に読むファイル

以下を最初から最後まで読む。

```text
tasks/antigravity_c1r0_metric_consistency_handover_v2_task.md
tasks/place_market_offset_catboost_c1r0_metric_consistency_audit_v1_task.md

docs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md
docs/place_market_offset_catboost_c1r0_feature_cleanup_v1_results.md

scripts/audit_c1r0_metric_consistency_v2.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.py
scripts/run_place_market_offset_catboost_c1r0_feature_cleanup_v1.py

config/place_market_offset_catboost_c1r0_metric_consistency_audit_v2.yaml
config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml

outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2/
outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1/

models/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1/
models/place_market_offset_catboost_c1r0_feature_cleanup_v1/
```

履歴特徴を生成している実装も追跡する。

候補:

```text
history_builder_v2_1.py
```

実際のファイル位置はリポジトリ内検索で確認する。

---

# 3. 絶対条件

- 2016年以降のみ使用
- random split禁止
- 2020～2024だけで採否判断
- 2025/2026は仕様固定後の診断のみ
- DB読込禁止
- DB更新禁止
- 元feature Parquet変更禁止
- feature dataset全体の再作成禁止
- tree countは300固定
- baseline方式変更禁止
- CatBoostの他hyperparameter変更禁止
- calibration方式変更禁止
- モデル選択にcalibrated probabilityを使わない
- ROI直接学習禁止
- Ability/ANA禁止
- Ranker禁止
- Kelly禁止
- 大規模Optuna禁止
- 自動購入禁止
- git add / commit / push禁止
- git reset / clean禁止
- 既存成果物の上書き・削除禁止

今回許可する再学習は、タスク内で明示したrate smoothing候補だけ。

---

# 4. Antigravityの安全条件

## 4.1 書き込み許可範囲

原則として以下だけに新規作成する。

```text
config/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.yaml
scripts/audit_c1r0_rate_features_v1.py
scripts/run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py
tests/test_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py
docs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1_results.md

outputs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1/
models/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1/
```

既存実装の修正が必要な場合も、まず新規ラッパー・新規ユーティリティで対応する。

## 4.2 停止条件

以下の場合は推測して続けず停止・報告する。

```text
rate列の意味を生成コードから確定できない
対応するstarts列を一意に決められない
成功数を安全に復元できない
現在レースの結果がrate計算へ混ざる
validation年からpriorを計算する必要がある
同日未来レースが混ざる
基準モデルのallowlistを再現できない
probability_rawを特定できない
モデル選択時にcalibrationが混ざる
既存成果物を上書きする必要がある
```

---

# Stage 0. 作業状態の確認

## 5. 開始時確認

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

既存の未追跡ファイルを勝手に削除・移動しない。

---

# Stage 1. 確率列スキーマの明確化

## 6. 新規予測成果物の必須列

今後このタスクで保存するfold/final予測には、最低限次を含める。

```text
probability_raw
probability_calibrated
probability_used_for_selection
is_calibrated
calibration_method
```

ただし今回、モデル選択で使うのは必ず:

```text
probability_raw
```

EVおよびROIの算出（補助診断）には、確率が適切にスケールされたキャリブレーション済み確率が数学的に必須である。
したがって、モデル選択自体は `probability_raw` を用いるが、EV/ROI計算用として必ず従来と同じ方式（isotonic等）でキャリブレーションをfitさせ、以下のように保存する。

```text
probability_calibrated = （キャリブレーション後の値）
is_calibrated = true
calibration_method = isotonic
probability_used_for_selection = probability_raw
```

既存の`probability`や`final_probability`を読み込む場合は、
manifestと生成コードから意味を確定し、読み込み直後に明示列へ変換する。

曖昧な場合は停止する。

## 7. manifest必須項目

```text
probability_raw_column
probability_calibrated_column
probability_used_for_selection
is_calibrated
calibration_method
calibrator_fit_period
selection_uses_calibrated_probability
```

`selection_uses_calibrated_probability`は必ず`false`。

---

# Stage 2. rate特徴の生成コード監査

## 8. 対象グループ

今回の対象は以下の3グループだけ。

```text
trainer
jockey
horse_surface
```

実際のallowlistと生成コードから対象rate列を列挙する。

候補例:

```text
trainer_*_win_rate
trainer_*_ren_rate
trainer_*_top3_rate
trainer_*_place_paid_rate

jockey_*_win_rate
jockey_*_ren_rate
jockey_*_top3_rate
jockey_*_place_paid_rate

horse_surface_*_win_rate
horse_surface_*_ren_rate
horse_surface_*_top3_rate
horse_surface_*_place_paid_rate
```

実際に存在しない列を推測で追加しない。

## 9. 対応する分母

例:

```text
trainer rate          -> trainer_past_starts
jockey rate           -> jockey_past_starts
horse_surface rate    -> horse_surface_past_starts
```

各rate列について次を確定する。

```text
rate_feature
group
success_definition
denominator_feature
source_file
source_function
current_race_excluded
same_day_future_excluded
zero_start_value
null_handling
raw_success_count_available
success_count_source
```

出力:

```text
rate_feature_inventory.csv
rate_denominator_mapping.csv
rate_generation_audit.md
```

## 10. 成功数の扱い

優先順位:

1. 生成コード内の履歴成功数を安全に利用できる
2. 保存済みの成功数列が存在する
3. `rate * starts`で成功数が正確に復元できることを検証できる

`rate * starts`を使う場合は、整数成功数との誤差を検証する。

```text
abs(rate * starts - round(rate * starts))
```

誤差が許容範囲を超える場合は使用しない。

成功数を安全に得られない場合は停止し、
「平滑化には履歴builderの拡張が必要」と報告する。
元Parquet全再生成は今回行わない。

---

# Stage 3. 平滑化方式

## 11. 基本式

Empirical-Bayes型の平滑化:

```text
smoothed_rate
=
(success_count + prior_strength * prior_rate)
/
(past_starts + prior_strength)
```

同値な形:

```text
smoothed_rate
=
(past_starts * raw_rate + prior_strength * prior_rate)
/
(past_starts + prior_strength)
```

0戦時:

```text
smoothed_rate = prior_rate
```

## 12. prior_rate

各foldの学習期間だけから計算する。

validation年、2025、2026を使わない。

成功定義別に分ける。

例:

```text
win prior
ren / top2 prior
top3 prior
place_paid prior
```

priorの算出方法は生成コードの意味に合わせる。

原則:

- 現在レース以前の履歴定義と整合
- fold train期間だけ
- validation行のtargetを使わない
- 同じイベントを重複集計しない

累積rate行をstartsで再加重してpriorを作ると、
同じ過去結果を何度も数える可能性がある。
その方法は原則禁止。

安全なpriorを作れない場合は停止する。

## 13. prior_strength候補

```text
5
10
20
```

大規模探索は行わない。

---

# Stage 4. 計算量を抑えた段階的比較

## 14. 正式基準

```text
BASE:
C1R0_fixed300_ablation_drop_person_codes
```

維持:

```text
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

BASEは既存モデル・予測を再利用する。

## 15. Screening

最初は`prior_strength = 10`だけで、各グループを個別比較する。

```text
T10: trainer ratesのみ平滑化
J10: jockey ratesのみ平滑化
H10: horse_surface ratesのみ平滑化
```

最大:

```text
3候補 × 5fold
```

この段階で明確に悪化したグループは終了する。

## 16. Strength refinement

T10 / J10 / H10のうち、有望またはほぼ同等だったグループだけについて追加する。

```text
prior_strength = 5
prior_strength = 20
```

無条件に全候補を学習しない。

有望判定の目安:

- Logloss/Brierが改善または実質同等
- ECEが大幅悪化しない
- residual tailが悪化しない
- worst-year性能が悪化しない

## 17. Drop control

各グループで平滑化が明確に悪化した場合に限り、
そのrate列群を削除したcontrolを最大1つ作ってよい。

```text
T_DROP
J_DROP
H_DROP
```

無条件に作らない。

## 18. 統合モデル

個別に有効と判定されたグループだけを統合する。

最大1モデル:

```text
C1R0_300_rate_smoothed_phase4_v1
```

全組合せ探索禁止。

---

# Stage 5. walk-forwardと前処理リーク防止

## 19. walk-forward

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

## 20. fold別パラメータ

各foldで保存:

```text
model_key
evaluation_year
train_start_year
train_end_year
rate_group
rate_feature
success_definition
denominator_feature
prior_rate
prior_strength
zero_start_policy
```

出力:

```text
rate_smoothing_params_by_fold.csv
```

final診断用も、学習期間だけからpriorを算出する。

---

# Stage 6. 評価

## 21. モデル選択に使う確率

必ず:

```text
probability_raw
```

calibrated probabilityで候補選択しない。

## 22. 主評価

```text
runner-weighted Logloss
runner-weighted Brier
fixed-bin ECE
calibration slope
calibration intercept
worst-year Logloss
worst-year Brier
```

## 23. 残差安定性

```text
residual mean
residual std
abs residual p90
abs residual p95
abs residual p99
年度CV
```

## 24. EV補助診断

```text
EV>=1件数
EV件数CV
market-only crossing
EV-ROI Spearman
```

ROIは補助。

```text
EV>=1 ROI
top1/top3/top5/top10払戻除外
```

ROIだけで採用しない。

---

# Stage 7. 採用ルール

## 25. 優先順位

1. runner-weighted Logloss
2. runner-weighted Brier
3. fixed-bin ECE
4. worst-year性能
5. residual tail
6. 年度安定性
7. EV件数の過度な膨張を避ける
8. EV件数安定性
9. EV-ROI Spearman
10. ROIは補助
11. 特徴の意味と単純さ

## 26. 統計確認

最終候補とBASEの差が小さい場合、
保存済みfold予測でrace単位paired bootstrapを行う。

```text
n_bootstrap = 5000
delta = candidate - base
sampling_unit = race
metric_weighting = runner-weighted
```

Logloss/Brierの95% CIが0をまたぎ、
実質差も極小ならBASEを維持する。

---

# Stage 8. 2025/2026固定診断

## 27. 対象

2020～2024で仕様固定後にのみ実施。

```text
最終的に採用された1つのモデル（selected_model）のみ
```

仕様固定後に複数の候補を2025/2026で比較することは、テストデータの覗き見（data leakage）にあたり致命的な評価の歪みを生むため厳禁である。必ず最終選択されたモデル1つだけを診断する。

## 28. 診断項目

```text
Logloss
Brier
ECE
calibration slope/intercept
residual distribution
EV>=1件数
EV-ROI Spearman
ROI
高配当除外後ROI
```

---

# Stage 9. 再利用とresume

## 29. 再利用

- BASEのfoldモデル・予測
- 既存allowlist
- 既存評価関数
- metric consistency auditの確率列判定
- 既存FI/SHAP

を再利用する。

## 30. resume

- fold単位
- model key
- feature hash
- config hash
- smoothing parameter hash
- prediction row count
- tree count

を検証し、不足foldだけ学習する。

`gpu_ram_part = 0.75`は実行環境制約として利用してよい。
モデル選択hyperparameterではないことをmanifestへ記録する。

---

# 10. 出力先

```text
outputs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1/
models/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1/
```

---

# 11. 実装ファイル

```text
config/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.yaml
scripts/audit_c1r0_rate_features_v1.py
scripts/run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py
tests/test_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py
docs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1_results.md
```

---

# 12. 必須成果物

```text
rate_feature_inventory.csv
rate_denominator_mapping.csv
rate_generation_audit.md

probability_schema_audit.csv
rate_smoothing_params_by_fold.csv
screening_comparison_by_fold.csv
screening_comparison_2020_2024.csv
strength_refinement_comparison.csv
rate_smoothing_residual_stability.csv
rate_smoothing_ev_stability.csv
rate_smoothing_roi_diagnostic.csv

selected_rate_smoothing.json
selected_feature_set_phase4.json
phase4_2025_2026_diagnostic.csv

selected_model_feature_importance.csv
selected_model_shap.csv
selected_model_shap_additivity.csv

paired_bootstrap_summary.csv
manifest.json
```

FI/SHAPとpaired bootstrapは選択候補だけに実施する。

---

# 13. 必須テスト

1. DBへ接続しない
2. 元Parquetを変更しない
3. feature dataset全体を再作成しない
4. random splitを使わない
5. 2015年以前を含めない
6. tree countが300
7. baselineが正しい
8. `KisyuCode` / `ChokyosiCode`が再混入しない
9. `Year` / 市場特徴が再混入しない
10. rateとstartsの対応を検証
11. current raceをrateへ含めない
12. same-day futureを含めない
13. priorはfold trainだけから計算
14. validation/2025/2026からpriorを計算しない
15. 0戦時はpriorへfallback
16. `probability_raw`を明示保存
17. calibrated列との混同を防止
18. model selectionは`probability_raw`
19. fold単位resume
20. feature/config/smoothing hashを記録
21. 既存成果物を上書きしない
22. 2025/2026を採否に使わない
23. seed固定
24. 自動commit/pushを行わない

---

# 14. 最終報告

日本語で以下を報告する。

1. 現在の基準モデルを正しく理解したか
2. 対象になったrate列の全一覧
3. 各rate列の成功定義
4. 対応するstarts列
5. 成功数をどのように取得したか
6. 時系列安全性
7. prior_rateの計算方法
8. fold別prior_rate
9. 確率列スキーマ
10. screening結果
11. refinementを実行したグループ
12. prior_strength 5/10/20比較
13. drop controlの有無
14. 個別に有効だったグループ
15. 統合モデルの有無
16. 2020～2024で固定した最終仕様
17. paired bootstrap結果
18. 2025/2026固定診断
19. FI/SHAP
20. 次に未補正タイム置換またはコース特徴へ進めるか
21. 再利用成果物
22. 新規学習モデル数
23. 実行時間
24. テスト結果
25. 作成・変更ファイル
26. `git status --short`
27. `git diff --stat`

自動commit/pushは行わない。
