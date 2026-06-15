# C1R0 Tree Count Audit / Fixed-Tree Comparison v1

## 目的

`C1R0_pure_market_offset`について、2025/2026で残差とEV>=1件数が急増した原因が、
final modelのtree count・early stopping・best iterationにあるか監査する。

その上で、2020〜2024のwalk-forward validationだけを使って固定tree countを比較し、
final modelへ適用する木数を決める。

350本は候補であり、先に正解と決めない。

## 必ず読むファイル

- `tasks/place_market_offset_catboost_c1r0_v1_task.md`
- `docs/place_market_offset_catboost_c1r0_v1_results.md`
- `config/place_market_offset_catboost_c1r0_v1.yaml`
- `scripts/run_place_market_offset_catboost_c1r0_v1.py`
- `scripts/train_place_market_offset_catboost_c1r0_v1.py`
- `scripts/evaluate_place_market_offset_catboost_c1r0_v1.py`
- `outputs/place_market_offset_catboost_c1r0_v1/`
- `models/place_market_offset_catboost_c1r0_v1/`

## 絶対条件

- 2016年以降のみ
- random split禁止
- 木数選択は2020〜2024のみ
- 2025/2026は固定診断のみ
- DB読込禁止
- feature dataset再作成禁止
- 新特徴量追加禁止
- C1R0 allowlist変更禁止
- tree count以外のhyperparameter変更禁止
- ROI直接学習、Ability/ANA、Ranker、Kelly、大規模Optuna禁止
- 自動購入、自動commit/push禁止
- 既存成果物を上書きしない

今回、固定tree count候補のC1R0再学習は許可する。

## 作業開始時

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

## 1. 保存済みモデルのtree count監査

対象:

- 2020評価用fold
- 2021評価用fold
- 2022評価用fold
- 2023評価用fold
- 2024評価用fold
- 2025 final model
- 2026 final model

各モデルについて以下を出す。

- train期間
- eval年
- train rows
- valid rows
- config上のiterations
- `tree_count_`
- best iteration
- eval_set有無
- `use_best_model`
- early stopping設定
- `od_type`
- `od_wait`
- learning rate
- depth
- random seed
- model SHA-256

出力:

`model_tree_count_audit.csv`

確認したいこと:

- foldではearly stoppingが有効か
- final modelだけeval_setなしで上限まで学習していないか
- 2024 foldから2025 finalでtree countが急増していないか
- データ量増加に対してtree count増加が連続的か不連続か

## 2. fold別best iteration要約

2020〜2024foldについて以下を集計する。

- min
- p25
- median
- mean
- p75
- max
- std

中央値を50本単位に丸めた参考値も出す。

出力:

`fold_best_iteration_summary.csv`

## 3. 固定tree count候補

2020〜2024で次を比較する。

- 250
- 300
- 350
- 400
- 450

必要ならfold中央値の近傍を追加してよいが、2020〜2024の情報だけを根拠にする。

変更するのはtree count / iterationsだけ。
以下はC1R0と同一にする。

- feature allowlist
- baseline
- learning rate
- depth
- loss
- random seed
- categorical features
- calibration方針
- その他hyperparameter

## 4. walk-forward比較

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

各候補で測定する。

### 確率指標

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

### EV安定性

- EV>=1件数
- EV>=1率
- market-only EV>=1件数
- EV<1からEV>=1へのcrossing
- EV>=1からEV<1へのcrossing
- EV-ROI Spearman

### ROI診断

- EV>=1 ROI
- top1/top3/top5/top10払戻除外
- bootstrap 95% CI

ROIだけで木数を選ばない。

出力:

- `fixed_tree_comparison_by_fold.csv`
- `fixed_tree_comparison_2020_2024.csv`
- `fixed_tree_residual_stability.csv`
- `fixed_tree_ev_stability.csv`
- `fixed_tree_roi_diagnostic.csv`

## 5. 木数選択ルール

2020〜2024だけで選ぶ。

優先順位:

1. Logloss / Brier
2. residual stdの年度安定性
3. abs residual p90/p95/p99の年度安定性
4. EV>=1件数の年度安定性
5. ECE / calibration
6. EV-ROI Spearman
7. ROIは補助

木数が多いほどわずかにLoglossが良くても、残差やEV件数が不安定なら小さい木数を優先してよい。

出力:

`selected_fixed_tree_count.json`

350本が選ばれなかった場合も、そのまま報告する。

## 6. ntree_end補助診断

保存済みfinal modelへ`ntree_end`を指定できる場合、再学習前に次で予測する。

- 250
- 300
- 350
- 400
- 450
- full tree count

目的:

- 後半の木が残差を膨らませているか確認
- fixed-tree再学習と方向が一致するか確認

これは原因分析用であり、正式な木数選択には使わない。

出力:

`ntree_end_diagnostic_2025_2026.csv`

## 7. 2025/2026固定診断

木数を2020〜2024で完全固定した後にのみ実施する。

### 2025

```text
train 2016-2024
iterations = selected_tree_count
predict 2025
```

### 2026

現行プロジェクトの固定評価ルールをコード/configから確認する。
2025結果を見て仕様を変更しない。

比較:

- 現在のC1R0 final model
- 選択した固定tree countのC1R0

測定:

- Logloss
- Brier
- ECE
- residual mean/std
- abs residual p90/p95/p99
- EV>=1件数
- crossing件数
- ROI
- 上位払戻除外後ROI

2025/2026結果で木数選択を変更しない。

出力:

- `fixed_tree_2025_2026_diagnostic.csv`
- `fixed_tree_residual_2025_2026.csv`
- `fixed_tree_ev_2025_2026.csv`

## 出力先

- `outputs/place_market_offset_catboost_c1r0_tree_count_v1/`
- `models/place_market_offset_catboost_c1r0_tree_count_v1/`

## 実装候補

- `config/place_market_offset_catboost_c1r0_tree_count_v1.yaml`
- `scripts/audit_c1r0_tree_count_v1.py`
- `scripts/run_place_market_offset_catboost_c1r0_tree_count_v1.py`
- `tests/test_place_market_offset_catboost_c1r0_tree_count_v1.py`
- `docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md`

既存C1R0コードを再利用し、大規模リファクタはしない。

## 必須テスト

1. DBへ接続しない
2. feature datasetを再作成しない
3. random splitを使わない
4. 2015年以前を含めない
5. C1R0 allowlistを変更しない
6. Year/p_market/market_logitをCatBoost特徴へ入れない
7. tree count以外の主要hyperparameterが同一
8. 2025/2026を選択に使わない
9. fixed-treeモデルのtree_countが指定値と一致
10. baselineが学習・推論で設定される
11. `final_logit = market_logit + residual_raw`
12. seed固定
13. 既存出力を上書きしない

## 最終報告

日本語で以下を報告する。

1. fold別tree count / best iteration
2. final modelのtree count
3. final modelだけ上限まで学習していたか
4. データ量増加で自然に木が増えたのか
5. early stopping有無
6. 250/300/350/400/450比較
7. 2020〜2024だけで選んだ木数
8. 350本が妥当だったか
9. residual安定性
10. EV>=1件数安定性
11. 2025/2026固定診断
12. 現行C1R0との比較
13. tree count問題が2025急増の主因だったか
14. 作成・変更ファイル
15. テスト結果
16. `git status --short`
17. `git diff --stat`

自動commit/pushは行わない。
