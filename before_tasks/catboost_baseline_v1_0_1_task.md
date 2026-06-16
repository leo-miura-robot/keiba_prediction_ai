# Task: CatBoost Baseline V1.0.1 安全性・再現性修正

## 目的

現在のCatBoost GPU分類ベースラインV1を保持しつつ、次工程の確率キャリブレーションと将来の単勝・複勝バックテストへ進む前に、評価・設定・resume・分析出力を安全かつ再現可能にする。

第一段階の最終目標:

```text
単勝回収率 >= 90%
複勝回収率 >= 90%
```

ただし今回はROI、EV、買い目、資金配分、確率キャリブレーション適用、Ability、ANA、Ranker、Optuna、Walk-forward変更は行わない。

---

## 1. 最初に確認するもの

現在のmainを確認する。

```text
scripts/train_catboost_baseline.py
src/models/catboost_data.py
src/models/catboost_metrics.py
src/models/catboost_runner.py
src/models/model_manifest.py
config/catboost_baseline_v1.yaml
config/feature_sets_v2_1_1.yaml
tests/test_catboost_*.py
outputs/model_training/catboost_baseline_v1/
models/catboost_baseline_v1/
```

最初に実行:

```bash
git status
git diff
python -m pytest -q
```

V1〜V2.1.1の特徴量生成コード、Parquet、stateは変更しない。

---

## 2. V1.0.1を別バージョンとして作る

推奨構成:

```text
config/catboost_baseline_v1_0_1.yaml

scripts/train_catboost_baseline_v1_0_1.py
scripts/analyze_catboost_baseline_v1_0_1.py

src/models/catboost_config.py
src/models/catboost_resume.py
src/models/catboost_analysis.py

tests/test_catboost_config.py
tests/test_catboost_resume.py
tests/test_catboost_analysis.py

models/catboost_baseline_v1_0_1/
outputs/model_training/catboost_baseline_v1_0_1/
logs/train_catboost_baseline_v1_0_1.log
```

既存V1成果物を無断上書きしない。

---

## 3. YAMLを実学習へ接続する

現在の学習パラメータの正本を以下に統一する。

```text
config/catboost_baseline_v1_0_1.yaml
```

要件:

1. YAMLを読み込む
2. schema validationを行う
3. CatBoostへ実際に渡すresolved paramsを生成する
4. CLI上書きは`task_type`、`devices`、`smoke_test`など明示項目だけ許可する
5. `training_config_resolved.json`を保存する
6. resolved configのSHA-256をmanifestへ保存する
7. YAML値とCatBoostへ渡した値が一致するテストを追加する

学習パラメータをコード内へ二重定義しない。

---

## 4. 市場比較を同一サンプルで再計算する

市場確率とCatBoostの比較は、同じentry_id集合で行う。

対象条件:

```text
eligible_for_win_training == True
tan_odds > 0
actualが0または1
market_free/history/awareの予測が全て存在
race内に有効オッズが存在
```

市場確率:

```text
raw_implied_probability = 1 / tan_odds
market_probability =
    raw_implied_probability /
    race内raw_implied_probability合計
```

同一集合で比較:

```text
market_probability
catboost_market_free
catboost_market_history
catboost_market_aware
```

指標:

- rows
- races
- positive count
- Logloss
- Brier score
- ROC-AUC
- PR-AUC
- Top1 winner accuracy
- Top3 winner inclusion rate
- race内確率合計

出力:

```text
market_comparison_same_sample.csv
market_comparison_sample_summary.csv
```

各比較対象のentry_id集合が完全一致するテストを追加する。

---

## 5. resumeをfingerprint対応にする

モデル単位で以下を保存する。

```text
target
feature_set
task_type
devices
random_seed

input_dataset_fingerprint
feature_set_yaml_sha256
training_config_resolved_sha256
code_bundle_sha256

python_version
catboost_version
numpy_version
pandas_version
sklearn_version
pyarrow_version

git_commit_sha
git_is_dirty

model_file_sha256
predictions_file_sha256
metrics_file_sha256
```

### `--resume`

- fingerprint一致かつ成果物完全: skip
- fingerprint不一致: そのモデルのみ再学習
- 成果物欠落: そのモデルのみ再学習

### `--resume --strict-resume`

- fingerprint不一致: exit 2
- 成果物欠落: exit 2
- 自動再学習しない

### `--force`

- 指定モデルのみ再学習
- 上書き対象を事前にログ出力
- 他モデルには触れない

Git SHAは記録用とし、Git SHAだけの変更では失敗させない。

---

## 6. 集計CSVを冪等化する

単純appendを禁止する。

以下の主キーでupsertするか、run_id単位で完全分離する。

```text
metrics_by_split:
  target, feature_set, data_split

race_metrics:
  target, feature_set, data_split

calibration_bins:
  target, feature_set, data_split, bin_type, bin_id

feature_importance:
  target, feature_set, importance_type, feature

model_comparison:
  target, feature_set
```

同じコマンドを2回実行しても行数・内容が増えないことをテストする。

---

## 7. 分析処理を再生成可能にする

次を既存modelとpredictionsから再生成する専用スクリプトを追加する。

```text
scripts/analyze_catboost_baseline_v1_0_1.py
```

再生成対象:

- metrics_by_split
- race_metrics
- race_probability_sum_diagnostics
- calibration bins
- feature importance
- SHAP importance
- 市場ベースライン
- 同一サンプル市場比較
- model comparison
- split summary
- class balance

分析スクリプトはモデル再学習を行わない。

不足成果物があれば明確に停止する。

---

## 8. calibration binを2種類出す

### fixed-width

```text
0.0〜0.1
0.1〜0.2
...
0.9〜1.0
```

### quantile

各binの件数が概ね等しくなるように分割する。

出力列:

```text
target
feature_set
data_split
bin_type
bin_id
lower_bound
upper_bound
count
mean_pred_probability
actual_rate
calibration_gap
```

同一値が多い場合にbin数が10未満になっても安全に動くこと。

今回はキャリブレーション補正を適用しない。

---

## 9. 既存V1モデルの再利用

V1とV1.0.1で以下が完全一致する場合、既存モデルを再利用してよい。

- resolved params
- 入力fingerprint
- feature set hash
- target
- feature columns
- categorical columns
- 年分割
- random seed
- task type

再利用時も以下を検証する。

1. model metadata
2. feature columns
3. categorical columns
4. predictionsの行数
5. entry_id整合性
6. 予測確率が0〜1
7. model再読込後の予測一致

manifestに記録:

```text
artifact_origin: reused_from_v1
source_model_path: ...
```

1項目でも不一致なら再学習する。

---

## 10. データ分割は今回は変更しない

暫定分割を維持する。

```text
train: 2016〜2023
validation: 2024
test: 2025
latest_holdout: 2026
```

レース数均等Walk-forward、最近データの追加学習、開催週、馬場進行、時間減衰は後日の別タスクとする。

---

## 11. ROI目標を文書化する

manifestと設計資料に記載する。

```yaml
phase1_goal:
  win_roi_min: 0.90
  place_roi_min: 0.90
```

ただし将来の合格条件はROIだけではない。

- 十分な購入件数
- validationで条件決定
- testで閾値変更禁止
- 高額払戻依存度
- オッズ帯別ROI
- 期間別安定性
- 上位払戻除外後ROI
- 穴馬1頭依存を合格扱いしない

今回は文書化のみでROI計算はしない。

---

## 12. 必須テスト

最低限以下を追加する。

### Config

1. V1.0.1 YAML読込
2. YAML値が実paramsへ反映
3. 不正configで停止
4. CLI上書き制限

### Resume

5. fingerprint一致でskip
6. input fingerprint不一致検出
7. feature set hash不一致検出
8. config hash不一致検出
9. code hash不一致検出
10. model欠落検出
11. metrics欠落検出
12. predictions欠落検出
13. strict resume停止
14. Git SHAのみでは失敗しない

### Idempotency

15. 同じ分析を2回実行しても行数不変
16. target / feature_set / split重複0

### Market comparison

17. 市場と3モデルのentry_id集合一致
18. rows / races一致
19. 市場確率のrace内合計が概ね1
20. 同一サンプルで全指標計算

### Calibration

21. fixed-width正常
22. quantile正常
23. 同一値が多くても正常
24. bin合計件数が元件数と一致

### Analysis and reuse

25. SHAPまたは標準importance再生成
26. 分析スクリプトが学習しない
27. 欠落成果物で停止
28. V1完全一致なら再利用可能
29. feature columns不一致なら再利用不可
30. model再読込後の予測一致

---

## 13. 実行手順

```bash
python -m py_compile scripts/train_catboost_baseline_v1_0_1.py
python -m py_compile scripts/analyze_catboost_baseline_v1_0_1.py
python -m pytest -q
```

GPUスモーク:

```bash
python scripts/train_catboost_baseline_v1_0_1.py   --target win   --feature-set market_free   --task-type GPU   --smoke-test
```

既存モデル再利用判定:

```bash
python scripts/train_catboost_baseline_v1_0_1.py   --all   --task-type GPU   --reuse-compatible-v1
```

strict resume:

```bash
python scripts/train_catboost_baseline_v1_0_1.py   --all   --task-type GPU   --resume   --strict-resume
```

分析再生成:

```bash
python scripts/analyze_catboost_baseline_v1_0_1.py --all
```

同じ分析コマンドをもう一度実行し、出力が増えないことを確認する。

GPU失敗時にCPUへ自動fallbackしない。

---

## 14. 出力

```text
config/catboost_baseline_v1_0_1.yaml

scripts/train_catboost_baseline_v1_0_1.py
scripts/analyze_catboost_baseline_v1_0_1.py

src/models/catboost_config.py
src/models/catboost_resume.py
src/models/catboost_analysis.py

tests/test_catboost_config.py
tests/test_catboost_resume.py
tests/test_catboost_analysis.py

models/catboost_baseline_v1_0_1/

outputs/model_training/catboost_baseline_v1_0_1/
  run_manifest.json
  model_comparison.csv
  split_summary.csv
  class_balance.csv
  metrics_by_split.csv
  race_metrics.csv
  race_probability_sum_diagnostics.csv
  calibration_bins.csv
  feature_importance.csv
  shap_importance.csv
  market_baseline_win.csv
  market_comparison_same_sample.csv
  market_comparison_sample_summary.csv
  predictions/

logs/train_catboost_baseline_v1_0_1.log

docs/catboost_baseline_v1_0_1_design.md
docs/catboost_baseline_v1_0_1_results.md
```

---

## 15. 完了条件

- pytest全件成功
- YAML値が実学習へ反映
- 6モデルartifact整合性確認
- 必要なモデルだけ再学習
- resume fingerprint成功
- strict resume正常終了
- 集計CSV重複0
- 同一コマンド再実行で出力不変
- 市場確率とCatBoostを同一entry_id集合で比較
- fixed-width / quantile bins生成
- SHAP・市場比較をコードから再生成可能
- V1〜V2.1.1特徴量生成資産を変更していない
- ROI、EV、買い目生成を行っていない
- Ability / Rankerを実装していない

1つでも満たさない場合、V1.0.1完了とは判定しない。

---

## 16. 最終報告

1. git status / git diff
2. 変更・追加ファイル
3. V1で確認した問題
4. YAML接続方法
5. resolved training config
6. pytest結果
7. GPUスモーク結果
8. 6モデルの再利用・再学習状況
9. 再利用判定根拠
10. resume fingerprint項目
11. strict resume結果
12. CSV冪等性結果
13. 同一サンプル市場比較のrows / races
14. 同一サンプル市場比較の各指標
15. 旧市場比較から変わった点
16. fixed-width calibration結果
17. quantile calibration結果
18. SHAP再生成結果
19. feature importance再生成結果
20. artifact hash
21. V1.0.1のvalidation / test指標
22. ROI90%目標の文書化
23. 次に確率キャリブレーションへ進める状態か
24. 未解決事項

完了後はここで停止する。
