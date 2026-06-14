# Task: CatBoost Baseline V1.0.2 Analysis Integrity Fix

## 目的

CatBoost Baseline V1.0.1のモデル本体と既存成果物を保持したまま、次工程の確率キャリブレーションへ進む前に、以下の分析・再利用・設定上の問題を修正する。

1. 市場比較を完全なレースだけで行う
2. 既存V1モデルを再利用する場合、現在データ全行を再予測する
3. split定義をYAMLへ接続し、一元管理する
4. quantile calibration binで同一確率値を分断しない
5. 分析CSVを全件置換し、古い行を残さない
6. 手編集ドキュメントを自動上書きしない
7. Git状態を正直にmanifestへ記録する

今回はモデル性能改善、確率キャリブレーション適用、ROI・EV・買い目生成、Ability、ANA、Ranker、Optunaは行わない。

Phase 1の最終目標は引き続き次のとおり。

```text
単勝回収率 >= 90%
複勝回収率 >= 90%
```

ただし、穴馬1頭や高額払戻1件への依存を達成扱いにしない。

---

# 1. 現在のmainを確認する

最初に以下を確認する。

```text
config/catboost_baseline_v1_0_1.yaml

scripts/train_catboost_baseline_v1_0_1.py
scripts/analyze_catboost_baseline_v1_0_1.py

src/models/catboost_data.py
src/models/catboost_metrics.py
src/models/catboost_runner.py
src/models/model_manifest.py
src/models/catboost_config.py
src/models/catboost_resume.py
src/models/catboost_analysis.py

tests/test_catboost_*.py

models/catboost_baseline_v1/
models/catboost_baseline_v1_0_1/

outputs/model_training/catboost_baseline_v1/
outputs/model_training/catboost_baseline_v1_0_1/

docs/catboost_baseline_v1_0_1_design.md
docs/catboost_baseline_v1_0_1_results.md
```

最初に実行する。

```bash
git status
git diff
python -m pytest -q
```

V1〜V2.1.1の特徴量生成コード、Parquet、stateは変更しない。

---

# 2. V1.0.2を別バージョンとして作る

推奨構成:

```text
config/catboost_baseline_v1_0_2.yaml

scripts/train_catboost_baseline_v1_0_2.py
scripts/analyze_catboost_baseline_v1_0_2.py

src/models/catboost_market_comparison.py
src/models/catboost_prediction_regeneration.py
src/models/catboost_atomic_output.py

tests/test_catboost_market_comparison.py
tests/test_catboost_prediction_regeneration.py
tests/test_catboost_atomic_output.py

models/catboost_baseline_v1_0_2/
outputs/model_training/catboost_baseline_v1_0_2/
logs/train_catboost_baseline_v1_0_2.log

docs/catboost_baseline_v1_0_2_design.md
docs/catboost_baseline_v1_0_2_results.md
```

既存モジュールを安全に共通化してもよいが、V1とV1.0.1の成果物を上書きしない。

---

# 3. 修正A: 完全なレースだけで市場比較する

## 現在の問題

一部の出走馬だけに有効オッズがあるレースでも、残った馬だけで市場確率を正規化するとrace内確率合計が1になる。

そのため、確率合計が1であっても完全なレースとは限らない。

## 比較母集団

各race_idについて、単勝学習の対象母集団を次とする。

```text
eligible_for_win_training == True
actual in {0, 1}
```

この母集団に属する全出走馬について、以下を要求する。

```text
entry_idが一意
tan_oddsが非欠損
tan_odds > 0
market_free予測が存在
market_history予測が存在
market_aware予測が存在
actualが0または1
```

さらにrace単位で次を要求する。

```text
有効比較行数 == 単勝学習対象母集団の行数
actualの合計 >= 1
```

同着などにより`actual`合計が2以上になることは許可する。

`actual`合計が0のレースは除外する。

## 完全レース判定

明示的な列を作成する。

```text
expected_runner_count
valid_odds_runner_count
valid_prediction_runner_count
actual_positive_count
is_complete_market_race
market_exclusion_reason
```

除外理由候補:

```text
missing_odds_runner
invalid_odds_runner
missing_model_prediction
duplicate_entry_id
invalid_actual
missing_winner
runner_count_mismatch
```

複数理由がある場合は、優先順位を固定するか、複数理由を別列で保存する。

## 市場確率

完全レースだけに対して計算する。

```text
raw_implied_probability = 1 / tan_odds

market_probability =
    raw_implied_probability /
    race内raw_implied_probability合計
```

完全レースではrace内market_probability合計が数値誤差内で1になることを確認する。

許容誤差例:

```text
abs(sum - 1.0) <= 1e-10
```

## 同一サンプル比較

以下を同じentry_id・race_id集合で比較する。

```text
market_probability
catboost_market_free
catboost_market_history
catboost_market_aware
```

分割:

```text
validation
test
latest_holdout
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
- race内確率合計統計

出力:

```text
market_comparison_complete_races.csv
market_comparison_complete_race_summary.csv
market_comparison_excluded_races.csv
market_comparison_exclusion_summary.csv
```

旧V1.0.1市場比較は保持するが、V1.0.2の正式比較には使用しない。

---

# 4. 修正B: 再利用モデルで現在データ全行を再予測する

## 現在の問題

既存予測Parquetをコピーするだけでは、現在の入力データ・actual・splitと完全に対応しているか保証できない。

## 方針

既存V1またはV1.0.1のモデル重みを再利用する場合も、予測ファイルは必ず現在のV2.1.1データから再生成する。

処理:

```text
既存model.cbmをロード
↓
現在のV2.1.1を読み込む
↓
現在のfeature columnsで前処理
↓
全eligible行を再予測
↓
現在のactual・split・metadataと結合
↓
V1.0.2 predictionsを新規生成
```

## 再利用前に確認する項目

- source run manifestが存在
- source input fingerprintと現在input fingerprintが一致
- feature set YAML hashが一致
- targetが一致
- feature columnsが完全一致
- categorical columnsが完全一致
- resolved training paramsが一致
- split定義が一致
- random seedが一致
- task typeが一致
- modelファイルが読める

source manifestで証明できない項目がある場合は、完全一致と推測しない。

その場合:

```text
strict mode: 再利用不可として停止
normal mode: 対象モデルだけ再学習
```

## 全行予測比較

旧予測と新規再予測をentry_idで結合し、全行について比較する。

保存する統計:

```text
compared_rows
missing_in_old
missing_in_new
max_abs_diff
mean_abs_diff
p99_abs_diff
mismatch_count
tolerance
```

推奨許容誤差:

```text
1e-10
```

CatBoost・GPU環境により微小差が出る場合は、根拠を記録して最小限変更する。

許容誤差を超えた場合:

- 旧予測を採用しない
- 対象モデルを再学習する
- 理由をmanifestへ記録する

## manifest記録

```text
model_origin: reused_from_v1 | reused_from_v1_0_1 | retrained_v1_0_2
predictions_origin: regenerated_from_current_v2_1_1
source_model_path
source_manifest_path
full_prediction_comparison
```

---

# 5. 修正C: split定義をYAMLへ接続する

## 正本

```text
config/catboost_baseline_v1_0_2.yaml
```

例:

```yaml
splits:
  train:
    years: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
  validation:
    years: [2024]
  test:
    years: [2025]
  latest_holdout:
    years: [2026]
```

文字列範囲を使う場合も、内部では明示的な年リストへ解決する。

## 要件

- `catboost_data.py`の固定`SPLIT_BY_YEAR`を学習の正本として使わない
- `load_dataset()`または分割関数へresolved split configを渡す
- 同じ年が複数splitに入らない
- 必須splitが欠落しない
- データに存在しない年を警告またはエラーにする
- splitごとのrows、races、entry_idを検証する
- race_idとentry_idがsplit間で重複しない

保存:

```text
split_definition_resolved.json
split_definition_sha256
```

split hashをmodel fingerprintへ含める。

今回は年分割の内容自体は変更しない。

---

# 6. 修正D: quantile binで同一確率値を分断しない

## 禁止

```python
rank(method="first")
```

で同じ予測確率を出現順に別binへ強制分割しない。

## 推奨

```python
pd.qcut(
    pred_probability,
    q=10,
    duplicates="drop",
)
```

または同等の、同一値を同一binへ保つ実装を使用する。

## 要件

- 同じpred_probabilityは同じbinに入る
- bin数が10未満でも正常
- 実bin数を保存
- 各binのlower/upper boundを保存
- binの件数合計が元split件数と一致
- fixed-width binは従来どおり別に出す

出力列:

```text
target
feature_set
data_split
bin_type
requested_bin_count
actual_bin_count
bin_id
lower_bound
upper_bound
count
mean_pred_probability
actual_rate
calibration_gap
```

今回は確率補正を適用しない。

---

# 7. 修正E: 分析CSVを毎回全件置換する

## 現在の問題

upsert方式では、新しい分析で消えたbinや特徴量行が古いCSVに残る可能性がある。

## 方針

分析スクリプトは毎回、対象6モデルから完全な表を再生成する。

書込方式:

```text
一時ファイルへ全件書込
↓
flush / fsync
↓
os.replaceで原子的に置換
```

単純appendや既存CSVとのmergeを行わない。

対象:

- model_comparison.csv
- split_summary.csv
- class_balance.csv
- metrics_by_split.csv
- race_metrics.csv
- race_probability_sum_diagnostics.csv
- calibration_bins.csv
- feature_importance.csv
- shap_importance.csv
- market comparison系CSV

テスト:

- 同じ分析を2回実行して内容hashが一致
- 行数が一致
- 古いダミー行が残らない
- 書込失敗時に既存正常ファイルを破壊しない

---

# 8. 修正F: 手編集ドキュメントを自動上書きしない

以下は人間が編集する資料として扱う。

```text
docs/catboost_baseline_v1_0_2_design.md
docs/catboost_baseline_v1_0_2_results.md
```

学習・分析スクリプトから無条件に上書きしない。

機械生成サマリーは別ファイルにする。

```text
outputs/model_training/catboost_baseline_v1_0_2/run_summary.md
outputs/model_training/catboost_baseline_v1_0_2/analysis_summary.md
```

既存V1.0.1の詳細docsも上書きしない。

テスト:

- strict resume後もdocsの内容hashが変わらない
- analyze再実行後もdocsの内容hashが変わらない

---

# 9. 修正G: Git状態を正直に記録する

manifestへ以下を保存する。

```text
git_commit_sha
git_is_dirty
git_status_summary
```

重要:

- dirty状態を偽ってfalseにしない
- Git SHAだけの変更でresumeを失敗させない
- code bundle hashが同じならdocs変更だけでモデル再学習しない
- Codexは自動commitしない
- clean manifestが必要な場合は、ユーザーがcommitした後に再度manifest更新またはstrict resumeを行う手順を記載する

V1.0.2作業中に`git_is_dirty=true`でも、それ自体は失敗ではない。

最終報告では、manifest生成時点のGit状態を明記する。

---

# 10. データ分割とモデル設定は変更しない

今回の分割:

```text
train: 2016〜2023
validation: 2024
test: 2025
latest_holdout: 2026
```

今回の6モデル:

```text
win × market_free
win × market_history
win × market_aware
place × market_free
place × market_history
place × market_aware
```

固定学習パラメータも原則変更しない。

モデル再学習が必要になった場合だけ、V1.0.2 YAMLの同一設定で再学習する。

---

# 11. ROI目標の扱い

manifestと設計資料に継続して保存する。

```yaml
phase1_goal:
  win_roi_min: 0.90
  place_roi_min: 0.90
```

補助条件:

- 十分な購入数
- validationで条件決定
- testで調整禁止
- 期間別安定性
- オッズ帯別安定性
- 高額払戻除外後ROI
- 穴馬1頭依存を達成扱いにしない

今回はROIを計算しない。

---

# 12. 必須テスト

## 完全レース市場比較

1. 全eligible runnerに有効オッズがあるレースだけ採用
2. 1頭でもオッズ欠損ならレース全体を除外
3. 1頭でもモデル予測欠損ならレース全体を除外
4. actual合計0なら除外
5. 同着でactual合計2以上は許可
6. entry_id重複を検出
7. expected countとvalid count一致
8. race内市場確率合計が1
9. 4比較対象のentry_id集合一致
10. 除外理由件数が正しい

## モデル再利用と再予測

11. source manifest fingerprint一致時だけ再利用
12. source fingerprint欠落時は再利用不可
13. 現在データ全行を再予測
14. 旧新予測を全行比較
15. tolerance超過時は対象モデル再学習
16. current actual / splitを予測出力へ使用
17. model再読込後の予測一致

## Split

18. YAML splitが実データ分割へ反映
19. 年重複を検出
20. split欠落を検出
21. race_id / entry_idのsplit間重複0
22. split definition hashを保存

## Calibration

23. 同じ確率値が同じquantile binへ入る
24. duplicates dropでbin数減少を許可
25. bin件数合計が元件数と一致
26. fixed-widthも維持

## Atomic output

27. 2回分析で内容hash一致
28. 古い行が残らない
29. 書込失敗時に既存CSVを保持

## Docs / Git

30. 学習・分析でdocsを上書きしない
31. Git dirty状態を正しく保存
32. Git SHAだけの変更ではresume失敗にしない

---

# 13. 実行手順

## 構文・テスト

```bash
python -m py_compile scripts/train_catboost_baseline_v1_0_2.py
python -m py_compile scripts/analyze_catboost_baseline_v1_0_2.py
python -m pytest -q
```

## GPUスモーク

```bash
python scripts/train_catboost_baseline_v1_0_2.py \
  --target win \
  --feature-set market_free \
  --task-type GPU \
  --smoke-test
```

CPUへ自動fallbackしない。

## 6モデル再利用・再予測

```bash
python scripts/train_catboost_baseline_v1_0_2.py \
  --all \
  --task-type GPU \
  --reuse-compatible-models
```

再利用可能なモデルは重みを再利用し、現在データ全行を再予測する。

不一致モデルだけ再学習する。

## strict resume

```bash
python scripts/train_catboost_baseline_v1_0_2.py \
  --all \
  --task-type GPU \
  --resume \
  --strict-resume
```

## 分析

```bash
python scripts/analyze_catboost_baseline_v1_0_2.py --all
```

同じ分析をもう一度実行し、CSV内容hashが一致することを確認する。

---

# 14. 出力

```text
config/catboost_baseline_v1_0_2.yaml

scripts/train_catboost_baseline_v1_0_2.py
scripts/analyze_catboost_baseline_v1_0_2.py

src/models/catboost_market_comparison.py
src/models/catboost_prediction_regeneration.py
src/models/catboost_atomic_output.py

tests/test_catboost_market_comparison.py
tests/test_catboost_prediction_regeneration.py
tests/test_catboost_atomic_output.py

models/catboost_baseline_v1_0_2/

outputs/model_training/catboost_baseline_v1_0_2/
  run_manifest.json
  split_definition_resolved.json
  training_config_resolved.json
  model_comparison.csv
  split_summary.csv
  class_balance.csv
  metrics_by_split.csv
  race_metrics.csv
  race_probability_sum_diagnostics.csv
  calibration_bins.csv
  feature_importance.csv
  shap_importance.csv
  market_comparison_complete_races.csv
  market_comparison_complete_race_summary.csv
  market_comparison_excluded_races.csv
  market_comparison_exclusion_summary.csv
  prediction_regeneration_comparison.csv
  run_summary.md
  analysis_summary.md
  predictions/

logs/train_catboost_baseline_v1_0_2.log

docs/catboost_baseline_v1_0_2_design.md
docs/catboost_baseline_v1_0_2_results.md
```

---

# 15. 完了条件

- pytest全件成功
- YAML splitが実処理へ接続
- 6モデルの再利用可否をsource manifestで検証
- 再利用モデルは現在データ全行を再予測
- 旧新予測を全行比較
- 不一致モデルだけ再学習
- 完全レースだけで市場比較
- 除外理由を保存
- 4比較対象のentry_id集合一致
- quantile binで同一値を分断しない
- 分析CSVを原子的に全置換
- 再分析で内容hash不変
- docsを自動上書きしない
- Git状態を正直にmanifestへ保存
- V1〜V2.1.1特徴量生成資産を変更していない
- ROI、EV、買い目、キャリブレーション適用を行っていない
- Ability / Rankerを実装していない

1つでも満たさない場合、V1.0.2完了とは判定しない。

---

# 16. 最終報告

1. git status / git diff
2. 変更・追加ファイル
3. V1.0.1で確認した問題
4. 完全レースの定義
5. 市場比較から除外したレース数と理由
6. 完全レース比較のrows / races / positives
7. 市場と3モデルの同一サンプル指標
8. V1.0.1比較から変わった点
9. source model再利用判定方法
10. 6モデルの再利用・再学習状況
11. 全行再予測件数
12. 旧新予測のmax / mean / p99差
13. split YAML接続方法
14. split definition hash
15. quantile tie処理
16. fixed-width / quantile bin結果
17. atomic CSV置換結果
18. 2回分析後の内容hash比較
19. docs非上書き確認
20. pytest結果
21. GPUスモーク結果
22. strict resume結果
23. artifact hash
24. manifest生成時のGit状態
25. validation / test / latest_holdout指標
26. ROI90%目標の文書化状況
27. 確率キャリブレーションへ進める状態か
28. 未解決事項

完了後はここで停止する。
