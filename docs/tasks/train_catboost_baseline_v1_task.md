# Task: CatBoost GPU Classification Baseline V1

## 目的

`keiba_prediction_ai` のV2.1.1特徴量データセットを使い、Phase 1の最初のモデルとしてCatBoost GPU分類ベースラインを構築・評価する。

今回の対象:

- 単勝的中確率
- 複勝的中確率
- `market_free`
- `market_history`
- `market_aware`

合計6モデルを、同一の時系列分割と評価方法で比較する。

今回は以下を行わない。

- LightGBM Ranker
- Optunaなどの大規模ハイパーパラメータ探索
- 確率キャリブレーションの適用
- 馬券購入戦略
- EV閾値最適化
- 資金配分
- 本番予測API
- 発走前時系列オッズ
- 2025年の結果を使ったモデル選択
- 2026年を通常テストへ混ぜること

---

## 参照する現行資産

最初に現在の`main`を確認する。

- `scripts/build_model_features_v2_1_1.py`
- `config/feature_sets_v2_1_1.yaml`
- `src/features/feature_sets_v2_1_1.py`
- `src/features/target_builder.py`
- `docs/model_feature_design_v2_1_1.md`
- `docs/feature_set_design_v2_1_1.md`
- `docs/target_definition_v2.md`
- `outputs/model_feature_dataset_v2_1_1/year=YYYY/data.parquet`
- `outputs/training_eligibility_summary_v2_1.csv`
- `outputs/feature_set_validation_v2_1_1.csv`
- `requirements.txt`
- `.gitignore`
- `README.md`

V1、V2、V2.1、V2.1.1の特徴量生成コード・データは変更しない。

---

# 1. 学習タスク

## 単勝モデル

正式ターゲット:

```text
target_win_paid
```

学習対象:

```text
eligible_for_win_training == True
```

## 複勝モデル

正式ターゲット:

```text
target_place_paid
```

学習対象:

```text
eligible_for_place_training == True
```

以下を正式ターゲットとして使わない。

- `target_win_rank`
- `target_ren_rank`
- `target_top3_rank`
- `target_place_by_rule`

これらは診断用として保持するだけにする。

---

# 2. データ分割

時系列順を厳守する。

```text
train:
  2016〜2023

validation:
  2024

test:
  2025

latest_holdout:
  2026
```

ルール:

- trainで学習する
- validationをearly stoppingとモデル比較に使う
- testは最終評価専用
- test結果を見てハイパーパラメータや特徴量を変更しない
- 2026は年途中の別ホールドアウトとして報告する
- 2026をearly stopping、モデル選択、閾値決定に使わない
- race_idが複数分割へ重複していないことを検証する
- entry_idが複数分割へ重複していないことを検証する

2025の評価後にtrainへ2024や2025を追加して再学習する処理は、今回は行わない。

---

# 3. feature set

必ず次の専用設定を読み込む。

```text
config/feature_sets_v2_1_1.yaml
```

対象:

- `market_free`
- `market_history`
- `market_aware`

YAMLに記載されたnumericとcategoricalの明示的許可リストだけを使う。

禁止:

- DataFrameの残り列を自動的に特徴量へ追加
- 列名パターンから暗黙に追加
- 欠損列を無断で削除
- feature setを学習スクリプト内へ二重定義

学習前に次を検証する。

- 指定列がデータセットに存在する
- 特徴量数がYAMLと一致する
- ターゲット列が含まれない
- 着順、払戻、確定後情報が含まれない
- race_id、entry_idを特徴量にしない
- 学習対象フラグや除外理由を特徴量にしない
- `market_free`に市場情報が含まれない
- `market_history`に当該レース市場情報が含まれない
- `market_aware`が`market_history`を包含する

検証失敗時は学習せず停止する。

---

# 4. 入力データの固定と追跡

モデル実行ごとに次を保存する。

- 使用した年別Parquet一覧
- 各Parquetのサイズ
- 各Parquetの更新日時
- 各ParquetのSHA-256または既存fingerprint
- feature set YAMLのSHA-256
- 学習コードbundle hash
- Git commit SHA
- Git dirty状態
- Pythonバージョン
- CatBoostバージョン
- pandas / numpy / scikit-learn / pyarrowバージョン
- GPU名
- CUDA利用可否
- 実行日時
- random seed

出力例:

```text
outputs/model_training/catboost_baseline_v1/run_manifest.json
```

同じrun IDの出力を無断で上書きしない。

---

# 5. CatBoost GPU

RTX 5070 Tiを使用する。

学習前にGPUを確認し、ログへ出力する。

基本設定:

```python
task_type="GPU"
devices="0"
loss_function="Logloss"
eval_metric="Logloss"
random_seed=42
allow_writing_files=False
```

GPU学習に失敗した場合:

- 黙ってCPUへ切り替えない
- エラー内容を記録して停止
- `--task-type CPU`が明示された場合だけCPUを許可

---

# 6. ベースラインの固定パラメータ

今回は大規模探索をしない。

初期値:

```yaml
iterations: 3000
learning_rate: 0.05
depth: 8
l2_leaf_reg: 5.0
random_strength: 1.0
bootstrap_type: Bayesian
bagging_temperature: 1.0
loss_function: Logloss
eval_metric: Logloss
od_type: Iter
od_wait: 200
random_seed: 42
task_type: GPU
devices: "0"
allow_writing_files: false
verbose: 100
```

現在のCatBoostで無効な組み合わせがある場合は、公式APIと実際のエラーを確認して最小限修正する。

変更した場合は、理由と最終設定を記録する。

今回は以下を行わない。

- grid search
- Optuna
- testデータを見た調整
- targetごとに恣意的に多数の設定を試すこと

---

# 7. クラス不均衡

正式ベースラインでは以下を使わない。

- oversampling
- undersampling
- SMOTE
- `auto_class_weights="Balanced"`
- 人為的なclass weight

理由:

- 今回は実的中率に近い確率出力を評価する
- class weightで確率の意味が変わることを避ける

positive率は分割・target別に必ず記録する。

将来、重み付きモデルを試す場合は別実験として分離する。

---

# 8. 欠損値と型

## numeric

- 数値型へ統一
- `inf`、`-inf`はNaNへ変換
- CatBoostの数値欠損処理を使用
- 全欠損列があれば警告し、勝手に削除せず報告

## categorical

- 文字列型へ統一
- NULL、NaN、空文字は`"__MISSING__"`へ統一
- 数値コードも文字列カテゴリとして扱う
- train/validation/testで同じ変換を使う

予測時も再利用できるよう、前処理定義をコードへ分離する。

---

# 9. 実装構成

推奨構成:

```text
scripts/train_catboost_baseline.py

src/models/catboost_data.py
src/models/catboost_metrics.py
src/models/catboost_runner.py
src/models/model_manifest.py

config/catboost_baseline_v1.yaml

tests/test_catboost_data.py
tests/test_catboost_metrics.py
tests/test_catboost_runner.py
```

処理を巨大な単一スクリプトへまとめない。

CLI例:

```powershell
python scripts\train_catboost_baseline.py --target win --feature-set market_free --task-type GPU
```

全6モデル:

```powershell
python scripts\train_catboost_baseline.py --all --task-type GPU
```

スモークテスト:

```powershell
python scripts\train_catboost_baseline.py --target win --feature-set market_free --task-type GPU --smoke-test
```

---

# 10. モデル組み合わせ

以下の6モデルを作る。

| target | feature set |
|---|---|
| win | market_free |
| win | market_history |
| win | market_aware |
| place | market_free |
| place | market_history |
| place | market_aware |

モデル名の例:

```text
catboost_win_market_free_v1
catboost_win_market_history_v1
catboost_win_market_aware_v1
catboost_place_market_free_v1
catboost_place_market_history_v1
catboost_place_market_aware_v1
```

---

# 11. early stopping

- train: 2016〜2023
- eval_set: 2024
- `use_best_model=True`
- 2024のLoglossでearly stopping
- best iterationを保存
- 2025と2026をeval_setに渡さない

モデル選択は2024の指標だけで行う。

---

# 12. 出力する予測

すべての分割について、1行=1出走馬で予測を出力する。

最低限の列:

```text
entry_id
race_id
race_date
Year
Umaban
KettoNum
target
feature_set
data_split
actual
pred_probability
eligible
tan_odds
fuku_odds_low
fuku_odds_high
place_rank_limit
```

市場列が特徴量セットに含まれない場合も、評価・将来のバックテスト用メタデータとして予測出力へ保持してよい。

ただし市場列を無断でモデル入力に加えない。

出力:

```text
outputs/model_training/catboost_baseline_v1/predictions/
  win_market_free.parquet
  win_market_history.parquet
  win_market_aware.parquet
  place_market_free.parquet
  place_market_history.parquet
  place_market_aware.parquet
```

予測確率はすべて0〜1であることを検証する。

---

# 13. 評価指標

分割別、target別、feature set別に以下を計算する。

## 確率評価

- Logloss
- Brier score
- ROC-AUC
- PR-AUC / Average Precision
- positive率
- 予測確率平均
- サンプル数
- positive数
- negative数

AUCが単一クラス等で計算不能な場合は、理由を記録してNULLとする。

## レース単位評価

### 単勝

各race_idで予測確率最大の1頭について:

- top1 winner accuracy
- top3 predicted horses内の勝馬包含率
- race数
- 同率最大がある場合の件数

### 複勝

各race_idで`place_rank_limit`頭を上位選択し:

- precision@k
- recall@k
- hit race rate
- race数

4頭以下など`place_rank_limit=0`のレースは、ルールを明記して除外する。

## 確率合計の診断

race_id単位で以下を集計する。

単勝:

```text
sum(pred_win_probability)
```

複勝:

```text
sum(pred_place_probability)
```

平均、中央値、標準偏差、最小、最大を分割別に出力する。

この段階では確率をレース内正規化しない。

---

# 14. calibration診断

今回は確率キャリブレーションを適用しない。

ただし診断として以下を出す。

- 予測確率10分位
- 各binの件数
- 各binの平均予測確率
- 各binの実的中率
- 予測と実績の差

validation、test、latest_holdoutを分けて出力する。

出力:

```text
outputs/model_training/catboost_baseline_v1/calibration_bins.csv
```

calibration手法の選択・適用は次タスクで行う。

---

# 15. feature importance

各モデルについて以下を出力する。

- CatBoost標準feature importance
- 上位30特徴量
- 可能ならvalidationから固定seedで最大5,000行を抽出したSHAP importance

SHAPがGPU・ライブラリ・メモリ上の理由で実行できない場合:

- 学習全体を失敗扱いにしない
- 理由をログへ記録
- 標準feature importanceは必ず出す

出力:

```text
outputs/model_training/catboost_baseline_v1/feature_importance.csv
outputs/model_training/catboost_baseline_v1/shap_importance.csv
```

---

# 16. モデル保存

モデルごとに以下を保存する。

```text
models/catboost_baseline_v1/<target>/<feature_set>/
  model.cbm
  model_metadata.json
  feature_columns.json
  categorical_columns.json
  training_config.json
  metrics.json
```

保存後にモデルを再読込し、同じサンプルに対する予測が一致することを確認する。

大容量モデルファイルは必要に応じて`.gitignore`へ追加する。

---

# 17. 比較表

以下の比較表を作成する。

```text
outputs/model_training/catboost_baseline_v1/model_comparison.csv
```

列例:

```text
target
feature_set
best_iteration
train_rows
validation_rows
test_rows
latest_holdout_rows
validation_logloss
test_logloss
latest_holdout_logloss
validation_brier
test_brier
validation_roc_auc
test_roc_auc
validation_pr_auc
test_pr_auc
validation_race_metric
test_race_metric
training_seconds
prediction_seconds
gpu_name
model_path
```

モデルの正式順位付けはvalidationを主に使う。

2025 testと2026 latest_holdoutは、順位変更のために使用しない。

---

# 18. 市場ベースライン

単勝について、`tan_odds > 0`の行だけを対象に比較用の市場確率を作る。

```text
raw_implied_probability = 1 / tan_odds
market_probability =
  raw_implied_probability /
  race内raw_implied_probability合計
```

validation、test、latest_holdoutで以下を比較する。

- 市場確率のLogloss
- 市場確率のBrier score
- 市場人気1位の勝率
- CatBoostとの指標差

これは比較用であり、モデル入力への追加ではない。

複勝は上下限オッズから一意の的中確率を作れないため、今回は市場確率ベースラインを無理に作らない。

---

# 19. 投資評価をまだ行わない

今回は以下を計算しない。

- EV
- 回収率
- 購入額
- 収支
- 最大ドローダウン
- Kelly基準
- オッズ閾値
- 買い目
- 推奨馬券

確定オッズは市場比較や予測出力メタデータには使えるが、バックテストは別タスクで行う。

---

# 20. テスト

最低限以下をpytestで確認する。

1. V2.1.1専用feature set YAMLを読み込む
2. feature setの全列がParquetに存在する
3. 禁止列がモデル入力に含まれない
4. 年分割が重複しない
5. race_idが分割間で重複しない
6. entry_idが分割間で重複しない
7. eligibilityで正しく絞り込む
8. winとplaceで正式ターゲットが異なる
9. categorical欠損が`__MISSING__`になる
10. 数値のinfがNaNになる
11. 予測確率が0〜1
12. metricsが既知の小データで正しく計算される
13. race単位top-k指標が正しく計算される
14. model保存・再読込後の予測が一致する
15. 2025をearly stoppingへ使っていない
16. 2026をモデル選択へ使っていない
17. market_freeへ市場列が混入しない
18. smoke-testが小規模データで完了する

GPUがないCI環境でもpytestが動くよう、GPU実機テストと純粋な単体テストを分離する。

---

# 21. 実行手順

## 依存関係

`requirements.txt`を確認し、不足分だけ追加する。

最低限候補:

```text
catboost
scikit-learn
```

既存パッケージの不要な大規模更新はしない。

## 構文・テスト

```powershell
python -m py_compile scripts\train_catboost_baseline.py
python -m pytest -q
```

## GPU確認

CatBoostがGPUを認識するか確認し、結果をログへ残す。

## スモークテスト

```powershell
python scripts\train_catboost_baseline.py --target win --feature-set market_free --task-type GPU --smoke-test
```

小規模データで以下を確認する。

- データ読込
- 前処理
- GPU学習
- 予測
- metrics
- model保存
- model再読込

## 1モデル試験

```powershell
python scripts\train_catboost_baseline.py --target win --feature-set market_free --task-type GPU
```

問題がなければ全6モデル:

```powershell
python scripts\train_catboost_baseline.py --all --task-type GPU
```

エラー発生時は原因を修正し、未完了モデルだけを再開できるようにする。

---

# 22. 再開機能

全6モデルの一部だけ完了した場合に再開できるようにする。

例:

```powershell
python scripts\train_catboost_baseline.py --all --task-type GPU --resume
```

モデルごとに以下を検証する。

- 入力fingerprint
- feature set hash
- training config hash
- code bundle hash
- modelファイル存在
- metrics存在
- predictions存在

不一致時はそのモデルだけ再学習対象とするか、strictモードでは停止する。

必要なら追加:

```text
--strict-resume
--force
```

既存成果物を無断上書きしない。

---

# 23. 出力

```text
config/catboost_baseline_v1.yaml

scripts/train_catboost_baseline.py
src/models/catboost_data.py
src/models/catboost_metrics.py
src/models/catboost_runner.py
src/models/model_manifest.py

tests/test_catboost_data.py
tests/test_catboost_metrics.py
tests/test_catboost_runner.py

models/catboost_baseline_v1/

outputs/model_training/catboost_baseline_v1/
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
  predictions/

logs/train_catboost_baseline_v1.log

docs/catboost_baseline_v1_design.md
docs/catboost_baseline_v1_results.md
```

---

# 24. README更新

READMEへ以下を追記する。

- Phase 1 CatBoost baselineの位置づけ
- 使用データはV2.1.1
- 3つのfeature set
- win/placeの正式ターゲット
- 年分割
- GPU実行方法
- 全6モデル実行方法
- モデル学習は実施済みだがバックテストは未実施であること
- 出力場所

---

# 25. 完了条件

以下をすべて満たす。

- pytest全件成功
- GPUスモークテスト成功
- 6モデルすべて学習成功
- 2024だけをearly stoppingに使用
- 2025をモデル選択に使っていない
- 2026をモデル選択に使っていない
- feature set検証成功
- 禁止列混入0
- entry_id重複0
- 分割重複0
- 全予測確率が0〜1
- 全モデル保存・再読込成功
- metricsと予測を保存
- market_free / market_history / market_awareを比較
- V1〜V2.1.1の特徴量データを変更していない
- バックテストや購入戦略へ進んでいない

1つでも満たさない場合、CatBoost baseline完了とは判定しない。

---

# 26. 最終報告

1. `git diff`と変更ファイル
2. 実装構成
3. 使用データとfingerprint
4. Python・CatBoost・CUDA・GPU情報
5. pytest結果
6. GPUスモークテスト結果
7. 学習データ分割件数
8. win/placeのpositive率
9. 6モデルの最終パラメータ
10. 各モデルのbest iteration
11. validation指標
12. test指標
13. latest_holdout指標
14. レース単位指標
15. race内確率合計診断
16. calibration診断
17. 単勝市場ベースライン比較
18. feature importance上位
19. model保存・再読込確認
20. 学習時間と予測時間
21. validation基準で最良だったfeature set
22. test結果がvalidationと大きく乖離していないか
23. 次に確率キャリブレーションへ進める状態か
24. 未解決事項

完了後はここで停止する。
