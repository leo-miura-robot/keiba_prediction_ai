# C1R0 Pure Market Offset 実装・比較タスク v1

## 0. 目的

現行の複勝市場残差CatBoostモデル `C1_market_offset_fundamental` を基準として、
市場情報をbaselineにのみ使用し、CatBoost残差側には市場以外の競馬情報だけを入力する
最も素直な市場残差モデルを新規実装・学習・比較する。

新モデル名:

```text
C1R0_pure_market_offset
```

基本構造:

```text
final_logit = market_logit + residual_fundamental
p_final = sigmoid(final_logit)
```

- `market_logit` はCatBoost Poolのbaselineとしてのみ使用する
- CatBoost側には市場情報を入れない
- `Year`は学習特徴から除外し、時系列分割・集計・監査専用とする
- 新特徴量は追加しない
- 既存feature datasetを再利用する

このタスクの目的はROI最大化ではなく、現行C1よりも意味が明確で年度間に安定した市場残差モデルを作ることである。

---

## 1. 前提資料

最初に以下を最初から最後まで読む。

```text
tasks/audit_place_market_offset_feature_importance_v1_task.md
docs/place_market_offset_feature_audit_v1_results.md
docs/place_market_offset_catboost_v1_design.md
docs/place_market_offset_catboost_v1_results.md
```

存在する場合は以下も読む。

```text
keiba_ai_handover_market_offset_v1.md
```

存在しない場合は、存在しないことだけを記録し、作業を止めない。

あわせて、現行C1のconfig、学習コード、評価コード、監査成果物を確認する。

```text
config/place_market_offset_catboost_v1.yaml
scripts/build_place_market_baseline_v1.py
scripts/train_place_market_offset_catboost_v1.py
scripts/evaluate_place_market_offset_catboost_v1.py
scripts/run_place_market_offset_catboost_v1.py

outputs/place_market_offset_catboost_v1/
outputs/place_market_offset_feature_audit_v1/
```

---

## 2. 絶対条件

- 2016年以降のみ使用する
- random splitは禁止
- 正式なwalk-forwardを維持する
- 2020～2024だけでモデル比較・仕様判断を行う
- 2025/2026をモデル選択、特徴量選択、閾値選択、calibration選択に使わない
- 2025/2026は固定評価・診断のみ
- 既存feature datasetを再利用する
- feature datasetを再作成しない
- DBを読まない
- 新特徴量を追加しない
- ROI直接学習を行わない
- Ability/ANA分離を行わない
- Learning to Rankを行わない
- Kelly基準を使わない
- 自動購入を行わない
- 大規模Optunaを行わない
- 自動commit/pushを行わない
- 既存成果物を上書きしない
- 現行C1の成果物を変更しない

今回、モデル再学習はC1R0の新規学習に限って許可する。

---

## 3. 作業開始時の確認

最初に以下を実行し、結果を記録する。

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

既存差分を勝手に戻さない。

---

## 4. 現行C1の問題認識

監査では次が確認されている。

- `market_logit`がPool baselineとして使われている
- 同時に`p_market`と`market_logit`がCatBoost残差側特徴にも入っている
- `p_market`と`market_logit`のSpearman相関は全年度で1.0
- `Year`、`p_market`、`market_logit`がFeature Importance/SHAP上位
- 2024から2025にEV>=1件数が22件から655件へ急増
- 2025の市場単体EV>=1は64件
- 2025の正式C1では残差によりEV<1からEV>=1へ642件移動
- 2024評価用CatBoost残差モデルを2025へ適用するとEV>=1は70件
- 正式2025モデルでは残差分布が大きく拡大

C1R0では、市場情報の二重利用と`Year`による年度外挿を除く。

---

## 5. C1R0の特徴量方針

### 5.1 baseline専用

以下はCatBoost Poolのbaseline作成・推論にだけ使う。

```text
market_logit
```

`p_market`は評価・監査用に保持してよいが、CatBoostの学習特徴には含めない。

### 5.2 CatBoost残差側から必ず除外

#### 年度・分割管理

```text
Year
fold
fold_id
split
split_name
dataset_index
row_id
```

実際の列名はdataset/config/manifestから確認する。

#### 市場確率

```text
p_market
market_logit
```

#### raw市場情報

存在する場合はすべて残差側から除外する。

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
market_rank
p_market_rank
rank_gap
```

#### 市場由来の派生特徴

以下に相当する列を名前と生成コードから確認して除外する。

```text
odds
ninki
market
vote
betting
implied_probability
inverse_odds
odds_ratio
race_odds_ratio
odds_width
odds_rank
popularity_rank
```

ただし、`fuku_odds_low`等はEV計算・評価には使用してよい。
CatBoostの特徴としてのみ禁止する。

#### 一意性の高い管理ID

```text
race_id
entry_id
row_id
file_name
```

#### 馬個体ID

以下に相当する列はCatBoostから除外する。

```text
KettoNum
horse_id
```

馬名等の個体識別列も除外する。

#### 結果・払戻・当該レース後情報

以下に相当する列は絶対に除外する。

```text
finish_position
KakuteiJyuni
payoff
payout
place_payout
win_payout
result
target
label
```

目的変数は学習ラベルとしてのみ使用する。

### 5.3 原則として残す特徴群

#### レース・コース条件

```text
JyoCD
Kyori
TrackCD
CourseKubunCD
SibaBabaCD
DirtBabaCD
TenkoCD
SyussoTosu
place_rank_limit
```

- `JyoCD`、`TrackCD`、`CourseKubunCD`等のコード列は、既存C1と同じcategorical扱いを維持する
- `Kyori`は数値として扱う
- `CourseKubunCD`の意味が曖昧でも、今回は新特徴追加ではなく既存C1からの純化が目的なので、既存型のまま残す
- ただし、明らかな型誤りが見つかった場合は勝手に変更せず報告する

#### 枠・馬番・斤量・馬体

```text
Wakuban
Umaban
Futan
BaTaijyu
horse_body_weight_diff_last
horse_futan_diff_last
```

#### 馬の近走

```text
horse_days_since_last
horse_last1_*
horse_last3_*
horse_last5_*
horse_distance_diff_last
horse_futan_diff_last
horse_body_weight_diff_last
```

#### 馬の条件別適性

```text
horse_jyo_*
horse_surface_*
horse_dist_band_*
horse_baba_*
```

#### 騎手・調教師

```text
KisyuCode
ChokyosiCode
jockey_*
trainer_*
horse_jockey_*
```

今回は騎手・調教師コードを残す。
コードなし比較は別タスクとする。

### 5.4 比較保留だが今回は変更しない列

以下はC1R0では原則として現行C1の扱いを維持し、追加削除しない。

```text
Month
Kaiji
Nichiji
RaceNum
SyussoTosu
place_rank_limit
BaTaijyu
raw time系
```

ただし`Year`は必ず除外する。

---

## 6. allowlist方式

C1R0では、ブラックリストだけでなく最終的な明示allowlistを生成する。

成果物:

```text
feature_allowlist_c1r0.json
feature_exclusion_c1r0.csv
```

`feature_exclusion_c1r0.csv`の列:

```text
feature
present_in_dataset
present_in_c1
included_in_c1r0
reason
category
```

カテゴリ例:

```text
allowed_fundamental
baseline_only
market_forbidden
time_management_only
id_forbidden
result_leakage_forbidden
not_present
```

モデルへ渡す特徴量は、生成したallowlistと完全一致させる。

将来datasetへ列が増えても、自動的にC1R0へ入らないようにする。

---

## 7. モデル構造

現行C1のbaseline/offset処理を再利用する。

```text
final_logit = market_logit + catboost_residual_raw
p_final = sigmoid(final_logit)
```

学習時・推論時ともにCatBoost Poolへ正しい`market_logit` baselineを設定する。

次を検証する。

```text
raw_prediction_with_baseline
≈ market_logit + residual_raw
```

最大誤差を記録する。

---

## 8. 時系列分割

正式なwalk-forward:

```text
train 2016-2019 -> validation 2020
train 2016-2020 -> validation 2021
train 2016-2021 -> validation 2022
train 2016-2022 -> validation 2023
train 2016-2023 -> validation 2024
```

固定評価:

```text
test: 2025
latest_holdout: 2026
```

2025評価用final modelは、既存方針と同じ期間ルールを使用する。
2026についても既存方針に合わせる。

期間ルールを推測せず、現行C1コードとconfigから確認する。

---

## 9. 学習条件

原則として、特徴量以外の条件は現行C1と同じにする。

維持するもの:

- CatBoost loss
- depth
- learning rate
- iterations上限
- early stopping
- random seed
- categorical処理
- class weight設定
- thread/GPU設定
- baseline model
- calibration方針
- evaluation metric
- fold構成

特徴量以外の条件を同時に変更しない。

現行C1との差は、原則として残差側特徴量だけに限定する。

---

## 10. 比較対象

最低限次を比較する。

```text
B_market_baseline
C1_market_offset_fundamental
C1R0_pure_market_offset
```

### B

市場baselineのみ。

### C1

既存成果物を再利用する。
再学習しない。

### C1R0

今回新規学習する。

---

## 11. 主評価指標

### 11.1 2020～2024正式評価

- Logloss
- Brier
- ECE
- calibration slope
- calibration intercept
- EV-ROI Spearman
- 年度別EV>=1件数
- 年度別EV>=1率
- 年度別EV>=1 ROI
- validation combined ROI
- market-onlyからのEV閾値crossing件数

ROIだけで採用判断しない。

### 11.2 残差安定性

年度別・fold別に以下を出す。

```text
residual_raw mean
residual_raw std
residual_raw min
residual_raw p01
residual_raw p05
residual_raw p10
residual_raw p25
residual_raw p50
residual_raw p75
residual_raw p90
residual_raw p95
residual_raw p99
residual_raw max

abs_residual_raw p50
abs_residual_raw p90
abs_residual_raw p95
abs_residual_raw p99
```

特にC1とC1R0の年度間変動を比較する。

### 11.3 EV閾値安定性

年別に次を出す。

```text
market_only_ev_ge_1
final_ev_ge_1
market_lt1_to_final_ge1
market_ge1_to_final_lt1
ev_ge_1_year_over_year_ratio
```

### 11.4 高配当依存

EV>=1について以下を出す。

```text
normal ROI
top1 payout removed
top3 payout removed
top5 payout removed
top10 payout removed
bootstrap 95% CI
max losing streak
max drawdown
```

---

## 12. 2025/2026の扱い

2025/2026は、2020～2024でモデル仕様と評価方法を完全固定した後にのみ評価する。

2025/2026の結果を見て以下を変更してはならない。

- 特徴量
- hyperparameter
- baseline
- calibration
- EV閾値
- betting条件
- residual shrinkage
- model selection

診断として以下を比較する。

- Logloss/Brier/ECE
- residual分布
- EV>=1件数
- market-onlyからのcrossing件数
- ROI
- 高配当除外後ROI
- 2024から2025の残差分布変化
- 2025から2026の残差分布変化

C1で発生した2025年655件の急増が、C1R0で抑制されるか確認する。

---

## 13. Feature Importance / SHAP

C1R0について以下を実施する。

### CatBoost Feature Importance

```text
PredictionValuesChange
LossFunctionChange
```

### SHAP

- 2020～2024 global mean absolute SHAP
- 年度別
- 中山全体
- 中山芝
- 中山ダート

市場情報がC1R0のCatBoost特徴に紛れ込んでいないことを確認する。

SHAP加法性:

```text
residual_raw
≈ shap_expected_value + sum(feature_shap)

final_logit
≈ market_logit + shap_expected_value + sum(feature_shap)
```

---

## 14. 必須の安全検証

最低限、次をテストする。

1. DBへ接続しない
2. feature datasetを再作成しない
3. 2015年以前を含めない
4. random splitを使用しない
5. `Year`がC1R0 feature namesに含まれない
6. `p_market`がC1R0 feature namesに含まれない
7. `market_logit`がC1R0 feature namesに含まれない
8. raw odds/rank/vote由来列がC1R0 feature namesに含まれない
9. race/entry/horse IDがC1R0 feature namesに含まれない
10. 結果・払戻列が含まれない
11. allowlistとモデルfeature namesが一致する
12. categorical feature順序・型が一致する
13. baselineが学習・推論の両方で設定される
14. `final_logit = market_logit + residual_raw`
15. `sigmoid(final_logit) = p_final`
16. SHAP加法性
17. 2025/2026を選択に使用していない
18. 既存出力を上書きしない
19. seed固定で再現する

---

## 15. 実装候補

```text
config/place_market_offset_catboost_c1r0_v1.yaml
scripts/train_place_market_offset_catboost_c1r0_v1.py
scripts/evaluate_place_market_offset_catboost_c1r0_v1.py
scripts/run_place_market_offset_catboost_c1r0_v1.py
tests/test_place_market_offset_catboost_c1r0_v1.py
docs/place_market_offset_catboost_c1r0_v1_results.md
```

既存C1コードを安全に再利用・共通化してよい。

ただし、現行C1の挙動を変えない。
大規模リファクタは避ける。

---

## 16. 出力先

新規ディレクトリ:

```text
outputs/place_market_offset_catboost_c1r0_v1/
```

既存出力を上書きしない。

最低限の成果物:

```text
manifest.json
feature_allowlist_c1r0.json
feature_exclusion_c1r0.csv
model_comparison_2020_2024.csv
model_comparison_2025_2026_diagnostic.csv
metrics_by_year.csv
metrics_summary.csv
residual_distribution_by_year.csv
ev_threshold_crossing_by_year.csv
ev_roi_by_year.csv
high_payout_dependency.csv
bootstrap_summary.csv
catboost_pvc_summary.csv
catboost_lfc_summary.csv
shap_global_2020_2024.csv
shap_by_year.csv
prediction_output.parquet
run_log.txt
```

fold別モデルも既存形式に合わせて保存する。

---

## 17. 採用判断

C1R0を次の基準モデルとして採用する条件は、2020～2024で総合判断する。

望ましい状態:

- Logloss/Brierが市場baselineより改善
- 現行C1に対する性能悪化が小さい
- calibrationが大きく悪化しない
- EV-ROI Spearmanが正で安定
- residual p90/p95/p99の年度変動がC1より小さい
- EV>=1件数の年度間変動がC1より小さい
- 特定年度だけ残差が暴走しない
- Feature Importance/SHAPが馬・近走・適性中心になる
- 高配当依存が現行C1より悪化しない

2020～2024だけで採用判断を出す。

2025/2026結果は採用判断を変更するために使わず、固定診断として別記する。

---

## 18. 最終報告

日本語で以下を報告する。

1. 読み込んだファイル
2. 使用したdataset、manifest、baseline、モデル
3. DB未使用
4. feature dataset再作成なし
5. C1R0で採用した全特徴量
6. 除外した全特徴量と理由
7. C1とC1R0のfeature count
8. B/C1/C1R0の2020～2024比較
9. 年度別Logloss/Brier/ECE
10. residual分布比較
11. EV>=1件数・crossing比較
12. EV-ROI Spearman
13. ROIと高配当依存
14. 2025/2026固定診断
15. C1の2025急増がC1R0で改善したか
16. Feature Importance/SHAP上位
17. C1R0を新基準モデルにすべきか
18. 作成・変更ファイル
19. テスト結果
20. `git status --short`
21. `git diff --stat`

自動commit/pushは行わない。

---

## 19. 今回行わないこと

- residual shrinkage
- Month等の追加ablation
- 騎手・調教師コード除外比較
- `SyussoTosu`と`place_rank_limit`の重複比較
- 新しいコース物理特徴
- 血統特徴
- hyperparameter探索
- threshold最適化
- betting strategy最適化

これらはC1R0の結果を確認した後の別タスクとする。
