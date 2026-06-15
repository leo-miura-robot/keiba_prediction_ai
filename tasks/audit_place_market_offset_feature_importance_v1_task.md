# C1市場残差型複勝モデル 特徴量・寄与監査タスク v1

## 0. タスクの目的

最新の複勝モデル `C1_market_offset_fundamental` について、**モデルを再学習せず**、既存のfeature dataset、feature manifest、config、学習・評価コード、保存済みCatBoostモデル、既存予測出力を監査する。

主目的は次のとおり。

1. CatBoost Feature Importanceを確認する
2. SHAPで残差モデルが何を見ているか確認する
3. 特徴群別permutation importanceを測定する
4. 競馬場、距離、馬場、騎手、近走、血統、市場情報の寄与を分解する
5. 中山の急坂、小回り、直線長、内外回り等が明示特徴として存在するか確認する
6. 中山カテゴリだけを覚えているのか、馬・騎手のコース適性まで利用しているのか確認する
7. 血統特徴量および血統の条件別過去成績が実装されているか確認する
8. C1のEV>=1件数が2024年22件から2025年655件へ急増した理由を分解する

このタスクでは、新しい特徴量の追加、既存モデルの再学習、閾値調整、calibration再選択は行わない。

---

## 1. 絶対条件

- 2016年以降のみ使用する
- random splitは禁止
- 2025/2026をモデル選択、閾値選択、特徴量選択、calibration選択に使わない
- 2025/2026は監査・診断結果としてのみ表示する
- ROIだけで結論を出さない
- ROI直接学習は行わない
- Ability/ANA分離は行わない
- Learning to Rankは行わない
- Kelly基準は使わない
- 自動購入は行わない
- 大規模Optunaは行わない
- 自動commit/pushは禁止
- 既存成果物を上書きしない
- DBは読まない
- 既存Parquet、manifest、保存済みモデル、保存済み予測を使う
- 必要成果物が欠けている場合もDBへフォールバックしない
- `.fit()`、`train()`、`grid_search()`、`randomized_search()`等の再学習処理を呼ばない

---

## 2. 作業開始時の確認

最初に以下を実行し、結果を作業報告へ残す。

```bash
git status --short
git diff --stat
git diff
git log -5 --oneline
```

特に次を確認する。

- `config/database_validation.yaml` の既存差分
- C1関連ファイルがtrackedかuntrackedか
- 今回の監査作業と無関係な差分
- 既存出力を変更しないこと

自動commit/pushは行わない。

---

## 3. 最初に読むファイル

少なくとも以下を確認する。

```text
keiba_ai_handover_market_offset_v1.md
config/place_market_offset_catboost_v1.yaml
config/feature_sets_v2_1_2.yaml
config/model_features_v2_1_2.yaml

scripts/build_model_features_v2_1_2.py
scripts/build_place_market_baseline_v1.py
scripts/train_place_market_offset_catboost_v1.py
scripts/evaluate_place_market_offset_catboost_v1.py
scripts/run_place_market_offset_catboost_v1.py

src/features/history_builder_v2_1.py
src/features/history_builder_v2_1_2.py
src/features/feature_sets_v2_1_2.py

docs/place_market_offset_catboost_v1_design.md
docs/place_market_offset_catboost_v1_results.md

outputs/model_feature_dataset_v2_1_2/manifest.json
outputs/model_feature_dataset_v2_1_2/feature_inventory.csv
outputs/model_feature_dataset_v2_1_2/feature_set_validation.csv

outputs/place_market_offset_catboost_v1/
```

保存済みモデルの格納場所はconfig、manifest、学習コード、既存出力から特定する。推測だけでパスを決めない。

---

## 4. 成果物の同一性監査

監査開始時に、次の情報を取得して `input_artifact_inventory.csv` と `audit_manifest.json` に保存する。

- ファイルパス
- ファイル種別
- サイズ
- 更新日時
- SHA-256
- モデル名
- fold名
- train期間
- evaluation年
- feature count
- categorical feature count
- tree count
- best iteration
- CatBoost params
- feature names
- baseline使用有無
- calibration使用有無
- 予測出力行数
- 年別行数
- race数
- entry_id重複数

モデルの `feature_names_`、config上の特徴量、feature manifest、実際のParquet列が一致するか検証する。

不一致がある場合は処理を続けて隠さず、`status=fail` として報告する。

---

## 5. 現行特徴量の棚卸し

### 5.1 列の実在確認

以下の4段階を区別する。

1. feature datasetに存在
2. feature manifestに存在
3. C1 configに採用
4. 保存済みC1モデルのfeature namesに採用

`feature_schema_inventory.csv` を作成し、最低限以下の列を持たせる。

```text
column_name
dtype
exists_in_parquet
exists_in_feature_inventory
included_in_c1_config
included_in_saved_model
numeric_or_categorical
null_rate_2020_2024
unique_count_2020_2024
first_year_non_null
last_year_non_null
proposed_group
notes
```

### 5.2 公開版V2.1.2で確認できる主要特徴

ローカル実体で必ず再確認するが、少なくとも以下の存在を重点監査する。

#### レース・コース条件

- `JyoCD`
- `Kyori`
- `TrackCD`
- `CourseKubunCD`
- `SibaBabaCD`
- `DirtBabaCD`
- `TenkoCD`
- `Wakuban`
- `Umaban`
- `Futan`
- `SyussoTosu`
- `place_rank_limit`

#### 馬の近走

- `horse_days_since_last`
- `horse_last1_avg_finish`
- `horse_last3_avg_finish`
- `horse_last5_avg_finish`
- `horse_last3_win_rate`
- `horse_last5_win_rate`
- `horse_last3_ren_rate`
- `horse_last5_ren_rate`
- `horse_last3_top3_rate`
- `horse_last5_top3_rate`
- `horse_last3_place_paid_rate`
- `horse_last5_place_paid_rate`
- `horse_last3_avg_haron_l3`
- `horse_last5_avg_haron_l3`
- `horse_last3_avg_time`
- `horse_last5_avg_time`
- `horse_distance_diff_last`
- `horse_futan_diff_last`
- `horse_body_weight_diff_last`

#### 馬の条件別過去成績

- `horse_jyo_past_starts`
- `horse_jyo_win_rate`
- `horse_jyo_top3_rate`
- `horse_surface_past_starts`
- `horse_surface_win_rate`
- `horse_surface_top3_rate`
- `horse_dist_band_past_starts`
- `horse_dist_band_win_rate`
- `horse_dist_band_top3_rate`
- `horse_baba_past_starts`
- `horse_baba_win_rate`
- `horse_baba_top3_rate`

#### 騎手・調教師

- `KisyuCode`
- `ChokyosiCode`
- `jockey_past_starts`
- `jockey_win_rate`
- `jockey_ren_rate`
- `jockey_top3_rate`
- `trainer_past_starts`
- `trainer_win_rate`
- `trainer_ren_rate`
- `trainer_top3_rate`
- `jockey_jyo_past_starts`
- `jockey_jyo_win_rate`
- `jockey_jyo_top3_rate`
- `jockey_dist_band_past_starts`
- `jockey_dist_band_win_rate`
- `jockey_dist_band_top3_rate`
- `horse_jockey_past_starts`
- `horse_jockey_win_rate`
- `horse_jockey_top3_rate`

---

## 6. 中山・コース構造特徴の監査

### 6.1 明示特徴の検索

feature dataset、manifest、config、コード全体に対して、次の概念を表す列・定数・マッピングを検索する。

```text
turn_direction
right_turn
left_turn
inner
outer
inner_outer
straight
straight_length
elevation
height_difference
slope
gradient
final_slope
steep_slope
corner_count
corner_radius
first_corner_distance
course_width
small_turn
small_course
course_id
```

日本語名、ローマ字、JRA-VAN系コード名も検索する。

### 6.2 判定区分

各項目を次のいずれかに分類し、`course_structure_feature_audit.csv` に出力する。

- `explicit`: 物理的意味が明示された独立列がある
- `encoded`: コード値に含まれることが仕様書またはコードで確認できる
- `ambiguous`: `CourseKubunCD`等があるが意味対応を確認できない
- `indirect_only`: `JyoCD`や競馬場別過去成績から間接的に学習可能
- `absent`: dataset/config/modelに存在しない

`CourseKubunCD`について、名前だけで内回り・外回りを表すと断定しない。値一覧、欠損率、競馬場・芝ダート・距離とのクロス表、コード仕様または実装マッピングを確認する。

### 6.3 中山カテゴリと適性の分離

特徴群を最低限、次のように分ける。

#### A. venue identity

- `JyoCD`

#### B. direct course context

- `Kyori`
- `TrackCD`
- `CourseKubunCD`
- `SibaBabaCD`
- `DirtBabaCD`
- `TenkoCD`
- 必要に応じて枠・馬番

#### C. horse course suitability

- `horse_jyo_*`
- `horse_surface_*`
- `horse_dist_band_*`
- `horse_baba_*`

#### D. jockey course suitability

- `jockey_jyo_*`
- `jockey_dist_band_*`

中山でAだけが効いているのか、C/Dまで寄与しているのかを、SHAP、特徴群permutation、必要ならCatBoost interaction importanceで確認する。

### 6.4 中山サブセット

2020-2024を主評価として次を出す。

- 中山全体
- 中山芝
- 中山ダート
- 中山距離別
- `CourseKubunCD`別
- 内外回りを安全に特定できた場合のみ内回り・外回り別

各表に行数、race数、陽性率を付ける。サンプル数が小さい区分は警告を付ける。

2025/2026も同じ表を診断用に出してよいが、モデル選択には使わない。

---

## 7. 血統特徴量の監査

以下の名前・類義語・DB由来コードを、dataset、manifest、config、学習コード、feature builder全体から検索する。

```text
sire
dam
damsire
broodmare_sire
father
mother
pedigree
bloodline
lineage
Ketto
Hansyoku
Bamei
父
母父
血統
種牡馬
繁殖
```

注意:

- `KettoNum`は馬個体IDとして使われている可能性が高く、父・母父特徴とは別物
- 馬個体IDがdatasetに存在しても、C1特徴として採用されていなければ血統特徴ではない
- 父名、母父名がraw datasetにあるだけでは「血統条件別過去成績が実装済み」と判定しない

以下を個別に判定する。

- 父ID・父名
- 母父ID・母父名
- 父系・母父系
- 種牡馬全体成績
- 母父全体成績
- 血統×競馬場
- 血統×距離帯
- 血統×芝ダート
- 血統×馬場状態
- 血統×急坂
- 血統×小回り

`pedigree_feature_audit.csv` に `implemented / raw_only / not_in_model / absent / unknown` を出す。

存在しない場合は「未実装」と明記し、新特徴量はこのタスクでは追加しない。

---

## 8. CatBoost Feature Importance

C1はwalk-forward foldごとに保存されたモデルを用いる。

### 8.1 PredictionValuesChange

各foldモデルについて次を取得する。

```python
model.get_feature_importance(type="PredictionValuesChange")
```

出力:

- `catboost_pvc_by_fold.csv`
- `catboost_pvc_summary.csv`

summaryには最低限以下を含める。

```text
feature
group
weighted_mean
unweighted_mean
median
min
max
std
rank_weighted_mean
rank_median
fold_count
```

PredictionValuesChangeは残差CatBoost部分のモデル内部重要度であり、市場baselineそのものの重要度ではないことをレポートに明記する。

### 8.2 LossFunctionChange

各foldの正式評価年Poolを使い、必ずそのfoldの正しい `market_logit` をbaselineへ設定する。

```python
model.get_feature_importance(
    data=evaluation_pool_with_baseline,
    type="LossFunctionChange"
)
```

出力:

- `catboost_lfc_by_fold.csv`
- `catboost_lfc_summary.csv`

2020-2024を主集計とし、2025/2026は別表に分離する。

---

## 9. SHAP監査

### 9.1 重要なbaseline/offsetの扱い

C1の構造は次である。

```text
final_logit = market_logit + residual_raw
p_final = sigmoid(final_logit)
```

CatBoostのSHAPは、基本的に残差木 `residual_raw` の特徴寄与として扱う。

各行で次の加法性を確認する。

```text
residual_raw ≈ shap_expected_value + sum(feature_shap)
final_logit ≈ market_logit + shap_expected_value + sum(feature_shap)
```

誤差の最大値、平均値、99.9 percentileを出す。

市場baselineをSHAP特徴の一つとして偽装しない。

### 9.2 出力

- `shap_global_2020_2024.csv`
- `shap_by_year.csv`
- `shap_by_jyo.csv`
- `shap_nakayama.csv`
- `shap_nakayama_turf.csv`
- `shap_nakayama_dirt.csv`
- `shap_nakayama_by_distance.csv`
- `shap_additivity_check.csv`

最低限の統計:

```text
feature
group
mean_abs_shap
mean_signed_shap
median_abs_shap
p90_abs_shap
p99_abs_shap
positive_share
sample_rows
```

### 9.3 サンプリング

SHAPが重い場合のみ、固定seedによる決定的サンプルを使う。

- 主集計: 年ごとに層化
- 陽性・陰性の比率を保持
- 中山は可能なら全件
- サンプル条件と抽出IDを保存
- 2020-2024の結果を主結果とする

サンプル上限はconfig化し、実行ログとmanifestへ保存する。

---

## 10. 市場baselineの寄与監査

市場情報は残差側から除外され、`market_logit`としてbaselineに入るため、Feature Importance/SHAPだけでは市場寄与を表せない。

次を別枠で測定する。

- market-only Logloss/Brier/ECE
- C1 final Logloss/Brier/ECE
- C1 residualを0にした場合の指標
- `market_logit` の平均、標準偏差、分位点
- `residual_raw` の平均、標準偏差、分位点
- `final_logit` の平均、標準偏差、分位点
- `p_market` と `p_final` のPearson/Spearman
- `abs(residual_raw)` の分布
- residualがmarket確率を上げた割合・下げた割合
- baselineからC1へのLogloss改善量
- baselineからC1へのBrier改善量

出力:

- `market_vs_residual_contribution_by_year.csv`
- `market_vs_residual_contribution_summary.csv`

「市場寄与率」を単純なSHAP比率として定義しない。指標改善とlogit分解の両方を示す。

---

## 11. 特徴群別permutation importance

### 11.1 基本原則

- 再学習しない
- 同じfoldモデルを使う
- 各グループ内の列を**同じpermutation indexで共同シャッフル**し、列間相関を可能な限り保持する
- 年をまたいで混ぜない
- 2020-2024を主評価
- seedを固定する
- 反復回数をconfig化する
- 元データを変更しない

### 11.2 特徴群

最低限次を定義する。

```text
market_baseline
horse_recent_form
horse_historical_performance
venue_identity
course_context
distance
surface_and_going
horse_course_suitability
jockey_overall
jockey_course_suitability
trainer
horse_jockey_pair
weight_and_gate
race_metadata
pedigree
```

`pedigree`が空なら空グループとして「未実装」を記録し、架空の結果を作らない。

`market_baseline`はfeature列のpermutationではなく、`market_logit`を年内でシャッフルして最終logitを再計算する。

### 11.3 race-level列

`JyoCD`、`Kyori`、`TrackCD`、`CourseKubunCD`、馬場、天候等のrace-level列は、可能ならrace単位で共同permutationする。同一race内で出走馬ごとに異なるコース条件を作らない。

runner-levelの適性特徴はrunner単位で共同permutationする。

### 11.4 測定指標

各group、fold、repeatについて次を測る。

- Logloss悪化
- Brier悪化
- ECE悪化
- EV-ROI Spearman悪化
- `p_final`変化の平均絶対値
- EV>=1件数変化

主順位はLogloss/Brierの悪化を中心とし、ROIだけで順位を決めない。

出力:

- `permutation_importance_by_fold_repeat.csv`
- `permutation_importance_summary_2020_2024.csv`
- `permutation_importance_2025_2026_diagnostic.csv`

---

## 12. 2024年22件 → 2025年655件の急増監査

### 12.1 まず定義を固定

学習・評価コードから、EVとeligible条件の正確な式を確認する。

推測で次を決めない。

- EVが `p_final * fuku_odds_low` か
- odds欠損時の扱い
- eligibility mask
- place bet availability
- calibration適用位置
- raw scoreへのbaseline加算位置
- 2025用モデルのtrain期間
- 2025用market baselineのtrain期間

`ev_definition_audit.md` にコード位置と式を記録する。

### 12.2 年別分布

2020-2026について以下を同一分位点で比較する。

- eligible rows
- races
- `p_market`
- `market_logit`
- `residual_raw`
- `p_final`
- `fuku_odds_low`
- `fuku_odds_high`
- EV
- `p_final - p_market`
- `final_logit - market_logit`
- EV>=1件数
- EV>=1率

分位点:

```text
min, p01, p05, p10, p25, p50, p75, p90, p95, p99, max
```

出力:

- `ev_component_distribution_by_year.csv`
- `ev_threshold_counts_by_year.csv`

### 12.3 閾値余裕の分解

EV定義が `p_final * fuku_odds_low` であることを確認できた場合、次を計算する。

```text
required_probability = 1 / fuku_odds_low
required_final_logit = logit(required_probability)
ev_logit_margin = market_logit + residual_raw - required_final_logit
```

EV>=1は概ね `ev_logit_margin >= 0` と一致するはずなので、最大不一致件数を検証する。

年別に次を出す。

- `required_final_logit`
- `market_logit`
- `residual_raw`
- `ev_logit_margin`
- marginが0付近にある件数
- market-onlyでEV>=1の件数
- residual追加によりEV<1からEV>=1へ移動した件数
- residual追加によりEV>=1からEV<1へ移動した件数

出力:

- `ev_crossing_decomposition_by_year.csv`
- `ev_margin_distribution_by_year.csv`

### 12.4 2024と2025のモデル差

各foldについて次を比較する。

- train期間
- train rows
- positive rate
- tree_count
- best_iteration
- learning_rate
- depth
- random_seed
- loss
- class weights
- feature count
- categorical count
- feature names hash
- model file hash
- residual平均・標準偏差
- calibration情報
- market baselineモデル情報

`model_fold_metadata.csv` に保存する。

### 12.5 前年モデルによる安全な反実仮想

保存済みで実行可能な場合、**2024評価用モデル（2016-2023学習）を2025データへ適用**し、正式2025モデルとの差を比較する。

これは2025を使った調整ではなく、モデル更新による急増を診断するための監査である。

比較:

- 正式2025モデルのEV>=1件数
- 前年モデルを2025へ適用したEV>=1件数
- residual分布
- EV分布
- crossing件数

2025学習モデルを2024へ適用してはならない。2024が学習期間に含まれるため、リークする。

market baselineモデルもfold別に保存されている場合、同様に前年baselineを2025へ適用して差を分解する。

出力:

- `counterfactual_previous_fold_model_on_2025.csv`
- `counterfactual_previous_fold_summary.csv`

### 12.6 原因分類

最終レポートでは急増原因を次の分類で整理する。

- eligibility件数の増加
- odds分布の変化
- market baseline分布の変化
- residual raw scoreの上方シフト
- model fold更新の影響
- calibration差
- feature欠損・型・カテゴリ出現率の変化
- コード・artifact不一致
- 複数要因
- 未確定

根拠のない「リーク」と断定はしない。

---

## 13. 出力先

新規ディレクトリを使用する。

```text
outputs/place_market_offset_feature_audit_v1/
```

最低限の成果物:

```text
audit_manifest.json
input_artifact_inventory.csv
feature_schema_inventory.csv
feature_group_map.csv
course_structure_feature_audit.csv
pedigree_feature_audit.csv

catboost_pvc_by_fold.csv
catboost_pvc_summary.csv
catboost_lfc_by_fold.csv
catboost_lfc_summary.csv

shap_global_2020_2024.csv
shap_by_year.csv
shap_by_jyo.csv
shap_nakayama.csv
shap_nakayama_turf.csv
shap_nakayama_dirt.csv
shap_nakayama_by_distance.csv
shap_additivity_check.csv

market_vs_residual_contribution_by_year.csv
market_vs_residual_contribution_summary.csv

permutation_importance_by_fold_repeat.csv
permutation_importance_summary_2020_2024.csv
permutation_importance_2025_2026_diagnostic.csv

ev_component_distribution_by_year.csv
ev_threshold_counts_by_year.csv
ev_crossing_decomposition_by_year.csv
ev_margin_distribution_by_year.csv
model_fold_metadata.csv
counterfactual_previous_fold_summary.csv

ev_definition_audit.md
audit_report.md
```

図は次の配下に保存する。

```text
outputs/place_market_offset_feature_audit_v1/figures/
```

既存出力を上書きしない。出力先が既に存在し内容がある場合は、上書きせずエラー終了するか、明示的な新規run directoryを作る。

---

## 14. 実装候補

```text
config/audit_place_market_offset_feature_importance_v1.yaml
scripts/audit_place_market_offset_feature_importance_v1.py
tests/test_audit_place_market_offset_feature_importance_v1.py
docs/place_market_offset_feature_audit_v1_results.md
```

必要なら処理を分割してよい。

```text
scripts/audit_c1_feature_schema_v1.py
scripts/audit_c1_feature_importance_v1.py
scripts/audit_c1_shap_v1.py
scripts/audit_c1_permutation_v1.py
scripts/audit_c1_ev_year_shift_v1.py
scripts/run_c1_feature_audit_v1.py
```

ただし無意味な重複実装は避け、既存の評価関数、metric関数、Pool構築処理を再利用する。

---

## 15. テスト

最低限次をテストする。

1. 再学習関数が呼ばれない
2. DB接続を行わない
3. 2015年以前を含まない
4. 2020-2024と2025/2026が別集計になる
5. feature names順序がモデルとPoolで一致する
6. categorical列の型が学習時と一致する
7. baselineがfoldごとに正しく設定される
8. `final_logit = market_logit + residual_raw`
9. `sigmoid(final_logit) = p_final`
10. SHAP加法性
11. EV計算の既存出力との一致
12. EV>=1年別件数が既存結果と一致する
13. 2024年22件、2025年655件を監査入力上で再現できる
14. permutationが年をまたがない
15. race-level permutationでrace内条件が壊れない
16. 出力先を上書きしない
17. seed固定で再現する

既存結果と一致しない場合は、監査を成功扱いにせず差分を報告する。

---

## 16. 最終報告で必ず答える質問

1. C1残差モデルで最も重要な特徴は何か
2. PredictionValuesChange、LossFunctionChange、SHAPで順位は一致するか
3. 市場baselineと残差モデルはそれぞれどの程度予測を担っているか
4. 近走、馬の過去成績、騎手、調教師、競馬場、距離、馬場、枠・斤量の寄与はどの程度か
5. 血統特徴量は実装されているか
6. 中山の急坂、直線長、小回り、右回り、内外回りは明示特徴として存在するか
7. `CourseKubunCD`は何を表すか確認できたか
8. 中山は単なる`JyoCD`カテゴリとして効いているのか
9. `horse_jyo_*`、`horse_surface_*`、`horse_dist_band_*`、`horse_baba_*`は中山予測に寄与しているか
10. `jockey_jyo_*`、`jockey_dist_band_*`は寄与しているか
11. 2024年22件から2025年655件へ増えた主因は何か
12. artifact、fold、baseline、calibration、eligibilityに異常はないか
13. 新特徴量追加前に修正すべきバグやデータ問題はあるか
14. 現行C1を次の基準モデルとして維持してよいか

---

## 17. 最終報告形式

最終メッセージには次を含める。

### A. 実行内容

- 読んだファイル
- 使用したモデル
- 使用したデータ
- DB未使用
- 再学習未実施
- 2025/2026未調整
- commit/push未実施

### B. 主要結論

- 特徴量寄与
- 中山監査
- 血統監査
- 2024→2025急増原因
- 異常・不一致

### C. 主要数値

- Feature Importance上位
- SHAP上位
- 特徴群permutation上位
- 市場baseline対残差の比較
- EV件数分解

### D. 生成ファイル

全ファイルパスを列挙する。

### E. git状態

```bash
git status --short
git diff --stat
```

自動commit/pushは行わない。

---

## 18. 禁止事項の再確認

このタスクでは以下を行わない。

- C1再学習
- 新特徴量追加
- feature dataset再作成
- DB読込
- random split
- 2025/2026での調整
- ROI直接学習
- Ability/ANA
- Ranker
- Kelly
- 自動購入
- 大規模Optuna
- 自動commit/push

監査完了後に改善案を提案してよいが、実装は別タスクとする。
