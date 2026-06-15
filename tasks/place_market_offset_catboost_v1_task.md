# Task: 市場確率をoffsetに使う複勝残差CatBoostの実装・比較検証

## 目的

現在の複勝モデルでは、高オッズ馬に対して予測複勝率を過大評価しやすく、

```text
EV >= 1.00 の大半が複勝下限2.0倍以上
EVと実ROIのSpearman相関が負
高EVほど実ROIが悪化
```

という問題が確認されている。

この問題に対し、複勝オッズ帯へ学習対象を限定せず、全データを使ったまま、

```text
市場が示す複勝確率
+
CatBoostによる市場からの補正
=
最終複勝確率
```

という市場残差型モデルを実装する。

モデル名の推奨:

```text
place_market_offset_catboost_v1
```

今回は複勝のみを対象とする。

---

# 1. 基本方針

市場残差型モデルは、対象を特定オッズ帯へ限定しない。

```text
全出走馬を学習対象にする
市場確率を強いベースラインにする
CatBoostは市場の上方・下方修正だけを学ぶ
購入判断はモデル学習と分離する
```

複勝下限1.2〜2.5倍のROIが良かったという結果を、学習対象の絞り込みには使用しない。

オッズ帯は評価・校正・購入ルール分析にのみ使用する。

---

# 2. 使用データ

入力:

```text
outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet
outputs/final_odds_two_models_v1/
outputs/place_odds_ev_surface_v1/
```

DB本体は原則再読込しない。

必要な場合のみDB validation cache HIT後にread-only利用する。

使用期間:

```text
2016〜2026
```

target:

```text
actual_place
```

既存のtarget定義、eligible条件、払戻定義を変更しない。

---

# 3. 比較するモデル

最低限、次の3方式を比較する。

## A. 現行モデル

```text
current_market_aware_place_model
```

既存の`final_odds_two_models_v1`の複勝market-awareモデル。

再学習せず、既存予測を比較対象に使う。

## B. 市場ベースラインモデル

市場情報だけから複勝確率を予測する。

推奨特徴量:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
field_size
place_slots
race内オッズ順位
race内オッズ比率
fuku_odds_width
log_tan_odds
log_fuku_low
log_fuku_high
```

`place_slots`は既存ルールに従い、レースごとの複勝対象頭数を正しく算出する。

市場ベースラインは、ロジスティック回帰または浅いCatBoostで作成してよい。

ただし市場ベースライン単体が過学習しないよう、複雑度を抑える。

## C. 市場offset残差CatBoost

市場ベースライン確率をlogitへ変換し、CatBoostのbaseline/offsetとして使用する。

```text
market_logit = log(p_market / (1 - p_market))
```

確率は数値安定性のためclipする。

推奨:

```text
epsilon = 1e-6
p_market = clip(p_market, epsilon, 1 - epsilon)
```

CatBoostの学習イメージ:

```text
Pool(
    data=X_residual,
    label=actual_place,
    baseline=market_logit
)
```

最終raw score:

```text
final_logit
=
market_logit
+
catboost_residual_score
```

最終確率:

```text
p_final = sigmoid(final_logit)
```

CatBoostのbaseline仕様を確認し、学習時・validation時・推論時に同じbaselineを渡す。

baselineが推論時に無視される実装になっていないことをテストする。

---

# 4. 市場ベースラインの作り方

単純に`1 / fuku_odds_low`だけを市場確率としない。

複勝オッズはレンジであり、単勝のように単純正規化できないため、市場ベースラインモデルを学習する。

market-onlyモデルは、各walk-forward foldで過去データだけを使って学習する。

例:

```text
train 2016〜2019 → market baseline for 2020
train 2016〜2020 → market baseline for 2021
train 2016〜2021 → market baseline for 2022
train 2016〜2022 → market baseline for 2023
train 2016〜2023 → market baseline for 2024
```

2025・2026についても、その時点より未来を使わない。

市場ベースライン確率を同じデータで学習・評価しない。

必ずout-of-foldまたはout-of-time予測を使う。

---

# 5. residual側の特徴量

residual CatBoostでは、原則として以下を使う。

```text
market_free特徴量
市場ベースライン確率
市場ベースラインlogit
市場順位との差を表す派生特徴量
```

ただしrawの複勝オッズを大量に再投入すると、市場モデルの二重学習になりやすい。

初回実装では次の2案を比較する。

## C1. fundamental residual

```text
market_free特徴量
p_market
market_logit
```

rawの`tan_odds`、`fuku_odds_low`、`fuku_odds_high`はresidual側へ直接入れない。

## C2. limited market residual

C1に加えて、限定した市場乖離特徴だけを追加する。

例:

```text
market_rank
model_free_rank
rank_gap
field_size
place_slots
```

高オッズそのものを直接強く再学習させない。

C1とC2を比較し、より安定する方を採用する。

---

# 6. 時系列walk-forward

ランダムsplitは禁止。

```text
fold 1:
train 2016〜2019
validation 2020

fold 2:
train 2016〜2020
validation 2021

fold 3:
train 2016〜2021
validation 2022

fold 4:
train 2016〜2022
validation 2023

fold 5:
train 2016〜2023
validation 2024
```

最終評価:

```text
test: 2025
latest_holdout: 2026
```

2025・2026をモデル選択、補正選択、閾値選択に使用しない。

---

# 7. CatBoost設定

既存V2.1.2複勝モデルの設定を初期値にする。

GPUを使用する。

```text
RTX 5070 Ti
CPU fallback禁止
```

大規模Optunaは行わない。

比較するのは少数の事前定義設定だけにする。

例:

```text
depth: 6, 8
learning_rate: 0.03, 0.05
l2_leaf_reg: 3, 7
```

候補総数を小さく保つ。

モデル選択は2020〜2024のみで行う。

---

# 8. 評価指標

ROIだけでモデルを選ばない。

## 確率精度

```text
Logloss
Brier score
ROC-AUC
ECE
MCE
calibration slope
calibration intercept
```

## 市場比較

```text
market baselineとの差
current market-awareとの差
market-freeとの差
```

## オッズ帯別校正

最低限:

```text
1.0〜1.2
1.2〜1.5
1.5〜2.0
2.0〜3.0
3.0〜5.0
5.0以上
```

各帯で:

```text
mean predicted probability
actual place rate
calibration gap
logloss
Brier score
```

## EV品質

補正後確率から以下を作る。

```text
adjusted_place_ev
=
p_final * fuku_odds_low
```

EV帯:

```text
EV < 0.85
0.85〜0.90
0.90〜0.95
0.95〜1.00
1.00〜1.02
1.02〜1.05
1.05〜1.10
1.10以上
```

各帯で:

```text
bets
actual ROI
hit rate
calibration gap
```

重要指標:

```text
EV帯順と実ROIのSpearman相関
EV >= 1.00件数
EV >= 1.00実ROI
EV >= 1.05件数
EV >= 1.05実ROI
```

現在の相関`-0.636`を改善できるか確認する。

最低目標:

```text
相関を0以上へ改善
```

---

# 9. calibration

市場offsetモデルの出力に対して、以下を比較する。

```text
none
Platt scaling
isotonic regression
```

ただし同一validation年でfitと評価を行わない。

時系列nested calibrationを使用する。

例:

```text
2020 OOFでfit → 2021評価
2020〜2021 OOFでfit → 2022評価
2020〜2022 OOFでfit → 2023評価
2020〜2023 OOFでfit → 2024評価
```

2025用calibratorは2020〜2024 OOFでfitする。

2026用calibratorは、2025の結果を見て方法を変更せず、同一仕様を維持する。

---

# 10. 購入戦略評価

モデル比較の主目的は確率とEVの品質改善である。

購入戦略は別レイヤーとして固定比較する。

最低限、次を比較する。

```text
戦略1:
複勝下限1.2〜2.5
raw/adjusted EV >= 0.85

戦略2:
adjusted EV >= 1.00

戦略3:
adjusted EV >= 1.05

戦略4:
オッズ制限なし adjusted EV >= 1.00
```

モデルごとに同じルールを適用し、公平に比較する。

ROI計算は実払戻`fuku_pay`を使用する。

100円均等買い。

---

# 11. 合格判定

## モデル改善合格

以下を総合判断する。

```text
2020〜2024平均Logloss改善
Brier改善
高オッズ帯calibration gap改善
EVと実ROIのSpearman相関改善
EV >= 1.00の実ROI改善
2025/2026で大崩れしない
```

## ROI目標

```text
2025 ROI >= 90%
2026 ROI >= 90%
2025+2026 ROI >= 90%
```

利益モデルとしては:

```text
2025+2026 ROI >= 100%
```

も参考表示する。

ただしROIだけでモデル採用を決めない。

---

# 12. 比較表

最低限、以下を同じ表へ出す。

```text
A current_market_aware
B market_baseline
C1 market_offset_fundamental
C2 market_offset_limited_market
```

列:

```text
validation logloss
validation Brier
validation ECE
validation calibration slope
high-odds calibration gap
EV-ROI Spearman
EV>=1 count
EV>=1 ROI
2025 ROI
2026 ROI
combined ROI
top5 removed ROI
bootstrap CI
```

---

# 13. 出力

```text
config/place_market_offset_catboost_v1.yaml

scripts/build_place_market_baseline_v1.py
scripts/train_place_market_offset_catboost_v1.py
scripts/evaluate_place_market_offset_catboost_v1.py
scripts/run_place_market_offset_catboost_v1.py

models/place_market_offset_catboost_v1/
outputs/place_market_offset_catboost_v1/
  market_baseline_oof.parquet
  residual_oof_predictions.parquet
  final_predictions_2025.parquet
  final_predictions_2026.parquet
  fold_metrics.csv
  calibration_metrics.csv
  odds_band_calibration.csv
  ev_band_roi.csv
  model_comparison.csv
  roi_comparison.csv
  bootstrap_ci.csv
  selected_model.json
  manifest.json

docs/place_market_offset_catboost_v1_design.md
docs/place_market_offset_catboost_v1_results.md

tests/test_place_market_offset_catboost_v1.py
```

---

# 14. 一括実行

```bash
python scripts/run_place_market_offset_catboost_v1.py \
  --config config/place_market_offset_catboost_v1.yaml
```

strict resume対応とする。

---

# 15. manifest

最低限保存する。

```text
input feature hash
split hash
market baseline config hash
residual feature hash
CatBoost config hash
calibration config hash
prediction hash
payout source hash
DB validation manifest hash
Git SHA
Git dirty
GPU
CatBoost version
random seed
```

---

# 16. テスト

最低限:

```text
market baseline OOF
no future leakage
place_slots calculation
market probability range
logit clipping
CatBoost baseline passed to train
CatBoost baseline passed to inference
raw score + baseline consistency
C1 feature exclusion
C2 limited market features
walk-forward split
nested calibration
EV calculation
EV band monotonicity calculation
ROI calculation
2025/2026 immutability
strict resume
GPU smoke
end-to-end smoke
```

実行:

```bash
python -m pytest -q
```

---

# 17. 禁止事項

```text
複勝1.2〜2.5倍だけに学習対象を限定
2025/2026でモデル・補正・閾値調整
ランダムsplit
直接ROIを目的変数にする
Ability/ANA/Ranker
大規模Optuna
Kelly基準
自動購入
既存モデル上書き
既存成果物上書き
自動commit
自動push
```

---

# 18. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 一括実行コマンド
5. 使用入力とhash
6. DB cache status
7. GPU/CatBoost情報
8. market baseline特徴量
9. market baselineモデル
10. market baseline fold別指標
11. p_marketの作り方
12. logit clip値
13. CatBoost baseline利用確認
14. inference時baseline利用確認
15. C1特徴量一覧
16. C2特徴量一覧
17. walk-forward fold一覧
18. leakage監査
19. fold別学習時間
20. fold別best iteration
21. A validation Logloss
22. B validation Logloss
23. C1 validation Logloss
24. C2 validation Logloss
25. 各モデルBrier
26. 各モデルECE
27. 各モデルcalibration slope/intercept
28. 1.0〜1.2倍校正
29. 1.2〜1.5倍校正
30. 1.5〜2.0倍校正
31. 2.0〜3.0倍校正
32. 3.0〜5.0倍校正
33. 5.0倍以上校正
34. 高オッズ過信改善判定
35. calibration候補
36. 採用calibration
37. EV帯別件数
38. EV帯別実ROI
39. EV-ROI Spearman
40. 現行-0.636との差
41. EV>=1年別件数
42. EV>=1実ROI
43. EV>=1.05年別件数
44. EV>=1.05実ROI
45. 戦略1 validation ROI
46. 戦略2 validation ROI
47. 戦略3 validation ROI
48. 戦略4 validation ROI
49. 選択モデル
50. 選択理由
51. 2025 bets/ROI
52. 2026 bets/ROI
53. 2025+2026 bets/ROI
54. top1/top3/top5/top10除外ROI
55. 最大連敗
56. 最大ドローダウン
57. bootstrap 95% CI
58. 現行モデルとの差
59. 市場ベースラインとの差
60. 90%目標達成判定
61. 100%参考達成判定
62. 2025/2026未調整確認
63. strict resume結果
64. end-to-end実行時間
65. pytest結果
66. 自動購入未実施確認
67. 未解決事項
68. 次の推奨手順

完了後は、市場offset残差CatBoostの学習・比較・固定評価まで報告し、自動購入や別モデル系統には進まず停止する。
