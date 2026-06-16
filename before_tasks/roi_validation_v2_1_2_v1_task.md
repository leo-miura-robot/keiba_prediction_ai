# Task: CatBoost V2.1.2の確度分析・EV計算・単勝複勝ROI検証

## 目的

学習済みのCatBoost V2.1.2モデルについて、予測の「確度」が実際の的中率・回収率とどう関係するかを分析し、そのうえで単勝・複勝のEVおよびROIを検証する。

対象モデル:

```text
win   × market_free
win   × market_history
win   × market_aware
place × market_free
place × market_history
place × market_aware
```

第一目標:

```text
単勝ROI >= 90%
複勝ROI >= 90%
```

ただし、一部の大穴や少数的中だけで達成したROIは合格としない。

今回は以下まで実施する。

```text
確度分析
キャリブレーション診断・適用候補比較
EV計算
validationでの購入条件選定
test/latest_holdoutでの固定ROI評価
安定性・大穴依存・ドローダウン検証
```

以下には進まない。

```text
自動購入
実運用API
資金配分最適化
Kelly基準
Ability/ANA/Ranker
大規模Optuna
モデル再学習
```

---

## 1. 前提

入力モデル:

```text
models/catboost_baseline_v2_1_2_v1/
```

予測・分析元:

```text
outputs/model_training/catboost_baseline_v2_1_2_v1/
```

split:

```text
train: 2016-2023
validation: 2024
test: 2025
latest_holdout: 2026
```

購入条件、補正方法、閾値の選択はvalidationだけで行う。

testとlatest_holdoutでは条件を固定し、結果を見て変更しない。

---

## 2. 最初に確認すること

```bash
git status
git diff
git rev-parse HEAD
python -m pytest -q
```

確認対象:

```text
config/catboost_baseline_v2_1_2_v1.yaml
models/catboost_baseline_v2_1_2_v1/
outputs/model_training/catboost_baseline_v2_1_2_v1/
```

各予測に最低限以下があることを確認する。

```text
entry_id
race_id
race_date
data_split
actual
prediction
target
feature_set
```

市場・払戻関連:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
actual_tan_pay
actual_fuku_pay
```

複勝払戻がrace_id + Umabanで正確に結合できない場合、複勝ROIは実行せず停止して報告する。

---

## 3. 確度の定義

確度を単一指標に限定しない。レース単位で以下を作る。

```text
top1_probability
top1_minus_top2_margin
prediction_entropy
top3_probability_sum
model_agreement_count
market_gap
```

### top1_probability

レース内1位評価馬の予測確率。

### top1_minus_top2_margin

```text
1位予測確率 - 2位予測確率
```

競馬でいう「◎と○の差」。

### prediction_entropy

予測が1頭に集中しているか、全馬へ散っているか。

```text
低entropy = 一強・軸堅め
高entropy = 混戦
```

比較可能にするため、必要なら出走頭数で正規化する。

### top3_probability_sum

上位3頭への確率集中度。

### model_agreement_count

market_free / market_history / market_awareの本命一致度。

```text
3モデル一致
2モデル一致
0-1モデル一致
```

### market_gap

単勝:

```text
market_gap = calibrated_win_probability - normalized_market_probability
```

---

## 4. 確度帯別分析

### 4.1 単勝

以下の帯を基本とする。

#### top1_probability

```text
< 0.10
0.10-0.15
0.15-0.20
0.20-0.25
0.25-0.30
0.30-0.40
>= 0.40
```

#### top1_minus_top2_margin

```text
< 0.02
0.02-0.05
0.05-0.10
0.10-0.20
>= 0.20
```

#### entropy

固定幅またはtie-safe quantile。

同じ値を別binへ分断しない。

各帯で算出:

```text
race_count
mean predicted probability
actual win rate
calibration gap
top1 hit rate
average tan_odds
median tan_odds
ROI
```

### 4.2 複勝

同様に以下を確認する。

```text
top1 place probability
top1-top2 margin
entropy
model agreement
```

各帯:

```text
race_count
mean predicted place probability
actual place rate
calibration gap
average fuku_odds_low
ROI
```

---

## 5. キャリブレーション

validationだけで補正器を学習・選択する。

候補:

```text
none
Platt scaling
isotonic regression
```

6モデルすべてを対象とする。

選定指標:

```text
validation logloss
validation Brier score
ECE
MCE
calibration slope
calibration intercept
```

サンプル数やbin数が不足する場合、isotonicを無理に採用しない。

test/latest_holdoutでは選定済み補正器を固定する。

出力:

```text
calibration_method_selection.csv
calibration_metrics.csv
calibrated_predictions/
```

---

## 6. 単勝市場確率とEV

単勝確定オッズから市場確率を作る。

```text
raw_market_probability = 1 / tan_odds
normalized_market_probability =
raw_market_probability / race内raw_market_probability合計
```

単勝EV:

```text
win_ev = calibrated_win_probability * tan_odds
```

market_free/historyでも、買い判断段階では確定オッズを参照してよい。

つまり:

```text
モデル予想には現在オッズを入れない
買うかどうかの判断にはオッズを使う
```

という構成にする。

---

## 7. 複勝EV

保守的に下限オッズを正本とする。

```text
place_ev_low = calibrated_place_probability * fuku_odds_low
place_ev_high = calibrated_place_probability * fuku_odds_high
```

購入条件は`place_ev_low`で決める。

ROI計算には実際の複勝払戻額を使う。

推定払戻やオッズ下限で実払戻を代用しない。

---

## 8. ROI計算

100円均等買いを基本とする。

### 単勝

```text
stake = 100
return = actual_tan_pay
ROI = return_sum / stake_sum * 100
```

### 複勝

```text
stake = 100
return = actual_fuku_pay
ROI = return_sum / stake_sum * 100
```

同一レースで複数購入を許す場合は、1頭ごとに100円として集計する。

---

## 9. validationでの購入条件探索

探索対象はvalidationのみ。

### 9.1 単勝

候補軸:

```text
EV threshold
minimum predicted probability
minimum odds
maximum odds
confidence margin
entropy
model agreement
market gap
top-N per race
```

小さな事前定義gridを使う。

例:

```text
EV >= 1.00, 1.05, 1.10, 1.15, 1.20, 1.30
予測勝率 >= 0.05, 0.08, 0.10, 0.15
オッズ帯 1.5-3, 3-5, 5-10, 10-20, 20+
margin >= 0, 0.02, 0.05, 0.10
```

### 9.2 複勝

```text
place_ev_low threshold
minimum place probability
fuku_odds_low帯
confidence margin
entropy
model agreement
top-N per race
```

### 9.3 過剰探索防止

総当たり数を制限する。

validationの最大ROIだけで選ばない。

以下を含む複合判定を行う。

```text
ROI
bet count
hit rate
月別安定性
オッズ帯安定性
最大払戻除外後ROI
最大ドローダウン
bootstrap下限
```

1 target × 1 modelにつき、採用候補は最大3ルールまで。

---

## 10. 採用基準

validationで最低限:

```text
ROI >= 90%
十分なbet count
単発大穴依存でない
最大払戻除外後も大崩れしない
月別で極端に偏らない
```

目安:

```text
単勝 bets >= 500
複勝 bets >= 500
```

データ量に応じた調整は可能だが、理由を明記する。

test/latest_holdoutを見て採用ルールを変更しない。

---

## 11. 固定評価

validationで選定したルールを、そのまま以下へ適用する。

```text
test: 2025
latest_holdout: 2026
```

各ルールで算出:

```text
bets
stake
return
profit
ROI
hit count
hit rate
average odds
median odds
max odds
average payout
max payout
max losing streak
max drawdown
```

---

## 12. 安定性分析

以下で分解する。

```text
year
month
racecourse
surface
distance band
popularity band
odds band
confidence band
model agreement
```

単勝・複勝、モデル別、split別に出す。

---

## 13. 大穴依存チェック

必須。

```text
全件ROI
最大払戻1件除外ROI
上位3件除外ROI
上位5件除外ROI
上位10件除外ROI
```

追加:

```text
利益上位1%の寄与率
利益上位5%の寄与率
```

ROIが90%以上でも、最大払戻除外で大幅悪化する場合は不安定と判定する。

---

## 14. 不確実性

レース単位bootstrapを使う。

```text
bootstrap 1000回
95% confidence interval
```

負荷が大きい場合は500回まで減らしてよい。

出力:

```text
ROI point estimate
ROI 2.5%
ROI 50%
ROI 97.5%
```

---

## 15. モデルの扱い

### market_free

市場情報なしの基準モデル。

### market_history

発走前実運用の主力候補。

### market_aware

確定オッズをモデル入力に使う理想条件モデル。

ROIを計算してよいが、実運用候補とは分けて表示する。

最終報告では:

```text
実運用候補
理想条件
```

を混同しない。

---

## 16. 推奨出力

```text
config/roi_validation_v2_1_2_v1.yaml

scripts/analyze_prediction_confidence_v2_1_2.py
scripts/calibrate_catboost_v2_1_2.py
scripts/backtest_roi_v2_1_2.py

outputs/roi_validation_v2_1_2_v1/
  confidence_analysis.csv
  confidence_calibration.csv
  calibration_method_selection.csv
  calibration_metrics.csv
  candidate_rules_validation.csv
  selected_rules.json
  roi_summary_test.csv
  roi_summary_latest_holdout.csv
  roi_by_year.csv
  roi_by_month.csv
  roi_by_track.csv
  roi_by_odds_band.csv
  roi_by_popularity_band.csv
  roi_by_confidence_band.csv
  payout_dependency.csv
  drawdown_summary.csv
  bootstrap_roi_ci.csv
  bet_details.parquet
  manifest.json

docs/roi_validation_v2_1_2_v1_design.md
docs/roi_validation_v2_1_2_v1_results.md
```

既存構造に合わせて調整してよいが、旧出力を上書きしない。

---

## 17. テスト

最低限:

```text
prediction loading
race-level confidence
top1-top2 margin
entropy
model agreement
market probability normalization
calibration fit/transform
validation-only rule selection
test rule immutability
win payout join
place payout join
ROI calculation
top payout removal
drawdown
bootstrap by race
atomic output
resume/fingerprint
```

実行:

```bash
python -m pytest -q
```

---

## 18. 再現性

manifestに保存:

```text
prediction input hash
model version
split hash
calibration config hash
rule grid hash
selected rule hash
payout source hash
code hash
Git SHA
Git dirty
Python version
library versions
random seed
```

分析を同条件で2回実行し、主要CSVのhash一致を確認する。

CSV/JSON/Parquetは一時ファイルから原子的に置換する。

---

## 19. 禁止事項

- test/latest_holdoutで閾値調整
- payout推定値でROI計算
- 複勝払戻欠損をオッズで代用
- 自動購入
- 資金配分最適化
- Kelly基準
- Ability/ANA/Ranker
- 大規模Optuna
- モデル再学習
- 旧出力上書き
- 自動commit
- 自動push

---

## 20. 完了条件

- 確度指標を作成
- 確度帯別予測確率・実績率を確認
- 3モデル一致度別成績を確認
- validationだけで補正方法を選択
- validationだけで購入条件を選択
- test/latest_holdoutで固定評価
- 単勝ROI算出
- 複勝ROI算出
- 単勝・複勝ROI 90%以上の達成可否判定
- 大穴依存確認
- 年別/月別/競馬場別安定性確認
- bootstrap CI算出
- market_historyを実運用候補として評価
- market_awareを理想条件として分離
- 分析再現性確認
- モデル未再学習

---

## 21. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 使用予測ファイル
5. split確認
6. payout source
7. 単勝払戻結合率
8. 複勝払戻結合率
9. 確度指標一覧
10. top1_probability帯別勝率
11. margin帯別勝率
12. entropy帯別勝率
13. 3モデル一致度別勝率
14. 複勝確度帯別複勝率
15. calibration候補
16. validationで選ばれた補正方法
17. calibration ECE/MCE
18. 単勝候補ルール数
19. 複勝候補ルール数
20. validation採用単勝ルール
21. validation採用複勝ルール
22. test単勝bets
23. test単勝ROI
24. latest_holdout単勝ROI
25. test複勝bets
26. test複勝ROI
27. latest_holdout複勝ROI
28. market_free結果
29. market_history結果
30. market_aware結果
31. 年別ROI
32. 月別ROI
33. 競馬場別ROI
34. 人気帯別ROI
35. オッズ帯別ROI
36. 確度帯別ROI
37. 最大連敗
38. 最大ドローダウン
39. 最大払戻除外ROI
40. 上位3件除外ROI
41. 上位5件除外ROI
42. 上位10件除外ROI
43. 利益上位1%寄与率
44. bootstrap 95% CI
45. 単勝ROI 90%以上達成可否
46. 複勝ROI 90%以上達成可否
47. 大穴依存の有無
48. test閾値未調整確認
49. 分析hash一致
50. pytest結果
51. モデル未再学習確認
52. 実運用候補モデル
53. 理想条件モデル
54. 未解決事項
55. 次の推奨手順

完了後はROI検証結果まで報告し、自動購入・資金配分・モデル再学習には進まず停止する。
