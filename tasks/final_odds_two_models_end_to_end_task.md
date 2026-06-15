# Task: 確定オッズ利用・単勝複勝2モデルの一括学習とROI検証

## 目的

2016年以降のJRAデータと確定オッズを使い、次の2モデルを新規作成する。

```text
単勝モデル
複勝モデル
```

今回は一つの実行フローで次まで行う。

```text
データ確認
ウォークフォワード学習
確率補正
予測生成
確度分析
保守的EV計算
購入候補抽出
2025 test評価
2026 latest_holdout評価
単勝・複勝ROI検証
安定性・高配当依存監査
```

第一目標:

```text
単勝ROI >= 90%
複勝ROI >= 90%
```

単年度だけ、一部の大穴だけ、少数的中だけで達成した結果は合格としない。

## 1. モデルの位置付け

今回の2モデルは対象レースの確定オッズを特徴量として使用する。

```text
確定オッズを事前に利用できたと仮定する理想条件モデル
発走前の実運用モデルではない
```

この点をconfig、manifest、docs、最終報告へ明記する。

## 2. 入力

```text
outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet
```

使用feature set:

```text
market_aware
```

主な市場列:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
```

既存V2.1.2の特徴量、target、eligible条件を変更しない。

## 3. モデル

### 単勝

```text
target = actual_win
```

### 複勝

```text
target = actual_place
```

複勝EVは保守的に`fuku_odds_low`を正本とする。

## 4. バージョン

既存成果物を上書きしない。

```text
config/final_odds_two_models_v1.yaml
scripts/run_final_odds_two_models_v1.py
scripts/train_final_odds_two_models_v1.py
scripts/analyze_final_odds_two_models_v1.py
scripts/backtest_final_odds_two_models_v1.py
models/final_odds_two_models_v1/
outputs/final_odds_two_models_v1/
docs/final_odds_two_models_v1_design.md
docs/final_odds_two_models_v1_results.md
```

`run_final_odds_two_models_v1.py`を一括実行の入口とする。

## 5. 一括実行

```bash
python scripts/run_final_odds_two_models_v1.py --config config/final_odds_two_models_v1.yaml
```

処理順:

```text
preflight
→ feature確認
→ walk-forward学習
→ calibration
→ validation予測
→ ルール設計
→ 2025 test固定評価
→ 2026 latest_holdout固定評価
→ ROI/安定性分析
→ 最終レポート
```

strict resume対応とする。

## 6. 時系列検証

ランダム分割は禁止。

```text
fold 1: train 2016-2019 → validation 2020
fold 2: train 2016-2020 → validation 2021
fold 3: train 2016-2021 → validation 2022
fold 4: train 2016-2022 → validation 2023
fold 5: train 2016-2023 → validation 2024
```

最終評価:

```text
test: 2025
latest_holdout: 2026
```

2025・2026を補正方法や購入ルール選択に使わない。

## 7. 学習

CatBoost GPUを使用する。

```text
CatBoost version
GPU name
CUDA smoke test
CPU fallback有無
```

GPUが使えない場合はCPUへ自動fallbackせず停止する。

既存V2.1.2 CatBoost設定を初期値とし、大規模Optunaは行わない。

各foldで単勝・複勝を独立学習する。

## 8. キャリブレーション

候補:

```text
none
Platt scaling
isotonic regression
```

2020〜2024のfold全体で選ぶ。

選定基準:

```text
平均logloss
平均Brier score
平均ECE
最悪年ECE
年度間ばらつき
```

単年度だけ良いisotonicを採用しない。

## 9. 市場確率

単勝:

```text
raw_market_probability = 1 / tan_odds
normalized_market_probability =
raw_market_probability / race内raw_market_probability合計
```

複勝は単勝と同じ正規化をしない。

## 10. 保守的確率

単勝:

```text
conservative_probability =
alpha * calibrated_probability
+ (1 - alpha) * normalized_market_probability
```

alpha候補:

```text
0.5, 0.6, 0.7, 0.8, 1.0
```

alphaは2020〜2024だけで選ぶ。

複勝は以下を比較する。

```text
calibrated probability
calibration bin実績下限
Wilsonまたはbootstrap下限
```

## 11. EV

単勝:

```text
win_ev = conservative_win_probability * tan_odds
```

複勝:

```text
place_ev_low = conservative_place_probability * fuku_odds_low
place_ev_high = conservative_place_probability * fuku_odds_high
```

購入判定は`place_ev_low`を使う。

## 12. 購入判断用指標

```text
calibrated_probability
conservative_probability
ev
edge
model_rank
market_rank
rank_gap
top1_probability
top1_minus_top2_margin
prediction_entropy
top3_probability_sum
```

単勝edge:

```text
edge = calibrated_win_probability - normalized_market_probability
```

rank_gap:

```text
rank_gap = market_rank - model_rank
```

## 13. 購入戦略

### 堅軸型

```text
model_rank = 1
top1_probability高
margin大
低〜中オッズ
conservative_ev > 1
```

### 妙味型

```text
model_rank <= 3
market_rank >= 4
rank_gap >= 1
edge > 0
conservative_ev > 1
```

### 大穴型

```text
高オッズ
model_rank上位
大きなedge
高いconservative_ev
```

大穴型は別集計とし、全購入件数の10%以下を目安に制限する。

## 14. オッズ帯別安全マージン

例:

```text
odds < 5       → conservative_ev >= 1.03
5 <= odds <10  → conservative_ev >= 1.08
10 <= odds <20 → conservative_ev >= 1.18
odds >=20      → conservative_ev >= 1.35
```

複勝も`fuku_odds_low`帯ごとに別基準を事前定義する。

## 15. ルール選択

2020〜2024だけで選ぶ。

評価:

```text
5年合算ROI
年別ROI平均
最低年ROI
年別ROI標準偏差
購入数
的中率
最大ドローダウン
上位払戻除外ROI
bootstrap下限
```

優先順位:

```text
最低年ROI
高配当除外後ROI
bootstrap下限
5年合算ROI
購入数
```

単勝・複勝それぞれ最大3ルールまで。

購入馬Jaccard similarityが0.8以上のルールは代表だけ残す。

## 16. ROI

100円均等買い。

単勝:

```text
return = tan_pay
```

複勝:

```text
return = fuku_pay
```

実払戻を使用し、払戻欠損をオッズで代用しない。

## 17. 評価

```text
bets
races
stake
return
profit
ROI
hit count
hit rate
average odds
median odds
average payout
max payout
max losing streak
max drawdown
```

分解:

```text
year
month
racecourse
surface
distance band
field size
popularity band
odds band
model rank
market rank
rank gap
edge band
confidence band
strategy type
```

## 18. 高配当依存

```text
全件ROI
最大払戻1件除外ROI
上位3件除外ROI
上位5件除外ROI
上位10件除外ROI
利益上位1%寄与率
利益上位5%寄与率
大穴型除外後ROI
```

## 19. bootstrap

レース単位1000回、95% CI。

CPU負荷対策として、DataFrame再構築をせず、レース単位集計配列をNumPyで再標本化する。

必要なら500回へ減らしてよい。

## 20. 2025・2026最終評価

2020〜2024で固定した以下をそのまま使う。

```text
モデル設定
calibration
alpha
購入戦略
EV基準
edge基準
rank条件
オッズ帯別条件
```

報告:

```text
2025 ROI
2026 ROI
2025/2026合算ROI
最低期間ROI
高配当除外後ROI
bootstrap CI
```

## 21. 合格基準

単勝・複勝それぞれ:

```text
2025 ROI >= 90%
2026 ROI >= 90%
2025/2026合算ROI >= 90%
十分な購入数
上位5払戻除外後も大崩れしない
```

一部戦略だけ超えた場合は「参考達成」とし、「安定達成」としない。

## 22. テスト

最低限:

```text
walk-forward split
future leakage
GPU smoke
two-model training
calibration selection across folds
market probability normalization
conservative probability
EV calculation
rank gap
odds-dependent threshold
strategy classification
rule overlap/Jaccard
validation-only rule selection
2025/2026 rule immutability
win payout join
place payout join
ROI calculation
top payout removal
drawdown
NumPy race bootstrap
strict resume
atomic output
end-to-end smoke
```

```bash
python -m pytest -q
```

## 23. 禁止事項

```text
2015年以前の追加
ランダムsplit
2025/2026でルール調整
確定オッズモデルを実運用モデルと表現
払戻推定
自動購入
Kelly基準
資金配分最適化
Ability/ANA/Ranker
大規模Optuna
旧成果物上書き
自動commit
自動push
```

## 24. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 一括実行コマンド
5. 入力データpath/hash
6. 使用feature set
7. walk-forward fold一覧
8. future leakage監査
9. GPU/CatBoost情報
10. fold別学習時間
11. fold別best iteration
12. 単勝fold別指標
13. 複勝fold別指標
14. calibration候補と採用法
15. 採用alpha
16. 確度指標
17. 単勝/複勝EV定義
18. 堅軸型・妙味型・大穴型条件
19. オッズ帯別安全マージン
20. 候補ルール数
21. 重複除去結果
22. 採用単勝ルール
23. 採用複勝ルール
24. 2020〜2024年別ROI
25. validation最低年ROI
26. validation合算ROI
27. 2025単勝bets/ROI
28. 2026単勝bets/ROI
29. 2025/2026単勝合算ROI
30. 2025複勝bets/ROI
31. 2026複勝bets/ROI
32. 2025/2026複勝合算ROI
33. 戦略別・人気帯別・オッズ帯別ROI
34. edge/rank_gap/確度帯別ROI
35. 最大連敗
36. 最大ドローダウン
37. 高配当除外ROI
38. 利益上位1%/5%寄与率
39. bootstrap 95% CI
40. 大穴型除外後ROI
41. 単勝90%安定達成判定
42. 複勝90%安定達成判定
43. 2025/2026未調整確認
44. strict resume結果
45. end-to-end実行時間
46. pytest結果
47. 既存成果物未変更確認
48. 自動購入未実施確認
49. 理想条件モデル明記
50. 未解決事項
51. 次の推奨手順

完了後は2モデルの学習からROI検証まで報告し、自動購入・資金配分・別モデル導入には進まず停止する。
