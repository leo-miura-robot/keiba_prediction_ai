# Task: 複勝オッズ帯別収益分析・重みづけ・固定評価

## 目的

既存の`final_odds_two_models_v1`および`roi_strategy_refinement_v1`の複勝予測・払戻結果を使用し、モデルを再学習せずに、重ならない複勝オッズ帯ごとの収益源を特定する。

2020〜2024年のwalk-forward validationだけを使って、以下を決定する。

```text
1. どの複勝オッズ帯が利益を生んでいるか
2. どの帯がROIを悪化させているか
3. 各帯を主力・補助・除外のどれにするか
4. 帯ごとに必要なEV・edge基準をどう変えるか
5. 帯別に購入金額を変えた場合に安定性が改善するか
```

2025 testと2026 latest_holdoutでは、2020〜2024で決めた条件を一切変更しない。

## 使用する入力

```text
outputs/final_odds_two_models_v1/
outputs/roi_strategy_refinement_v1/
outputs/model_feature_dataset_v2_1_2/
```

必要列:

```text
race_id
entry_id
race_date
data_split
actual_place
predicted_place_probability
calibrated_probability
conservative_probability
model_rank
market_rank
rank_gap
fuku_odds_low
fuku_odds_high
fuku_pay
place_ev_low
place_edge_low
```

列名が異なる場合は、既存成果物を調査し、意味が同一の列を使う。

DB本体は原則読み直さず、既存Parquet・CSV・manifestを使用する。DBへ接続する必要がある場合は、DB validation cacheのHITを確認してからread-onlyで利用する。

## 重ならない複勝オッズ帯

`fuku_odds_low`を以下の排他的な帯へ分類する。

```text
1.0以上 1.1未満
1.1以上 1.2未満
1.2以上 1.3未満
1.3以上 1.5未満
1.5以上 2.0未満
2.0以上 2.5未満
2.5以上 3.0未満
3.0以上
```

境界値が二つの帯へ重複しないようにする。各購入馬は必ず一つのオッズ帯だけに所属させる。

## 各帯で計算する指標

2020〜2024について、年別および5年合算で以下を出す。

```text
bets
races
stake
return
profit
ROI
hit count
hit rate
平均予測複勝率
実複勝率
calibration gap
平均fuku_odds_low
平均fuku_odds_high
平均払戻
最大払戻
最大連敗
最大ドローダウン
上位1件払戻除外ROI
上位3件払戻除外ROI
上位5件払戻除外ROI
bootstrap 95% CI
```

特に以下を明確にする。

```text
2.0〜2.5倍のROI
2.5〜3.0倍のROI
3.0倍以上のROI
```

## オッズ帯の信頼度スコア

ROIだけで重みを決めない。各帯の信頼度を、以下の指標から0〜1で算出する。

```text
5年合算ROI
最低年ROI
年別ROI標準偏差
合計bets
最低年bets
bootstrap 2.5%下限
上位5件除外後ROI
最大ドローダウン
```

各指標を正規化し、configで重みを管理する。

```yaml
band_reliability_weights:
  combined_roi: 0.15
  minimum_year_roi: 0.25
  bootstrap_lower: 0.20
  top5_removed_roi: 0.20
  sample_size: 0.10
  roi_stability: 0.05
  drawdown: 0.05
```

この重み自体を大規模探索しない。

## 帯の分類

各帯を次の3区分へ分類する。

```text
主力帯
補助帯
除外帯
```

主力帯の最低条件例:

```text
5年中4年以上に購入あり
合計bets >= 300
最低年bets >= 30
5年合算ROI >= 95%
上位5件除外後ROI >= 90%
bootstrap下限が他帯より良好
```

補助帯:

```text
十分な購入数はあるが、年度安定性またはbootstrap下限が主力未満
```

除外帯:

```text
長期ROIが低い
最低年ROIが著しく低い
購入数が少なすぎる
高配当に強く依存する
```

閾値はconfigへ置き、2020〜2024だけで判定する。

## 選別重み

主戦略では100円均等買いを維持する。帯別信頼度は、必要なEV・edge基準へ反映する。

```text
主力帯
→ place_ev_low >= 1.02
→ place_edge_low >= 指定下限

補助帯
→ place_ev_low >= 1.08
→ 主力帯より大きなedgeを要求

除外帯
→ 購入しない
```

高オッズ帯ほど予測誤差が大きくなるため、原則として高いEV安全余裕を要求する。無制限に閾値を最適化しない。

候補:

```text
place_ev_low:
1.00, 1.02, 1.05, 1.08, 1.10, 1.15

place_edge_low:
-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03
```

候補総数に上限を設ける。

## 賭け金重みの比較

主結果は100円均等買いとする。参考実験として、帯別信頼度による賭け金配分も比較する。

```text
均等型:
全購入100円

3段階型:
主力帯100円
補助帯50円
除外帯0円

安定度比例型:
信頼度から50円・100円・150円のいずれかへ丸める
```

連続値の賭け金最適化、Kelly基準、資金曲線最大化探索は禁止する。賭け金重みの成績は、均等買いと明確に分けて報告する。

## 重複購入の処理

同じ馬が複数の旧ルールへ該当していても、今回の統合戦略では一頭一回だけ購入する。

```text
一意キー = race_id + entry_id
```

同一レースで複数馬が選ばれた場合は、以下を比較する。

```text
全該当馬を購入
レース内place_ev_low最高馬だけ購入
レース内conservative_probability最高馬だけ購入
```

主戦略は原則1レース1頭を優先する。

## ルール選択

ルール選択には2020〜2024だけを使う。単純な5年合算ROI最大では選ばない。

優先順位:

```text
1. 十分な購入数
2. 最低年ROI
3. 上位5件除外後ROI
4. bootstrap下限
5. 最大ドローダウン
6. 5年合算ROI
```

最終候補は最大3戦略まで残す。購入集合のJaccard similarityが0.8以上なら代表戦略だけ残す。

## 2025・2026の固定評価

2020〜2024で決定した以下を完全に固定する。

```text
オッズ帯区分
帯別分類
信頼度
EV基準
edge基準
1レース1頭ルール
重複排除方法
賭け金方式
```

その後にのみ以下を評価する。

```text
2025 test
2026 latest_holdout
2025+2026合算
```

2025・2026の結果を見て設定を変更しない。

## 最終評価

均等100円戦略を正式評価とする。

最低限報告:

```text
2025 bets / ROI
2026 bets / ROI
2025+2026 bets / ROI
年別的中率
最大連敗
最大ドローダウン
上位1/3/5/10件除外ROI
bootstrap 95% CI
オッズ帯別利益寄与率
オッズ帯別購入数
1レース複数購入率
```

安定達成条件:

```text
2025 ROI >= 90%
2026 ROI >= 90%
2025+2026 ROI >= 90%
十分な購入数
上位5件除外後も大崩れしない
高オッズ帯一つだけに利益が集中しすぎない
```

2026のbootstrap下限が90%未満の場合は、点推定達成と統計的安定達成を分けて表現する。

## 出力

```text
config/place_odds_band_weighting_v1.yaml
scripts/analyze_place_odds_bands_v1.py
scripts/build_place_band_strategy_v1.py
scripts/evaluate_place_band_strategy_v1.py
scripts/run_place_odds_band_weighting_v1.py
outputs/place_odds_band_weighting_v1/
docs/place_odds_band_weighting_v1_design.md
docs/place_odds_band_weighting_v1_results.md
tests/test_place_odds_band_weighting_v1.py
```

主な出力:

```text
odds_band_summary.csv
odds_band_yearly.csv
odds_band_bootstrap.csv
odds_band_payout_dependency.csv
odds_band_reliability.csv
candidate_strategies.csv
selected_strategies.json
validation_summary.csv
test_2025_summary.csv
latest_2026_summary.csv
combined_2025_2026_summary.csv
equal_stake_comparison.csv
weighted_stake_comparison.csv
bet_details.parquet
manifest.json
```

## 一括実行

```bash
python scripts/run_place_odds_band_weighting_v1.py \
  --config config/place_odds_band_weighting_v1.yaml
```

strict resumeへ対応する。既存DB validation cache、入力成果物hash、split hash、予測hash、選択ルールhashをmanifestへ保存する。

## 性能要件

モデル再学習を行わない。目標実行時間は通常5〜15分以内。

bootstrapはレース単位のNumPy再標本化を使い、ループ内でDataFrameを再構築しない。全pytestは最後に1回だけ実行し、実装中は対象テストだけを実行する。

## 禁止事項

```text
モデル再学習
特徴量再生成
2015年以前の利用
2025/2026での調整
無制限な閾値探索
連続的な賭け金最適化
Kelly基準
自動購入
単勝戦略の追加
Ability/ANA/Ranker
既存成果物上書き
自動commit
自動push
```

## 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 一括実行コマンド
5. 使用入力とhash
6. DB cache HIT確認
7. モデル未再学習確認
8. 排他的オッズ帯定義
9. 各帯の購入数
10. 各帯の年別ROI
11. 各帯の5年合算ROI
12. 各帯の最低年ROI
13. 各帯の的中率
14. 各帯のcalibration gap
15. 各帯の最大ドローダウン
16. 各帯の上位5件除外ROI
17. 各帯のbootstrap CI
18. 2.0〜2.5倍の詳細
19. 2.5〜3.0倍の詳細
20. 3.0倍以上の詳細
21. 信頼度計算式
22. 各帯の信頼度
23. 主力・補助・除外分類
24. 帯別EV基準
25. 帯別edge基準
26. 候補戦略数
27. サンプル不足除外数
28. Jaccard除去結果
29. 最終採用戦略
30. 重複馬除去数
31. 同一レース複数購入率
32. 1レース1頭戦略結果
33. validation 2020 ROI
34. validation 2021 ROI
35. validation 2022 ROI
36. validation 2023 ROI
37. validation 2024 ROI
38. validation合算ROI
39. validation最低年ROI
40. 2025 bets / ROI
41. 2026 bets / ROI
42. 2025+2026 bets / ROI
43. 上位1/3/5/10件除外ROI
44. 最大連敗
45. 最大ドローダウン
46. bootstrap 95% CI
47. オッズ帯別利益寄与率
48. 均等100円戦略
49. 3段階賭け金戦略
50. 安定度比例戦略
51. 正式採用する賭け金方式
52. 2025/2026未調整確認
53. 複勝90%点推定達成判定
54. 複勝90%統計的安定達成判定
55. 高配当依存判定
56. strict resume結果
57. end-to-end実行時間
58. pytest結果
59. 自動購入未実施確認
60. 未解決事項
61. 次の推奨手順

完了後は、複勝オッズ帯分析・重みづけ・固定評価までで停止し、自動購入やモデル再学習には進まない。
