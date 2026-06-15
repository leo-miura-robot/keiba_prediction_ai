# Task: 複勝下限オッズ・EV件数・実ROIの三軸探索

## 目的

既存の複勝予測結果を使い、複勝下限オッズを固定せず、以下の3軸を同時に探索する。

```text
1. 複勝下限オッズの範囲
2. EV >= 1.00となる候補数
3. 実際のROI
```

モデル再学習は行わない。

対象は既存の以下の成果物とする。

```text
outputs/final_odds_two_models_v1/
outputs/roi_strategy_refinement_v1/
outputs/place_odds_band_weighting_v1/
outputs/model_feature_dataset_v2_1_2/
```

今回の目的は、単純にROI最大の条件を探すことではない。

以下を明らかにする。

```text
どの複勝下限オッズ範囲でEV >= 1.00の候補がどれだけ存在するか
その候補群の実ROIはどうか
EV基準を上げるほど実ROIも改善するか
複勝下限オッズとEVの組合せに単調性があるか
件数・安定性・ROIのバランスが最も良い領域はどこか
```

2020〜2024だけで探索・選択する。

2025 testと2026 latest_holdoutでは条件を変更しない。

---

# 1. 使用列

最低限、以下を使用する。

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

既存列名が異なる場合は意味が同一の列を使う。

ROI計算は必ず実払戻`fuku_pay`を使う。

---

# 2. DB利用

原則として既存Parquet・CSV・manifestだけを使う。

DBを直接読まない場合、manifestへ以下を記録する。

```text
database_accessed = false
database_validation_reused = true
```

DBへ接続する必要がある場合は、既存DB validation cacheのHITを確認してからread-onlyで利用する。

---

# 3. 複勝下限オッズの探索

固定の帯ではなく、下限・上限の候補を組み合わせて探索する。

## 下限候補

```text
1.0
1.1
1.2
1.3
1.4
1.5
1.6
1.8
2.0
2.2
2.5
3.0
```

## 上限候補

```text
1.1
1.2
1.3
1.4
1.5
1.6
1.8
2.0
2.2
2.5
3.0
4.0
5.0
上限なし
```

条件:

```text
lower <= fuku_odds_low < upper
```

上限なしの場合:

```text
fuku_odds_low >= lower
```

下限以上・上限未満で統一し、境界の重複を避ける。

探索する範囲はconfigへ定義する。

---

# 4. EV閾値の探索

以下の候補を比較する。

```text
0.85
0.90
0.95
1.00
1.02
1.05
1.08
1.10
1.15
1.20
```

ただし今回の中心は以下。

```text
EV >= 1.00
```

各オッズ範囲について、最低限以下を必ず出す。

```text
EV制限なし
EV >= 0.90
EV >= 0.95
EV >= 1.00
EV >= 1.05
EV >= 1.10
```

---

# 5. 年別候補数

各条件について、2020〜2026年で以下を出す。

```text
対象馬数
対象レース数
1年あたり平均候補数
月あたり平均候補数
1開催日あたり平均候補数
EV >= 1.00候補数
EV >= 1.00候補率
```

特に次を明確にする。

```text
年間EV >= 1.00候補数
年間EV >= 1.05候補数
年間EV >= 1.10候補数
```

---

# 6. 実ROI

各条件について以下を計算する。

```text
bets
races
stake
return
profit
ROI
hit count
hit rate
average fuku_odds_low
average fuku_pay
median fuku_odds_low
median fuku_pay
max payout
max losing streak
max drawdown
```

さらに以下を出す。

```text
top1 payout removed ROI
top3 payout removed ROI
top5 payout removed ROI
top10 payout removed ROI
bootstrap 95% CI
```

---

# 7. EVと実ROIの単調性

重要な監査項目とする。

以下のEV帯へ排他的に分ける。

```text
EV < 0.85
0.85 <= EV < 0.90
0.90 <= EV < 0.95
0.95 <= EV < 1.00
1.00 <= EV < 1.02
1.02 <= EV < 1.05
1.05 <= EV < 1.10
1.10 <= EV < 1.15
1.15 <= EV < 1.20
EV >= 1.20
```

各EV帯について、年別・合算で以下を出す。

```text
bets
ROI
hit rate
average predicted probability
actual place rate
calibration gap
average fuku_odds_low
```

確認事項:

```text
EVが高いほど実ROIも高いか
EV >= 1.00で実ROIが100%以上か
EV >= 1.00でも実ROIが低い範囲はどこか
EV < 1.00でも実ROIが高い範囲はどこか
```

Spearman相関などを用いて、EV帯順と実ROIの関係を参考表示してよい。

ただし相関だけで採用判断しない。

---

# 8. 候補戦略の評価

各候補戦略は以下の組合せで構成する。

```text
複勝下限オッズ範囲
EV閾値
必要ならmodel_rank条件
必要なら1レース1頭条件
```

主探索は以下に限定する。

```text
オッズ範囲
EV閾値
```

model_rankなどの追加条件は、既存主戦略と同じ固定条件を使う。

新しい複雑な条件を増やさない。

---

# 9. サンプル数条件

候補を採用可能とする最低条件:

```text
2020〜2024合計bets >= 300
5年中4年以上で購入あり
各年bets >= 30を目安
2025/2026は選択に使わない
```

EV >= 1.00候補が少ない場合は、以下のカテゴリに分ける。

```text
年間100件以上
年間30〜99件
年間10〜29件
年間10件未満
```

少数候補で高ROIでも、安定戦略と判定しない。

---

# 10. 選択基準

単純なROI最大では選ばない。

優先順位:

```text
1. 十分なEV >= 1.00候補数
2. 2020〜2024の最低年ROI
3. 上位5件除外後ROI
4. bootstrap下限
5. 5年合算ROI
6. 最大ドローダウン
7. 条件の単純さ
```

最終候補は最大5つまで残す。

似た購入集合はJaccard similarityで除去する。

```text
Jaccard >= 0.8
```

---

# 11. 2025・2026固定評価

2020〜2024で選択した条件を固定して、以下を評価する。

```text
2025 test
2026 latest_holdout
2025+2026 combined
```

変更禁止:

```text
複勝下限オッズ範囲
EV閾値
model_rank条件
1レース1頭条件
重複排除方法
```

---

# 12. 必須比較

以下を同じ表で比較する。

```text
現在の採用戦略
EV >= 1.00だけの戦略
EV >= 1.05だけの戦略
EV >= 1.10だけの戦略
最良オッズ範囲 × EV >= 1.00
最良オッズ範囲 × 最適EV閾値
```

---

# 13. 重要な問いへの回答

最終報告では、以下へ明確に答える。

```text
EV >= 1.00の馬は1年に何頭いるか
EV >= 1.00の馬は1年に何レースあるか
EV >= 1.05の馬は1年に何頭いるか
EV >= 1.10の馬は1年に何頭いるか
EV >= 1.00戦略の実ROIはいくらか
どの複勝下限オッズ範囲でEV >= 1.00が最も多いか
どの複勝下限オッズ範囲で実ROIが最も高いか
件数とROIのバランスが最も良い範囲はどこか
EVと実ROIは単調に改善しているか
EV 1未満を許容する合理性があるか
```

---

# 14. 出力

```text
config/place_odds_ev_surface_v1.yaml

scripts/analyze_place_odds_ev_surface_v1.py
scripts/select_place_odds_ev_strategy_v1.py
scripts/evaluate_place_odds_ev_strategy_v1.py
scripts/run_place_odds_ev_surface_v1.py

outputs/place_odds_ev_surface_v1/
  odds_range_summary.csv
  odds_range_yearly.csv
  ev_threshold_summary.csv
  ev_threshold_yearly.csv
  ev_band_summary.csv
  ev_band_yearly.csv
  odds_ev_grid.csv
  odds_ev_yearly.csv
  candidate_strategies.csv
  selected_strategies.json
  validation_summary.csv
  test_2025_summary.csv
  latest_2026_summary.csv
  combined_2025_2026_summary.csv
  payout_dependency.csv
  bootstrap_ci.csv
  bet_details.parquet
  manifest.json

docs/place_odds_ev_surface_v1_design.md
docs/place_odds_ev_surface_v1_results.md

tests/test_place_odds_ev_surface_v1.py
```

---

# 15. 一括実行

```bash
python scripts/run_place_odds_ev_surface_v1.py \
  --config config/place_odds_ev_surface_v1.yaml
```

strict resume対応とする。

---

# 16. 性能

既存予測を使う。

モデル再学習は禁止。

目標実行時間:

```text
5〜15分以内
```

bootstrapはNumPyのレース単位再標本化を使う。

ループ内でDataFrameを再構築しない。

---

# 17. 禁止事項

```text
モデル再学習
特徴量再生成
2015年以前の利用
2025/2026での条件調整
無制限な閾値探索
連続的な賭け金最適化
Kelly基準
自動購入
単勝戦略追加
Ability/ANA/Ranker
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
7. DB accessed有無
8. モデル未再学習確認
9. 探索した複勝下限候補
10. 探索した上限候補
11. 探索したEV閾値
12. 候補戦略総数
13. EV >= 1.00の年別馬数
14. EV >= 1.00の年別レース数
15. EV >= 1.00の年平均馬数
16. EV >= 1.05の年別馬数
17. EV >= 1.10の年別馬数
18. 各オッズ範囲の件数
19. 各オッズ範囲の年別ROI
20. 各オッズ範囲の合算ROI
21. 各EV閾値の件数
22. 各EV閾値の年別ROI
23. 各EV閾値の合算ROI
24. EV帯別件数
25. EV帯別実ROI
26. EV帯別的中率
27. EV帯別calibration gap
28. EVと実ROIの単調性
29. EVと実ROIのSpearman相関
30. 2.0倍未満のEV >= 1.00件数
31. 2.0倍以上のEV >= 1.00件数
32. EV >= 1.00で最も件数が多いオッズ範囲
33. EV >= 1.00で最もROIが高いオッズ範囲
34. 件数とROIのバランス最良範囲
35. サンプル不足除外数
36. Jaccard除去結果
37. 最終採用候補
38. validation 2020 ROI
39. validation 2021 ROI
40. validation 2022 ROI
41. validation 2023 ROI
42. validation 2024 ROI
43. validation合算ROI
44. validation最低年ROI
45. 2025 bets / ROI
46. 2026 bets / ROI
47. 2025+2026 bets / ROI
48. top1/top3/top5/top10除外ROI
49. 最大連敗
50. 最大ドローダウン
51. bootstrap 95% CI
52. 現戦略との比較
53. EV >= 1.00戦略との比較
54. EV >= 1.05戦略との比較
55. EV >= 1.10戦略との比較
56. EV 1未満許容の合理性
57. 2025/2026未調整確認
58. 複勝90%点推定達成判定
59. 複勝100%点推定達成判定
60. 統計的安定達成判定
61. 高配当依存判定
62. strict resume結果
63. end-to-end実行時間
64. pytest結果
65. 自動購入未実施確認
66. 未解決事項
67. 次の推奨手順

完了後は、複勝下限オッズ・EV件数・実ROIの三軸探索と固定評価までで停止する。
