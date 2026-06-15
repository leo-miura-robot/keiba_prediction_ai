# Task: 確定オッズ2モデルのROI失速要因分析・購入戦略再設計

## 目的

`final_odds_two_models_v1` の結果を基に、単勝・複勝ROIが90%へ届かなかった原因を分解し、2020〜2024年のvalidationだけを使って購入戦略を再設計する。

今回はモデルを再学習しない。

現状:

```text
単勝 validation合算ROI: 81.50%
単勝 2025 ROI: 78.23%
単勝 2026 ROI: 82.72%
単勝 2025/2026合算ROI: 79.59%

複勝 validation合算ROI: 85.06%
複勝 2025 ROI: 88.23%
複勝 2026 ROI: 89.21%
複勝 2025/2026合算ROI: 88.52%
```

第一目標:

```text
単勝ROI >= 90%
複勝ROI >= 90%
```

単年度、少数購入、一部高配当だけで超えた結果は安定達成としない。

---

## 1. 基本方針

```text
1. 複勝の低オッズ過剰購入を特定
2. レース種別ごとの得意・不得意を特定
3. 単勝は本命coreと妙味馬戦略を分離
4. 類似ルールを削減
5. 2020〜2024だけで戦略選択
6. 2025・2026は固定評価
```

禁止:

```text
モデル再学習
特徴量変更
2015年以前の追加
ランダムsplit
2025/2026でルール調整
大規模Optuna
自動購入
資金配分最適化
Kelly基準
Ability/ANA/Ranker
旧成果物上書き
自動commit/push
```

---

## 2. 入力

```text
outputs/final_odds_two_models_v1/
models/final_odds_two_models_v1/
outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet
```

必要列:

```text
race_id, entry_id, race_date, data_split
actual_win, actual_place
prediction, calibrated_probability, conservative_probability
tan_odds, tan_ninki, fuku_odds_low, fuku_odds_high, fuku_ninki
tan_pay, fuku_pay
model_rank, market_rank, rank_gap, edge, margin, entropy
```

V2.1.2から最低限次も結合する。

```text
競馬場
芝/ダート
距離
クラス/グレード
頭数
牝馬限定
年齢条件
ハンデ/別定/定量
新馬/未勝利/条件戦/オープン/重賞
```

---

## 3. 複勝の低オッズ過剰購入分析

`fuku_odds_low`を次の帯で集計する。

```text
1.0-1.1
1.1-1.2
1.2-1.3
1.3-1.5
1.5-2.0
2.0-3.0
3.0+
```

各帯で:

```text
bets
races
hit rate
mean predicted probability
actual place rate
calibration gap
mean fuku_odds_low
mean payout
ROI
max drawdown
top5 payout removal ROI
bootstrap 95% CI
```

人気帯も確認する。

```text
1番人気
2〜3番人気
4〜6番人気
7番人気以下
```

---

## 4. 複勝の安全余裕

```text
break_even_probability = 1 / fuku_odds_low
place_edge_low = conservative_place_probability - break_even_probability
place_ev_low = conservative_place_probability * fuku_odds_low
```

購入選別は下限オッズを使い、ROIは実払戻`fuku_pay`で計算する。

候補:

```text
place_edge_low >= 0.01, 0.02, 0.03, 0.05
place_ev_low >= 1.02, 1.05, 1.08, 1.10, 1.15
```

低オッズほど大きなedgeを要求する設計も比較する。

```text
fuku_odds_low < 1.3           → edge >= 0.05
1.3 <= fuku_odds_low < 1.8    → edge >= 0.03
fuku_odds_low >= 1.8          → edge >= 0.02
```

---

## 5. レース種別分析

単勝・複勝を次で分解する。

```text
芝 / ダート / 障害
短距離 / マイル / 中距離 / 長距離
新馬 / 未勝利 / 1勝 / 2勝 / 3勝 / OP-L / G3 / G2 / G1
〜8頭 / 9〜12頭 / 13〜16頭 / 17頭以上
牝馬限定
ハンデ / 定量 / 別定
競馬場
月
```

各セグメントで:

```text
bets
hit rate
ROI
average odds
max drawdown
top5 removal ROI
bootstrap CI
year count
minimum yearly bets
minimum yearly ROI
```

採用条件:

```text
2020〜2024のうち3年以上で購入あり
合計bets >= 300
各年bets >= 30を目安
validation合算ROIが基準戦略より改善
最低年ROIが悪化しすぎない
top5 removal後も改善が残る
```

複数条件の掛け合わせは原則2軸まで。

---

## 6. 単勝戦略の分離

### 6.1 本命core

```text
model_rank = 1
```

市場人気、オッズ、edge、margin、entropy、レース種別別に分析する。

### 6.2 妙味型

```text
model_rank <= 3
market_rank >= 4
rank_gap >= 2
edge >= 0.02
conservative_ev >= オッズ帯別基準
```

AI順位2〜3位を必ず独立評価する。

### 6.3 人気逆転型

```text
rank_gap >= 2
```

本命coreと混ぜず独立集計する。

### 6.4 大穴型

```text
tan_odds >= 20
```

別枠とし、主戦略購入件数の10%以下を目安とする。

---

## 7. 候補ルール

総当たりは禁止。意味のある少数候補のみ。

```text
単勝候補 <= 200
複勝候補 <= 200
```

単勝軸:

```text
strategy type
model_rank
market_rank
rank_gap
edge
conservative_ev
odds band
race segment
```

複勝軸:

```text
fuku_odds_low band
place_edge_low
place_ev_low
model_rank
market_rank
race segment
```

---

## 8. ルール選択

2020〜2024だけで選ぶ。

評価:

```text
validation合算ROI
最低年ROI
年別ROI標準偏差
購入数
最低年購入数
top1 removal ROI
top5 removal ROI
最大ドローダウン
bootstrap 2.5%下限
```

優先順位:

```text
1. 十分な購入数
2. 最低年ROI
3. top5 removal ROI
4. bootstrap下限
5. validation合算ROI
```

2024年だけ高いルールは採用しない。

---

## 9. 類似ルール除去

購入馬集合のJaccard similarityを使う。

```text
Jaccard >= 0.8
```

なら代表のみ残す。

代表選択:

```text
最低年ROI
top5 removal ROI
購入数
ルールの単純さ
```

最終採用:

```text
単勝 最大3ルール
複勝 最大3ルール
```

---

## 10. 2025・2026固定評価

2020〜2024で決めた条件を変更せず適用する。

```text
2025 test
2026 latest_holdout
```

報告:

```text
2025 ROI
2026 ROI
2025/2026合算ROI
bets
hit rate
max drawdown
top5 removal ROI
bootstrap CI
```

---

## 11. 比較対象

```text
旧単勝core
新単勝core
新単勝妙味型
新単勝人気逆転型

旧複勝core
低オッズ除外型
edge要求型
レース種別選別型
```

---

## 12. 合格基準

安定達成:

```text
2025 ROI >= 90%
2026 ROI >= 90%
2025/2026合算ROI >= 90%
十分なbets
top5 payout removal後も大崩れしない
bootstrap下限が旧戦略より改善
```

合算だけ90%以上、片年未達、少数購入、高配当依存は参考達成とする。

---

## 13. 出力

```text
config/roi_strategy_refinement_v1.yaml
scripts/analyze_roi_failure_segments_v1.py
scripts/refine_final_odds_rules_v1.py
scripts/evaluate_refined_rules_v1.py
scripts/run_roi_strategy_refinement_v1.py

outputs/roi_strategy_refinement_v1/
  baseline_summary.csv
  place_low_odds_analysis.csv
  place_edge_analysis.csv
  race_segment_summary.csv
  race_segment_yearly.csv
  win_core_analysis.csv
  win_value_analysis.csv
  win_rank_reversal_analysis.csv
  rule_candidates_win.csv
  rule_candidates_place.csv
  rule_overlap_matrix.csv
  selected_rules.json
  validation_rule_summary.csv
  test_2025_summary.csv
  latest_2026_summary.csv
  combined_2025_2026_summary.csv
  payout_dependency.csv
  drawdown_summary.csv
  bootstrap_ci.csv
  bet_details.parquet
  manifest.json

docs/roi_strategy_refinement_v1_design.md
docs/roi_strategy_refinement_v1_results.md
```

---

## 14. 一括実行

```bash
python scripts/run_roi_strategy_refinement_v1.py --config config/roi_strategy_refinement_v1.yaml
```

処理順:

```text
preflight
→ 既存結果確認
→ 低オッズ複勝分析
→ レース種別分析
→ 単勝戦略分離
→ 候補ルール生成
→ 類似ルール除去
→ validation選択
→ 2025固定評価
→ 2026固定評価
→ 安定性監査
→ レポート
```

strict resume対応。

---

## 15. bootstrap

前回のCPU停止を再発させない。

```text
レース単位集計配列
NumPy再標本化
bootstrapループ内でDataFrame再構築禁止
```

1000回を基本とし、負荷が高ければ500回へ減らしてよい。

---

## 16. テスト

最低限:

```text
input loading
split immutability
place odds bands
break-even probability
place edge
race segment mapping
class normalization
win strategy separation
rank reversal
odds-dependent EV threshold
candidate count limit
minimum sample filter
Jaccard overlap
validation-only selection
2025/2026 immutability
ROI
payout removal
drawdown
NumPy bootstrap
strict resume
atomic output
end-to-end smoke
```

```bash
python -m pytest -q
```

---

## 17. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 一括実行コマンド
5. 使用入力
6. 旧単勝core成績
7. 旧複勝core成績
8. 複勝オッズ帯別購入数・ROI
9. 低オッズ複勝が全体ROIへ与えた影響
10. place_edge_low帯別ROI
11. 芝/ダート別ROI
12. 距離帯別ROI
13. クラス別ROI
14. 頭数別ROI
15. 競馬場別ROI
16. 月別ROI
17. 有効な単一セグメント
18. 有効な2軸セグメント
19. サンプル不足セグメント
20. 単勝本命core結果
21. 単勝妙味型結果
22. 単勝人気逆転型結果
23. 単勝AI順位2〜3位結果
24. 大穴型・大穴除外後結果
25. 単勝候補ルール数
26. 複勝候補ルール数
27. サンプル条件で除外した数
28. Jaccard除去前後ルール数
29. 採用単勝ルール
30. 採用複勝ルール
31. 2020〜2024年別ROI
32. validation最低年ROI
33. validation合算ROI
34. 2025単勝ROI/bets
35. 2026単勝ROI/bets
36. 2025/2026単勝合算ROI
37. 2025複勝ROI/bets
38. 2026複勝ROI/bets
39. 2025/2026複勝合算ROI
40. top1/top3/top5/top10除外ROI
41. 最大連敗
42. 最大ドローダウン
43. bootstrap 95% CI
44. 旧戦略との差
45. 2025/2026未調整確認
46. 単勝90%安定達成判定
47. 複勝90%安定達成判定
48. 参考達成の有無
49. strict resume結果
50. end-to-end実行時間
51. pytest結果
52. モデル未再学習確認
53. 自動購入未実施確認
54. 未解決事項
55. 次の推奨手順

完了後は購入戦略の再設計と固定評価まで報告し、モデル再学習、自動購入、資金配分には進まず停止する。
