# Codex Phase 6B Task v1
## Conservative EV Threshold Selection for Place Betting

## 0. 目的

Phase 6A認証監査で、以下が認証された。

```text
Champion raw model:
ROLLING_10Y

Champion calibration candidate:
PLATT_SCALING

Challenger shadow:
ROLLING_15Y + ISOTONIC
```

状態:

```text
operationally_activated = false
Champion = ROLLING_10Y
```

ROLLING_10Y + PLATT_SCALINGは正式な運用候補として認証可能だが、
2020～2024のPlatt vs RAW改善は小さく、
Logloss / Brier bootstrap CIは0をまたいでいる。

一方、2025+2026診断ではPlattがrawより改善し、
EV>=1.00の合算ROIは96.69%だった。

ただし、top10払戻をゼロにしたstress ROIは大きく低下し、
高配当依存が残っている。

本タスクでは新規モデル学習を行わず、
既存の時系列安全なcalibrated predictionだけを使い、
複勝EV閾値を保守的に選択・診断する。

第一段階目標:

```text
複勝回収率 90%以上
```

ただし、単純な最大ROIではなく、
件数・年度安定性・高配当依存・bootstrap下限を重視する。

---

# 1. 絶対条件

禁止:

```text
CatBoost再学習
DB接続
calibrator再fit
calibrator family再選択
特徴量変更
Champion変更
15YをChampionへ昇格
2025/2026を閾値選択へ使用
ROI直接学習
Kelly
自動購入
Optuna
大規模探索
commit/push
```

使用可能:

```text
Phase 6A既存calibrated predictions
Phase 6A certification成果物
軽量な再集計
race単位bootstrap
threshold grid評価
pytest
py_compile
artifact audit
```

---

# 2. 対象構成

## 正式選択対象

```text
ROLLING_10Y + PLATT_SCALING
```

## Shadow診断対象

```text
ROLLING_15Y + ISOTONIC
```

15Yの結果は比較・監視用であり、
本タスクでChampion変更に使用しない。

---

# 3. データ期間の役割

## Development / threshold selection

```text
2020～2024
```

EV閾値選択はこの期間だけで行う。

## Diagnostic only

```text
2025～2026
```

2025/2026は閾値・ルール選択へ使用しない。

2026が途中年度の場合は:

```text
evaluation_start_date
evaluation_end_date
latest_race_date
rows
races
```

を明記する。

---

# 4. 使用確率とEV

Champion:

```text
probability = ROLLING_10YのPLATT_SCALING済み確率
```

Shadow:

```text
probability = ROLLING_15YのISOTONIC済み確率
```

EV:

```text
EV = probability_calibrated * fuku_odds_low
```

購入単位:

```text
1件100円均等購入
```

払戻:

```text
fuku_pay
```

target:

```text
target_place_paid
```

---

# 5. Threshold grid

固定grid:

```text
1.00, 1.01, 1.02, ..., 1.30
```

31候補のみ。

追加の細かい局所探索は禁止。

各thresholdについて、
strategy別・年度別・2020～2024合算を出す。

必須列:

```text
strategy
calibration_method
threshold
Year
bet_count
race_count_with_bet
stake
payout
roi
hit_count
hit_rate
average_odds
median_odds
average_probability
average_ev
median_ev
```

合算ROIは:

```text
total payout / total stake
```

年度ROIの単純平均は禁止。

---

# 6. Nested walk-forward threshold policy

全期間を見て1つの閾値を決めるだけでなく、
閾値選択自体の時系列頑健性を確認する。

## 2021 validation

```text
threshold selection data = 2020
```

## 2022 validation

```text
threshold selection data = 2020～2021
```

## 2023 validation

```text
threshold selection data = 2020～2022
```

## 2024 validation

```text
threshold selection data = 2020～2023
```

2020はprior selection dataがないため、
nested評価から除外するか、事前固定1.00をbaselineとして明示する。

各foldで、下記の保守的選択ルールを同じまま適用する。
未来年の結果を使ってルール変更しない。

出力:

```text
selection_end_year
selected_threshold
selection_bet_count
selection_combined_roi
validation_year
validation_bet_count
validation_roi
validation_payout
validation_stake
```

---

# 7. Threshold eligibility

2020～2024の最終候補として適格になるには、
最低限次を満たす。

```text
combined bet_count >= 300
bet years >= 5
minimum yearly bet_count >= 30
ROI >= 90% の年が3年以上
```

さらに次を記録する。

```text
minimum yearly ROI
median yearly ROI
ROI standard deviation
worst-year drawdown from 100%
```

適格候補が0件の場合:

```text
status = NO_THRESHOLD_CERTIFIED
```

として停止し、無理に選ばない。

---

# 8. 高配当依存stress

各thresholdで2020～2024合算について:

## 8.1 Row removed

払戻額上位の的中行を除外する。

```text
top1
top3
top5
top10
```

## 8.2 Payout zeroed

対象行は残し、払戻だけ0にする。

```text
top1
top3
top5
top10
```

必須不変条件:

```text
payout_zeroed_stress_roi <= normal_roi
```

必須出力:

```text
normal_roi
row_removed_roi
payout_zeroed_roi
roi_drop_point
remaining_bet_count
removed_or_zeroed_payout_share
```

---

# 9. Race bootstrap

各thresholdについてrace単位bootstrap 5000回。

2020～2024のraceを復元抽出し、
各raceに紐づく購入・stake・payoutをまとめて再集計する。

出力:

```text
threshold
point_roi
bootstrap_mean_roi
roi_ci_lower
roi_ci_upper
probability_roi_ge_90
probability_roi_ge_100
```

購入がないraceもsampling populationへ含める。

少数購入による過大評価を避ける。

---

# 10. 保守的な最終選択ルール

単純な最大ROIは禁止。

まずeligibilityを満たす候補だけ残す。

候補順位は以下の順に決める。

```text
1. probability_roi_ge_90 が最大
2. ROI 95% CI lower が最大
3. top5 payout-zeroed ROI が最大
4. minimum yearly ROI が最大
5. bet_count が多い
6. thresholdが低い
```

ただし、最終認証には追加条件を置く。

```text
combined ROI >= 90%
probability_roi_ge_90 >= 0.70
top3 payout-zeroed ROI >= 70%
top5 payout-zeroed ROI >= 60%
nested walk-forward validationの合算ROI >= 85%
```

すべて満たす場合:

```text
status = THRESHOLD_CANDIDATE_CERTIFIED
```

一部を満たさない場合:

```text
status = DIAGNOSTIC_ONLY
```

閾値を無理に有効化しない。

---

# 11. 2025/2026固定診断

2020～2024だけで選んだ最終thresholdを固定し、
2025・2026へ適用する。

2025/2026でthresholdを変更しない。

出力:

```text
strategy
threshold
Year
bet_count
race_count_with_bet
stake
payout
roi
hit_count
hit_rate
average_odds
average_probability
average_ev
```

2025+2026合算はtotal payout / total stake。

同じthresholdで次を出す。

```text
row_removed top1/3/5/10
payout_zeroed top1/3/5/10
race bootstrap ROI CI
```

2025/2026結果は認証判定には使用せず、
diagnostic列として分離する。

---

# 12. Shadow comparison

ROLLING_15Y + ISOTONICにも、
Championで選んだ同じthresholdを適用する。

さらに15Y単独の2020～2024最適thresholdも診断として算出してよいが、
以下を守る。

```text
shadow_selected_threshold
operationally_activated = false
Champion変更なし
```

比較:

```text
Champion同一thresholdでの10Y vs 15Y
各strategy独自thresholdでの診断
bet overlap
common bets
10Y-only bets
15Y-only bets
Jaccard
```

ROIだけで昇格判断しない。

---

# 13. セグメント診断

最終Champion thresholdについて、
2020～2024と2025/2026を分けて出す。

最低限:

```text
Year
JyoCD
surface
distance_bucket
field_size_bucket
odds_bucket
EV_bucket
```

各segment:

```text
bet_count
stake
payout
roi
hit_rate
payout_share
```

小標本:

```text
bet_count < 20
```

にはsmall_sample=trueを付け、
強い結論を出さない。

セグメント除外ルールは作らない。

---

# 14. Operational activation

本タスクでは自動有効化しない。

```text
operationally_activated = false
```

出力する判定:

```text
threshold_status
recommended_threshold
activation_recommended
reason
```

`activation_recommended=true`でも、
ユーザー承認まではfalseのまま。

---

# 15. 出力先

```text
outputs/place_market_offset_ev_threshold_phase6b_v1/
```

コード・設定・文書:

```text
config/place_market_offset_ev_threshold_phase6b_v1.yaml
scripts/run_place_market_offset_ev_threshold_phase6b_v1.py
scripts/audit_place_market_offset_ev_threshold_phase6b_v1.py
tests/test_place_market_offset_ev_threshold_phase6b_v1.py
docs/place_market_offset_ev_threshold_phase6b_v1_results.md
docs/place_market_offset_ev_threshold_phase6b_v1_runbook.md
```

---

# 16. 必須成果物

```text
threshold_grid_by_year.csv
threshold_grid_combined_2020_2024.csv
threshold_nested_walk_forward.csv
threshold_eligibility.csv

threshold_roi_bootstrap.csv
threshold_row_removed_stress.csv
threshold_payout_zeroed_stress.csv

selected_threshold.json
diagnostic_2025_2026.csv
diagnostic_2025_2026_stress.csv
diagnostic_2025_2026_bootstrap.csv

shadow_threshold_comparison.csv
bet_overlap.csv
segment_diagnostic.csv

manifest.json
audit_report.md
```

---

# 17. 必須監査・テスト

1. 新規学習なし
2. DB接続なし
3. calibration再fitなし
4. ChampionはROLLING_10Y
5. Champion methodはPLATT_SCALING
6. ChallengerはROLLING_15Y
7. Challenger methodはISOTONIC
8. selectionは2020～2024のみ
9. 2025/2026を選択へ使用しない
10. threshold gridは1.00～1.30、step 0.01
11. EVはprobability_calibrated * fuku_odds_low
12. stakeは1件100円
13. combined ROIはtotal payout/total stake
14. 年度ROI平均をcombined ROIにしない
15. eligibility条件固定
16. nested walk-forwardで未来年を使わない
17. bootstrapはrace単位
18. 購入なしraceもbootstrap母集団に含む
19. payout-zeroed ROI <= normal ROI
20. betなしROIはN/A
21. top payout依存を記録
22. threshold最大ROIだけで選ばない
23. 2025/2026でthreshold変更なし
24. shadow結果でChampion変更なし
25. ensembleなし
26. operationally_activated=false
27. 既存成果物上書きなし
28. commit/pushなし

---

# 18. 実行範囲

Codex側で以下まで実行する。

```text
py_compile
pytest
threshold grid集計
nested walk-forward
race bootstrap 5000回
stress test
2025/2026固定診断
shadow比較
segment診断
artifact audit
結果レポート
```

新規学習がないため、Codex側で最後まで実行してよい。

---

# 19. 最終報告

簡潔に以下を報告する。

1. 作成・変更ファイル
2. 2020～2024 threshold grid概要
3. eligibility通過候補
4. nested walk-forward結果
5. race bootstrap
6. 高配当stress
7. 推奨threshold
8. threshold_status
9. activation_recommended
10. 2025結果
11. 2026結果
12. 2025+2026合算
13. shadow比較
14. bet overlap
15. segment診断
16. operationally_activated=false
17. py_compile
18. pytest
19. audit checks / failed
20. git status --short

commit/pushは行わない。
