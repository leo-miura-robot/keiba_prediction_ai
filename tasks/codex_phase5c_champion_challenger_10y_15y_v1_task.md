# Codex Phase 5C Task v1
## Champion–Challenger Parallel Evaluation: ROLLING_10Y vs ROLLING_15Y

## 0. 目的

Phase 5Bの2020～2024比較では、以下が上位だった。

```text
Champion candidate: ROLLING_10Y
Challenger candidate: ROLLING_15Y
```

結果概要:

```text
ROLLING_10Y:
mean Logloss = 0.405659
mean Brier   = 0.130333

ROLLING_15Y:
mean Logloss = 0.405697
mean Brier   = 0.130341
```

直接paired bootstrapではROLLING_10Yが点推定で優位だったが、
95% CIは0をまたぎ、差は小さい。

また、ROLLING_15Yはworst-year性能でわずかに優れていた。

したがって、今後は次の運用とする。

```text
Champion:
ROLLING_10Y

Challenger / shadow:
ROLLING_15Y
```

本タスクでは両戦略を2025・2026へ固定適用し、
比較・監視・将来の入替基準を整備する。

---

# 1. 重要な位置づけ

2025・2026は既に複数回確認されているため、
完全な未使用holdoutではない。

したがって本タスクの2025・2026結果は:

```text
diagnostic only
```

であり、以下に使用してはいけない。

```text
年度戦略の再選択
特徴量選択
EV閾値選択
hyperparameter調整
10Yと15Yの即時入替
```

正式運用上は引き続き:

```text
Champion = ROLLING_10Y
Challenger = ROLLING_15Y
```

とする。

---

# 2. 実行対象

## ROLLING_10Y

```text
2025 validation:
train 2015～2024

2026 validation:
train 2016～2025
```

## ROLLING_15Y

```text
2025 validation:
train 2010～2024

2026 validation:
train 2011～2025
```

履歴特徴生成の基盤は両方とも2006年以降を使用する。

2026が途中年度の場合は、以下を記録する。

```text
evaluation start date
evaluation end date
row count
race count
latest available race date
```

---

# 3. 固定条件

Phase 5Bの安全仕様を変更しない。

```text
target = target_place_paid
iterations = 300
outer validationをeval_setへ渡さない
use_best_model = False
early stoppingなし
overfitting detectorなし
official C1R0 feature allowlist
market model window = residual model window
probability_rawのみ
calibrationなし
```

禁止:

```text
新しい特徴量
hyperparameter変更
target変更
market baseline変更
EV閾値変更
2025/2026を使った選択
ensemble作成
10Y/15Y平均確率の採用
自動購入
commit/push
```

---

# 4. 実行範囲

Codex側で次まで実行してよい。

```text
py_compile
pytest
ROLLING_10Y 2025/2026
ROLLING_15Y 2025/2026
成果物監査
既存成果物だけを使った後処理
レポート作成
```

4foldだけなので、Codex側で実行してよい。

30分を大きく超える場合のみ停止し、
resume可能なローカル実行コマンドを提示する。

---

# 5. 主評価指標

各戦略・各年・2025+2026合算について出す。

```text
rows
races
Logloss
Brier
ECE
calibration slope
calibration intercept
race-wise Spearman
residual mean
residual std
abs residual p90
abs residual p95
abs residual p99
```

2025+2026合算指標は、行を結合して再計算する。
年度指標の単純平均は禁止。

---

# 6. 直接比較

同じrace / runner keys上で10Yと15Yを比較する。

## 6.1 Probability comparison

```text
mean absolute probability difference
p50 / p90 / p95 / p99 absolute difference
Pearson correlation
Spearman correlation
maximum absolute difference
```

## 6.2 Ranking agreement

レース単位で:

```text
top1 predicted horse agreement
top3 set overlap
rank correlation
```

## 6.3 Error comparison

```text
10Y better row count
15Y better row count
tie count

10Y better race count
15Y better race count
tie race count
```

## 6.4 Direct paired bootstrap

2025、2026、2025+2026についてrace単位5000回。

対象:

```text
Logloss delta = 10Y - 15Y
Brier delta   = 10Y - 15Y
```

出力:

```text
point estimate
bootstrap mean
95% CI lower
95% CI upper
10Y better probability
```

2025/2026結果でChampionを変更しない。

---

# 7. ROI補助診断

使用:

```text
probability_raw
EV >= 1.00
100円均等購入
```

各戦略・各年・合算について:

```text
bet_count
race_count_with_bet
stake
payout
ROI
hit_count
hit_rate
average odds
average predicted probability
average EV
```

合算ROI:

```text
total payout / total stake
```

年度ROIの平均は禁止。

## 7.1 Bet overlap

10Yと15Yについて:

```text
common bets
10Y-only bets
15Y-only bets
Jaccard similarity
common-bet ROI
10Y-only ROI
15Y-only ROI
```

## 7.2 High-payout robustness

両戦略で:

```text
row_removed top1 / 3 / 5 / 10
payout_zeroed top1 / 3 / 5 / 10
```

必須不変条件:

```text
payout_zeroed_stress_roi <= normal_roi
```

ROIは選択基準ではなく補助診断。

---

# 8. セグメント差の診断

10Yと15Yの差がどこで出るか、
既存列だけで軽量集計する。

最低限:

```text
year
track / JyoCD
surface
distance bucket
field size bucket
odds bucket
predicted probability bucket
```

出力:

```text
row count
race count
10Y Logloss
15Y Logloss
delta Logloss
10Y Brier
15Y Brier
delta Brier
```

小標本は明示し、結論を強くしない。

新規特徴量は作らない。

---

# 9. Champion–Challenger運用ルール

次の運用ルールをJSONとMarkdownで保存する。

## Champion

```text
ROLLING_10Y
```

## Challenger

```text
ROLLING_15Y
```

## 即時入替禁止

2025/2026診断の結果だけでは入れ替えない。

## 将来の前向き比較開始

```text
freeze date:
本タスク完了日

forward evaluation:
freeze date以降に確定した新規レースのみ
```

## 最低観測条件

入替検討は、次をすべて満たしてから。

```text
6か月以上
1000レース以上
各戦略のEV>=1候補が200件以上
```

不足時は比較継続。

## 入替検討条件

Challengerの昇格を検討できるのは:

```text
race-paired bootstrapでLogloss差が改善方向
95% CI upper < 0
Brierも悪化しない
worst-monthがChampionより大幅悪化しない
residual p99が10%以上悪化しない
```

ROIは補助条件であり、
ROIだけで昇格させない。

## 入替後

Championを変更する場合も、
旧Championを最低3か月はshadowで継続する。

---

# 10. 予測保存仕様

今後の前向き比較用に、
両戦略の予測を同一行へ保存できる形式を定義する。

必須列:

```text
prediction_generated_at
model_version
strategy
entry_id
race_id
race_date
probability_raw
market_logit
residual_raw
odds_available_at_prediction
EV_at_prediction
feature_hash
model_hash
data_cutoff_date
```

実際の自動運用はまだ実装しない。
保存スキーマと手動運用runbookだけ作る。

---

# 11. 出力先

```text
outputs/place_market_offset_champion_challenger_phase5c_v1/
models/place_market_offset_champion_challenger_phase5c_v1/
```

コード・文書:

```text
config/place_market_offset_champion_challenger_phase5c_v1.yaml
scripts/run_place_market_offset_champion_challenger_phase5c_v1.py
scripts/audit_place_market_offset_champion_challenger_phase5c_v1.py
tests/test_place_market_offset_champion_challenger_phase5c_v1.py
docs/place_market_offset_champion_challenger_phase5c_v1_results.md
docs/place_market_offset_champion_challenger_forward_runbook_v1.md
```

---

# 12. 必須成果物

```text
metrics_2025_2026_by_strategy.csv
metrics_2025_2026_combined.csv
direct_pairwise_bootstrap.csv
probability_agreement.csv
ranking_agreement.csv
error_win_loss.csv

roi_by_strategy_year.csv
roi_combined.csv
bet_overlap.csv
roi_row_removed.csv
roi_payout_zeroed_stress.csv

segment_comparison.csv
champion_challenger_policy.json
forward_prediction_schema.json

manifest.json
audit_report.md
```

---

# 13. 必須テスト

1. 10Y 2025 train=2015～2024
2. 10Y 2026 train=2016～2025
3. 15Y 2025 train=2010～2024
4. 15Y 2026 train=2011～2025
5. validation年をtrainへ含めない
6. target_place_paid固定
7. iterations=300
8. outer eval_setなし
9. use_best_model=False
10. early stoppingなし
11. feature allowlist固定
12. market/residual window一致
13. calibration fitなし
14. 2025/2026で選択変更しない
15. direct bootstrapはrace単位
16. 合算指標は行結合で再計算
17. 合算ROIはtotal payout/total stake
18. payout-zeroed stress <= normal ROI
19. bet overlapの集合整合
20. 予測なしをROI 0扱いしない
21. 既存成果物を上書きしない
22. 自動git操作なし

---

# 14. 最終報告

簡潔に以下を報告する。

1. 作成・変更ファイル
2. 4foldの実行結果
3. 2025指標
4. 2026指標
5. 2025+2026合算指標
6. 10Y vs 15Y direct bootstrap
7. probability / ranking agreement
8. bet overlap
9. ROIと高配当依存
10. segment差
11. Champion維持判定
12. forward比較ルール
13. py_compile結果
14. pytest結果
15. audit結果
16. git status --short

2025/2026の結果にかかわらず、
本タスク内でChampionを変更しない。

commit/pushは行わない。
