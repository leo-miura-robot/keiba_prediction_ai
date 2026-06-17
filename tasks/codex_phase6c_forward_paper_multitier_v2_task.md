# Codex Phase 6C Task v2
## Forward Paper Trading with Fixed Multi-Tier EV Monitoring

## 0. 目的

Phase 6Bでは、複勝EV閾値について以下が得られた。

```text
Champion:
ROLLING_10Y + PLATT_SCALING

Certified threshold candidate:
EV >= 1.00

Challenger shadow:
ROLLING_15Y + ISOTONIC
```

2020～2024では`EV >= 1.00`のみがeligibilityを通過し、
`THRESHOLD_CANDIDATE_CERTIFIED`となった。

一方で、将来的にEVの安全余裕を大きくした方が
回収率・安定性・高配当依存の面で有利かを確認する価値がある。

ただし、過去データを見ながら閾値を後付けで変更すると、
threshold overfittingが発生する。

そのため、今後の前向きpaper trading開始前に、
次の4段階を固定して同時監視する。

```text
CORE      : EV >= 1.00
MARGIN    : EV >= 1.05
HIGH      : EV >= 1.10
VERY_HIGH : EV >= 1.15
```

正式閾値は引き続き`EV >= 1.00`とする。

1.05、1.10、1.15はshadow diagnosticであり、
実購入や正式閾値変更には使用しない。

---

# 1. 固定運用状態

```text
Champion:
ROLLING_10Y + PLATT_SCALING

Official threshold:
EV >= 1.00

Challenger:
ROLLING_15Y + ISOTONIC

Threshold tiers:
1.00 / 1.05 / 1.10 / 1.15

paper_trade_enabled:
true

real_money_betting:
false

operationally_activated:
false
```

出力上は次を分離する。

```text
threshold_candidate_certified = true
official_threshold = 1.00
shadow_thresholds = [1.05, 1.10, 1.15]
paper_trade_recommended = true
real_money_activation_recommended = false
operationally_activated = false
```

---

# 2. 絶対禁止事項

```text
自動購入
実馬券購入API
Kelly
資金配分最適化
正式threshold変更
calibrator再fit
CatBoost再学習
Champion変更
2025/2026を使った再選択
ensemble
過去レースへの後付け予測
結果確定後の予測生成
prediction rowの上書き
過去成績を見て新しいtierを追加
commit/push
```

---

# 3. 固定EV tier

以下の4段階だけを使用する。

| tier | threshold | role |
|---|---:|---|
| CORE | 1.00 | 正式threshold candidate |
| MARGIN | 1.05 | shadow diagnostic |
| HIGH | 1.10 | shadow diagnostic |
| VERY_HIGH | 1.15 | shadow diagnostic |

追加thresholdは禁止。

例:

```text
EV = 1.12

CORE      = bet
MARGIN    = bet
HIGH      = bet
VERY_HIGH = no bet
```

---

# 4. Tier包含関係

必須不変条件:

```text
VERY_HIGH bets ⊆ HIGH bets
HIGH bets ⊆ MARGIN bets
MARGIN bets ⊆ CORE bets
```

strategy別・prediction run別に検証する。

違反が1件でもあればaudit failureとする。

---

# 5. Forward評価の原則

前向き実績へ含める予測は、
必ずレース結果確定前に保存する。

予測時点で次を固定する。

```text
prediction_generated_at
data_cutoff_at
odds_observed_at
source_data_latest_at
model_version
calibrator_version
threshold_policy_version
feature_hash
model_hash
calibrator_hash
config_hash
```

レース結果・払戻は予測生成時には保存しない。

結果取り込みは別コマンドで行う。

---

# 6. 2段階ワークフロー

## 6.1 Before-race prediction

例:

```powershell
python scripts/run_place_market_offset_forward_paper_phase6c_v2.py predict `
  --race-date 2026-06-20 `
  --output-root outputs/place_market_offset_forward_paper_phase6c_v2
```

必須処理:

1. 対象レース・出走馬を取得
2. data cutoffを検証
3. Champion raw probability生成
4. Platt calibration適用
5. Challenger raw probability生成
6. Isotonic calibration適用
7. EVを計算
8. 4 tierのpaper bet flagを同時生成
9. append-onlyで保存
10. 同一prediction runの重複を拒否
11. 結果確定後の予測生成を拒否

## 6.2 After-race settlement

例:

```powershell
python scripts/run_place_market_offset_forward_paper_phase6c_v2.py settle `
  --race-date 2026-06-20 `
  --output-root outputs/place_market_offset_forward_paper_phase6c_v2
```

必須処理:

1. 既存prediction runを読む
2. 結果・払戻を取得
3. `target_place_paid`を付与
4. tierごとにpaper stake=100円でsettle
5. prediction tableを変更しない
6. settlement tableへappend
7. `settled_at > prediction_generated_at`を検証
8. 未確定レースはpendingのまま残す

---

# 7. 保存形式

SQLiteを中心にし、Parquet/CSV exportも作る。

```text
outputs/place_market_offset_forward_paper_phase6c_v2/
  forward_paper.sqlite
  predictions_export.parquet
  prediction_tiers_export.parquet
  settlements_export.parquet
  daily_summary.csv
  monthly_summary.csv
  cumulative_summary.csv
  threshold_comparison.csv
  audit_report.md
  manifest.json
```

## 7.1 prediction_runs

```text
prediction_run_id
prediction_generated_at
race_date
data_cutoff_at
odds_observed_at
source_data_latest_at
code_version
config_hash
feature_hash
threshold_policy_version
is_fixture
created_at
```

## 7.2 predictions

```text
prediction_run_id
strategy
calibration_method
entry_id
race_id
race_date
horse_no
probability_raw
probability_calibrated
market_logit
residual_raw
fuku_odds_low_at_prediction
ev_at_prediction
model_version
model_hash
calibrator_version
calibrator_hash
```

## 7.3 prediction_tiers

```text
prediction_run_id
strategy
entry_id
race_id
threshold
threshold_tier
paper_bet_flag
paper_stake_if_bet
```

一意キー:

```text
prediction_run_id + strategy + entry_id + threshold
```

## 7.4 settlements

```text
prediction_run_id
strategy
entry_id
race_id
race_date
threshold
threshold_tier
settled_at
target_place_paid
fuku_pay
paper_stake
paper_payout
paper_profit
settlement_status
```

## 7.5 不変条件

```text
prediction rows immutable
tier rows immutable
settlement append-only
odds_at_predictionと確定後oddsを混同しない
fixtureをforward実績へ含めない
結果確定後に生成された予測をforward実績へ含めない
```

---

# 8. モデル参照

既存の認証済みartifactだけを利用する。

Champion:

```text
ROLLING_10Y
PLATT_SCALING
```

Challenger:

```text
ROLLING_15Y
ISOTONIC
```

必要artifactが不足している場合:

```text
status = BLOCKED
```

として、不足ファイルを列挙する。

推測による再学習・再fitは禁止。

---

# 9. 日次・月次・累積レポート

strategy × threshold tierごとに出す。

```text
period
strategy
calibration_method
threshold
threshold_tier
races
entries
paper_bets
stake
payout
profit
ROI
hit_count
hit_rate
average_odds
median_odds
average_probability
average_EV
median_EV
```

betなしの場合:

```text
ROI = N/A
```

0%として表示しない。

---

# 10. Threshold比較レポート

Championについて:

```text
CORE vs MARGIN
CORE vs HIGH
CORE vs VERY_HIGH
```

必須:

```text
bet_count
stake
payout
ROI
hit_rate
average_odds
average_EV
common bets
higher-tier-only concept is impossible
lower-tier-only bets
Jaccard similarity
```

tierは包含関係なので、
高tierは低tierの部分集合として比較する。

追加で:

```text
incremental bets between tiers
incremental stake
incremental payout
incremental ROI
```

例:

```text
CORE追加部分 = CORE - MARGIN
MARGIN追加部分 = MARGIN - HIGH
HIGH追加部分 = HIGH - VERY_HIGH
```

---

# 11. High-payout stress

strategy × threshold tierごとに:

```text
top1
top3
top5
top10
```

について、

```text
row_removed
payout_zeroed
```

を計算する。

必須:

```text
payout_zeroed_stress_roi <= normal_roi
```

出力:

```text
normal_roi
stress_roi
roi_drop_point
removed_or_zeroed_payout_share
remaining_bet_count
```

---

# 12. Forward statistical monitoring

各tierは最低観測条件に達するまで結論を出さない。

## CORE

```text
minimum elapsed period = 6 months
minimum races = 1000
minimum paper bets = 200
```

## MARGIN / HIGH / VERY_HIGH

```text
minimum elapsed period = 6 months
minimum paper bets = 200
```

200件未満の場合:

```text
decision_status = CONTINUE_MONITORING
```

条件達成後のみ:

```text
race bootstrap ROI CI
P(ROI >= 90%)
P(ROI >= 100%)
monthly worst ROI
monthly median ROI
maximum drawdown
top3 payout-zeroed ROI
top5 payout-zeroed ROI
```

を正式に解釈する。

---

# 13. Threshold昇格ルール

正式thresholdは1.00のまま。

1.05、1.10、1.15の昇格を検討できるのは、
前向きデータで以下をすべて満たした後だけ。

```text
elapsed >= 6 months
paper bets >= 200
combined ROI >= 90%
P(ROI >= 90%) >= 0.80
top3 payout-zeroed ROI >= 70%
top5 payout-zeroed ROI >= 60%
monthly worst ROIがCOREより著しく悪化しない
```

さらに、COREと比較して:

```text
ROI改善
または
同程度ROIで高配当依存が低下
または
同程度ROIでdrawdownが改善
```

のどれかが必要。

本タスク内では昇格しない。

---

# 14. Champion–Challenger比較

各threshold tierで10Yと15Yを比較する。

```text
common bets
10Y-only bets
15Y-only bets
Jaccard similarity
common-bet ROI
10Y-only ROI
15Y-only ROI
```

ROIだけでChampion変更しない。

---

# 15. Activation gate

本タスクでは:

```text
paper_trade_enabled = true
real_money_activation_candidate = false
operationally_activated = false
```

を固定する。

以下を満たしても自動購入は作らない。

```text
6か月以上
1000 races以上
CORE 200 bets以上
P(ROI >= 90%) >= 0.80
高配当stress基準通過
timestamp違反なし
```

最終的な実購入判断はユーザー承認を必要とする。

---

# 16. Manual runbook

Windows PowerShell向けに、初心者でも実行できる形で記載する。

```text
事前準備
予測前コマンド
保存結果の確認
tier別bet確認
レース後settle
pending確認
日次レポート
月次レポート
累積レポート
threshold比較
バックアップ
resume
エラー対応
```

---

# 17. Optional PowerShell wrappers

```text
scripts/run_forward_predict_phase6c_v2.ps1
scripts/run_forward_settle_phase6c_v2.ps1
scripts/run_forward_report_phase6c_v2.ps1
```

デフォルトでは自動スケジュール登録しない。

秘密情報をコードへ書かない。

---

# 18. 実装ファイル

```text
config/place_market_offset_forward_paper_phase6c_v2.yaml
scripts/run_place_market_offset_forward_paper_phase6c_v2.py
scripts/audit_place_market_offset_forward_paper_phase6c_v2.py
tests/test_place_market_offset_forward_paper_phase6c_v2.py

scripts/run_forward_predict_phase6c_v2.ps1
scripts/run_forward_settle_phase6c_v2.ps1
scripts/run_forward_report_phase6c_v2.ps1

docs/place_market_offset_forward_paper_phase6c_v2_design.md
docs/place_market_offset_forward_paper_phase6c_v2_runbook.md
```

出力先:

```text
outputs/place_market_offset_forward_paper_phase6c_v2/
```

---

# 19. Fixture smoke test

実未来レースが使えない場合はfixtureでテストする。

fixtureはforward実績へ混ぜない。

必須smoke:

1. predict成功
2. 4 tier生成
3. tier包含関係通過
4. 同一run重複拒否
5. settle成功
6. prediction row不変
7. settlement append-only
8. timestamp violation拒否
9. betなしROI=N/A
10. fixture除外
11. daily report生成
12. monthly report生成
13. cumulative report生成
14. threshold comparison生成

---

# 20. 必須監査

1. 新規モデル学習なし
2. calibration再fitなし
3. Champion固定
4. Challenger固定
5. official threshold=1.00
6. shadow thresholds=1.05/1.10/1.15
7. 追加thresholdなし
8. tier包含関係
9. prediction before settlement
10. prediction timestampあり
11. data cutoffあり
12. odds observed timestampあり
13. prediction immutable
14. tier immutable
15. settlement append-only
16. duplicate key拒否
17. odds snapshot保存
18. paper stake=100円
19. betなしROI=N/A
20. fixture除外
21. real money bettingなし
22. auto-buyなし
23. Kellyなし
24. threshold自動昇格なし
25. operationally_activated=false
26. Champion変更なし
27. 既存成果物上書きなし
28. commit/pushなし

---

# 21. 実行範囲

Codex側で以下まで実行する。

```text
設計
実装
py_compile
pytest
fixture smoke
artifact audit
runbook作成
```

実未来レースへのpredictは、
結果未確定の対象データが安全に利用できる場合のみ1回実施してよい。

結果未確定ならsettleしない。

---

# 22. 最終報告

簡潔に以下を報告する。

1. 作成・変更ファイル
2. SQLite schema
3. 4 tier定義
4. tier包含関係監査
5. predictコマンド
6. settleコマンド
7. reportコマンド
8. fixture smoke結果
9. timestamp・immutability監査
10. official threshold=1.00確認
11. shadow thresholds確認
12. paper_trade_enabled=true
13. real_money_activation_candidate=false
14. operationally_activated=false
15. py_compile
16. pytest
17. audit checks / failed
18. 実未来レース予測の有無
19. git status --short

commit/pushは行わない。
