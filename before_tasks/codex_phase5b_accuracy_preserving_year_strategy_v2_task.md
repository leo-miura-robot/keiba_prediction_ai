# Codex Phase 5B Task v2
## Accuracy-Preserving Year-Usage Strategy Framework
## Codex implements and tests; local terminal performs long training

## 0. このタスクの目的

競馬予想AIプロジェクトを、次の分担で進める。

```text
GPT:
実験設計、Codexへの指示、結果の採否判断

Codex:
危険な評価コードの実装、回帰テスト、実行コマンド作成

ローカル端末:
長時間のモデル学習・評価を実行

ユーザー:
実行結果をGPTへ共有
```

重要:

> Codex利用量を節約するために、科学的な比較範囲・候補戦略・評価精度を削ってはいけない。

節約対象は以下だけ。

```text
Codexが長時間学習の完了を待つ時間
Codexによるログの継続監視
CodexによるMarkdownの過剰な整形
Codexによる既知結果の長い再説明
```

年度利用戦略の全候補は実装可能な共通runnerで支え、
長時間実行はローカル端末から行う。

---

# 1. 直前の軽量監査で確定した問題

## Critical

Phase 4のIsotonic Regressionで、既に0/1の`actual_place`へ:

```python
train["actual_place"].le(3).astype(int)
```

を適用し、全件を1としてfitしていた。

影響:

```text
probability_calibrated ≈ 0.999999
calibrated EV / ROIは無効
```

## High

1. Phase 4のmanifestと実際のEV/ROI使用確率列が不一致
2. Phase 5でouter validation年をCatBoostの`eval_set`へ渡していた
3. Phase 5の旧StressROIがEV>=1集合ではなく別母集団を使っていた

## 再利用禁止

```text
Phase 4 probability_calibrated
Phase 4 calibrated EV/ROI
Phase 4 Isotonic実装
Phase 5旧StressROI関数
outer validationをeval_setへ渡すPhase 5 training実装
```

## 再利用候補

```text
src/features/history_builder_v2_1.py
scripts/audit_phase5_v2.py の安全な evaluate_stress_roi()
Phase 5の履歴拡張データ（hashと期間を再確認した場合）
```

---

# 2. 現在の正式BASE

```text
C1R0_fixed300_ablation_drop_person_codes
```

基本構造:

```text
final_logit = market_logit + residual_raw
probability_raw = sigmoid(final_logit)
```

固定条件:

```text
tree count = 300
KisyuCode除外
ChokyosiCode除外
Year除外
p_market除外
market_logit除外
raw odds / 人気 / 市場順位除外
結果・払戻・管理ID除外
rate smoothing不採用
```

Phase 5Bでは、特徴量セットやCatBoost hyperparameterを変更しない。

---

# 3. 最重要原則

## 3.1 精度を落とさない

「省コスト」は以下を意味しない。

```text
候補戦略を減らす
foldを減らす
評価年を減らす
指標を簡略化する
bootstrapを省略する
旧BASEとの再現性確認を省略する
```

最終比較では全候補を2020～2024の5foldで評価する。

## 3.2 新runnerの互換性ゲート

新しいPhase 5B runnerが、正式BASEの`LEGACY_2016`を再現できなければ、
他戦略の比較へ進んではいけない。

新runnerによるLEGACY_2016と既存正式BASEについて、同じvalidation年で:

```text
row key完全一致
予測行数一致
feature名・順序一致
target一致
market_logit一致
tree count一致
probability_raw分布一致
Logloss / Brier一致
```

を検証する。

推奨許容差:

```text
row key match = 100%
feature list match = exact
target match = exact
market_logit p99 abs diff <= 1e-8
probability_raw p99 abs diff <= 1e-4
Logloss abs diff <= 1e-5
Brier abs diff <= 1e-5
```

GPU非決定性などで超える場合は、原因を確認して停止する。
許容差を勝手に広げない。

---

# 4. 年度利用戦略

共通runnerで以下をすべて実装可能にする。

## S0: LEGACY_2016

```text
history start = 2016
model train start = 2016
```

## S1: WARMUP_2006_TRAIN_2016

```text
history start = 2006
model train start = 2016
```

2006～2015は履歴ウォームアップ専用。

## S2: EXPANDING_FULL_2006

```text
history start = 2006
model train = 2006 ～ outer validation前年
```

## S3: ROLLING_10Y

```text
history start = 2006
model train = outer validation前年までの直近10年間
```

例:

```text
validation 2020 -> train 2010～2019
validation 2021 -> train 2011～2020
validation 2022 -> train 2012～2021
validation 2023 -> train 2013～2022
validation 2024 -> train 2014～2023
```

## S4: ROLLING_15Y

```text
history start = 2006
model train = outer validation前年までの直近15年間
```

データ開始年より前へ遡る場合は2006で切る。

## S5: FULL_2006_TIME_DECAY_HL5

```text
history start = 2006
model train = 2006 ～ outer validation前年
half_life_years = 5
```

## S6: FULL_2006_TIME_DECAY_HL10

```text
history start = 2006
model train = 2006 ～ outer validation前年
half_life_years = 10
```

TIME_DECAYの重み:

```text
age_years = max(0, (validation_start_date - race_date).days / 365.25)
weight = 2 ** (-age_years / half_life_years)
```

各foldのtraining rowsで:

```text
mean(weight) = 1
```

となるよう正規化する。

保存:

```text
weight min
weight p1
weight p50
weight p99
weight max
effective sample size
```

大規模なhalf-life探索は禁止。
固定候補は5年と10年だけ。

---

# 5. Market baseline

`market_logit`は、単純なオッズ変換ではなく、
Logistic Regression等の学習済み市場モデルであることが監査で確認された。

Phase 5Bでは、候補戦略ごとに原則:

```text
market model training rows
=
residual model training rows
```

とする。

例:

```text
ROLLING_10Y / validation 2024:
market model train 2014～2023
residual model train 2014～2023
```

保存必須:

```text
strategy
validation_year
market_train_start
market_train_end
market_train_rows
market_target
market_input_columns
market_model_config
residual_train_start
residual_train_end
residual_train_rows
```

ただし、新runnerのLEGACY互換性を再現するために、
既存正式BASEが別のmarket windowを使っている場合は、
以下を明示的に分ける。

```text
LEGACY_COMPAT:
既存BASEと完全同一のmarket window

ALIGNED_STRATEGY:
market windowをresidual windowへ揃える
```

比較要因を混ぜない。
LEGACY再現確認を先に行い、その後にaligned戦略を評価する。

---

# 6. CatBoost学習の安全条件

outer validationを`eval_set`へ渡さない。

必須:

```text
iterations = 300
use_best_model = False
early_stopping_rounds = None
od_type / od_waitを無効化
```

禁止:

```text
model.fit(train_pool, eval_set=outer_validation_pool)
outer validationをbest iteration選択に使う
outer validationをearly stoppingへ使う
```

inner splitによるhyperparameter調整も今回は行わない。
300本固定で公平比較する。

---

# 7. Calibrationの扱い

Phase 5Bではcalibrationを実装・fitしない。

主評価・モデル選択:

```text
probability_raw
```

のみ。

理由:

- Phase 4 calibration成果物は破損
- 年度戦略選択前にcalibrationを混ぜると比較要因が増える
- calibrationは年度戦略確定後にゼロから別タスクで構築する

したがってPhase 5Bでは:

```text
probability_calibrated = not generated
```

とmanifestへ明記する。

ROIもPhase 5Bでは`probability_raw`による補助診断だけ。
正式なEV閾値選択はPhase 6で行う。

---

# 8. 評価期間

## 主比較

```text
2020
2021
2022
2023
2024
```

全7戦略を5foldで評価する。

## 補助評価

```text
2016～2019
```

2006年から学習可能な戦略だけの長期安定性確認。
正式選択には直接使わず、regime drift診断とする。

## 診断

```text
2025
2026
```

2020～2024で戦略固定後にだけ実行する。

2025/2026は既に複数回確認済みのため、
完全未使用holdoutではなく診断期間として扱う。

---

# 9. 主評価

すべて`probability_raw`。

```text
runner-weighted Logloss
runner-weighted Brier
fixed-bin ECE
calibration slope
calibration intercept
race-wise Spearman
top probability hit rate
```

年度安定性:

```text
mean
median
std
CV
best year
worst year
5年間のstrategy別勝敗数
```

残差:

```text
residual mean
residual std
abs residual p90
abs residual p95
abs residual p99
```

---

# 10. ROI補助診断

ROIだけで年度戦略を選ばない。

使用:

```text
probability_raw
EV >= 1.00
100円均等購入
```

安全な`evaluate_stress_roi()`を再利用または独立実装する。

必須:

```text
normal ROI
row_removed_roi
payout_zeroed_stress_roi
```

k:

```text
1
3
5
10
```

不変条件:

```text
payout_zeroed_stress_roi <= normal ROI
```

合算ROI:

```text
total payout / total stake
```

年度ROIの単純平均は禁止。

予測なし:

```text
N/A
```

ROI 0%にしない。

---

# 11. 統計比較

各候補対LEGACY_2016でrace単位paired bootstrap。

```text
n_bootstrap = 5000
seed fixed
sampling unit = race
```

対象:

```text
Logloss delta
Brier delta
```

出力:

```text
point estimate
bootstrap mean
95% CI lower
95% CI upper
candidate better probability
```

平均だけで「明確に優位」と判断しない。

---

# 12. 採用基準

優先順位:

1. mean Logloss
2. mean Brier
3. worst-year Logloss / Brier
4. 年度別勝敗
5. residual p95 / p99
6. ECE / slope / intercept
7. strategyの単純さ
8. ROIは補助

候補を正式採用できる条件の目安:

```text
Logloss/Brierの点推定が改善
worst-yearが悪化しない
5年のうち少なくとも3年で改善
residual tailが大幅悪化しない
bootstrapで支持される、またはCIが0を跨いでも全年度一貫性がある
```

差が極小かつCIが0を跨ぎ、年度一貫性もない場合:

```text
LEGACY_2016維持
```

---

# 13. Codex利用量を節約する実行フロー

## Stage A: Codexが実装・テスト

Codexが行う:

```text
新規generic runner
config
year-window builder
market/residual provenance
ROI helper接続
parity gate
pytest
実行コマンド作成
```

Codexは長時間学習を実行しない。

## Stage B: ユーザーがローカルでLEGACY parity smoke

対象:

```text
validation year = 2024
strategy = LEGACY_2016
```

parity gateが通るまで他戦略へ進まない。

## Stage C: ユーザーがローカルで2024全戦略smoke

対象:

```text
validation year = 2024
all 7 strategies
```

確認:

```text
train window
row count
market window
weight distribution
probability_raw
metric computation
ROI invariants
```

## Stage D: ユーザーがローカルで2020～2024本実行

```text
all 7 strategies × 5 folds
```

## Stage E: 結果だけGPTへ共有

共有対象:

```text
metrics summary
worst-year summary
bootstrap summary
strategy provenance
ROI diagnostic
git status
```

Codexに長時間ログ解析をさせない。

---

# 14. 新規ファイル

推奨:

```text
config/place_market_offset_year_strategy_phase5b_v2.yaml
scripts/run_place_market_offset_year_strategy_phase5b_v2.py
scripts/audit_place_market_offset_year_strategy_phase5b_v2.py
tests/test_place_market_offset_year_strategy_phase5b_v2.py
docs/place_market_offset_year_strategy_phase5b_v2_runbook.md
```

出力:

```text
outputs/place_market_offset_year_strategy_phase5b_v2/
models/place_market_offset_year_strategy_phase5b_v2/
```

既存成果物を上書きしない。

---

# 15. 必須成果物

```text
strategy_definition.csv
walk_forward_folds.csv
legacy_parity_check.csv

market_model_window_by_strategy.csv
residual_model_window_by_strategy.csv
sample_weight_summary.csv

metrics_by_strategy_fold.csv
metrics_by_strategy_2020_2024.csv
metrics_by_strategy_2016_2019_aux.csv

yearly_win_loss_matrix.csv
worst_year_summary.csv
residual_stability_by_strategy.csv
paired_bootstrap_summary.csv

roi_diagnostic_raw.csv
roi_row_removed_raw.csv
roi_payout_zeroed_stress_raw.csv

selected_year_strategy.json
phase5b_2025_2026_diagnostic.csv

manifest.json
audit_report.md
```

---

# 16. 必須テスト

1. outer validationをeval_setへ渡さない
2. iterations=300
3. early stopping無効
4. use_best_model=False
5. validation年をtrain rowsへ含めない
6. 各strategyのyear windowが正しい
7. history startとmodel train startを分離
8. market/residual windowを記録
9. LEGACY parity gate
10. feature allowlist完全一致
11. target完全一致
12. probability_rawだけで選択
13. calibrationをfitしない
14. current race leakageなし
15. same-day future leakageなし
16. TIME_DECAY weight式
17. TIME_DECAY mean weight=1
18. stress ROI母集団一致
19. payout-zeroed stress <= normal ROI
20. 合算ROIはtotal payout/total stake
21. predictionなしをN/A
22. bootstrapはrace単位
23. seed固定
24. resume時hash検証
25. 既存成果物を上書きしない
26. 自動git操作なし

---

# 17. Resume / キャッシュ

各foldについて保存:

```text
strategy
validation_year
config_hash
feature_hash
source_hash
train_key_hash
validation_key_hash
market_window_hash
sample_weight_hash
model_path
prediction_path
status
```

全hashとrow countが一致する場合だけ再利用。
不足foldだけ実行する。

---

# 18. 停止条件

以下の場合は推測せず停止する。

```text
LEGACY parity gate失敗
正式BASE予測を一意に特定できない
feature allowlist不一致
market model実装を特定できない
train/validation key重複
outer validationがeval_setへ入る
StressROI不変条件違反
history startとtrain startを分離できない
source hash不一致
```

---

# 19. Codexの今回の完了条件

Codexは以下まで行って停止する。

```text
実装
軽量pytest
LEGACY 2024 smoke用コマンド提示
全戦略2024 smoke用コマンド提示
2020～2024本実行コマンド提示
出力確認コマンド提示
```

Codex自身は長時間学習を開始しない。

---

# 20. 最終報告

簡潔に以下だけを報告する。

1. 作成・変更ファイル
2. 再利用した安全な既存部品
3. 再利用禁止にした部品
4. LEGACY parity gateの内容
5. 実装した7戦略
6. market window設計
7. TIME_DECAY式
8. pytest結果
9. LEGACY 2024 smokeコマンド
10. 全戦略2024 smokeコマンド
11. 2020～2024本実行コマンド
12. resumeコマンド
13. 成果物確認コマンド
14. git status --short
15. git diff --stat

自動commit/pushは行わない。
