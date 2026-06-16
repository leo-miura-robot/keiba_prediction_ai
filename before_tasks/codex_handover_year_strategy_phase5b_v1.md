# Codex Handover: C1R0 Horse Racing AI
## Phase 5B — Year-Usage Strategy Reassessment and Safe Continuation

## 0. この文書の目的

この文書は、競馬予想AIプロジェクトをGemini / AntigravityからCodexへ戻すための引き継ぎ資料である。

次の目的を持つ。

1. これまでの実装・監査・評価結果を整理する
2. 現在の正式BASEを明確にする
3. Gemini実装で発生した重大な問題を共有する
4. 2006～2015年DB追加後の年度利用戦略をフラットに再設計する
5. 次にCodexが実施すべき作業範囲を定義する
6. 既存成果物を壊さず、安全に再開する

---

# 1. プロジェクトの第一目標

第一段階の目標:

```text
単勝ROI >= 90%
複勝ROI >= 90%
```

現在は主に複勝モデルを改善している。

重要:

- ROIは最終目標
- ただしモデル選択をROIだけで行わない
- 確率モデルの比較はLogloss / Brier / calibration / worst-yearを優先
- 購入戦略はモデル選択後の別レイヤーとして扱う

---

# 2. 現在のモデル構造

正式なモデル系統はC1R0。

基本式:

```text
final_logit = market_logit + residual_raw
probability_raw = sigmoid(final_logit)
```

意味:

- market_logit:
  市場オッズ由来の市場基準確率
- residual_raw:
  CatBoostが市場からのズレを補正
- CatBoostには市場情報を直接入れない

正式BASE:

```text
C1R0_fixed300_ablation_drop_person_codes
```

固定内容:

```text
tree count = 300
KisyuCode除外
ChokyosiCode除外
Year除外
p_market除外
market_logit除外
raw odds除外
人気除外
市場順位除外
結果・払戻・管理ID除外
```

当面維持している特徴:

```text
trainer_past_starts
jockey_past_starts
MonthDay
Kaiji
Nichiji
RaceNum
horse_last3_avg_time
horse_last5_avg_time
BaTaijyu
各種raw rate特徴
```

---

# 3. これまでの主要経緯

## 3.1 初期C1の問題

初期C1は市場情報をbaselineとして使う一方、CatBoost特徴にも以下が入っていた。

```text
Year
p_market
market_logit
```

そのため市場情報の二重使用とYear shortcutが疑われた。

また、2025でEV>=1件数が急増した。

例:

```text
2024: 22件
2025: 655件
```

最終モデル更新による残差拡大が主因だった。

## 3.2 C1R0 pure market offset

市場情報はbaselineのみとし、CatBoostから市場情報を除外。

3000本モデルでは残差が膨らみ過学習傾向。

その後tree count監査で300本を採用。

重要:

```text
250本の方がabs residual p95は低い
300本は安定性とSpearmanのバランスで採用
```

300本BASEの既知結果:

```text
2025:
Logloss 0.402056
Brier 0.129081
ECE 0.003542
EV>=1件数 217
ROI 82.58%

2026:
Logloss 0.383807
Brier 0.122618
ECE 0.006291
EV>=1件数 126
ROI 89.21%
```

## 3.3 人物コード除外

`KisyuCode`と`ChokyosiCode`は高カーディナリティかつ未知率が増加。

除外モデル:

```text
C1R0_fixed300_ablation_drop_person_codes
```

既知診断:

```text
2025 ROI 73.35%
2026 ROI 104.02%
```

のちにcalibration定義の再整理が入っているため、
各数値は必ず使用確率列と成果物を確認すること。

## 3.4 累積出走数変換

trainer/jockey past startsについて:

```text
raw
log1p
clip p99
clip p99 + log1p
drop
```

を比較。

Phase 2ではclip_p99_log1pが微小改善と報告されたが、
Phase 3ではrawが明確に良いという矛盾が出た。

監査により原因判明:

```text
Phase 2:
calibrated vs calibrated

Phase 3:
calibrated raw vs uncalibrated clip
```

公平なuncalibrated comparisonでは差は極小でCIが0を跨いだため:

```text
raw維持
```

## 3.5 Rate smoothing

trainer / jockey / horse_surface rateへEmpirical-Bayes smoothingを実施。

候補:

```text
strength 5
strength 10
strength 20
```

一部の候補で微小な改善やROI上振れが見られたが、
統合モデルのLogloss差は極小でbootstrap CIが0を跨いだ。

正式採用:

```text
rate smoothingなし
raw rate維持
```

---

# 4. Gemini / Antigravityで発生した重大な問題

Codexは以下を既知リスクとして扱うこと。

## 4.1 calibrated / uncalibrated混在

同じような列名で:

```text
probability
final_probability
probability_raw
probability_calibrated
```

が混在。

比較対象ごとに意味が違う事故が発生した。

今後は必ず:

```text
probability_raw
probability_calibrated
probability_used_for_model_selection
probability_used_for_ev
is_calibrated
calibration_method
```

を明示する。

## 4.2 in-fold Isotonic fit

Phase 4で致命的なバグが発生。

```text
学習内予測
↓
同じ行でIsotonic fit
```

により確率がほぼ1.0へ歪み、
全頭近くが高EVになる異常が発生した。

正しい構造:

```text
outer training期間内でtime-based OOF予測を作る
↓
そのOOF予測だけでcalibratorをfit
↓
outer validation年へ適用
```

outer validation年の正解は絶対に見ない。

## 4.3 StressROIの不可能な逆転

過去に:

```text
normal ROI = 84.0%
payout-zeroed StressROI = 84.2%
```

という不可能な結果が出た。

原因は母集団フィルタ条件の誤り。

必須不変条件:

```text
payout_zeroed_stress_roi <= normal_roi
```

## 4.4 未評価をROI 0%と表示

予測が存在しないBASEの2025/2026を:

```text
ROI 0.0%
```

と誤表記。

正しくは:

```text
N/A
not evaluated
```

## 4.5 統計的に有意でない差を「明確」と報告

FULL_2006のLogloss改善:

```text
95% CI = [-0.00050, +0.00013]
```

ゼロを跨ぐにもかかわらず、
初期報告では「明確に高精度」と結論した。

修正版では撤回済み。

## 4.6 最終報告の省略

ROI、bootstrap、使用確率列、再利用数など、
必須項目が最終要約から抜けることが複数回あった。

Codexは成果物実体を確認し、
報告文だけを信頼しないこと。

---

# 5. 2006～2015 DB統合 Phase 5

2006～2015年のDBを取得済み。

比較した候補:

```text
BASE_2016
WARMUP_2006_TRAIN_2016
FULL_2006
```

定義:

## BASE_2016

```text
履歴生成開始: 2016
モデル学習開始: 2016
```

## WARMUP_2006_TRAIN_2016

```text
履歴生成開始: 2006
モデル学習開始: 2016
2006～2015は履歴ウォームアップ専用
```

## FULL_2006

```text
履歴生成開始: 2006
モデル学習開始: 2006
```

初期報告ではFULL_2006を強く推奨したが、
追加audit v2で正式採用は撤回。

Phase 5 audit v2の最終結論:

```text
FULL_2006は有望候補
正式BASEは従来C1R0を維持
```

既知結果:

```text
BASE_2016:
2020～2024 ROI 107.5%（旧報告）
2025+2026 combined ROI 88.17%（audit v2）

FULL_2006:
2025+2026 combined ROI 76.82%

BASE vs FULL Logloss delta bootstrap:
95% CI [-0.00050, +0.00013]
```

したがって:

```text
FULLの確率精度は微小改善方向
ただし統計的優位性なし
購入ROIではBASEが上
```

---

# 6. 重要な方針修正

従来BASE維持は:

```text
2016開始が最適と確定した
```

という意味ではない。

正確には:

```text
FULL_2006が現BASEを置換する十分な根拠がなかった
```

だけである。

2006年以降を使わない前提にしてはいけない。

今後は年度利用戦略をフラットに比較する。

---

# 7. 次に比較すべき年度利用戦略

履歴生成開始年とモデル学習期間を分離する。

## S0 LEGACY_2016

```text
history start = 2016
train start = 2016
```

正式BASE比較用。

## S1 WARMUP_2006_TRAIN_2016

```text
history start = 2006
train start = 2016
```

純粋なウォームアップ効果。

## S2 EXPANDING_FULL_2006

```text
history start = 2006
train = 2006 ～ validation前年
```

全期間expanding。

## S3 ROLLING_10Y

```text
history start = 2006
train = validation前年から遡る10年間
```

例:

```text
2020評価: 2010～2019
2021評価: 2011～2020
2022評価: 2012～2021
2023評価: 2013～2022
2024評価: 2014～2023
```

## S4 FULL_2006_TIME_DECAY

```text
history start = 2006
train rows = 2006～validation前年
古い行ほどsample_weightを小さくする
```

大規模探索は禁止。

固定候補例:

```text
half_life_years = 5
half_life_years = 10
```

最初は1候補だけでもよい。
追加候補は明確な理由がある場合のみ。

## S5 ROLLING_15Y

FULLとROLLING_10Yの結果が大きく異なる場合だけ追加。

無条件実行禁止。

---

# 8. 市場baselineの年度戦略

market_logitはLogistic Regression由来である可能性が高い。

Phase 5では2016～2019で学習した市場モデルを2006～2015へ外挿したと報告された。

これはFULLだけに特殊条件を作る。

Codexは実コードを確認し、
以下を明確化する。

```text
market_logitは単純オッズ変換か
学習済みLogistic Regressionか
```

学習モデルなら原則として:

```text
market model training window
=
residual model training window
```

を比較候補ごとに揃える。

例:

```text
ROLLING_10Y 2020評価:
market model train 2010～2019
residual model train 2010～2019
```

ただし市場モデル変更自体が別要因になるため、
必要に応じて以下を分ける。

```text
A: market window fixed
B: market window aligned
```

最初から候補を爆発させない。

---

# 9. 評価期間

## 9.1 主比較

```text
2020～2024
```

全候補を同条件で比較。

## 9.2 補助評価

2006年があるため:

```text
2016～2019
```

を長期安定性確認に使える。

ただし候補によって学習可能期間が異なるため、
主選択とは分けて扱う。

用途:

- 2020～2024だけへの適応確認
- regime drift確認
- early-period stability

## 9.3 2025 / 2026

これらは既に何度も確認済み。

厳密には完全未使用holdoutではない。

扱い:

```text
diagnostic period
```

今後の真の最終評価は、
仕様固定後の未来レースによるforward testとする。

---

# 10. 年度利用戦略の選択基準

主優先順位:

1. probability_raw runner-weighted Logloss
2. probability_raw Brier
3. worst-year Logloss
4. worst-year Brier
5. 年度別安定性
6. residual abs p95 / p99
7. calibration slope / intercept
8. race-wise ranking
9. モデル複雑性
10. ROIは補助診断

差が小さい場合:

```text
race-paired bootstrap
n_bootstrap = 5000
```

CIが0を跨ぎ、差が極小なら単純な戦略を優先。

ただし:

```text
全年度で一貫して微小改善
worst-yearも改善
tailも改善
```

なら実質的採用を検討してよい。

---

# 11. EV閾値 Phase 6について

すでに以下のタスクが作成されている可能性がある。

```text
tasks/antigravity_place_ev_threshold_phase6_v1_task.md
```

内容:

```text
EV threshold:
1.00
1.03
1.05
1.07
1.10
1.12
1.15
1.20
1.25
1.30
```

ただし年度利用戦略が未確定なため、
正式なEV閾値選択はまだ実行しない。

正しい順番:

```text
Phase 5B:
year-usage strategy selection
↓
Phase 6:
EV threshold robustness
↓
Phase 7:
feature engineering
```

既存Phase 6タスクは保留する。

---

# 12. 次のCodexタスク

## タスク名

```text
Phase 5B:
Year-Usage Strategy Audit
```

## 目的

以下を公平に比較する。

```text
LEGACY_2016
WARMUP_2006_TRAIN_2016
EXPANDING_FULL_2006
ROLLING_10Y
FULL_2006_TIME_DECAY
```

## 固定条件

```text
同じ特徴allowlist
tree count = 300
同じCatBoost hyperparameters
同じseed
同じtarget
同じouter validation rows
同じevaluation functions
model selection uses probability_raw
```

## 禁止

- 2025/2026で選択
- ROIだけで選択
- random split
- large Optuna
- Ability / ANA
- Ranker
- Kelly
- automatic purchase
- automatic git commit / push
- existing artifact overwrite

---

# 13. Codexが最初に行うこと

1. `git status --short`
2. `git diff --stat`
3. `git log -10 --oneline`
4. `tasks/`一覧確認
5. Phase 1～5関連成果物の実在確認
6. Phase 5 audit_v2の出力先確認
7. Geminiが変更したtracked files確認
8. source DB / derived Parquetのhash確認
9. current official BASE artifact確認
10. market_logit生成実装確認
11. calibration実装確認
12. 既存year-strategy候補の再利用可否確認

不明なパスを推測しない。

---

# 14. 信頼してよい結果 / 再監査が必要な結果

## 比較的信頼してよい

```text
tree count 300採用
人物コード除外
rate smoothing不採用
raw trainer/jockey starts維持
FULL_2006に明確な統計優位性なし
正式BASE維持
```

## 必ず成果物を再確認

```text
全ROI数値
calibrated ROI
Phase 4結果
Phase 5 StressROI
2025/2026 merged ROI
market_logit extrapolation
calibration provenance
Geminiの最終報告だけにある数値
```

---

# 15. ファイル候補

過去に作成済みの可能性があるタスク:

```text
tasks/audit_place_market_offset_feature_importance_v1_task.md
tasks/place_market_offset_catboost_c1r0_v1_task.md
tasks/place_market_offset_catboost_c1r0_tree_count_v1_task.md
tasks/place_market_offset_catboost_c1r0_feature_cleanup_v1_task.md
tasks/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_task.md
tasks/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_task.md
tasks/place_market_offset_catboost_c1r0_metric_consistency_audit_v1_task.md
tasks/antigravity_c1r0_metric_consistency_handover_v2_task.md
tasks/antigravity_c1r0_rate_smoothing_phase4_v1_task.md
tasks/antigravity_history_extension_2006_phase5_v1_task.md
tasks/antigravity_place_ev_threshold_phase6_v1_task.md
```

実在確認してから読む。

---

# 16. 出力方針

Phase 5Bでは新規出力先を使う。

推奨:

```text
config/place_market_offset_year_strategy_phase5b_v1.yaml
scripts/run_place_market_offset_year_strategy_phase5b_v1.py
scripts/audit_place_market_offset_year_strategy_phase5b_v1.py
tests/test_place_market_offset_year_strategy_phase5b_v1.py
docs/place_market_offset_year_strategy_phase5b_v1_results.md

outputs/place_market_offset_year_strategy_phase5b_v1/
models/place_market_offset_year_strategy_phase5b_v1/
```

既存成果物を上書きしない。

---

# 17. 必須成果物

```text
strategy_definition.csv
walk_forward_folds.csv
market_model_window_by_strategy.csv
calibration_provenance_by_fold.csv

metrics_by_strategy_fold.csv
metrics_by_strategy_2020_2024.csv
metrics_by_strategy_2016_2019_aux.csv

residual_stability_by_strategy.csv
worst_year_summary.csv
yearly_win_loss_matrix.csv

paired_bootstrap_summary.csv

roi_diagnostic_raw.csv
roi_diagnostic_calibrated.csv
roi_high_payout_row_removed.csv
roi_high_payout_zeroed_stress.csv

selected_year_strategy.json
phase5b_2025_2026_diagnostic.csv

manifest.json
audit_report.md
```

---

# 18. テスト要件

1. current race leakageなし
2. same-day future leakageなし
3. random splitなし
4. strategyごとのtrain windowが正しい
5. market model windowが記録される
6. residual model windowが記録される
7. probability_raw / calibrated分離
8. outer validationをcalibratorが見ない
9. in-fold calibration禁止
10. model selectionはraw
11. 2025/2026を選択に使わない
12. payout-zeroed stress ROI <= normal ROI
13. missing predictionをROI 0扱いしない
14. source artifact hash記録
15. resume時hash検証
16. tree count 300
17. feature allowlist固定
18. git自動操作なし

---

# 19. 最終判断の表現

Codexは以下のような断定を避ける。

禁止例:

```text
最も高精度
明確に優位
正式採用すべき
```

bootstrapやworst-year根拠がない場合は使わない。

推奨表現:

```text
点推定では改善
統計的優位性は未確認
実質差は小さい
採用保留
```

---

# 20. 最終目標

このPhase 5Bで決めるのは:

```text
履歴生成開始年
モデル学習期間戦略
市場baseline学習期間
time decayの有無
```

これを固定した後に:

```text
Phase 6:
EV threshold robustness

Phase 7:
relative time / course features / pedigree
```

へ進む。

現時点の正式BASEは維持するが、
2016年開始を最適と仮定しない。
