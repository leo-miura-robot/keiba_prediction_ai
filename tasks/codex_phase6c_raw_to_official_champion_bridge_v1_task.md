# Codex Task: Phase 6C Raw Pre-Race to Official Champion End-to-End Bridge v1

## 0. 背景

Phase 6C v2では、以下の正式Champion予測経路が実装・監査済み。

```text
ROLLING_10Y CatBoost
→ probability_raw
→ official Platt artifact
→ probability_calibrated
→ EV
→ 4 tier
→ Phase 6C SQLiteへimmutable登録
```

正式artifact:

```text
CatBoost:
models/place_market_offset_champion_challenger_phase5c_v1/
ROLLING_10Y/validation_2026/model.cbm

CatBoost SHA256:
4c6f1b9e236391bd84b9d75a14f7ea8ea3fe44761737bb645b8f21d74ed38256

Official Platt:
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json

Platt SHA256:
ffee1efc19c38f3a76a1efa93488153429e8463bc65f693b331617274e208e98
```

現在の残課題は、jrvltsql側から出力されるpre-race raw CSVを、
モデルが必要とする正式79特徴へ変換し、
official Champion予測・Phase 6C登録までを1コマンドで実行できるか確認・実装すること。

本タスクでは、raw pre-race CSVからmodel-ready feature CSVまでのbridgeを完成させる。

---

# 1. 目的

以下を一気通貫で実現する。

```text
jrvltsql pre-race raw CSV
→ raw schema監査
→ 長期履歴DBをread-only参照
→ pre-day履歴特徴量生成
→ 正式79特徴へ整形
→ feature allowlist / order / dtype監査
→ official ROLLING_10Y CatBoost推論
→ official Platt適用
→ EV・4 tier生成
→ Phase 6C SQLite登録
→ run manifest・監査レポート生成
```

最終的に、ユーザーが実行するコマンドは1本にする。

---

# 2. 絶対条件

禁止:

```text
CatBoost再学習
Platt再fit
Isotonic再fit
OOFからparameter再生成
Champion変更
EV閾値変更
特徴量定義変更
学習済みallowlist変更
未来結果の利用
当日結果の履歴混入
raw probabilityへの黙ったfallback
実購入
commit/push
```

許可:

```text
既存履歴feature builderの再利用
既存長期DBのread-only参照
既存CatBoost artifactのread-only load
official Platt artifactのread-only load
pre-race推論
EV計算
paper trading登録
py_compile
pytest
fixture smoke
artifact audit
```

---

# 3. 最初に行う監査

既存の以下を調査する。

```text
scripts/prepare_place_forward_predictions_phase6c_v2.py
scripts/run_forward_predict_official_champion_phase6c_v2.ps1
src/features/
config/feature_sets_v2_1_2.yaml
feature_allowlist_c1r0.json
Phase 5B / 5Cの推論処理
history builder
```

最優先確認:

```text
prepare_place_forward_predictions_phase6c_v2.py が
1. raw pre-race CSVを受け取るのか
2. 完成済み79特徴CSVを要求するのか
```

判定:

```text
RAW_INPUT_SUPPORTED
MODEL_READY_INPUT_ONLY
PARTIAL_SUPPORT
BLOCKED_UNKNOWN
```

監査結果を保存:

```text
outputs/phase6c_raw_to_model_ready_bridge_v1/
input_contract_audit.json
```

---

# 4. 入力仕様

想定入力:

```text
inputs/forward/pre_race_YYYYMMDD.csv
```

jrvltsql由来の最低限raw列:

```text
race_id
entry_id
race_date
Year
MonthDay
JyoCD
Kaiji
Nichiji
RaceNum
Umaban
Wakuban
KettoNum

YoubiCD
GradeCD
SyubetuCD
JyokenCD1
JyokenCD2
JyokenCD3
JyokenCD4
JyokenCD5
TrackCD
CourseKubunCD
Kyori
TenkoCD
SibaBabaCD
DirtBabaCD
TorokuTosu
SyussoTosu

Barei
SexCD
Futan
BaTaijyu
ZogenSa
ZogenFugo
KisyuCode
ChokyosiCode

tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki

odds_observed_at
odds_snapshot_type
source_updated_at
retrospective_only
```

結果列は含めてはいけない。

禁止列例:

```text
KakuteiJyuni
target_place_paid
fuku_pay
払戻
確定結果
HaronTimeL3の当日確定値
Timeの当日確定値
```

禁止列が1つでも含まれていたらfail-closed。

---

# 5. 長期履歴DB

既存履歴DB:

```text
D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

read-onlyで使用する。

必要に応じて当日より前の短期追加DBを結合する。

原則:

```text
予測日Dの特徴量は D-1 までの履歴のみ使用
```

例:

```text
2026-06-20予測:
2026-06-19までの履歴だけ使用
```

禁止:

```text
予測日自身の結果
予測日より未来の結果
settlement済み当日情報
```

各runで以下を保存:

```text
history_source_paths
history_start_date
history_end_date
history_rows
history_races
history_cutoff_date
```

---

# 6. 履歴特徴量生成

既存history builderを再利用し、学習時と同じ定義で生成する。

最低限確認対象:

```text
horse_past_starts
horse_days_since_last

horse_last1_avg_finish
horse_last3_avg_finish
horse_last5_avg_finish

horse_last3_win_rate
horse_last5_win_rate
horse_last3_ren_rate
horse_last5_ren_rate
horse_last3_top3_rate
horse_last5_top3_rate
horse_last3_place_paid_rate
horse_last5_place_paid_rate

horse_jyo_past_starts
horse_surface_past_starts
horse_dist_band_past_starts
horse_baba_past_starts

jockey_past_starts
jockey_win_rate
jockey_top3_rate

trainer_past_starts
trainer_win_rate
trainer_top3_rate

horse_jockey_past_starts
horse_jockey_win_rate
horse_jockey_top3_rate
```

ゼロ率・欠損率を出す。

特に:

```text
horse_past_starts == 0
jockey_past_starts == 0
trainer_past_starts == 0
```

をrun manifestへ記録する。

---

# 7. Model-ready 79特徴

正式allowlistをartifactから読む。

候補:

```text
outputs/place_market_offset_catboost_c1r0_v1/
feature_allowlist_c1r0.json
```

Phase 5Cの実運用modelに紐づく正式allowlistが別にある場合は、
model provenanceから正しいものを特定する。

必須監査:

```text
79列すべて存在
missingなし
extraなし
duplicateなし
列順完全一致
dtype互換
all-nullなし
禁止列なし
```

CatBoost入力から除外されるもの:

```text
Year
p_market
market_logit
odds
popularity
result
payout
ID
KisyuCode
ChokyosiCode
```

ただしKisyuCode / ChokyosiCodeは履歴集計用raw列としては使用可。

出力:

```text
outputs/phase6c_raw_to_model_ready_bridge_v1/
model_ready/model_ready_features_YYYYMMDD.parquet
```

必要ならCSVも併記。

---

# 8. Market baseline生成

学習時と同じmarket model・scaler・変換を使用する。

入力候補:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
SyussoTosu
place_rank_limit
```

派生列候補:

```text
market_rank
tan_rank
fuku_odds_width
log_tan_odds
log_fuku_low
log_fuku_high
fuku_low_inverse
fuku_mid_inverse
fuku_low_to_race_min
fuku_low_to_race_mean
```

必須:

```text
学習時と同じclip
学習時と同じ欠損処理
学習時と同じscaler
学習時と同じLogisticRegression artifact
```

推測実装は禁止。

market artifactが見つからない場合はBLOCKED。

---

# 9. Official Champion推論

CatBoost:

```text
models/place_market_offset_champion_challenger_phase5c_v1/
ROLLING_10Y/validation_2026/model.cbm
```

SHA256:

```text
4c6f1b9e236391bd84b9d75a14f7ea8ea3fe44761737bb645b8f21d74ed38256
```

Raw probability:

```text
final_logit = market_logit + residual_raw
probability_raw = sigmoid(final_logit)
```

Official Platt:

```text
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json
```

SHA256:

```text
ffee1efc19c38f3a76a1efa93488153429e8463bc65f693b331617274e208e98
```

既存loader:

```text
src/calibration/official_calibrator_loader.py
```

を必ず使う。

---

# 10. EV・Tier

EV:

```text
expected_value = probability_calibrated * fuku_odds_low
```

Tier:

```text
CORE:      EV >= 1.00
MARGIN:    EV >= 1.05
HIGH:      EV >= 1.10
VERY_HIGH: EV >= 1.15
```

包含:

```text
VERY_HIGH ⊆ HIGH ⊆ MARGIN ⊆ CORE
```

---

# 11. 一気通貫スクリプト

作成候補:

```text
scripts/run_phase6c_raw_to_official_champion.py
scripts/run_phase6c_raw_to_official_champion.ps1
```

PowerShell例:

```powershell
.\scripts\run_phase6c_raw_to_official_champion.ps1 `
  -RaceDate 2026-06-20 `
  -RawPreRaceCsv "inputs\forward\pre_race_20260620.csv" `
  -OutputRoot "outputs\place_market_offset_forward_paper_phase6c_v2"
```

内部処理:

```text
raw input audit
→ history cutoff audit
→ feature generation
→ 79列schema audit
→ market baseline
→ CatBoost raw inference
→ official Platt
→ EV / tier
→ Phase 6C immutable registration
→ artifact audit
→ run manifest
```

---

# 12. 出力

```text
outputs/phase6c_raw_to_model_ready_bridge_v1/
```

必須成果物:

```text
input_contract_audit.json
raw_input_audit.csv
history_source_audit.json
history_feature_completeness.csv
feature_schema_parity.csv
market_input_audit.csv
model_ready/model_ready_features_YYYYMMDD.parquet
predictions/pre_race_predictions_YYYYMMDD.csv
run_manifest_YYYYMMDD_<id>.json
artifact_audit.json
pipeline_report.md
```

Phase 6C登録先:

```text
outputs/place_market_offset_forward_paper_phase6c_v2/
```

---

# 13. Run manifest

最低限:

```text
prediction_run_id
race_date
created_at
raw_input_path
raw_input_sha256
history_source_paths
history_cutoff_date
history_rows
history_races
feature_allowlist_path
feature_allowlist_sha256
feature_count
model_artifact_path
model_artifact_sha256
market_artifact_paths
market_artifact_sha256
calibrator_artifact_path
calibrator_artifact_sha256
odds_snapshot_type
retrospective_only
row_count
horse_history_zero_rate
jockey_history_zero_rate
trainer_history_zero_rate
probability_raw_min
probability_raw_max
probability_calibrated_min
probability_calibrated_max
core_count
margin_count
high_count
very_high_count
phase6c_registration_performed
fixture
```

---

# 14. Fail-closed条件

以下では停止する。

```text
raw input不存在
必須raw列不足
結果列混入
race_date不一致
重複entry
履歴DB不存在
履歴cutoff違反
future leakage検出
allowlist不存在
79列不一致
列順不一致
dtype重大不一致
all-null feature
market artifact不存在
market変換不一致
model artifact不存在
model hash不一致
Platt artifact不存在
Platt hash不一致
fit/refit検出
NaN / inf
probability範囲外
odds欠損
odds <= 0
odds snapshot不明
timestamp違反
duplicate prediction
raw fallback検出
```

---

# 15. Fixture smoke

実未来入力がない場合、
fixtureでraw CSVから最後まで通す。

fixture条件:

```text
fixture = true
report対象外
実績集計対象外
```

確認:

```text
raw CSV読込
履歴特徴量生成
79列一致
market生成
CatBoost推論
official Platt適用
EV / tier生成
SQLite登録
duplicate拒否
immutable確認
fixture report除外
```

---

# 16. 必須テスト

1. raw必須列検証
2. 結果列混入拒否
3. race_date不一致拒否
4. 重複entry拒否
5. history cutoff D-1
6. 同日結果利用なし
7. future leakageなし
8. 79特徴一致
9. feature order一致
10. dtype重大不一致拒否
11. all-null feature拒否
12. market artifact read-only
13. model artifact hash一致
14. model hash不一致拒否
15. Platt artifact hash一致
16. Platt hash不一致拒否
17. fit呼び出しなし
18. fit_transform呼び出しなし
19. CatBoost再学習なし
20. raw fallbackなし
21. probability_raw範囲
22. probability_calibrated範囲
23. EV再計算一致
24. tier包含関係
25. duplicate予測拒否
26. prediction immutable
27. timestamp violation拒否
28. FINAL_ODDSならretrospective_only=true
29. PRE_RACE_SNAPSHOTならretrospective_only=false
30. fixture report除外
31. 15Y calibrated登録なし
32. commit/pushなし

---

# 17. 変更ファイル候補

```text
scripts/run_phase6c_raw_to_official_champion.py
scripts/run_phase6c_raw_to_official_champion.ps1
scripts/prepare_place_forward_predictions_phase6c_v2.py
src/features/history_builder_v2_1.py
src/calibration/official_calibrator_loader.py
tests/test_phase6c_raw_to_official_champion_bridge.py
docs/phase6c_raw_to_official_champion_bridge_v1.md
config/place_market_offset_forward_paper_phase6c_v2.yaml
```

既存コードを再利用し、必要最小限の変更にする。

---

# 18. 最終判定

候補:

```text
PHASE6C_RAW_TO_OFFICIAL_CHAMPION_BRIDGE_PASSED
RAW_INPUT_ALREADY_SUPPORTED
BLOCKED_RAW_SCHEMA
BLOCKED_HISTORY_SOURCE
BLOCKED_HISTORY_LEAKAGE
BLOCKED_FEATURE_SCHEMA
BLOCKED_MARKET_ARTIFACT
BLOCKED_MODEL_ARTIFACT
BLOCKED_CALIBRATOR_ARTIFACT
BLOCKED_ODDS_INPUT
BLOCKED_REFIT_DETECTED
MULTIPLE_BLOCKERS
```

boolean:

```text
raw_pre_race_supported
history_generation_performed
feature_79_parity_passed
official_model_loaded
official_platt_loaded
refit_performed
raw_fallback_used
phase6c_registration_performed
fixture_only
ready_for_real_forward_prediction
```

---

# 19. Codex側実行範囲

実施:

```text
既存入力契約監査
raw-to-feature bridge実装
履歴cutoff監査
79列parity監査
market artifact接続
official Champion推論
Phase 6C登録
py_compile
pytest
fixture smoke
artifact audit
runbook作成
```

実未来pre-race入力がなければ、
実レース登録は行わずfixture smokeまで。

---

# 20. 最終報告

簡潔に以下を報告する。

1. 既存prepare scriptの入力契約
2. raw input対応の有無
3. 使用履歴DB
4. history cutoff
5. 履歴特徴量生成の有無
6. 79特徴parity
7. market artifact path / hash
8. model artifact path / hash
9. Platt artifact path / hash
10. probability_raw生成結果
11. probability_calibrated生成結果
12. EV定義
13. tier件数
14. Phase 6C登録結果
15. duplicate / immutable監査
16. fixture smoke
17. 実未来予測の実施有無
18. ready_for_real_forward_prediction
19. final status
20. py_compile
21. pytest
22. artifact checks / failed
23. 作成・変更ファイル
24. git status --short

commit/pushは行わない。
