# Task: V2.1.2入力によるCatBoost 6モデル再学習・旧モデル比較

## 目的

O1修正版DBから再生成したV2.1.2特徴量を使用し、次の6モデルをGPUで新規学習する。

```text
win   × market_free
win   × market_history
win   × market_aware
place × market_free
place × market_history
place × market_aware
```

今回行うこと:

1. V2.1.2入力で6モデルを新規学習
2. validation / test / latest_holdoutを評価
3. 旧CatBoost V1.0.2と公平に比較
4. 単勝は完全レースで市場確率と比較
5. 複勝はオッズ帯別に診断
6. キャリブレーションは診断のみ実施
7. 次のキャリブレーション・EV・ROI段階へ進めるか判定

今回行わないこと:

```text
キャリブレーション適用
EV計算
ROIバックテスト
買い目生成
資金配分
Ability / ANA / Ranker
大規模Optuna
```

---

## 1. 入力データ

```text
outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet
```

期待値:

```text
rows: 505,881
races: 36,269
columns: 160
market_free: 83 columns
market_history: 85 columns
market_aware: 96 columns
```

split:

```text
train: 2016-2023
validation: 2024
test: 2025
latest_holdout: 2026
```

split hash:

```text
f14d81dc372c700c71684998c17809a05fcd1552a3f925e261b3a99fe9766424
```

O1品質:

```text
TanOdds coverage: 501,241 / 505,881 = 99.0828%
FukuOddsLow coverage: 501,241 / 505,881 = 99.0828%
FukuOddsHigh coverage: 501,241 / 505,881 = 99.0828%
SE.Odds / O1.TanOdds exact match: 501,241 / 501,241 = 100%
```

時系列リーク監査:

```text
same race: 0
same day: 0
future: 0
cutoff violation: 0
```

---

## 2. 旧比較対象

```text
outputs/model_training/catboost_baseline_v1_0_2/
models/catboost_baseline_v1_0_2/
```

旧モデル・旧予測・旧分析を削除または上書きしない。

V1.0.2はO1大量欠損時のV2.1.1入力で学習されているため、V2.1.2モデルは6本とも新規学習を原則とする。旧重みの再利用は禁止する。

---

## 3. 新バージョン

推奨名:

```text
catboost_baseline_v2_1_2_v1
```

推奨ファイル:

```text
config/catboost_baseline_v2_1_2_v1.yaml
scripts/train_catboost_baseline_v2_1_2_v1.py
scripts/analyze_catboost_baseline_v2_1_2_v1.py

models/catboost_baseline_v2_1_2_v1/
outputs/model_training/catboost_baseline_v2_1_2_v1/

docs/catboost_baseline_v2_1_2_v1_design.md
docs/catboost_baseline_v2_1_2_v1_results.md
```

既存共通moduleは再利用してよいが、V1.0.2の挙動を壊さない。

---

## 4. 最初に確認

```bash
git status
git diff
git rev-parse HEAD
python -m pytest -q
```

確認対象:

```text
config/catboost_baseline_v1_0_2.yaml
scripts/train_catboost_baseline_v1_0_2.py
scripts/analyze_catboost_baseline_v1_0_2.py
src/models/catboost_*.py
config/model_features_v2_1_2.yaml
config/feature_sets_v2_1_2.yaml
outputs/model_feature_dataset_v2_1_2/
```

自動commit・自動pushは禁止。

---

## 5. 学習条件

- CatBoost binary classification
- RTX 5070 Tiを使用
- GPU smoke test必須
- CPU fallbackは禁止。GPU利用不可なら停止
- seed固定
- validationによるearly stopping
- best iteration保存
- 旧V1.0.2と同じ基本ハイパーパラメータを初期条件とする
- 今回はOptunaを行わない
- target定義・eligible条件を変更しない
- splitはYAMLを正本とし、hard-codeしない

market-awareは**今回レースの確定オッズを事前に利用できたと仮定する理想条件モデル**であり、発走前実運用モデルではない。config、manifest、docs、結果に必ず明記する。

---

## 6. 学習対象行

### market_free / market_history

既存eligible条件を維持する。

### win × market_aware

最低限確認:

```text
eligible_for_win_training = True
actual_win in {0,1}
tan_odds valid and > 0
tan_ninki valid
```

### place × market_aware

最低限確認:

```text
eligible_for_place_training = True
actual_place in {0,1}
fuku_odds_low valid and > 0
fuku_odds_high valid and > 0
```

既存設計がCatBoost欠損処理を利用している場合、勝手に行削除せず、次の件数を両方記録する。

```text
既存eligible全行数
市場列完全行数
```

主な市場比較・EV前段評価は完全レースへ限定する。

---

## 7. 完全レース定義

### 単勝完全レース

全対象出走馬について:

```text
entry_id unique
tan_odds valid and > 0
3つのwinモデル予測あり
actual_win valid
runner count一致
勝馬あり
```

### 複勝完全レース

全対象出走馬について:

```text
entry_id unique
fuku_odds_low valid and > 0
fuku_odds_high valid and > 0
3つのplaceモデル予測あり
actual_place valid
runner count一致
複勝的中馬数がtarget定義と整合
```

出力:

```text
complete_race_summary_win.csv
complete_race_summary_place.csv
excluded_races_win.csv
excluded_races_place.csv
```

---

## 8. 予測出力

6モデルそれぞれについて全splitの予測を保存する。

最低限の列:

```text
entry_id
race_id
race_date
year
data_split
target
feature_set
actual
prediction
model_version
```

market-awareでは追加:

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
```

出力例:

```text
predictions/win_market_free.parquet
predictions/win_market_history.parquet
predictions/win_market_aware.parquet
predictions/place_market_free.parquet
predictions/place_market_history.parquet
predictions/place_market_aware.parquet
```

---

## 9. 基本評価

各 target × feature_set × split で算出:

```text
logloss
Brier score
ROC-AUC
PR-AUC
race-level top1 hit rate
mean prediction
actual positive rate
prediction min/max
best iteration
training time
prediction rows
race count
```

主指標:

```text
logloss
Brier score
race-level top1 hit rate
calibration
```

AUCだけで良否を判断しない。

---

## 10. 旧V1.0.2との公平比較

validation / test / latest_holdoutで比較する。

```text
old logloss / new logloss / delta
old Brier / new Brier / delta
old AUC / new AUC / delta
```

共通entry_idで予測差も保存する。

```text
entry_id
old prediction
new prediction
absolute difference
actual
split
target
feature_set
```

期待:

- market_free: 大幅に変わらない可能性が高い
- market_history: 過去オッズcoverage改善で変化する可能性あり
- market_aware: 大きく変化する可能性あり

大きな差が出た場合は、feature coverage・対象行・欠損率の変化から説明する。

出力:

```text
old_new_model_metric_comparison.csv
old_new_prediction_comparison.csv
old_new_prediction_diff_summary.csv
```

---

## 11. 単勝市場確率との比較

単勝完全レースだけを使う。

```text
raw_market_probability = 1 / tan_odds
normalized_market_probability
  = raw_market_probability / race内raw_market_probability合計
```

同一レース・同一entry集合で比較:

```text
market
market_free
market_history
market_aware
```

算出:

```text
logloss
Brier score
race count
runner rows
positive count
```

不完全レースを混ぜない。

---

## 12. 複勝市場診断

複勝は上下限オッズがあるため、単勝と同じ方法で市場確率へ正規化しない。

今回は次を出す。

```text
fuku_odds_low band
fuku_odds_high band
prediction mean
actual place rate
count
split
feature_set
```

将来のEVでは安全側として原則:

```text
place probability × fuku_odds_low
```

を使う予定だが、今回はEV計算をしない。

---

## 13. キャリブレーション診断

適用はせず診断だけ行う。

binning:

```text
fixed-width
quantile
```

quantileは同値予測を分断しない。

```python
pd.qcut(..., duplicates="drop")
```

各bin:

```text
count
mean prediction
actual rate
absolute calibration error
split
target
feature_set
bin method
```

追加指標:

```text
ECE
MCE
calibration slope
calibration intercept
```

出力:

```text
calibration_bins.csv
calibration_summary.csv
```

---

## 14. 予測分布

各 target × feature_set × split と年別で:

```text
mean
std
p01
p05
p25
p50
p75
p95
p99
min
max
```

market-awareがO1改善後に極端化していないか確認する。

出力:

```text
prediction_distribution_summary.csv
prediction_distribution_by_year.csv
```

---

## 15. 特徴量重要度

各モデルで既存baselineと同じ重要度方式を保存する。

最低限:

```text
PredictionValuesChange
```

market-awareで次の順位・重要度を確認する。

```text
tan_odds
tan_ninki
fuku_odds_low
fuku_odds_high
fuku_ninki
```

SHAPは既存実装が安全に動く場合のみ、固定seedのサンプルで生成する。

---

## 16. fingerprint / resume

fingerprintへ含める:

```text
V2.1.2 input fingerprint
feature config hash
feature set hash
target definition hash
split hash
CatBoost params
seed
task type
feature columns
categorical columns
code hash
```

strict resumeは全fingerprint一致・成果物存在・metadata一致の場合のみskipする。

V1.0.2モデルを再利用しない。

---

## 17. 原子的出力・冪等性

CSV / JSON / Parquetは一時ファイルへ書き込み、flush・fsync後に`os.replace`する。

append/upsertで古い行を残さない。

分析を2回実行し、生成CSVの内容hashが一致することを確認する。

---

## 18. テスト

最低限追加・確認:

```text
V2.1.2 input loading
feature set columns
target eligibility
split YAML
model fingerprint
strict resume
complete win race
complete place race
market probability normalization
same-sample comparison
quantile ties
calibration bins
atomic output
old/new model comparison
market-aware ideal-condition metadata
```

全テスト:

```bash
python -m pytest -q
```

---

## 19. GPU確認

学習前にsmoke testを実行する。

記録:

```text
CatBoost version
CUDA device
GPU name
smoke result
CPU fallback有無
```

GPUが使えない場合は停止し、CPUで本学習へ進まない。

---

## 20. Phase 1目標

config・manifest・docsへ記録する。

```text
単勝ROI >= 90%
複勝ROI >= 90%
```

今回はROIを計算しない。

次段階で必須の過学習防止観点も記録する。

```text
十分な購入数
年別・月別安定性
競馬場別安定性
人気帯・オッズ帯安定性
最大払戻除外後ROI
上位払戻除外後ROI
drawdown
confidence interval
testで閾値調整しない
```

---

## 21. 禁止事項

- 旧モデル・旧予測・旧分析の上書き
- V2.1.1 / V2.1.2特徴量の変更
- target定義変更
- split変更
- Ability / ANA / Ranker
- 大規模Optuna
- calibration適用
- EV計算
- ROI計算
- 買い目生成
- 自動commit
- 自動push

---

## 22. 完了条件

- V2.1.2入力を使用
- 6モデルをGPUで新規学習
- validation/test/latest予測保存
- 基本指標算出
- 旧V1.0.2との公平比較
- 単勝完全レースで市場比較
- 複勝オッズ帯診断
- calibration診断
- 予測分布診断
- 特徴量重要度保存
- strict resume確認
- 分析出力の冪等性確認
- 旧モデル・旧出力未変更
- ROI/EV未実行
- 次段階へ進めるか判定

---

## 23. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. 使用V2.1.2 input path
5. input fingerprint
6. feature set列数
7. split件数
8. split hash
9. GPU smoke結果
10. CatBoost version / GPU名
11. 6モデルの学習成否
12. 各モデルのbest iteration
13. 各モデルの学習時間
14. validation指標
15. test指標
16. latest_holdout指標
17. win top1 hit rate
18. place top1 hit rate
19. 旧V1.0.2とのmetric差
20. 旧新prediction差
21. market_freeの変化
22. market_historyの変化
23. market_awareの変化
24. 単勝完全レース数
25. 単勝市場とのlogloss/Brier比較
26. 複勝完全レース数
27. 複勝オッズ帯別actual rate
28. calibration ECE/MCE
29. calibration slope/intercept
30. prediction distribution
31. feature importance上位
32. market特徴量の重要度
33. strict resume結果
34. 分析2回のhash一致
35. pytest結果
36. 旧モデル未変更確認
37. V2.1.1/V2.1.2未変更確認
38. ROI/EV未実行確認
39. market-aware理想条件の明記
40. キャリブレーションへ進めるか
41. ROI/EV前に残る課題
42. 未解決事項
43. 次の推奨手順

完了後は学習・予測・評価・比較結果まで報告し、キャリブレーション適用、EV、ROI、買い目生成には進まず停止する。
