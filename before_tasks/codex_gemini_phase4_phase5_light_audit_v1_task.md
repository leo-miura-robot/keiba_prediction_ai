# Codex Light Audit Task
## Review Gemini / Antigravity Phase 4–5 Changes Before Phase 5B

## 0. 目的

Gemini / AntigravityがPhase 4～5で追加・変更したコードについて、
Phase 5Bの年度利用戦略へ進む前に軽量なコード監査を行う。

本タスクは再実験や再学習を目的としない。

目的:

1. Geminiが変更したファイルを特定する
2. 重大な評価バグが残っていないか確認する
3. Phase 5Bで再利用してよいコードと、再利用禁止コードを分類する
4. 修正が必要なら最小限の修正案だけを提示する

---

# 1. 絶対条件

- CatBoost fit/train禁止
- calibrator fit禁止
- DB接続禁止
- DB更新禁止
- Parquet変更禁止
- 既存成果物の上書き・削除禁止
- 長時間処理禁止
- 自動commit/push禁止
- git add/reset/clean禁止

許可:

- git差分確認
- コード静的レビュー
- CSV/JSON/Markdown成果物の読込
- 既存pytestの実行
- 軽量な整合性チェック
- 新規監査レポートの作成

---

# 2. 最初に実行

```bash
git status --short
git diff --stat
git diff
git log -10 --oneline
```

未追跡ファイルも含め、Gemini / Antigravityが追加した可能性のあるPhase 4～5関連ファイルを一覧化する。

確認対象候補:

```text
config/*phase4*
config/*phase5*
scripts/*phase4*
scripts/*phase5*
scripts/*history_extension*
scripts/*rate_smoothing*
scripts/*metric_consistency*
tests/*phase4*
tests/*phase5*
docs/*phase4*
docs/*phase5*
outputs/*phase4*
outputs/*phase5*
models/*phase4*
models/*phase5*
walkthrough.md
task.md
```

実在するファイルだけを対象にする。

---

# 3. 最優先レビュー項目

## 3.1 キャリブレーション

以下を確認する。

- in-fold予測でIsotonic Regressionをfitしていない
- outer validation年の正解をfitへ使っていない
- training期間内のtime-based OOFだけでfitしている
- `probability_raw`と`probability_calibrated`を混同していない
- 同じ列名に異なる意味を持たせていない
- 2025/2026のキャリブレーター由来が明示されている

重大な問題があれば、該当ファイル・関数・行番号を報告する。

## 3.2 ROI / EV

以下を確認する。

```text
EV = probability * odds
```

- probability列が明示されている
- odds列と単位が明示されている
- 100円払戻との単位が一致
- EV>=1フィルタがROIとStressROIで同一
- `row_removed_roi`で分母を減らしている
- `payout_zeroed_stress_roi`で分母を維持している
- `payout_zeroed_stress_roi <= original_roi`
- 予測なしをROI 0%としていない
- 年度合算ROIを年度ROIの単純平均で計算していない
- 合算ROIが`total payout / total stake`である

## 3.3 年度利用戦略

以下を確認する。

- BASE / WARMUP / FULLの学習期間がコード上で定義どおり
- 履歴生成開始年とモデル学習開始年を混同していない
- validation年を学習行へ含めていない
- 2025/2026をモデル選択へ使っていない
- market modelとresidual modelの学習期間が記録されている
- 2006～2015へのmarket_logit外挿がどこで行われるか明確
- FULLだけに不公平な前処理が入っていない

## 3.4 履歴特徴

以下を確認する。

- 現在レース結果を特徴生成前に更新していない
- 同日未来レースが混入しない
- ソートキーが安定している
- 2006年と2016年の境界で履歴をリセットしていない
- horse / jockey / trainer IDの扱いが期間を跨いで一貫
- 累積履歴値の合計を実出走数として誤使用していない

## 3.5 市場baseline

`market_logit`が以下のどちらかを特定する。

```text
A. オッズから直接計算
B. Logistic Regression等の学習モデル
```

Bの場合:

- fit期間
- target
- input columns
- validationへの適用方法
- 2006～2015への外挿
- candidate間の公平性

を確認する。

---

# 4. テストレビュー

既存pytestを確認し、可能なら軽量に実行する。

確認:

- テストが実装の同じ関数をそのまま呼んでいるだけではないか
- 不変条件を独立に検証しているか
- ダミーデータだけで本番列名の違いを見逃していないか
- StressROIの不変条件があるか
- calibration leakageを検出するテストがあるか
- missing predictionをN/Aにするテストがあるか
- year windowの境界テストがあるか

テストが通ってもロジックが保証されない場合は明記する。

---

# 5. 成果物との整合

コードと以下の成果物を照合する。

```text
walkthrough.md
audit_v2/
manifest.json
paired bootstrap JSON/CSV
ROI CSV
calibration provenance CSV
```

確認:

- walkthroughの数値がCSV/JSONと一致
- 報告文のモデル名が実ファイルと一致
- 1000回/5000回bootstrapの区別
- raw/calibratedの区別
- N/Aと0%の区別
- normal ROIとStressROIの母集団一致

---

# 6. 出力

新規レポートだけを作成する。

推奨:

```text
docs/codex_gemini_phase4_phase5_light_audit_v1.md
outputs/codex_gemini_phase4_phase5_light_audit_v1/
```

必須内容:

```text
reviewed_files.csv
issues.csv
reusable_components.csv
do_not_reuse_components.csv
test_results.txt
audit_summary.md
```

issues.csv列:

```text
severity
file
function
line
category
description
impact
recommended_action
```

severity:

```text
critical
high
medium
low
info
```

---

# 7. 最終判定

以下を分類する。

## A. そのまま再利用可能

例:

- DB schema audit
- read-only inventory
- deterministic data loading
- output formatting

## B. 条件付き再利用

例:

- ROI helper
- calibration helper
- bootstrap helper
- year-window builder

必要な条件を明記する。

## C. 再利用禁止

例:

- in-fold calibration
- 母集団が異なるStressROI
- missing predictionを0扱い
- validationを含むtrain window

---

# 8. 最終報告

日本語で以下を報告する。

1. 確認したファイル一覧
2. critical / high問題
3. medium / low問題
4. calibrationの安全性
5. ROI / StressROIの安全性
6. 年度利用戦略の安全性
7. 履歴生成の安全性
8. market_logit実装
9. テスト結果
10. 再利用可能なコード
11. 再利用禁止コード
12. Phase 5Bへ進めるか
13. 修正が必要な場合の最小修正案
14. git status --short
15. git diff --stat

コード修正や再学習は行わず、まず監査結果だけを報告する。
