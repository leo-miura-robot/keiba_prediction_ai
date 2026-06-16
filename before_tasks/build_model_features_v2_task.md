# Task: Build Model Feature Dataset V2

## 目的

`keiba_prediction_ai` リポジトリの既存V1を残したまま、モデル学習前の時系列特徴量生成処理をV2として修正する。

今回は **Phase 1: 時系列オッズなし** とする。

以下は実施しない。

- モデル学習
- CatBoost / LightGBM / Ranker
- Optuna
- 確率キャリブレーション
- バックテスト
- EV閾値最適化
- `TS_O1` など発走前オッズ時系列データの利用

## 参照ファイル

最初に以下を確認する。

- `scripts/build_model_features.py`
- `scripts/build_full_runner_dataset.py`
- `docs/model_feature_design.md`
- `docs/target_definition.md`
- `docs/time_leakage_validation.md`
- `outputs/training_eligibility_summary.csv`
- `outputs/label_mismatch_cases.csv`
- `outputs/time_leakage_validation_samples.csv`

入力:

- `outputs/base_runner_dataset/year=2016..2026/data.parquet`

既存のV1コード・出力・資料は削除または上書きしない。

---

## 修正対象

1. 同一レース内の別馬結果が履歴特徴量へ混入する可能性
2. 同日レース結果が後続レースへ混入する可能性
3. 未確定・異常結果が履歴統計へ追加される問題
4. 単勝・複勝・ランキングの学習対象条件が共通
5. 欠損IDを空文字などの共通キーとして集計する問題
6. `horse_past_starts` が直近履歴上限で頭打ちになる問題
7. 少頭数レースの複勝ルール
8. raw列と正規化済み市場列の重複
9. `--resume` が全年度を再計算する問題

---

## 推奨構成

```text
scripts/build_model_features_v2.py
src/features/history_builder.py
src/features/target_builder.py
src/features/feature_sets.py
tests/test_history_builder.py
tests/test_target_builder.py
config/feature_sets.yaml
```

巨大な単一スクリプトにまとめず、履歴生成、ターゲット、特徴量セットを分離する。

---

## 1. 予測時点

V2は前日終了時点の `pre_day` モデルとする。

同じ開催日の結果は、その日のどのレースの特徴量にも使わない。

処理順:

1. `race_date` 単位で当日全出走馬を取得
2. 前日以前の履歴だけで当日全行の特徴量を作成
3. 当日全行の特徴量作成を完了
4. 当日の有効な確定結果をまとめて履歴へ追加
5. 次の日へ進む

追加列:

- `feature_snapshot_mode = "pre_day"`
- `history_cutoff_date`

必須条件:

```text
historical_source_race_date < current_race_date
```

---

## 2. 同一レース内リーク防止

行ごとに特徴量作成直後に履歴へ追加しない。

同じ `race_id` の全馬の特徴量を作り終えてから、そのレース結果を履歴追加候補へ保持する。

当日分の履歴更新は、その日の全レースの特徴量作成後にまとめて行う。

対象となる履歴:

- 馬
- 騎手
- 調教師
- 馬×騎手
- 騎手×競馬場
- 騎手×距離帯
- 馬×競馬場
- 馬×芝ダート
- 馬×距離帯
- 馬×馬場状態

---

## 3. 履歴更新条件

履歴へ追加できる行は原則として以下を満たすものに限定する。

- `race_has_result=True`
- `IJyoCD == "0"`
- `KakuteiJyuni > 0`
- 有効な `KettoNum`
- 対象集計に必要なID・条件キーが有効

以下は出走数や負けとして履歴に追加しない。

- 未確定
- 取消
- 除外
- 競走中止
- 失格
- 正常な確定着順がない行

履歴更新から除外した理由を集計する。

---

## 4. 欠損IDの扱い

次の無効値を共通キーとして集計しない。

- NULL
- 空文字
- `"0"`
- 不正なID

対象:

- `KettoNum`
- `KisyuCode`
- `ChokyosiCode`
- 条件別・組み合わせ別キー

無効な場合、その履歴特徴量はNULLまたは未経験扱いにする。

---

## 5. 馬の通算出走数

直近1・3・5走計算用の履歴は最大20件程度でもよい。

ただし `horse_past_starts` は独立した通算カウンターで管理し、20で頭打ちにしない。

```text
recent_history:
  直近1・3・5走特徴量用

horse_total_stats:
  通算出走数・通算成績用
```

---

## 6. 学習対象フラグ

### 単勝

```text
eligible_for_win_training =
  race_has_result
  AND race_has_win_payout
  AND IJyoCD == "0"
  AND 有効なUmaban
  AND 有効なKettoNum
  AND target_win_paidが確定
```

### 複勝

```text
eligible_for_place_training =
  race_has_result
  AND race_has_place_payout
  AND IJyoCD == "0"
  AND 有効なUmaban
  AND 有効なKettoNum
  AND target_place_paidが確定
```

### ランキング

```text
eligible_for_ranking_training =
  race_has_result
  AND KakuteiJyuni > 0
  AND IJyoCD == "0"
  AND 有効なUmaban
  AND 有効なKettoNum
```

追加列:

- `win_training_exclusion_reason`
- `place_training_exclusion_reason`
- `ranking_training_exclusion_reason`

3種類の対象行数は同じでなくてよい。

---

## 7. 複勝ルール

診断用のルール:

```text
4頭以下:
  place_bet_available_by_rule = False
  place_rank_limit = 0

5〜7頭:
  place_bet_available_by_rule = True
  place_rank_limit = 2

8頭以上:
  place_bet_available_by_rule = True
  place_rank_limit = 3
```

追加・修正列:

- `place_bet_available_by_rule`
- `place_rank_limit`
- `target_place_by_rule`

正式ターゲットは引き続き以下とする。

```text
target_place_paid = is_place_paid
```

---

## 8. 履歴率の名称

曖昧な `place_rate` を避け、可能なら以下を区別する。

- `top3_rate`: 確定着順3着以内率
- `place_paid_rate`: 実際の複勝払戻対象率

単勝も必要に応じて、着順1着率と単勝払戻対象率を分ける。

既存列を残す場合は意味を資料に明記する。

---

## 9. 特徴量セット

`config/feature_sets.yaml` を作成する。

```yaml
market_free:
  numeric: []
  categorical: []

market_aware:
  numeric: []
  categorical: []
```

### market_free

オッズ、人気、票数を含めない。

### market_aware

`market_free` に以下を追加する。

- `tan_odds`
- `tan_ninki`
- `fuku_odds_low`
- `fuku_odds_high`
- `fuku_ninki`
- `TanVote`
- `FukuVote`
- `win_odds_available`
- `place_odds_available`
- `win_votes_available`
- `place_votes_available`

`TanOdds` と `tan_odds` のような同義列を同時に入れない。原則として正規化済みの小文字列を正式採用する。

特徴量は除外方式ではなく、明示的な許可リストで管理する。

以下は絶対に含めない。

- 着順
- タイム
- 上がり
- 通過順位
- 払戻
- ターゲット
- レース確定判定
- 学習対象フラグ
- 除外理由
- データ分割列
- 将来情報由来の列

---

## 10. 市場取得フラグ

以下を分ける。

- `win_odds_available`
- `place_odds_available`
- `win_votes_available`
- `place_votes_available`

NULLでないだけでなく、0や不正なプレースホルダーを除外して判定する。

---

## 11. 真のresume

現在のように全年度を再計算して出力だけスキップする方式をやめる。

年末時点の履歴状態を保存する。

```text
outputs/model_feature_dataset_v2_checkpoint/
  history_state_after_2016.pkl
  history_state_after_2017.pkl
  ...
```

チェックポイントには以下を保存する。

- 完了年
- 履歴状態ファイル
- 出力行数
- 入力ファイル情報
- 設定ハッシュまたは設定バージョン
- スクリプトバージョン
- 完了日時

`--resume` 時は最後に正常完了した年の履歴を読み込み、次年から処理する。

入力・設定・コードが変わった場合は古いチェックポイントを無条件に使わない。

---

## 12. 出力

```text
outputs/model_feature_dataset_v2/year=YYYY/data.parquet
outputs/model_feature_dataset_v2_sample.csv
outputs/training_eligibility_summary_v2.csv
outputs/label_mismatch_cases_v2.csv
outputs/history_leakage_validation_v2.csv
outputs/feature_inventory_v2.csv
outputs/model_feature_v1_v2_comparison.csv

docs/model_feature_design_v2.md
docs/target_definition_v2.md
docs/time_leakage_validation_v2.md
docs/feature_set_design.md

logs/build_model_features_v2.log
```

既存V1は上書きしない。

---

## 13. 必須テスト

小さな合成データを使うpytestを作成する。

1. 同一レース内の別馬結果を参照しない
2. 同日1Rの結果を同日12Rが参照しない
3. 翌日は前日の結果を参照できる
4. 未確定・異常結果を履歴へ追加しない
5. 欠損騎手・調教師IDを共有統計にしない
6. `horse_past_starts` が20を超える
7. 4・5・7・8頭の複勝ルール
8. 単勝・複勝・ランキングの学習対象条件が別々に動く
9. feature setにリーク列が含まれない
10. raw市場列と正規化市場列が重複しない
11. resumeで完了済み年度を再計算しない

---

## 14. 実行手順

### 構文確認とテスト

```bash
python -m py_compile scripts/build_model_features_v2.py
python -m pytest -q
```

### 2016〜2017年の試験実行

```bash
python scripts/build_model_features_v2.py --years 2016-2017 --force
```

確認項目:

- `entry_id` 重複0
- 同一race_id参照0
- 同日参照0
- 未来日参照0
- 券種別の学習対象件数が個別に出る
- `horse_past_starts > 20` の行がある
- `feature_sets.yaml` にリーク列がない
- 年末履歴チェックポイントが作成される

### 全期間

問題がなければ以下を実行する。

```bash
python scripts/build_model_features_v2.py --resume
```

実行中はログを定期確認する。

エラー時は原因を修正し、正常なチェックポイントから再開する。

---

## 今回実施しないこと

- CatBoost学習
- LightGBM学習
- ランキング学習
- Optuna
- 確率キャリブレーション
- バックテスト
- EV閾値最適化
- 発走前時系列オッズの利用

---

## 最終報告

1. `git diff` と変更ファイル
2. V1で確認した問題
3. 修正内容
4. pytest結果
5. V1/V2の行数・列数比較
6. 券種別学習対象行数
7. 同一レース参照件数
8. 同日参照件数
9. 未来日参照件数
10. 異常・未確定結果の履歴除外件数
11. 無効IDの履歴除外件数
12. `horse_past_starts` 最大値
13. market_freeの数値・カテゴリ特徴量数
14. market_awareの数値・カテゴリ特徴量数
15. V2処理時間
16. 真のresumeが動作したか
17. モデル学習へ進める状態か
18. 未解決事項

V2データセットの正当性確認までで停止する。
