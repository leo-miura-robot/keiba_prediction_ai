# Task: Build Model Feature Dataset V2.1

## 目的

`keiba_prediction_ai` のV2実装を残したまま、モデル学習前の最終調整としてV2.1を作成する。

今回は以下だけを行う。

- 監査・リーク検証の強化
- `history_cutoff_date` の修正
- `--resume` の安全性向上
- feature set の再整理
- テスト・資料・READMEの更新
- V2.1データセットの再生成と検証

以下は行わない。

- CatBoost / LightGBM / Ranker の学習
- Optuna
- 確率キャリブレーション
- バックテスト
- EV閾値最適化
- 発走前時系列オッズの利用

---

## 参照ファイル

最初に現在の `main` を確認する。

- `scripts/build_model_features_v2.py`
- `src/features/history_builder.py`
- `src/features/target_builder.py`
- `src/features/feature_sets.py`
- `config/feature_sets.yaml`
- `tests/test_history_builder.py`
- `tests/test_target_builder.py`
- `tests/test_feature_sets.py`
- `docs/model_feature_design_v2.md`
- `docs/target_definition_v2.md`
- `docs/time_leakage_validation_v2.md`
- `docs/feature_set_design.md`
- `outputs/history_leakage_validation_v2.csv`
- `outputs/model_feature_v1_v2_comparison.csv`
- `outputs/training_eligibility_summary_v2.csv`

V1とV2は削除・上書きしない。

---

# 1. history_cutoff_date の修正

現在レース日そのものが入っている場合は修正する。

V2.1では、`history_cutoff_date` を「その行の特徴量生成時に利用可能だった履歴の最終日」とする。

原則:

```text
history_cutoff_date < race_date
```

定義:

- 過去履歴がある場合: 履歴ストア内の最大 `race_date`
- 過去履歴がない場合: NULL
- `feature_snapshot_mode`: `"pre_day"`

同日の結果を履歴として利用しない。

---

# 2. リーク検証を固定値ではなく実測にする

`same_race_reference=False` の固定出力だけで0件扱いしない。

履歴ストアごとに、最後に更新元となった以下を保持できるようにする。

- `last_source_race_id`
- `last_source_race_date`

確認対象:

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

各特徴量生成時に次を実測する。

```text
last_source_race_id != current_race_id
last_source_race_date < current_race_date
```

少なくとも以下を集計する。

- 同一race_id参照件数
- 同日参照件数
- 未来日参照件数
- 正常な過去日参照件数
- 履歴なし件数
- 検証対象件数
- 履歴ストア種別ごとの違反件数

出力:

```text
outputs/history_leakage_validation_v2_1.csv
outputs/history_leakage_validation_by_store_v2_1.csv
outputs/history_leakage_validation_samples_v2_1.csv
```

違反が1件でもある場合は全期間処理を成功扱いにしない。

---

# 3. resume の安全性を強化

チェックポイントに以下を保存する。

- 完了年
- 入力Parquetのパス
- 入力ファイルサイズ
- 入力ファイル更新時刻
- 入力ファイルのSHA-256または軽量fingerprint
- `config/feature_sets.yaml` のハッシュ
- V2.1コードバージョン
- Git commit SHA
- 履歴状態ファイル
- 出力行数
- 完了日時

`--resume` 時に必ず整合性を確認する。

次のいずれかが変わった場合、該当年以降のチェックポイントを無効化する。

- 入力Parquet
- feature set設定
- V2.1コードバージョン
- 履歴状態形式

危険な状態で無条件に再開しない。

必要なら以下を追加する。

```text
--strict-resume
--rebuild-from-year YYYY
```

推奨動作:

- `--strict-resume`: 不一致時に停止
- `--rebuild-from-year`: 指定年以降を再生成

---

# 4. feature set を3系統に整理

現在の定義を確認し、以下の3系統へ整理する。

## market_free

現在レースだけでなく、過去レースの市場情報も使わない。

除外対象例:

- 過去人気
- 過去単勝オッズ
- 過去複勝オッズ
- 過去票数
- 現在レースの人気・オッズ・票数

最低限、以下を `market_free` から除外する。

- `horse_last3_avg_ninki`
- `horse_last5_avg_ninki`
- `horse_last3_avg_tan_odds`
- `horse_last5_avg_tan_odds`

同義列が他にもあれば列名と生成元を確認して除外する。

## market_history

過去レースの市場情報は使うが、当該レースの市場情報は使わない。

例:

- 過去人気
- 過去単勝オッズ
- 過去複勝オッズ
- 過去市場順位

当該レースの以下は含めない。

- `tan_odds`
- `tan_ninki`
- `fuku_odds_low`
- `fuku_odds_high`
- `fuku_ninki`
- `TanVote`
- `FukuVote`

## market_aware

`market_history` に当該レースの市場情報を追加する。

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

`config/feature_sets.yaml` は明示的な許可リストとする。

raw列と正規化済み列を同時に正式特徴量へ入れない。

---

# 5. feature set の安全検証

以下を自動検証する。

- ターゲット列が含まれない
- 払戻列が含まれない
- 確定着順・タイム・通過順位・上がりが含まれない
- 学習対象フラグが含まれない
- 除外理由列が含まれない
- データ分割列が含まれない
- raw市場列と正規化市場列が重複しない
- market_freeに市場由来列が含まれない
- market_historyに当該レース市場列が含まれない
- market_awareがmarket_historyを包含する

出力:

```text
outputs/feature_inventory_v2_1.csv
outputs/feature_set_validation_v2_1.csv
```

---

# 6. テスト追加

pytestへ最低限以下を追加する。

1. `history_cutoff_date < race_date`
2. 過去履歴なしの場合 `history_cutoff_date` がNULL
3. 同一race_id参照を実測で検出できる
4. 同日参照を実測で検出できる
5. 未来日参照を実測で検出できる
6. 全履歴ストア種別で監査情報を保持する
7. 入力Parquet変更時にstrict resumeが停止する
8. feature_sets.yaml変更時にstrict resumeが停止する
9. `--rebuild-from-year` で指定年以降を再処理する
10. market_freeに過去人気・過去オッズがない
11. market_historyに当該レース市場情報がない
12. market_awareがmarket_historyを包含する
13. リーク列がfeature setに含まれない

既存テストもすべて成功させる。

---

# 7. V2.1出力

既存V2を上書きしない。

```text
scripts/build_model_features_v2_1.py

outputs/model_feature_dataset_v2_1/year=YYYY/data.parquet
outputs/model_feature_dataset_v2_1_checkpoint/
outputs/model_feature_dataset_v2_1_sample.csv
outputs/training_eligibility_summary_v2_1.csv
outputs/label_mismatch_cases_v2_1.csv
outputs/history_leakage_validation_v2_1.csv
outputs/history_leakage_validation_by_store_v2_1.csv
outputs/history_leakage_validation_samples_v2_1.csv
outputs/feature_inventory_v2_1.csv
outputs/feature_set_validation_v2_1.csv
outputs/model_feature_v2_v2_1_comparison.csv

docs/model_feature_design_v2_1.md
docs/time_leakage_validation_v2_1.md
docs/feature_set_design_v2_1.md
docs/resume_design_v2_1.md

logs/build_model_features_v2_1.log
```

必要に応じて共通モジュールはV2と共有してよいが、V2の再現性を壊さない。

---

# 8. README更新

READMEへ以下を追記する。

- V1 / V2 / V2.1 の違い
- Phase 1は時系列オッズなし
- V2.1の実行方法
- `--resume`
- `--strict-resume`
- `--rebuild-from-year`
- feature set 3系統
- 現在はモデル学習前であること

---

# 9. 実行順

## 構文確認

```bash
python -m py_compile scripts/build_model_features_v2_1.py
```

## テスト

```bash
python -m pytest -q
```

## 2016〜2017年の試験実行

```bash
python scripts/build_model_features_v2_1.py --years 2016-2017 --force
```

確認:

- entry_id重複0
- 同一race_id参照0
- 同日参照0
- 未来日参照0
- history_cutoff_date違反0
- 3つのfeature set検証成功
- 年末チェックポイント作成

## resume検証

```bash
python scripts/build_model_features_v2_1.py --resume --strict-resume
```

次をテストする。

- 正常時は次年から再開
- 入力Parquet変更を検出
- feature_sets.yaml変更を検出
- 指定年以降を `--rebuild-from-year` で再生成可能

## 全期間

試験が成功した場合のみ全期間を処理する。

```bash
python scripts/build_model_features_v2_1.py --resume --strict-resume
```

---

# 10. 完了条件

以下をすべて満たすこと。

- pytest全件成功
- entry_id重複0
- 同一race_id参照0
- 同日参照0
- 未来日参照0
- history_cutoff_date違反0
- strict resumeの不一致検知が動作
- market_freeに市場情報がない
- market_historyに当該レース市場情報がない
- market_awareがmarket_historyを包含
- V1/V2を上書きしていない

1つでも満たさない場合、モデル学習可能とは判定しない。

---

# 11. 最終報告

1. `git diff` と変更ファイル
2. 修正した問題
3. pytest結果
4. V2/V2.1の行数・列数比較
5. history_cutoff_date違反件数
6. 同一race_id参照件数
7. 同日参照件数
8. 未来日参照件数
9. 履歴ストア別の違反件数
10. market_free特徴量数
11. market_history特徴量数
12. market_aware特徴量数
13. feature set検証結果
14. strict resume検証結果
15. rebuild-from-year検証結果
16. 全期間処理時間
17. CatBoost学習へ進める状態か
18. 未解決事項

V2.1の再生成と検証が完了した時点で停止する。
