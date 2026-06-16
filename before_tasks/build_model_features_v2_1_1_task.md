# Task: V2.1.1 Resume Safety Fix

## 目的

`keiba_prediction_ai` のV2.1を保持したまま、resume関連の不具合と再現性の問題を修正した **V2.1.1** を別実装する。

今回は以下だけを行う。

1. strict resume時の`feature_validation`未定義修正
2. resumeの連続年度・state・出力・Git SHA・コードhash検証
3. `--rebuild-from-year`の前年度state読込修正
4. feature set設定ファイルのバージョン分離
5. テスト・README・設計資料更新
6. V2.1.1データの再生成と検証

以下は行わない。

- CatBoost / LightGBM / Ranker学習
- Optuna
- 確率キャリブレーション
- バックテスト
- EV閾値最適化
- 時系列オッズ利用

---

## 基本方針

- V1、V2、V2.1は削除・上書きしない
- V2.1.1を別コード・別設定・別出力として作る
- resume不整合時は自動再生成せず停止する
- 再生成は`--rebuild-from-year`を明示した場合のみ行う
- Git SHAは記録用、実際のresume判定はコードhash・設定hash・入力fingerprintで行う

---

## 参照ファイル

最初に現在の`main`を確認する。

- `scripts/build_model_features_v2_1.py`
- `src/features/history_builder_v2_1.py`
- `src/features/feature_sets_v2_1.py`
- `src/features/target_builder.py`
- `config/feature_sets.yaml`
- `tests/test_v2_1_history_and_resume.py`
- `docs/resume_design_v2_1.md`
- `docs/feature_set_design_v2_1.md`
- `README.md`
- `outputs/model_feature_dataset_v2_1_checkpoint/checkpoint.json`

現在の実装を確認してから修正する。推測だけで作り直さない。

---

# 1. V2.1.1を別実装する

新規追加の基本構成:

```text
scripts/build_model_features_v2_1_1.py
src/features/history_builder_v2_1_1.py
src/features/feature_sets_v2_1_1.py
config/feature_sets_v2_1_1.yaml
tests/test_v2_1_1_resume_safety.py

docs/model_feature_design_v2_1_1.md
docs/resume_design_v2_1_1.md
docs/feature_set_design_v2_1_1.md

outputs/model_feature_dataset_v2_1_1/
outputs/model_feature_dataset_v2_1_1_checkpoint/
logs/build_model_features_v2_1_1.log
```

共通コードの再利用は可。ただしV2.1の挙動と再現性を壊さない。

---

# 2. feature_validation未定義を修正

strict resume時でも`feature_validation`を必ず初期化・検証する。

禁止:

```python
if not (args.resume and args.strict_resume):
    feature_validation = validate_feature_sets()
```

その後に無条件で`feature_validation`を使う構造。

推奨:

```python
feature_validation = validate_feature_sets(...)
```

を必ず実行する。

以下を分離する。

1. feature set定義の生成
2. feature set定義の読み込み
3. feature set安全検証
4. 検証結果のCSV出力

strict resume時も検証結果を再取得し、未定義変数を発生させない。

---

# 3. versioned feature set設定

共有の`config/feature_sets.yaml`をV2.1.1から使用しない。

V2.1.1は必ず次だけを参照する。

```text
config/feature_sets_v2_1_1.yaml
```

既存の設定は保持する。

必要なら以下も追加してよい。

```text
config/feature_sets_v2.yaml
config/feature_sets_v2_1.yaml
```

ただし既存コードの参照先を勝手に変更しない。

V2.1.1のチェックポイントには、次のSHA-256を保存する。

```text
config/feature_sets_v2_1_1.yaml
```

---

# 4. resumeの連続年度検証

2016年から連続して正常完了した年度だけをresume対象にする。

例:

```text
2016 complete
2017 complete
2018 missing
2019 complete
```

この場合、有効な最終年度は2017年。

2019年のstateから再開してはいけない。

実装要件:

- 完了年度を昇順で検証
- 開始年度から連続しているか確認
- 途中の欠落を検出
- 欠落後の年度を有効扱いしない
- strict resumeでは不整合理由を表示してexit 2

---

# 5. 各年度のstate・出力検証

チェックポイントでcompleteとなっている各年度について、以下を検証する。

- state pickleが存在する
- 年別Parquetが存在する
- Parquetの行数がチェックポイントと一致
- state versionが一致
- state内の完了年が対象年と一致
- 出力年が連続している
- stateファイルが破損していない
- Parquetが読み込める

不一致時:

- strict resume: 理由を表示してexit 2
- 非strict resume: 自動再生成せず停止
- 再生成は`--rebuild-from-year`を要求する

---

# 6. 入力・設定・コードの整合性

年度ごとに以下をチェックポイントへ保存する。

## 入力Parquet

- path
- file size
- mtime
- SHA-256または軽量fingerprint

## feature set

- 設定ファイルpath
- SHA-256

## コード

Git SHAは記録用として保存する。

resume判定には、少なくとも以下の内容hashを使う。

- `scripts/build_model_features_v2_1_1.py`
- `src/features/history_builder_v2_1_1.py`
- `src/features/feature_sets_v2_1_1.py`
- `src/features/target_builder.py`
- `config/feature_sets_v2_1_1.yaml`

`code_bundle_hash`として保存する。

READMEやdocsだけの変更でresumeを無効化しない。

---

# 7. Git SHAの扱い

チェックポイントへ以下を保存する。

```text
git_commit_sha
git_is_dirty
```

ただしGit SHA単独の変更をresume失敗理由にしない。

用途:

- 実行履歴の追跡
- 監査
- 再現性記録

resume判定の主条件は以下。

- 入力fingerprint
- feature set hash
- code bundle hash
- state version

---

# 8. strict resumeの動作

通常運用:

```bash
python scripts/build_model_features_v2_1_1.py --resume --strict-resume
```

以下の場合は停止してexit 2。

- 年度欠落
- state欠落
- 出力Parquet欠落
- 行数不一致
- state version不一致
- 入力fingerprint不一致
- feature set hash不一致
- code bundle hash不一致
- state破損
- output破損

エラーには最低限以下を含める。

- 不一致年度
- 不一致項目
- 保存値
- 現在値
- 推奨する再実行コマンド

例:

```text
Resume validation failed at year 2020:
input fingerprint mismatch.

Run:
python scripts/build_model_features_v2_1_1.py --resume --rebuild-from-year 2020
```

---

# 9. rebuild-from-yearの前年度state読込

`--rebuild-from-year YYYY`指定時、`--years`に前年が含まれるかどうかに関係なく、前年末stateを読み込む。

ルール:

## 2016から再生成

```text
空stateから開始
```

## 2017以降から再生成

```text
YYYY-1年末stateが必須
```

例:

```text
--years 2019-2021 --rebuild-from-year 2019
```

この場合も2018年末stateを読み込む。

禁止:

- 前年stateがない状態で空stateから途中年度を生成
- `prev in years`を前提にする
- 不足した前年stateを黙って無視する

前年stateがない場合はexit 2で停止し、より前の再生成年を案内する。

---

# 10. rebuild時の無効化範囲

`--rebuild-from-year 2020`なら以下を削除または無効化する。

- 2020年以降のcomplete記録
- 2020年以降のstate
- 2020年以降のV2.1.1 Parquet
- 2020年以降に依存する集計結果

2019年末stateは保持し、そこから再開する。

削除前にログへ対象一覧を出す。

V1、V2、V2.1のファイルには触れない。

---

# 11. 必須テスト

既存テストに加え、最低限以下を追加する。

1. strict resume正常系で`feature_validation`未定義にならない
2. 2016〜2018が連続completeなら2018から再開できる
3. 2017が欠落して2018がcompleteでも2018から再開しない
4. state欠落を検出する
5. Parquet欠落を検出する
6. Parquet行数不一致を検出する
7. state version不一致を検出する
8. 入力fingerprint不一致を検出する
9. feature set hash不一致を検出する
10. code bundle hash不一致を検出する
11. Git SHAだけの変更ではresume失敗にしない
12. `--rebuild-from-year 2019`で2018年stateを読む
13. 前年stateがない場合は停止する
14. rebuild対象年以降だけを無効化する
15. V2.1.1が専用YAMLだけを参照する
16. V1/V2/V2.1の出力を変更しない

---

# 12. 実行手順

## 構文確認

```bash
python -m py_compile scripts/build_model_features_v2_1_1.py
```

## テスト

```bash
python -m pytest -q
```

## 小規模生成

```bash
python scripts/build_model_features_v2_1_1.py --years 2016-2017 --force
```

確認:

- entry_id重複0
- 同一race_id参照0
- 同日参照0
- 未来日参照0
- feature set検証pass
- 2016・2017のstateとParquet生成
- チェックポイントにhash類を保存

## strict resume正常系

```bash
python scripts/build_model_features_v2_1_1.py --resume --strict-resume
```

2016〜2017を再処理せず、2018から進むことを確認する。

## 不一致テスト

実データを破壊せず、コピーまたは一時ファイルで次を検証する。

- 入力fingerprint変更
- feature set hash変更
- code bundle hash変更
- state欠落
- Parquet欠落
- 年度欠落

すべてexit 2になること。

## rebuild検証

```bash
python scripts/build_model_features_v2_1_1.py --resume --rebuild-from-year 2017
```

2016年末stateを読み込み、2017年以降だけを再生成する。

## 全期間

すべて成功した場合のみ実行。

```bash
python scripts/build_model_features_v2_1_1.py --resume --strict-resume
```

---

# 13. 出力

```text
outputs/model_feature_dataset_v2_1_1/year=YYYY/data.parquet
outputs/model_feature_dataset_v2_1_1_checkpoint/checkpoint.json
outputs/model_feature_dataset_v2_1_1_checkpoint/history_state_after_YYYY.pkl

outputs/model_feature_v2_1_v2_1_1_comparison.csv
outputs/resume_validation_v2_1_1.csv
outputs/feature_set_validation_v2_1_1.csv
outputs/history_leakage_validation_v2_1_1.csv

logs/build_model_features_v2_1_1.log
```

---

# 14. README・資料

READMEへ追記する。

- V2.1.1の位置づけ
- V2.1.1はresume安全性修正版
- 専用feature set YAML
- strict resumeの標準利用
- rebuild-from-yearの利用方法
- 不整合時は自動再生成しない方針

追加資料:

```text
docs/resume_design_v2_1_1.md
docs/feature_set_design_v2_1_1.md
docs/model_feature_design_v2_1_1.md
```

---

# 15. 完了条件

以下をすべて満たすこと。

- pytest全件成功
- strict resume正常系成功
- `feature_validation`未定義なし
- 完了年度の連続性検証成功
- state・出力存在確認成功
- 入力fingerprint検証成功
- feature set hash検証成功
- code bundle hash検証成功
- rebuild-from-yearで前年state読込成功
- V2.1.1専用YAMLを使用
- V1/V2/V2.1を上書きしていない
- V2.1.1全期間データ生成成功
- entry_id重複0
- リーク違反0

1つでも満たさない場合、モデル学習可能とは判定しない。

---

# 16. 最終報告

1. `git diff`と変更ファイル
2. V2.1で確認した問題
3. 修正内容
4. pytest結果
5. strict resume正常系結果
6. `feature_validation`未定義修正結果
7. 連続年度検証結果
8. state欠落検出結果
9. 出力Parquet欠落・行数不一致検出結果
10. 入力fingerprint不一致検出結果
11. feature set hash不一致検出結果
12. code bundle hash不一致検出結果
13. Git SHA記録結果
14. rebuild-from-yearの前年state読込結果
15. feature set設定分離結果
16. V2.1/V2.1.1の行数・列数比較
17. entry_id重複件数
18. リーク検証結果
19. 全期間処理時間
20. CatBoost学習へ進める状態か
21. 未解決事項

V2.1.1の実装・再生成・検証が完了した時点で停止する。
