# Task: SQLite DB完全検査キャッシュ・軽量fingerprint共通基盤の実装

## 目的

競馬予想AIの各処理で毎回実行されている、約18.9GBのSQLite DBに対する

```sql
PRAGMA integrity_check;
```

を、**精度・安全性を落とさずに必要時だけ実行する仕組み**へ変更する。

現在、`integrity_check`には約20分かかっている。

同一DBが変更されていない場合は、過去の完全検査結果を共通キャッシュとして再利用し、通常実行では数秒程度の軽量検証だけを行う。

対象DB:

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

今回の目的は、DB検証処理の共通基盤化と既存runnerへの統合である。

モデル学習、特徴量再生成、ROI再計算は原則行わない。必要な範囲のsmoke testだけ行う。

---

# 1. 基本方針

## 初回またはDB変更時

以下を実行する。

```text
PRAGMA integrity_check
DB全体SHA-256
DBメタデータ取得
主要テーブル確認
検証manifest保存
```

## 通常実行時

以下だけを実行する。

```text
read-only接続
SQLite header確認
file size
mtime_ns
page_size
page_count
schema_version
user_version
freelist_count
主要テーブル存在
先頭・中央・末尾の部分hash
WAL/journal有無
保存済みmanifestとの照合
```

すべて一致した場合:

```text
full integrity check skipped
reason: unchanged database verified by cached validation manifest
```

として処理を続行する。

---

# 2. 精度・安全性に関する要件

`integrity_check`の省略は、モデル精度・特徴量・ROI計算へ影響してはならない。

省略可能なのは、次をすべて満たす場合だけとする。

```text
前回のfull integrity_checkがok
DB path一致
file size一致
mtime_ns一致
page_size一致
page_count一致
schema_version一致
user_version一致
軽量fingerprint一致
主要テーブル存在
WAL/journal不在
前回の検証manifestが正常完了
```

1つでも不一致なら、キャッシュを信用せず完全検査を要求する。

---

# 3. 共通モジュール

推奨:

```text
src/database/db_validation_cache.py
```

責務:

```text
DB metadata取得
軽量fingerprint生成
全体SHA-256生成
full integrity_check実行
検証manifest読み書き
キャッシュ有効性判定
read-only接続
CLI用結果整形
```

既存プロジェクト構造に適切なdatabase/commonディレクトリがある場合は、それに合わせてよい。

---

# 4. CLI

推奨:

```text
scripts/validate_database.py
```

## 通常確認

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db"
```

動作:

```text
軽量fingerprint確認
キャッシュ一致ならfull check省略
不一致なら停止してfull checkが必要と報告
```

通常確認では、勝手に20分の完全検査を開始しない。

## 初回・強制完全検査

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" \
  --full
```

または:

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" \
  --force-integrity-check
```

動作:

```text
PRAGMA integrity_check
全体SHA-256
軽量fingerprint
主要テーブル確認
manifest更新
```

## キャッシュ状態表示

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" \
  --status
```

表示:

```text
cache found
cache valid/invalid
last full check
DB path
DB size
mtime
full SHA-256
light fingerprint
integrity result
invalid reason
```

---

# 5. キャッシュ保存先

推奨:

```text
outputs/db_validation_cache/
```

DBごとに分ける。

例:

```text
outputs/db_validation_cache/
  keiba_5b9485aa/
    validation_manifest.json
    integrity_check.txt
    metadata.json
```

または、DB絶対パスのhashをディレクトリ名に使う。

キャッシュファイル名にDBパス文字列を直接入れない。

---

# 6. validation manifest

最低限保存する。

```json
{
  "manifest_version": 1,
  "db_path": "D:/keiba/new_jra_2016-2026_fixed/keiba.db",
  "db_path_hash": "...",
  "file_size": 18918297600,
  "mtime_ns": 0,
  "sqlite_header_sha256": "...",
  "head_chunk_sha256": "...",
  "middle_chunk_sha256": "...",
  "tail_chunk_sha256": "...",
  "light_fingerprint": "...",
  "full_file_sha256": "...",
  "page_size": 4096,
  "page_count": 0,
  "schema_version": 0,
  "user_version": 0,
  "freelist_count": 0,
  "journal_mode": "...",
  "wal_exists": false,
  "journal_exists": false,
  "integrity_check": "ok",
  "integrity_checked_at": "...",
  "validation_completed": true,
  "required_tables": [],
  "table_presence": {},
  "validator_code_hash": "...",
  "python_version": "...",
  "sqlite_version": "..."
}
```

原子的に保存する。

```text
temporary file
flush
fsync
os.replace
```

---

# 7. 軽量fingerprint

毎回DB全体を読まない。

最低限:

```text
SQLite header
先頭64MiB
中央64MiB
末尾64MiB
file size
mtime_ns
page_size
page_count
schema_version
user_version
freelist_count
```

これらを正規化JSONへまとめ、SHA-256を計算する。

DBサイズが小さい場合はチャンク範囲が重複しても正しく処理する。

チャンクサイズはconfig化する。

推奨:

```text
64 MiB
```

---

# 8. 全体SHA-256

初回full validation時だけDB全体SHA-256を計算する。

ストリーミング読み込みを使う。

```text
chunk size: 8〜64 MiB
```

メモリへDB全体を読み込まない。

進捗を表示する。

例:

```text
hashing database: 10%
hashing database: 20%
...
```

5分以上ログが止まらないようにする。

---

# 9. SQLite接続

read-only:

```python
sqlite3.connect(
    "file:D:/keiba/new_jra_2016-2026_fixed/keiba.db?mode=ro",
    uri=True,
)
```

`immutable=1`はconfigオプションとする。

既定値:

```text
false
```

DBが外部から絶対に更新されない運用であることが明示された場合だけ有効化する。

---

# 10. 完全検査を再実行する条件

以下のどれかでキャッシュ無効。

```text
DB path変更
file size変更
mtime_ns変更
SQLite header変更
head/middle/tail hash変更
page_size変更
page_count変更
schema_version変更
user_version変更
WAL存在
journal存在
前回manifest未完了
前回integrity_check != ok
required table不足
validator version非互換
manifest破損
ユーザーがforce指定
```

キャッシュ無効時は、通常runnerでは自動full checkを始めず、明確に停止して次を案内する。

```bash
python scripts/validate_database.py --db ... --full
```

ただしconfigで明示的に:

```yaml
database_validation:
  auto_full_check_on_cache_miss: true
```

とした場合のみ自動実行を許可する。

既定値は`false`。

---

# 11. 主要テーブル

最低限:

```text
NL_RA
NL_SE
NL_O1
NL_HR
```

既存処理が要求する場合:

```text
NL_H1
NL_H6
NL_UM
NL_KS
NL_CH
```

required tablesはconfigから指定する。

テーブル行数の全件COUNTは通常確認では実行しない。

初回full validationでも、既存manifestで必要なら主要テーブルのみ件数を記録してよいが、必須ではない。

---

# 12. 既存runnerへの統合

少なくとも以下のrunnerが存在する場合は共通validatorを利用する。

```text
scripts/build_full_runner_dataset_o1_fixed.py
scripts/build_model_features_v2_1_2.py
scripts/run_final_odds_two_models_v1.py
scripts/run_roi_strategy_refinement_v1.py
```

存在しないrunnerは無理に作らない。

各runnerで独自に`PRAGMA integrity_check`を実行している箇所を探索する。

```bash
rg -n "integrity_check|quick_check|PRAGMA" .
```

統合後の通常動作:

```text
validator cache確認
→ validなら数秒で通過
→ invalidなら停止
→ full checkは専用CLIまたはforce指定時のみ
```

---

# 13. 共通config

推奨:

```text
config/database_validation.yaml
```

例:

```yaml
database_validation:
  cache_dir: outputs/db_validation_cache
  required_tables:
    - NL_RA
    - NL_SE
    - NL_O1
    - NL_HR

  light_hash_chunk_mib: 64
  full_hash_chunk_mib: 32

  require_mtime_match: true
  require_full_sha256_in_cache: true
  reject_wal: true
  reject_journal: true

  auto_full_check_on_cache_miss: false
  immutable_read: false
```

各runnerのconfigから参照できるようにする。

---

# 14. runnerオプション

各主要runnerへ共通オプションを追加する。

```text
--force-integrity-check
--skip-db-validation
--db-validation-config
```

ただし`--skip-db-validation`は危険なので、既定では禁止または警告付きにする。

推奨:

```text
--skip-db-validation
```

を使う場合:

```text
明示警告
manifestへ記録
本番用途非推奨
```

通常利用では使わない。

---

# 15. ログ

通常キャッシュhit:

```text
database validation cache: HIT
light fingerprint: matched
last full integrity check: 2026-...
full integrity_check: skipped
elapsed: 2.4s
```

キャッシュmiss:

```text
database validation cache: MISS
reason: mtime changed
full integrity_check not started automatically
run: python scripts/validate_database.py --db ... --full
```

force:

```text
database validation: FULL
integrity_check start
integrity_check result: ok
full SHA-256 start
...
cache updated
```

---

# 16. strict resumeとの関係

モデル・特徴量・ROI処理のfingerprintには、次を含める。

```text
DB validation manifest path
DB light fingerprint
DB full SHA-256
integrity_checked_at
validator manifest_version
```

DBが同じならstrict resumeを維持する。

DBが変わったらresumeしない。

---

# 17. 性能目標

通常確認:

```text
目標 1〜10秒
```

環境により数十秒以内は許容。

初回full:

```text
integrity_check 約20分
全体SHA-256 数分〜十数分
```

初回は時間がかかってよい。

通常実行ではfull checkを繰り返さない。

---

# 18. テスト

最低限:

```text
manifest create
manifest atomic write
cache hit
cache miss by size
cache miss by mtime
cache miss by head hash
cache miss by middle hash
cache miss by tail hash
cache miss by page_count
cache miss by schema_version
cache miss by WAL
cache miss by journal
corrupt manifest
failed integrity manifest
required table missing
full hash streaming
read-only connection
force full check
auto full check default false
runner integration
strict resume DB fingerprint
```

テスト用に小さいSQLite DBを生成する。

18GB本番DBでfull checkをテストごとに実行しない。

実行:

```bash
python -m pytest -q
```

---

# 19. 実DBでの受け入れ確認

本番DBでは次を行う。

## 1回目

既に成功済みの過去integrity結果が信頼できる形でmanifest化できない場合は、専用CLIでfull validationを1回実施する。

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" \
  --full
```

## 2回目

同じコマンドを通常モードで実行。

```bash
python scripts/validate_database.py \
  --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db"
```

確認:

```text
cache HIT
integrity_check skipped
light fingerprint matched
処理時間大幅短縮
```

## runner確認

主要runnerをpreflightまたはdry-runで起動し、full checkが再実行されないことを確認する。

---

# 20. 既存検査結果の移行

既存成果物に以下がある場合:

```text
integrity_check = ok
DB path
DB size
DB fingerprint
検査日時
```

それだけでfull SHA-256が不足している場合は、過去結果を無条件で完全キャッシュとして扱わない。

安全な選択肢:

```text
A. full integrity_checkをもう一度行い正式manifest作成
B. integrity_check結果を再利用し、全体SHA-256だけ今回作成
```

Bを採用する場合は、過去検査時からDBが未変更と強く確認できること。

判断根拠をdocsへ記載する。

---

# 21. 出力

```text
src/database/db_validation_cache.py
scripts/validate_database.py
config/database_validation.yaml
tests/test_db_validation_cache.py

outputs/db_validation_cache/
  <db_path_hash>/
    validation_manifest.json
    integrity_check.txt
    metadata.json

docs/database_validation_cache_design.md
docs/database_validation_cache_results.md
```

---

# 22. docs

## design

```text
なぜ毎回integrity_check不要か
精度へ影響しない理由
full check条件
cache hit条件
fingerprint設計
安全上の制約
runner統合方法
運用コマンド
```

## results

```text
初回full所要時間
通常確認所要時間
短縮時間
manifest内容
runner動作
pytest
未解決事項
```

---

# 23. 禁止事項

```text
DB更新
DB上書き
DB VACUUM
DBコピー
モデル再学習
特徴量再生成
ROI再計算
既存成果物削除
検証失敗時の無条件続行
キャッシュ不一致の無視
自動commit
自動push
```

---

# 24. 完了条件

```text
共通DB validator実装
初回full validation対応
通常時cache hit対応
軽量fingerprint対応
full SHA-256対応
キャッシュ無効条件実装
runner統合
strict resume統合
通常実行でintegrity_check省略
不一致時は安全停止
pytest成功
実DBでcache hit確認
精度・データ内容未変更
```

---

# 25. 最終報告

1. git status
2. git diff
3. 追加・変更ファイル
4. validator module path
5. CLI path
6. config path
7. cache dir
8. 対象DB path
9. DB size
10. manifest path
11. full integrity result
12. full integrity所要時間
13. full SHA-256
14. full hash所要時間
15. light fingerprint
16. light fingerprint対象
17. 通常確認所要時間
18. cache hit確認
19. full check skipped確認
20. DB変更検知テスト
21. size変更検知
22. mtime変更検知
23. head/middle/tail変更検知
24. page_count変更検知
25. schema_version変更検知
26. WAL/journal検知
27. required table確認
28. atomic manifest確認
29. read-only確認
30. immutable設定
31. auto full check既定値
32. runner統合一覧
33. 既存integrity_check削除・置換箇所
34. strict resume統合
35. preflight/dry-run結果
36. pytest結果
37. 通常時の短縮時間
38. モデル精度への影響なし確認
39. DB未変更確認
40. モデル未再学習確認
41. ROI未再計算確認
42. 自動commit/push未実施
43. 未解決事項
44. 今後の運用手順

完了後はDB検証キャッシュ基盤の実装と確認まで報告し、モデル学習・特徴量再生成・ROI分析には進まず停止する。
