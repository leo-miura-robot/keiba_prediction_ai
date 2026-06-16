# Codex Phase 5B Patch Task v1
## Fix parity key dtype mismatch and complete all short validations

## 0. 目的

Phase 5BのLEGACY_2016 / validation 2024 parity smokeで、学習完了後の比較処理が以下で停止した。

```text
ValueError:
You are trying to merge on datetime64[us] and str columns
for key 'race_date'
```

学習自体は300 iterationsまで正常完了している。
問題は旧予測と新予測の`race_date`型が異なることによるparity merge失敗である。

本タスクでは、parity keyの型を安全に正規化し、
Codex側で文法確認・pytest・短時間smokeまで完了させる。

長時間の2020～2024全7戦略本実行は行わない。

---

# 1. 作業分担

Codexが行う:

```text
コード修正
py_compile
pytest
LEGACY_2016 / 2024 parity smoke
parity成果物確認
短時間で完了する場合のみ2024全7戦略smoke
```

ユーザーがローカルで行う:

```text
2020～2024 × 全7戦略の長時間本実行
```

目安:

```text
10分以内で完了する確認はCodexが実行してよい
10分を大きく超える学習は実行せず、コマンドだけ提示
```

---

# 2. 絶対条件

- CatBoost設定を変更しない
- feature allowlistを変更しない
- tree count 300を維持
- market model / residual modelのwindowを変更しない
- parity許容差を広げない
- calibrationを追加しない
- DB接続禁止
- 既存Parquet変更禁止
- 既存成果物の削除・上書き禁止
- git add / commit / push / reset / clean禁止

今回の修正でモデル予測値を変えてはいけない。

---

# 3. 原因

対象:

```text
scripts/run_place_market_offset_year_strategy_phase5b_v2.py
parity_gate()
```

旧予測:

```text
race_date dtype = str / object
```

新予測:

```text
race_date dtype = datetime64[us]
```

そのまま`DataFrame.merge()`しているため失敗した。

Pandasのエラー文にある`pd.concat`へ置き換えてはいけない。
parityはキーによる1対1対応比較が必要なので、正規化後にmergeする。

---

# 4. 実装方針

## 4.1 parity専用key正規化関数

新規helperを作る。

例:

```python
def normalize_parity_keys(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

元DataFrameを破壊せずcopyを返す。

最低限`race_date`を両DataFrameで同一のcanonical形式へ変換する。

推奨:

```python
parsed = pd.to_datetime(df["race_date"], errors="raise")
df["race_date"] = parsed.dt.strftime("%Y-%m-%d")
```

時刻情報が本来意味を持たない日付キーであることを、
既存データ定義・KEY_COLUMNSから確認してから適用する。

timezone付きの場合も同じ日付表現へ正規化する。

## 4.2 他のKEY_COLUMNS監査

merge前に旧・新の各KEY_COLUMNSについて以下を比較する。

```text
column
old dtype
new dtype
old null count
new null count
old unique count
new unique count
```

`race_date`以外にも型差がある場合、
列の意味に応じた明示的変換を行う。

禁止:

```text
全KEY_COLUMNSを無条件にastype(str)
```

理由:

- 数値キーの小数化
- ゼロ埋めコードの消失
- 欠損値が文字列"nan"になる
- 意味の異なる値が同一化する

## 4.3 merge安全条件

正規化後に以下をassertする。

```text
KEY_COLUMNSにnullなし
旧側KEY_COLUMNS重複なし
新側KEY_COLUMNS重複なし
旧側行数とunique key数一致
新側行数とunique key数一致
```

merge:

```python
merge(..., how="outer", indicator=True, validate="one_to_one")
```

を推奨。

以下を記録する。

```text
old_only count
new_only count
both count
key match rate
```

parity gateでは:

```text
old_only = 0
new_only = 0
key match rate = 100%
```

が必須。

## 4.4 日付変換失敗

`errors="coerce"`で黙ってNaT化しない。

```text
errors="raise"
```

または、失敗値を具体的に列挙して停止する。

---

# 5. 追加テスト

対象:

```text
tests/test_place_market_offset_year_strategy_phase5b_v2.py
```

最低限追加する。

## Test 1

旧`race_date`が`"2024-01-06"`文字列、
新`race_date`が`Timestamp("2024-01-06")`でも正常に1対1mergeできる。

## Test 2

datetime64[us] / datetime64[ns]の差でも一致する。

## Test 3

不正日付がある場合は明示的に失敗する。

## Test 4

KEY_COLUMNS重複がある場合は失敗する。

## Test 5

正規化によって予測値・target・market_logitが変更されない。

## Test 6

outer mergeでold_only / new_onlyが存在する場合はparity failureとなる。

既存の以下も維持する。

```text
outer validationをeval_setへ渡さない
iterations=300
early stopping無効
use_best_model=False
StressROI不変条件
```

---

# 6. Codexが実行する確認

## 6.1 文法

```powershell
python -m py_compile `
  scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  scripts\audit_place_market_offset_year_strategy_phase5b_v2.py
```

## 6.2 pytest

```powershell
python -m pytest `
  tests\test_place_market_offset_year_strategy_phase5b_v2.py `
  tests\test_v2_1_history_and_resume.py `
  -q
```

## 6.3 LEGACY 2024 parity smoke

既存の途中成果物を誤再利用しないため、
新しいoutput/model rootを使用する。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016 `
  --years 2024 `
  --parity-check `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_parity_2024_v2 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_parity_2024_v2
```

## 6.4 parity確認

以下を報告する。

```text
row count old/new
both count
old_only count
new_only count
key match rate
feature match
target match
market_logit p99 abs diff
probability_raw p99 abs diff
Logloss abs diff
Brier abs diff
overall parity pass/fail
```

許容差は既存仕様から変更しない。

## 6.5 2024全7戦略smoke

LEGACY parityが通過し、
実行時間が10分以内と見込める場合のみCodexが実行してよい。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2024 `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_smoke_2024_v2 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_smoke_2024_v2
```

10分を超えそうな場合は実行せずコマンドだけ提示する。

---

# 7. 長時間本実行

Codexは実行しない。

ユーザー向けコマンドだけ提示する。

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2020,2021,2022,2023,2024 `
  --parity-check `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2
```

resume:

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2020,2021,2022,2023,2024 `
  --parity-check `
  --resume `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2
```

---

# 8. 最終報告

簡潔に以下を報告する。

1. 原因
2. 修正ファイル
3. key正規化方法
4. 全KEY_COLUMNSのdtype監査結果
5. 追加テスト
6. py_compile結果
7. pytest結果
8. LEGACY 2024 parity結果
9. 2024全戦略smokeを実行したか
10. 長時間本実行コマンド
11. resumeコマンド
12. git status --short
13. git diff --stat

長時間本実行、追加実験、commit/pushは行わない。
