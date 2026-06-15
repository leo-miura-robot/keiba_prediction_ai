# Keiba Prediction AI

JRA 2016-2026年データを使った競馬予想AIの検証リポジトリです。  
SQLite DBのスキーマ調査、払戻データ修正確認、1行=1出走馬データセット作成、時系列特徴量作成、CatBoost学習、確率補正、EV/ROI検証まで進めています。

現時点の重要な前提は次の通りです。

- 学習・検証対象はJRA 2016年以降です。
- 目的変数は着順ベースではなく、原則として払戻ベースの `target_win_paid` / `target_place_paid` を使います。
- 2025年はtest、2026年はlatest_holdoutとして扱い、モデル選択や購入条件選択には使いません。
- `market_aware` は対象レースの確定オッズを入力に使う理想条件モデルです。発走前実運用モデルではありません。
- `outputs/` と `models/` は巨大な生成物なのでGit管理から除外しています。

## 使用DB

現在の正式DBは、O1オッズ欠損修正版です。

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

旧DBは比較・調査用です。

```text
D:\keiba\new_jra_2016-2026\keiba.db
D:\keiba\jra_2016-2026\keiba.db
```

DBは読み取り専用で扱う方針です。SQLite DB自体をこのリポジトリへコミットしません。

## データ処理の流れ

大まかな流れは次の通りです。

1. SQLite DB調査
2. 複勝払戻スロット確認
3. O1修正版DBへの切替
4. base runner dataset作成
5. V2.1.2特徴量作成
6. CatBoost V2.1.2 6モデル学習
7. ROI/EV検証
8. 確定オッズ2モデルのend-to-end検証

## 主要データセット

### Base Runner Dataset

1行=1出走馬の基礎データセットです。

```text
outputs/base_runner_dataset_o1_fixed/
```

主な構造:

- base table: `NL_SE`
- race metadata: `NL_RA`
- odds: `NL_O1`
- win/place payouts: `NL_HR`
- join key: `race_id + Umaban`

生成スクリプト:

```powershell
python scripts\build_full_runner_dataset_o1_fixed.py --resume
```

### Model Feature Dataset V2.1.2

現在の正式特徴量データセットです。

```text
outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet
```

特徴:

- 2016-2026年を年別Parquetで保存
- 履歴特徴量は当該レース以前のみ参照
- same race / same day / future leakage監査済み
- O1修正版DB由来の単勝・複勝オッズを使用
- splitは固定

split:

```text
train:          2016-2023
validation:     2024
test:           2025
latest_holdout: 2026
```

生成スクリプト:

```powershell
python scripts\build_model_features_v2_1_2.py --resume --strict-resume
```

関連docs:

- `docs/o1_fixed_ai_data_migration.md`
- `docs/feature_set_design_v2_1_2.md`
- `docs/model_features_v2_1_2_results.md`
- `docs/resume_design_v2_1_2.md`

## Feature Sets

`config/feature_sets_v2_1_2.yaml` を正式な特徴量定義として使います。

### market_free

現在レースのオッズ・人気・票数を使わない特徴量群です。  
発走前実運用に最も近い比較基準です。

### market_history

過去走の人気など、履歴市場情報を含む特徴量群です。  
現在レースの確定オッズは使いません。

### market_aware

`market_history` に加えて、現在レースの確定オッズ・人気・票数を使う特徴量群です。

主な列:

- `tan_odds`
- `tan_ninki`
- `fuku_odds_low`
- `fuku_odds_high`
- `fuku_ninki`
- `TanVote`
- `FukuVote`

これは理想条件モデル用です。発走前に確定オッズを知っている前提になるため、実運用モデルとは分けて評価します。

## 目的変数

正式ターゲット:

```text
win:   target_win_paid
place: target_place_paid
```

補助ターゲット:

```text
target_win_rank
target_ren_rank
target_top3_rank
target_place_by_rule
```

単勝・複勝のROI検証では、実払戻に対応する `target_win_paid` / `target_place_paid` を使います。

## 払戻データ

修正版DBでは `NL_HR` に複勝払戻の複数スロットが入り、`race_id + Umaban` で各出走馬へ結合できるようになっています。

単勝:

```text
TanUmaban / TanPay
TanUmaban2 / TanPay2
TanUmaban3 / TanPay3
```

複勝:

```text
FukuUmaban / FukuPay
FukuUmaban2 / FukuPay2
FukuUmaban3 / FukuPay3
FukuUmaban4 / FukuPay4
FukuUmaban5 / FukuPay5
```

検証スクリプト:

```powershell
python scripts\validate_new_db_fuku_payout_full.py
```

関連docs:

- `docs/fuku_payout_design.md`
- `docs/new_db_fuku_payout_validation.md`

## CatBoost Baseline V2.1.2

V2.1.2特徴量を使って、次の6モデルを学習済みです。

```text
win   x market_free
win   x market_history
win   x market_aware
place x market_free
place x market_history
place x market_aware
```

学習コマンド:

```powershell
python scripts\train_catboost_baseline_v2_1_2_v1.py --all --task-type GPU --devices 0 --force
```

分析コマンド:

```powershell
python scripts\analyze_catboost_baseline_v2_1_2_v1.py --all
```

出力:

```text
models/catboost_baseline_v2_1_2_v1/
outputs/model_training/catboost_baseline_v2_1_2_v1/
```

これらは `.gitignore` 対象です。

主な結果:

- `market_aware` は予測性能では最も強い
- 単勝market比較では、市場確率と `market_aware` はかなり近い水準
- 複勝 `market_aware` も強いが、確定オッズ入力なので実運用候補ではない
- CatBoost GPU smoke成功
- CPU fallbackなし

関連docs:

- `docs/catboost_baseline_v2_1_2_v1_design.md`
- `docs/catboost_baseline_v2_1_2_v1_results.md`

## ROI Validation V2.1.2 V1

学習済み6モデルの予測を使い、確率補正・EV・ROI検証を実施しました。

実行コマンド:

```powershell
python scripts\roi_validation_v2_1_2_v1.py --config config\roi_validation_v2_1_2_v1.yaml
```

出力:

```text
outputs/roi_validation_v2_1_2_v1/
```

主な結果:

- validation 2024だけで補正方法と購入条件を選択
- calibration候補は `none`, Platt, isotonic
- test 2025 / latest_holdout 2026では条件を固定適用
- 100円均等買いで実払戻を使ってROI算出
- 最大払戻除外、上位3/5/10件除外、bootstrap CIを出力

結論:

- 単勝・複勝とも一部条件では90%超えがある
- ただしlatest_holdoutや高配当除外後まで含めると、安定的に90%以上とは言えない
- 大穴や一部期間への依存を排除すると、まだ改善余地が大きい

関連docs:

- `docs/roi_validation_v2_1_2_v1_design.md`
- `docs/roi_validation_v2_1_2_v1_results.md`

## Final Odds Two Models V1

確定オッズを使う理想条件モデルとして、単勝・複勝の2本だけをend-to-endで作成・評価しました。

入口:

```powershell
python scripts\run_final_odds_two_models_v1.py --config config\final_odds_two_models_v1.yaml
```

処理内容:

1. preflight
2. feature確認
3. walk-forward学習
4. calibration比較
5. alpha選択
6. final model学習
7. 2025/2026予測
8. EV計算
9. 購入ルール選択
10. ROI/安定性/高配当依存分析

walk-forward:

```text
2016-2019 -> 2020
2016-2020 -> 2021
2016-2021 -> 2022
2016-2022 -> 2023
2016-2023 -> 2024
```

最終評価:

```text
test:           2025
latest_holdout: 2026
```

出力:

```text
models/final_odds_two_models_v1/
outputs/final_odds_two_models_v1/
```

主な結果:

| Target | 2025 ROI | 2026 ROI | 2025+2026 ROI | 判定 |
|---|---:|---:|---:|---|
| win | 78.23% | 82.72% | 79.59% | 90%未達 |
| place | 88.23% | 89.21% | 88.52% | 90%未達 |

補足:

- 単勝・複勝ともcalibration採用は `none`
- 単勝の採用alphaは `0.5`
- 複勝の採用alphaは `1.0`
- 最終採用ルールは緩和されたcoreルール
- 厳格なEV/edge条件では十分な購入数が残らなかった
- 確定オッズを使っても、単純なtop1/core戦略では90%安定達成には届いていない

関連docs:

- `docs/final_odds_two_models_v1_design.md`
- `docs/final_odds_two_models_v1_results.md`

## 実行環境メモ

GPU:

```text
NVIDIA GeForce RTX 5070 Ti
```

CatBoost:

```text
CatBoost GPU
CPU fallback disabled
```

テスト:

```powershell
python -m pytest -q
```

直近では `82 passed` を確認しています。

## Git管理方針

GitHubへ上げる対象:

- `config/`
- `docs/`
- `scripts/`
- `src/`
- `tasks/`
- `tests/`
- `README.md`
- `.gitignore`

GitHubへ上げない対象:

- `outputs/`
- `models/`
- `*.parquet`
- `*.pkl`
- `*.cbm`
- SQLite DB

`.gitignore` で次を除外しています。

```gitignore
outputs/
models/
*.parquet
*.db
*.sqlite
*.sqlite3
models/**/*.cbm
```

既にindexへ載った成果物は、ローカルファイルを残したまま次で追跡解除します。

```powershell
git rm -r --cached --ignore-unmatch outputs models
```

## 現時点の結論

ここまでの検証では、特徴量生成・払戻結合・CatBoost学習・ROI検証の一通りのパイプラインは構築できています。

ただし、ROIの第一目標である単勝・複勝それぞれ90%以上の安定達成は、まだ確認できていません。

特に重要な観察:

- `market_aware` は予測性能が高いが、確定オッズ入力なので発走前実運用とは別物
- 確定オッズ理想条件でも、単純なtop1やEV閾値では90%安定達成に届かない
- 複勝は単勝より90%に近いが、2025/2026合算で88.5%程度
- 高配当依存を除くとROIは下がるため、表面的なROIだけでは判断できない
- 次はレース条件別・オッズ帯別・人気帯別に、より保守的な購入除外ルールを設計する必要がある

## 次の推奨方針

1. `market_history` を発走前実運用候補として再評価する
2. 確定オッズモデルは、市場の歪み診断用・上限性能確認用として扱う
3. 複勝の低オッズ過剰購入を抑制する
4. 単勝は市場確率との差分だけでなく、レース条件別の信頼区間を見る
5. 購入ルールは平均ROIではなく、最低年ROI・高配当除外後ROI・bootstrap下限を重視する
6. 2025/2026を見てルールを変えない

## よく使うコマンド

V2.1.2特徴量生成:

```powershell
python scripts\build_model_features_v2_1_2.py --resume --strict-resume
```

CatBoost 6モデル学習:

```powershell
python scripts\train_catboost_baseline_v2_1_2_v1.py --all --task-type GPU --devices 0 --force
```

CatBoost 6モデル分析:

```powershell
python scripts\analyze_catboost_baseline_v2_1_2_v1.py --all
```

ROI検証:

```powershell
python scripts\roi_validation_v2_1_2_v1.py --config config\roi_validation_v2_1_2_v1.yaml
```

確定オッズ2モデルend-to-end:

```powershell
python scripts\run_final_odds_two_models_v1.py --config config\final_odds_two_models_v1.yaml
```

strict resume確認:

```powershell
python scripts\run_final_odds_two_models_v1.py --config config\final_odds_two_models_v1.yaml --resume --strict-resume
```

テスト:

```powershell
python -m pytest -q
```
