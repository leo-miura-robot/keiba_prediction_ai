# Keiba Prediction AI Dataset

JRA 2016-2026 データを使った競馬予想AI用のデータセット構築リポジトリです。

このリポジトリでは、現時点ではモデル学習・バックテスト・買い目生成は行いません。SQLite DBの調査、1行=1出走馬の基礎データセット作成、モデル学習前のターゲット修正と時系列特徴量作成までを扱います。

## 使用DB

最新版DBのみを使用します。

```text
D:\keiba\new_jra_2016-2026\keiba.db
```

旧DB `D:\keiba\jra_2016-2026\keiba.db` は使用しません。

## 主要スクリプト

- `scripts/build_full_runner_dataset.py`: SQLite DBから年別の基礎出走馬データセットを作成
- `scripts/build_model_features.py`: 基礎データセットからターゲット修正と時系列特徴量を作成
- `scripts/validate_new_db_fuku_payout_full.py`: 新DBの複勝払戻展開を検証

## 主要出力

- `outputs/base_runner_dataset/year=YYYY/data.parquet`
- `outputs/model_feature_dataset/year=YYYY/data.parquet`
- `outputs/training_eligibility_summary.csv`
- `outputs/label_mismatch_cases.csv`
- `outputs/historical_feature_quality.csv`

Parquetやログは容量が増えるため `.gitignore` でGit管理から除外しています。ローカル作業成果物としてはコピー済みです。

## 実行例

基礎データセット作成:

```powershell
python scripts\build_full_runner_dataset.py --resume
```

モデル学習前特徴量作成:

```powershell
python scripts\build_model_features.py --resume
```

2016-2017年だけ再作成:

```powershell
python scripts\build_model_features.py --years 2016,2017 --force
```

## 現在の状態

- 基礎データセット作成済み
- モデル特徴量データセット作成済み
- モデル学習は未実施
- バックテストは未実施
