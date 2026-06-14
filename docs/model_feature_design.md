# Model Feature Design

## 入力と出力

入力は `outputs/base_runner_dataset/year=YYYY/data.parquet` の年別Parquetです。出力は `outputs/model_feature_dataset/year=YYYY/data.parquet` に同じく年別で保存します。

今回の出力は 1行=1出走馬を維持し、以下を追加しています。

- レース確定判定: `race_has_result`, `race_has_win_payout`, `race_has_place_payout`, `race_is_finalized`
- 修正ターゲット: `target_win_rank`, `target_ren_rank`, `target_top3_rank`, `target_win_paid`, `target_place_paid`, `target_place_by_rule`
- 学習対象フラグ: `eligible_for_win_training`, `eligible_for_place_training`, `eligible_for_ranking_training`
- データ分割: `data_split`
- 市場データ取得フラグ: `market_odds_available`, `market_votes_available`
- 時系列特徴量: 馬、条件別、騎手、調教師、組み合わせ実績

## 実行結果

- 総行数: 505,881
- 総レース数: 36,269
- 出力列数: 151
- `entry_id` 重複: 0
- 単勝払戻対象行: 36,111
- 複勝払戻対象行: 107,588
- 市場オッズ取得行: 142,925
- 市場票数取得行: 143,692

## データ分割

- 2016-2023: `train`、388,111行、27,761レース
- 2024: `validation`、47,212行、3,456レース
- 2025: `test`、48,058行、3,468レース
- 2026: `latest_holdout`、22,500行、1,584レース

2026年は年途中データのため、通常のテスト期間には混ぜません。

## 時系列特徴量

すべての時系列特徴量は `race_date`, `Year`, `MonthDay`, `JyoCD`, `Kaiji`, `Nichiji`, `RaceNum`, `Umaban`, `entry_id` の順でソートし、現在行の特徴量を作成してから履歴ストアへ現在行を追加する one-pass 方式で作成しています。そのため当該レース自身は集計に含まれません。

主な特徴量は以下です。

- 馬の近走: `horse_days_since_last`, `horse_past_starts`, `horse_last1_avg_finish`, `horse_last3_avg_finish`, `horse_last5_avg_finish`
- 馬の近走率: `horse_last3_win_rate`, `horse_last5_win_rate`, `horse_last3_ren_rate`, `horse_last5_ren_rate`, `horse_last3_place_rate`, `horse_last5_place_rate`
- 馬の近走平均: `horse_last3_avg_ninki`, `horse_last5_avg_ninki`, `horse_last3_avg_tan_odds`, `horse_last5_avg_tan_odds`, `horse_last3_avg_haron_l3`, `horse_last5_avg_haron_l3`, `horse_last3_avg_time`, `horse_last5_avg_time`
- 前走差分: `horse_distance_diff_last`, `horse_futan_diff_last`, `horse_body_weight_diff_last`
- 条件別: `horse_jyo_*`, `horse_surface_*`, `horse_dist_band_*`, `horse_baba_*`
- 騎手・調教師: `jockey_*`, `trainer_*`
- 組み合わせ: `jockey_jyo_*`, `jockey_dist_band_*`, `horse_jockey_*`

## market_free / market_aware

`market_free` はオッズ、人気、票数を使わない特徴量セットです。レース条件、出走馬属性、時系列成績、騎手・調教師の過去成績を中心に使います。

`market_aware` は `market_free` に加えて以下を許可します。

- `TanOdds`, `TanNinki`, `TanVote`
- `FukuOddsLow`, `FukuOddsHigh`, `FukuNinki`, `FukuVote`
- 正規化済み列: `tan_odds`, `tan_ninki`, `fuku_odds_low`, `fuku_odds_high`, `fuku_ninki`
- 取得フラグ: `market_odds_available`, `market_votes_available`

オッズ欠損行は削除していません。後続モデルでは、欠損値処理と取得フラグを併用してください。

## 実行スクリプト

`scripts/build_model_features.py` が今回の処理本体です。

- `--years`: 対象年を `2016,2017` または `2016-2026` の形式で指定
- `--resume`: 完了済み年のParquet再出力をスキップ
- `--force`: 完了済み年も再出力

チェックポイントは `outputs/model_feature_dataset_checkpoint.json` に保存します。一時ファイルへ書き出してから正式Parquetに置換します。
