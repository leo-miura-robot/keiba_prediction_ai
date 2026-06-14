# Time Leakage Validation

## 方針

時系列特徴量は、当該レースより前のデータだけを使って作成します。処理は全年度を結合したうえで日付とレース順にソートし、各行の特徴量を作成した後に、その行の結果を履歴ストアへ追加します。

これにより以下を防ぎます。

- 当該レース自身の着順、タイム、払戻を履歴特徴量に含める
- 同日後続レースの結果を同日前半レースに使う
- 2024年以降の結果を2016-2023年のtrain特徴量に混入する

## ソート順

使用した順序は以下です。

1. `race_date`
2. `Year`
3. `MonthDay`
4. `JyoCD`
5. `Kaiji`
6. `Nichiji`
7. `RaceNum`
8. `Umaban`
9. `entry_id`

## 検証結果

`outputs/time_leakage_validation_samples.csv` に、過去走を持つ馬のサンプル200件を出力しました。

- 検証サンプル件数: 200
- `previous_is_before_current=False`: 0
- `entry_id` 重複: 0
- 分割期間の重複: なし

## 特徴量に使わない列

以下はレース後に確定するため、モデル特徴量には使いません。

- `NyusenJyuni`
- `KakuteiJyuni`
- `Time`
- `ChakusaCD`
- `HaronTimeL3`
- `Jyuni1c`, `Jyuni2c`, `Jyuni3c`, `Jyuni4c`
- `tan_pay`, `fuku_pay`
- `is_win_paid`, `is_place_paid`
- `target_win_rank`, `target_ren_rank`, `target_top3_rank`
- `target_win_paid`, `target_place_paid`, `target_place_by_rule`
- `race_has_result`, `race_has_win_payout`, `race_has_place_payout`, `race_is_finalized`
- `eligible_for_win_training`, `eligible_for_place_training`, `eligible_for_ranking_training`

## 注意が必要な列

以下は予測時点の取得タイミングによって扱いが変わります。

- `TanOdds`, `tan_odds`
- `TanNinki`, `tan_ninki`
- `FukuOddsLow`, `fuku_odds_low`
- `FukuOddsHigh`, `fuku_odds_high`
- `FukuNinki`, `fuku_ninki`
- `TanVote`, `FukuVote`

確定オッズだけを持つ場合、締切前のリアルタイム予測では未来情報に近くなります。後続では `market_free` と `market_aware` を分けて比較してください。

## 欠損について

新馬や履歴不足の馬では、馬の過去走特徴量が欠損します。これはデータ不備ではなく、過去情報が存在しないことを示します。欠損率は `outputs/historical_feature_quality.csv` に年別・列別で出力しています。
