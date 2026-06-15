# ROI Validation V2.1.2 V1

- 入力は学習済み `catboost_baseline_v2_1_2_v1` の予測Parquetのみです。
- モデル再学習は行っていません。
- 補正方法と購入条件はvalidation 2024だけで選び、test 2025 / latest_holdout 2026には固定適用します。
- 単勝EVは `calibrated_win_probability * tan_odds`、複勝EVは保守的に `calibrated_place_probability * fuku_odds_low` です。
- ROIは100円均等買いで、実払戻 `tan_pay` / `fuku_pay` のみを使います。
- `market_history` は発走前実運用候補、`market_aware` は確定オッズ入力の理想条件モデルとして別枠で扱います。
- 自動購入、Kelly、資金配分最適化、Ability/ANA/Rankerは対象外です。
