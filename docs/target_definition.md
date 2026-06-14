# Target Definition

## レース確定判定

未実施・未確定・払戻未取得レースの全馬を負例にしないため、レース単位で以下を作成しています。

- `race_has_result`: 同一 `race_id` 内に `KakuteiJyuni > 0` が存在する
- `race_has_win_payout`: 同一 `race_id` 内に `tan_pay > 0` が存在する
- `race_has_place_payout`: 同一 `race_id` 内に `fuku_pay > 0` が存在する
- `race_is_finalized`: `race_has_result` かつ、単勝または複勝払戻が確認できる

`race_is_finalized=False` の行は、原則として学習対象から除外します。

## 着順ベースターゲット

着順由来のターゲットは比較・診断用として残します。

- `target_win_rank`: 通常出走馬かつ `KakuteiJyuni = 1`
- `target_ren_rank`: 通常出走馬かつ `KakuteiJyuni <= 2`
- `target_top3_rank`: 通常出走馬かつ `KakuteiJyuni <= 3`

通常出走馬は `IJyoCD = "0"` かつ `KakuteiJyuni > 0` で判定しています。

## 払戻ベースターゲット

単勝・複勝の回収率最大化では、正式ターゲットは払戻ベースです。

- `target_win_paid`: `is_win_paid`
- `target_place_paid`: `is_place_paid`

少頭数レースの複勝ルール確認用に以下も作成しています。

- `place_rank_limit`: `SyussoTosu >= 8` なら3、それ以外は2
- `target_place_by_rule`: 通常出走馬かつ `KakuteiJyuni <= place_rank_limit`

同着、降着、失格、取消、少頭数、払戻側の特殊処理を正確に反映する正式な複勝ターゲットは `target_place_paid` です。

## 学習対象フラグ

以下を満たす行を学習候補にしています。

- `race_is_finalized=True`
- `IJyoCD = "0"` の通常出走馬
- `Umaban` が有効
- `KettoNum` が有効
- 目的変数が確定している

作成列は以下です。

- `eligible_for_win_training`
- `eligible_for_place_training`
- `eligible_for_ranking_training`

## 全期間集計

- 単勝学習対象行: 498,926
- 複勝学習対象行: 498,926
- ランキング学習対象行: 498,926
- 未確定扱い行: 2,819
- `target_win_rank` と `target_win_paid` の不一致: 2行
- `target_top3_rank` と `target_place_paid` の不一致: 679行
- `target_place_by_rule` と `target_place_paid` の不一致: 46行

不一致行は削除していません。`outputs/label_mismatch_cases.csv` に出力しています。
