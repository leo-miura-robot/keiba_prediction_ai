# Target Definition V2

- `target_win_rank`: normal runner and `KakuteiJyuni = 1`
- `target_ren_rank`: normal runner and `KakuteiJyuni <= 2`
- `target_top3_rank`: normal runner and `KakuteiJyuni <= 3`
- `target_win_paid`: `is_win_paid`
- `target_place_paid`: `is_place_paid`; formal target for place ROI modeling

Place rule diagnostics:

- `SyussoTosu <= 4`: `place_bet_available_by_rule=False`, `place_rank_limit=0`
- `SyussoTosu 5..7`: `place_bet_available_by_rule=True`, `place_rank_limit=2`
- `SyussoTosu >= 8`: `place_bet_available_by_rule=True`, `place_rank_limit=3`

Eligibility flags are separated for win, place, and ranking. Exclusion reasons are stored in `*_training_exclusion_reason` columns.