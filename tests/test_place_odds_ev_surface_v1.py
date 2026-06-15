from __future__ import annotations

import pandas as pd

from scripts.run_place_odds_ev_surface_v1 import apply_one_per_race, mask_odds, odds_ranges, summarize_bets


def test_odds_ranges_are_lower_inclusive_upper_exclusive() -> None:
    cfg = {"odds_lower_candidates": [1.0, 1.1], "odds_upper_candidates": [1.1, 1.2, None]}
    ranges = odds_ranges(cfg)
    spec = next(r for r in ranges if r["odds_range"] == "1.0-1.1")
    df = pd.DataFrame({"fuku_odds_low": [1.0, 1.09, 1.1]})
    assert mask_odds(df, spec).tolist() == [True, True, False]


def test_one_per_race_prefers_probability() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "entry_id": ["a", "b", "c"],
            "conservative_probability": [0.5, 0.7, 0.4],
            "place_ev_low": [1.2, 1.1, 1.3],
        }
    )
    got = apply_one_per_race(df, "max_probability")
    assert got["entry_id"].tolist() == ["b", "c"]


def test_summarize_bets_uses_actual_fuku_pay() -> None:
    df = pd.DataFrame(
        {
            "race_id": ["r1", "r2"],
            "fuku_pay": [150, 0],
            "fuku_odds_low": [1.2, 2.0],
            "actual_place": [1, 0],
            "conservative_probability": [0.8, 0.3],
        }
    )
    got = summarize_bets(df)
    assert got["stake"] == 200
    assert got["return"] == 150
    assert got["roi"] == 75.0


def test_open_ended_range() -> None:
    cfg = {"odds_lower_candidates": [2.0], "odds_upper_candidates": [None]}
    spec = odds_ranges(cfg)[0]
    df = pd.DataFrame({"fuku_odds_low": [1.9, 2.0, 9.9]})
    assert mask_odds(df, spec).tolist() == [False, True, True]
