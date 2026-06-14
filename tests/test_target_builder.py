from __future__ import annotations

import polars as pl

from src.features.target_builder import add_target_columns, is_valid_id


def base_row(race_id: str, entry_id: str, syusso: int, rank: int, horse_id: str = "2016000001") -> dict:
    return {
        "race_id": race_id,
        "entry_id": entry_id,
        "race_date": "2016-01-01",
        "Year": 2016,
        "MonthDay": 101,
        "JyoCD": "06",
        "Kaiji": 1,
        "Nichiji": 1,
        "RaceNum": 1,
        "Umaban": int(entry_id[-2:]),
        "KettoNum": horse_id,
        "IJyoCD": "0",
        "KakuteiJyuni": rank,
        "SyussoTosu": syusso,
        "tan_pay": 100 if rank == 1 else 0,
        "fuku_pay": 100 if rank <= 3 else 0,
        "is_win_paid": 1 if rank == 1 else 0,
        "is_place_paid": 1 if rank <= 3 else 0,
        "tan_odds": 12.3,
        "fuku_odds_low": 1.1,
        "fuku_odds_high": 1.6,
        "TanVote": 1000,
        "FukuVote": 2000,
    }


def test_place_rank_limit_rules() -> None:
    rows = [
        base_row("r4", "r401", 4, 1),
        base_row("r5", "r501", 5, 2),
        base_row("r7", "r701", 7, 3),
        base_row("r8", "r801", 8, 3),
    ]
    out = add_target_columns(pl.DataFrame(rows)).sort("race_id")
    assert out["place_rank_limit"].to_list() == [0, 2, 2, 3]
    assert out["place_bet_available_by_rule"].to_list() == [False, True, True, True]
    assert out["target_place_by_rule"].to_list() == [0, 1, 0, 1]


def test_eligibility_reasons_are_separate_by_bet_type() -> None:
    rows = [
        base_row("r1", "r101", 8, 1),
        {**base_row("r2", "r201", 8, 1), "tan_pay": 0, "is_win_paid": 0},
        {**base_row("r3", "r301", 8, 1), "KettoNum": "0"},
    ]
    out = add_target_columns(pl.DataFrame(rows)).sort("race_id")
    assert out["eligible_for_win_training"].to_list() == [True, False, False]
    assert out["eligible_for_place_training"].to_list() == [True, True, False]
    assert out["win_training_exclusion_reason"].to_list()[1] == "no_win_payout"
    assert out["place_training_exclusion_reason"].to_list()[2] == "invalid_horse_id"


def test_invalid_id_values() -> None:
    assert not is_valid_id(None)
    assert not is_valid_id("")
    assert not is_valid_id("0")
    assert not is_valid_id("000000")
    assert is_valid_id("2016000001")

