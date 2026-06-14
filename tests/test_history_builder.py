from __future__ import annotations

import polars as pl

from src.features.history_builder import build_pre_day_history_features
from src.features.target_builder import add_target_columns


def row(race_id: str, entry_id: str, race_date: str, race_num: int, horse: str, rank: int, ijyo: str = "0") -> dict:
    return {
        "race_id": race_id,
        "entry_id": entry_id,
        "race_date": race_date,
        "Year": int(race_date[:4]),
        "MonthDay": int(race_date[5:7] + race_date[8:10]),
        "JyoCD": "06",
        "Kaiji": 1,
        "Nichiji": 1,
        "RaceNum": race_num,
        "Wakuban": 1,
        "Umaban": 1,
        "KettoNum": horse,
        "KisyuCode": "01001",
        "ChokyosiCode": "02001",
        "TrackCD": "10",
        "Kyori": 1600,
        "Futan": 560,
        "BaTaijyu": 480,
        "SibaBabaCD": "1",
        "DirtBabaCD": "0",
        "IJyoCD": ijyo,
        "KakuteiJyuni": rank,
        "SyussoTosu": 8,
        "Ninki": 1,
        "HaronTimeL3": 35.5,
        "Time": 95.0,
        "tan_pay": 100 if rank == 1 else 0,
        "fuku_pay": 100 if rank <= 3 else 0,
        "is_win_paid": 1 if rank == 1 else 0,
        "is_place_paid": 1 if rank <= 3 else 0,
        "tan_odds": 2.0,
        "fuku_odds_low": 1.1,
        "fuku_odds_high": 1.4,
        "TanVote": 100,
        "FukuVote": 200,
    }


def test_same_day_results_are_not_used_but_next_day_can_use_previous_day() -> None:
    df = pl.DataFrame([
        row("r1", "e1", "2016-01-01", 1, "h1", 1),
        row("r2", "e2", "2016-01-01", 12, "h1", 2),
        row("r3", "e3", "2016-01-02", 1, "h1", 3),
    ])
    labeled = add_target_columns(df)
    out, _state, leakage = build_pre_day_history_features(labeled)
    by_entry = {r["entry_id"]: r for r in out.to_dicts()}
    assert by_entry["e1"]["horse_past_starts"] == 0
    assert by_entry["e2"]["horse_past_starts"] == 0
    assert by_entry["e3"]["horse_past_starts"] == 2
    assert all(r["source_before_current"] for r in leakage)
    assert not any(r["same_day_reference"] for r in leakage)


def test_abnormal_and_invalid_rows_are_not_added_to_history() -> None:
    df = pl.DataFrame([
        row("r1", "e1", "2016-01-01", 1, "h1", 1, ijyo="1"),
        row("r2", "e2", "2016-01-02", 1, "h1", 2),
        row("r3", "e3", "2016-01-03", 1, "0000", 1),
        row("r4", "e4", "2016-01-04", 1, "0000", 2),
    ])
    out, state, _leakage = build_pre_day_history_features(add_target_columns(df))
    by_entry = {r["entry_id"]: r for r in out.to_dicts()}
    assert by_entry["e2"]["horse_past_starts"] == 0
    assert by_entry["e4"]["horse_past_starts"] is None
    assert state["excluded_history_rows"]["abnormal_or_cancelled"] == 1
    assert state["excluded_history_rows"]["invalid_horse_id"] == 2


def test_horse_past_starts_is_not_capped_at_20() -> None:
    rows = [row(f"r{i}", f"e{i}", f"2016-01-{i:02d}", 1, "h1", 1) for i in range(1, 23)]
    out, _state, _leakage = build_pre_day_history_features(add_target_columns(pl.DataFrame(rows)))
    last = out.sort("race_date").tail(1).to_dicts()[0]
    assert last["horse_past_starts"] == 21

