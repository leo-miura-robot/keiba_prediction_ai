from __future__ import annotations

from pathlib import Path

import polars as pl

import scripts.build_model_features_v2_1 as runner
from src.features.feature_sets_v2_1 import feature_sets, validate_feature_sets
from src.features.history_builder_v2_1 import audit_store, build_pre_day_history_features_v2_1, new_state
from src.features.target_builder import add_target_columns


def row(race_id: str, entry_id: str, race_date: str, horse: str = "h1") -> dict:
    return {
        "race_id": race_id, "entry_id": entry_id, "race_date": race_date,
        "Year": int(race_date[:4]), "MonthDay": int(race_date[5:7] + race_date[8:10]),
        "JyoCD": "06", "Kaiji": 1, "Nichiji": 1, "RaceNum": 1, "Wakuban": 1,
        "Umaban": 1, "KettoNum": horse, "KisyuCode": "01001", "ChokyosiCode": "02001",
        "TrackCD": "10", "Kyori": 1600, "Futan": 560, "BaTaijyu": 480,
        "SibaBabaCD": "1", "DirtBabaCD": "0", "IJyoCD": "0", "KakuteiJyuni": 1,
        "SyussoTosu": 8, "Ninki": 1, "HaronTimeL3": 35.0, "Time": 95.0,
        "tan_pay": 100, "fuku_pay": 100, "is_win_paid": 1, "is_place_paid": 1,
        "tan_odds": 2.0, "tan_ninki": 1, "fuku_odds_low": 1.1, "fuku_odds_high": 1.3,
        "fuku_ninki": 1, "TanVote": 100, "FukuVote": 200,
    }


def test_history_cutoff_date_is_last_history_date_or_null() -> None:
    df = add_target_columns(pl.DataFrame([
        row("r1", "e1", "2016-01-01"),
        row("r2", "e2", "2016-01-02"),
    ]))
    out, _state, audit, _samples = build_pre_day_history_features_v2_1(df)
    by_entry = {r["entry_id"]: r for r in out.to_dicts()}
    assert by_entry["e1"]["history_cutoff_date"] is None
    assert by_entry["e2"]["history_cutoff_date"] == "2016-01-01"
    assert audit["horse"]["same_race"] == 0
    assert audit["horse"]["same_day"] == 0
    assert audit["horse"]["future"] == 0


def test_audit_store_detects_same_race_same_day_and_future() -> None:
    state = new_state()
    key = ("horse", "h1")
    current = row("r1", "e1", "2016-01-02")
    state["sources"][key] = {"race_id": "r1", "race_date": "2016-01-01"}
    assert audit_store(current, state, "horse", key)["status"] == "same_race"
    state["sources"][key] = {"race_id": "r0", "race_date": "2016-01-02"}
    assert audit_store(current, state, "horse", key)["status"] == "same_day"
    state["sources"][key] = {"race_id": "r3", "race_date": "2016-01-03"}
    assert audit_store(current, state, "horse", key)["status"] == "future"


def test_all_store_names_are_audited() -> None:
    df = add_target_columns(pl.DataFrame([row("r1", "e1", "2016-01-01")]))
    _out, _state, audit, _samples = build_pre_day_history_features_v2_1(df)
    assert set(audit) == set(runner.STORE_NAMES)


def test_strict_resume_detects_input_and_config_changes(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    year_dir = base / "year=2016"
    year_dir.mkdir(parents=True)
    parquet = year_dir / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(parquet)
    monkeypatch.setattr(runner, "BASE_DIR", base)
    good_fp = runner.file_fingerprint(parquet)
    checkpoint = {"years": {"2016": {"status": "complete", "input": good_fp, "feature_set_hash": "good", "script_version": runner.SCRIPT_VERSION, "state_version": runner.STATE_VERSION}}}
    assert runner.resume_mismatches(checkpoint, [2016], "good") == []
    parquet.write_bytes(parquet.read_bytes() + b"changed")
    assert runner.resume_mismatches(checkpoint, [2016], "good")
    checkpoint["years"]["2016"]["input"] = runner.file_fingerprint(parquet)
    assert runner.resume_mismatches(checkpoint, [2016], "changed")


def test_rebuild_from_year_invalidates_later_years() -> None:
    checkpoint = {"years": {str(y): {"status": "complete"} for y in range(2016, 2020)}}
    runner.invalidate_from_year(checkpoint, 2018)
    assert checkpoint["years"]["2017"]["status"] == "complete"
    assert checkpoint["years"]["2018"]["status"] == "invalidated"
    assert checkpoint["years"]["2019"]["status"] == "invalidated"


def test_v2_1_feature_sets_are_safe_and_nested() -> None:
    sets = feature_sets()
    assert validate_feature_sets()[0]["status"] == "pass"
    free = set(sets["market_free"]["numeric"] + sets["market_free"]["categorical"])
    hist = set(sets["market_history"]["numeric"] + sets["market_history"]["categorical"])
    aware = set(sets["market_aware"]["numeric"] + sets["market_aware"]["categorical"])
    assert "horse_last3_avg_ninki" not in free
    assert "horse_last5_avg_ninki" not in free
    assert "tan_odds" not in hist
    assert "TanVote" not in hist
    assert hist <= aware
