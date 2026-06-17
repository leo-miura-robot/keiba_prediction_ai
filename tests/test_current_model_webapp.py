from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.data.normalization import (
    classify_source,
    grouped_roi,
    normalize_prediction_frame,
    race_summary,
    summarize_bets,
    tier_for_ev,
)
from webapp.data.repository import sqlite_tables_readonly
from webapp.data.schema import REQUIRED_NORMALIZED_COLUMNS


def fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_date": ["2026-01-01"] * 7,
            "race_id": ["R1"] * 7,
            "entry_id": [f"R1_{i}" for i in range(1, 8)],
            "JyoCD": ["05"] * 7,
            "RaceNum": [9] * 7,
            "Umaban": list(range(1, 8)),
            "Bamei": [""] + [f"Horse{i}" for i in range(2, 8)],
            "KettoNum": [f"K{i}" for i in range(1, 8)],
            "probability_raw": [0.50, 0.30, 0.20, 0.10, 0.20, 0.05, 0.04],
            "probability_official_platt": [0.55, 0.30, 0.20, 0.10, 0.20, 0.05, 0.04],
            "fuku_odds_low": [2.0, 1.5, 2.0, 2.0, 5.0, 3.0, 3.0],
            "fuku_odds_high": [2.4, 1.8, 2.6, 2.8, 6.0, 4.0, 4.0],
            "KakuteiJyuni": [1, 2, 3, 4, 5, 6, 7],
            "target_place_paid": [1, 1, 0, 0, 0, 0, 0],
            "fuku_pay": [180, 120, 0, 0, 0, 0, 0],
            "fuku_ninki": [1, 2, 3, 4, 5, 6, 7],
        }
    )


def normalized() -> pd.DataFrame:
    return normalize_prediction_frame(fixture_frame(), "outputs/example_validation.parquet")


def test_roi_calculation_and_fixed_100_yen_stake() -> None:
    df = normalized()
    summary = summarize_bets(df)
    assert summary["bets"] == 2
    assert summary["total_stake_yen"] == 200
    assert summary["total_payout_yen"] == 180
    assert summary["total_profit_yen"] == -20
    assert summary["roi"] == pytest.approx(90.0)


def test_zero_stake_roi_is_nan() -> None:
    df = normalized()
    df["selected_for_bet"] = False
    df["stake_yen"] = 0
    df["payout_yen"] = 0
    summary = summarize_bets(df)
    assert summary["bets"] == 0
    assert np.isnan(summary["roi"])


def test_target_place_paid_priority_prevents_third_place_error() -> None:
    df = normalized()
    third = df[df["actual_finish_position"].eq(3)].iloc[0]
    assert third["target_place_paid"] == 0
    assert third["payout_yen"] == 0


def test_fuku_pay_fallback_when_target_absent() -> None:
    raw = fixture_frame().drop(columns=["target_place_paid"])
    df = normalize_prediction_frame(raw, "outputs/example_validation.parquet")
    assert df.loc[df["Umaban"].eq(1), "target_place_paid"].iloc[0] == 1
    assert df.loc[df["Umaban"].eq(3), "target_place_paid"].iloc[0] == 0


def test_fixture_exclusion_and_source_classification() -> None:
    df = normalize_prediction_frame(fixture_frame(), "outputs/place_market_offset_forward_paper_phase6c_v2_fixture/predictions_export.parquet")
    assert classify_source("outputs/place_market_offset_forward_paper_phase6c_v2_fixture/predictions_export.parquet") == "FIXTURE"
    assert df["fixture"].all()
    assert df[~df["fixture"]].empty


def test_tier_filtering() -> None:
    df = normalized()
    assert tier_for_ev(1.16) == "VERY_HIGH"
    assert tier_for_ev(1.11) == "HIGH"
    assert tier_for_ev(1.06) == "MARGIN"
    assert tier_for_ev(1.01) == "CORE"
    assert (df[df["tier"].eq("CORE") | df["tier"].eq("VERY_HIGH")]["selected_for_bet"]).all()


def test_date_and_race_filtering() -> None:
    df = normalized()
    assert len(df[df["race_date"].eq("2026-01-01")]) == 7
    assert len(df[df["race_id"].eq("R1")]) == 7


def test_duplicate_rows_are_removed() -> None:
    raw = pd.concat([fixture_frame(), fixture_frame()], ignore_index=True)
    df = normalize_prediction_frame(raw, "outputs/example_validation.parquet")
    assert len(df) == 7


def test_sqlite_readonly_mode(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table predictions (id integer)")
    con.commit()
    con.close()
    tables = sqlite_tables_readonly(db)
    assert tables["predictions"]["row_count"] == 0
    uri = "file:" + str(db.resolve()).replace("\\", "/") + "?mode=ro"
    ro = sqlite3.connect(uri, uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("insert into predictions values (1)")
    ro.close()


def test_horse_name_fallback_and_schema() -> None:
    df = normalized()
    assert "馬番1 / K1" in df["horse_name"].tolist()
    assert REQUIRED_NORMALIZED_COLUMNS.issubset(set(df.columns))


def test_empty_data_summary() -> None:
    summary = summarize_bets(pd.DataFrame(columns=normalized().columns))
    assert summary["bets"] == 0
    assert summary["races"] == 0


def test_race_level_summary_and_dashboard_summary() -> None:
    df = normalized()
    races = race_summary(df)
    assert len(races) == 1
    assert races["selected_horses"].iloc[0] == 2
    assert races["actual_place_horses"].iloc[0] == 2
    assert races["profit_yen"].iloc[0] == -20
    by_tier = grouped_roi(df, "tier")
    assert not by_tier.empty
