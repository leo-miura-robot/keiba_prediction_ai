from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.run_place_market_offset_champion_challenger_phase5c_v1 as phase5c
import scripts.run_place_market_offset_year_strategy_phase5b_v2 as phase5b


def test_champion_challenger_windows_are_fixed() -> None:
    cfg = phase5c.load_yaml(phase5c.ROOT / "config/place_market_offset_champion_challenger_phase5c_v1.yaml")
    windows = phase5b.build_windows(cfg, ["ROLLING_10Y", "ROLLING_15Y"], [2025, 2026])
    by_key = {(w.strategy, w.validation_year): w for w in windows}
    assert by_key[("ROLLING_10Y", 2025)].train_years == tuple(range(2015, 2025))
    assert by_key[("ROLLING_10Y", 2026)].train_years == tuple(range(2016, 2026))
    assert by_key[("ROLLING_15Y", 2025)].train_years == tuple(range(2010, 2025))
    assert by_key[("ROLLING_15Y", 2026)].train_years == tuple(range(2011, 2026))
    assert all(w.validation_year not in w.train_years for w in windows)


def test_pair_merge_requires_one_to_one_keys_and_same_target() -> None:
    base = pd.DataFrame(
        {
            "entry_id": ["e1", "e1"],
            "race_id": ["r1", "r1"],
            "race_date": ["2025-01-01", pd.Timestamp("2025-01-01")],
            "Year": [2025, 2025],
            "actual_place": [1, 1],
            "probability_raw": [0.4, 0.5],
            "catboost_residual_score": [0.1, 0.2],
            "strategy": ["ROLLING_10Y", "ROLLING_15Y"],
        }
    )
    merged = phase5c.merged_pair(base)
    assert len(merged) == 1
    assert merged["actual_place"].iloc[0] == 1
    bad = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        phase5c.merged_pair(bad)


def test_direct_bootstrap_is_race_unit_and_prefers_lower_loss() -> None:
    rows = []
    for race in range(4):
        for horse in range(3):
            y = int(horse == 0)
            rows.append(
                {
                    "entry_id": f"e{race}_{horse}",
                    "race_id": f"r{race}",
                    "race_date": "2025-01-01",
                    "Year": 2025,
                    "actual_place_10y": y,
                    "actual_place_15y": y,
                    "actual_place": y,
                    "probability_raw_10y": 0.8 if y else 0.1,
                    "probability_raw_15y": 0.5,
                }
            )
    out = phase5c.direct_bootstrap(pd.DataFrame(rows), iterations=50, seed=42)
    ll = out[(out["Year"].eq("2025_2026")) & (out["metric"].eq("logloss"))].iloc[0]
    assert ll["delta_10y_minus_15y"] < 0
    assert ll["champion_10y_better_probability"] == 1.0
    assert ll["n_bootstrap"] == 50


def test_roi_overlap_and_payout_zeroed_invariant() -> None:
    cfg = {
        "stake_yen": 100,
        "odds_column": "fuku_odds_low",
        "payout_column": "fuku_pay",
    }
    rows = []
    for strategy in ["ROLLING_10Y", "ROLLING_15Y"]:
        for i in range(3):
            rows.append(
                {
                    "strategy": strategy,
                    "Year": 2025,
                    "race_id": f"r{i}",
                    "entry_id": f"e{i}",
                    "race_date": "2025-01-01",
                    "actual_place": int(i == 0),
                    "probability_raw": 0.8,
                    "fuku_odds_low": 2.0,
                    "fuku_pay": 300 if i == 0 else 0,
                }
            )
    pred = pd.DataFrame(rows)
    overlap = phase5c.bet_overlap(pred, cfg)
    assert overlap.loc[overlap["Year"].eq("2025_2026"), "common_bets"].iloc[0] == 3
    _year, _combined, _rr, pz = phase5c.roi_tables(pred, cfg)
    assert np.all(pz["roi"].fillna(-np.inf) <= pz["normal_roi"].fillna(np.inf) + 1e-12)
