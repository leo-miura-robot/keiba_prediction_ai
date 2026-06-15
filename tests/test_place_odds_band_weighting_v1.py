from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_place_odds_band_weighting_v1 import assign_odds_band, bootstrap_ci, summarize_bets, top_removed_roi


def test_odds_bands_are_exclusive() -> None:
    cfg = {"odds_bands": [
        {"label": "1.0-1.1", "min": 1.0, "max": 1.1},
        {"label": "1.1-1.2", "min": 1.1, "max": 1.2},
        {"label": "3.0+", "min": 3.0, "max": 999.0},
    ]}
    df = pd.DataFrame({"fuku_odds_low": [1.0, 1.099, 1.1, 3.0]})
    out = assign_odds_band(df, cfg)
    assert list(out["place_odds_band"]) == ["1.0-1.1", "1.0-1.1", "1.1-1.2", "3.0+"]


def test_roi_uses_actual_fuku_pay() -> None:
    bets = pd.DataFrame({
        "fuku_pay": [120, 0],
        "race_id": ["r1", "r2"],
        "actual_place": [1, 0],
        "conservative_probability": [0.8, 0.8],
        "fuku_odds_low": [9.9, 9.9],
        "fuku_odds_high": [9.9, 9.9],
    })
    row = summarize_bets(bets)
    assert row["return"] == 120
    assert np.isclose(row["roi"], 60)


def test_top_removed_roi() -> None:
    bets = pd.DataFrame({
        "fuku_pay": [400, 200, 0],
        "race_id": ["a", "b", "c"],
        "actual_place": [1, 1, 0],
        "conservative_probability": [0.5, 0.5, 0.5],
        "fuku_odds_low": [2, 2, 2],
        "fuku_odds_high": [2, 2, 2],
    })
    assert np.isclose(top_removed_roi(bets, 1), 100)


def test_bootstrap_ci_order() -> None:
    bets = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "fuku_pay": [100, 0, 300]})
    lo, mid, hi = bootstrap_ci(bets, 20, 1)
    assert lo <= mid <= hi
