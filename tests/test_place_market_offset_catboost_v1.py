from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_place_market_offset_catboost_v1 import (
    EV_LABELS,
    add_eval_columns,
    logit,
    residual_features,
    roi_of,
    summarize_bets,
)


def test_logit_clipping_keeps_finite_values() -> None:
    out = logit(np.array([0.0, 0.5, 1.0]), 1e-6)
    assert np.isfinite(out).all()
    assert out[0] < 0
    assert out[-1] > 0


def test_c1_excludes_raw_market_odds() -> None:
    numeric, cat = residual_features(
        {},
        "fundamental",
        ["horse_past_starts", "tan_odds", "fuku_odds_low", "fuku_odds_high", "fuku_ninki"],
        ["JyoCD"],
    )
    assert "horse_past_starts" in numeric
    assert "p_market" in numeric
    assert "market_logit" in numeric
    assert "tan_odds" not in numeric
    assert "fuku_odds_low" not in numeric
    assert cat == ["JyoCD"]


def test_c2_adds_only_limited_market_features() -> None:
    numeric, _ = residual_features({}, "limited_market", ["horse_past_starts"], [])
    for col in ["market_rank", "p_market_rank", "rank_gap", "SyussoTosu", "place_rank_limit"]:
        assert col in numeric
    assert "fuku_odds_low" not in numeric


def test_ev_band_and_roi_use_actual_fuku_pay() -> None:
    df = pd.DataFrame({
        "final_probability": [0.5, 0.25],
        "fuku_odds_low": [2.0, 5.0],
        "fuku_pay": [200, 0],
        "actual_place": [1, 0],
    })
    out = add_eval_columns(df, "final_probability")
    assert list(out["adjusted_place_ev"]) == [1.0, 1.25]
    assert set(out["ev_band"]).issubset(set(EV_LABELS))
    assert roi_of(df) == 100.0
    assert summarize_bets(df, {})["max_losing_streak"] == 1
