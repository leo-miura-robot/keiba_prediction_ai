from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_final_odds_two_models_v1 import (
    add_market_and_confidence,
    bootstrap_ci,
    remove_similar_rules,
    summarize_bets,
    threshold_for_odds,
)


def test_market_probability_normalizes_by_race() -> None:
    df = pd.DataFrame({
        "race_id": ["r1", "r1"],
        "entry_id": ["a", "b"],
        "calibrated_probability": [0.6, 0.4],
        "tan_odds": [2.0, 4.0],
        "TanNinki": [1, 2],
    })
    out = add_market_and_confidence(df, "win", 0.5)
    assert np.isclose(out["normalized_market_probability"].sum(), 1.0)
    assert "edge" in out.columns


def test_roi_uses_actual_pay() -> None:
    bets = pd.DataFrame({"tan_pay": [300, 0], "tan_odds": [2.0, 100.0], "race_id": ["r1", "r2"]})
    row = summarize_bets(bets, "win")
    assert row["return"] == 300
    assert row["roi"] == 150.0


def test_odds_thresholds_are_band_dependent() -> None:
    cfg = {"rule_selection": {"odds_thresholds": {"win": [
        {"min_odds": 1, "max_odds": 5, "ev_min": 1.03},
        {"min_odds": 5, "max_odds": 999, "ev_min": 1.20},
    ]}}}
    s = threshold_for_odds(cfg, "win", pd.Series([2.0, 10.0]))
    assert list(s) == [1.03, 1.20]


def test_bootstrap_uses_race_arrays() -> None:
    bets = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "fuku_pay": [100, 0, 200]})
    lo, mid, hi = bootstrap_ci(bets, "place", 20, 1)
    assert lo <= mid <= hi


def test_remove_similar_rules_keeps_non_overlapping(monkeypatch) -> None:
    rules = pd.DataFrame({"target": ["win", "win"], "strategy_type": ["core", "core"]})
    candidates = pd.DataFrame()
    calls = [pd.DataFrame({"entry_id": ["a", "b"]}), pd.DataFrame({"entry_id": ["c", "d"]})]
    import scripts.run_final_odds_two_models_v1 as mod

    def fake_apply_rule(_c, _r):
        return calls.pop(0)

    monkeypatch.setattr(mod, "apply_rule", fake_apply_rule)
    kept = remove_similar_rules(rules, candidates, 0.8)
    assert len(kept) == 2
