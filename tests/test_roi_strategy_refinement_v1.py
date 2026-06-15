from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_roi_strategy_refinement_v1 import (
    add_place_edge_cols,
    bootstrap_ci,
    remove_similar_rules,
    summarize_bets,
    top_payout_removed_roi,
)


def test_place_edge_uses_low_odds_break_even() -> None:
    df = pd.DataFrame({"fuku_odds_low": [1.25], "conservative_probability": [0.85]})
    out = add_place_edge_cols(df)
    assert np.isclose(out["break_even_probability"].iloc[0], 0.8)
    assert np.isclose(out["place_edge_low"].iloc[0], 0.05)
    assert np.isclose(out["place_ev_low"].iloc[0], 1.0625)


def test_roi_uses_actual_fuku_pay_not_low_odds() -> None:
    bets = pd.DataFrame({"fuku_pay": [110, 0], "fuku_odds_low": [9.9, 9.9], "race_id": ["r1", "r2"]})
    row = summarize_bets(bets, "place")
    assert row["return"] == 110
    assert np.isclose(row["roi"], 55.0)


def test_top_payout_removal() -> None:
    bets = pd.DataFrame({"tan_pay": [1000, 200, 0], "tan_odds": [10, 2, 5], "race_id": ["a", "b", "c"]})
    assert np.isclose(top_payout_removed_roi(bets, "win", 1), 100.0)


def test_bootstrap_ci_order() -> None:
    bets = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "tan_pay": [100, 0, 300]})
    lo, mid, hi = bootstrap_ci(bets, "win", 20, 7)
    assert lo <= mid <= hi


def test_jaccard_pruning_keeps_distinct_rules(monkeypatch) -> None:
    rules = pd.DataFrame({"target": ["win", "win"], "strategy_type": ["win_core", "win_core"]})
    calls = [pd.DataFrame({"entry_id": ["a", "b"]}), pd.DataFrame({"entry_id": ["c", "d"]})]
    import scripts.run_roi_strategy_refinement_v1 as mod

    def fake_apply_rule(_df, _rule):
        return calls.pop(0)

    monkeypatch.setattr(mod, "apply_rule", fake_apply_rule)
    kept, overlap = remove_similar_rules(rules, pd.DataFrame(), {"jaccard_duplicate_threshold": 0.8})
    assert len(kept) == 2
    assert not overlap.empty
