from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.roi_validation_v2_1_2_v1 import (
    add_race_confidence,
    apply_rule,
    ece_mce,
    payout_dependency,
    summarize_roi,
)


def test_roi_uses_actual_payout_not_odds() -> None:
    bets = pd.DataFrame({
        "tan_pay": [250, 0],
        "tan_odds": [2.0, 99.0],
    })
    row = summarize_roi(bets, "win")
    assert row["stake"] == 200
    assert row["return"] == 250
    assert row["roi"] == 125.0


def test_place_roi_uses_fuku_pay() -> None:
    bets = pd.DataFrame({
        "fuku_pay": [110, 0, 300],
        "fuku_odds_low": [1.1, 2.0, 3.0],
    })
    row = summarize_roi(bets, "place")
    assert row["return"] == 410
    assert round(row["roi"], 6) == round(410 / 300 * 100, 6)


def test_race_confidence_adds_margin_entropy_and_agreement() -> None:
    df = pd.DataFrame({
        "target": ["win"] * 6,
        "feature_set": ["market_free"] * 2 + ["market_history"] * 2 + ["market_aware"] * 2,
        "race_id": ["r1"] * 6,
        "entry_id": ["a", "b"] * 3,
        "calibrated_probability": [0.6, 0.4, 0.7, 0.3, 0.2, 0.8],
    })
    out = add_race_confidence(df)
    assert {"top1_probability", "top1_minus_top2_margin", "prediction_entropy", "model_agreement_count"} <= set(out.columns)
    assert out[(out["feature_set"] == "market_free") & (out["entry_id"] == "a")]["model_agreement_count"].iloc[0] == 2
    assert np.isfinite(out["prediction_entropy"]).all()


def test_apply_rule_is_split_fixed_and_uses_validation_selected_values() -> None:
    df = pd.DataFrame({
        "target": ["win", "win"],
        "feature_set": ["market_history", "market_history"],
        "data_split": ["test", "validation"],
        "model_rank_in_race": [1, 1],
        "ev": [1.2, 1.2],
        "calibrated_probability": [0.2, 0.2],
        "tan_odds": [6.0, 6.0],
        "top1_minus_top2_margin": [0.05, 0.05],
        "model_agreement_count": [2, 2],
        "prediction_entropy": [0.4, 0.4],
        "market_gap": [0.01, 0.01],
    })
    rule = pd.Series({
        "target": "win", "feature_set": "market_history", "top_n": 1,
        "ev_min": 1.1, "probability_min": 0.1, "odds_min": 3.0, "odds_max": 10.0,
        "margin_min": 0.02, "model_agreement_min": 2, "entropy_max": 0.5,
        "market_gap_min": 0.0,
    })
    assert len(apply_rule(df, rule, "test")) == 1
    assert len(apply_rule(df, rule, "latest_holdout")) == 0


def test_ece_mce_basic() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    ece, mce = ece_mce(y, p, bins=2)
    assert 0 <= ece <= 1
    assert 0 <= mce <= 1


def test_payout_dependency_reports_top_removal() -> None:
    bets = pd.DataFrame({"tan_pay": [1000, 200, 0], "tan_odds": [10.0, 2.0, 5.0]})
    rows = payout_dependency(bets, "win", {"rule_id": "r"})
    assert {r["removed_top_payouts"] for r in rows} >= {0, 1, 3, 5, 10, "dependency"}
