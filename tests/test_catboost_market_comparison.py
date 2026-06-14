from __future__ import annotations

import polars as pl

from src.models.catboost_market_comparison import build_complete_market_frame, complete_market_metrics, exclusion_summary


def _pred(fs: str) -> pl.DataFrame:
    base = pl.DataFrame({
        "entry_id": ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"],
        "race_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "data_split": ["validation"] * 8,
        "actual": [1, 0, 1, 0, 0, 0, 1, 0],
        "eligible": [True] * 8,
        "tan_odds": [2.0, 3.0, None, 4.0, 2.0, 3.0, 2.0, 3.0],
        "pred_probability": [0.6, 0.4, 0.7, 0.3, 0.2, 0.1, 0.8, 0.2],
    })
    if fs == "market_history":
        return base.filter(pl.col("entry_id") != "d2")
    return base


def test_complete_market_race_filters_missing_odds_prediction_and_winner() -> None:
    complete, race_status, excluded = build_complete_market_frame({
        "market_free": _pred("market_free"),
        "market_history": _pred("market_history"),
        "market_aware": _pred("market_aware"),
    })
    assert set(complete["race_id"].to_list()) == {"a"}
    reasons = {r["race_id"]: r["market_exclusion_reason"] for r in excluded.to_dicts()}
    assert reasons["b"] == "missing_odds_runner"
    assert reasons["c"] == "missing_winner"
    assert reasons["d"] == "missing_model_prediction"
    sums = complete.group_by("race_id").agg(pl.col("market_probability").sum().alias("s"))
    assert abs(sums["s"][0] - 1.0) <= 1e-10
    summary = exclusion_summary(excluded)
    assert sum(r["races"] for r in summary) == 3


def test_complete_market_metrics_same_sample() -> None:
    complete, _, _ = build_complete_market_frame({
        "market_free": _pred("market_free").filter(pl.col("race_id") == "a"),
        "market_history": _pred("market_history").filter(pl.col("race_id") == "a"),
        "market_aware": _pred("market_aware").filter(pl.col("race_id") == "a"),
    })
    rows = complete_market_metrics(complete)
    assert {r["model"] for r in rows} == {"market_probability", "catboost_market_free", "catboost_market_history", "catboost_market_aware"}
    assert all(r["rows"] == 2 and r["races"] == 1 for r in rows)
