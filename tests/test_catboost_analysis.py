from __future__ import annotations

import polars as pl

from src.models.catboost_analysis import calibration_bins_v101, market_comparison_rows, market_probability_frame


def _pred(feature_set: str) -> pl.DataFrame:
    return pl.DataFrame({
        "entry_id": ["r1_1", "r1_2", "r2_1", "r2_2"],
        "race_id": ["r1", "r1", "r2", "r2"],
        "data_split": ["validation", "validation", "test", "test"],
        "actual": [1, 0, 0, 1],
        "eligible": [True, True, True, True],
        "tan_odds": [2.0, 4.0, 3.0, 3.0],
        "pred_probability": [0.6, 0.4, 0.3, 0.7] if feature_set == "market_free" else [0.55, 0.45, 0.4, 0.6],
    })


def test_market_comparison_same_entry_set_and_metrics() -> None:
    same, summary = market_probability_frame({
        "market_free": _pred("market_free"),
        "market_history": _pred("market_history"),
        "market_aware": _pred("market_aware"),
    })
    assert same.height == 4
    assert summary["rows"] == 4
    assert summary["races"] == 2
    sums = same.group_by("race_id").agg(pl.col("market_probability").sum().alias("s"))
    assert all(abs(v - 1.0) < 1e-12 for v in sums["s"].to_list())
    rows = market_comparison_rows(same)
    assert {r["model"] for r in rows} == {"market_probability", "catboost_market_free", "catboost_market_history", "catboost_market_aware"}
    assert all(r["rows"] in {2} for r in rows)
    assert all(r["races"] == 1 for r in rows)


def test_calibration_fixed_width_and_quantile_counts() -> None:
    pred = pl.DataFrame({
        "data_split": ["validation"] * 20,
        "actual": [0, 1] * 10,
        "pred_probability": [i / 20 for i in range(20)],
    })
    fixed = calibration_bins_v101(pred, "fixed_width", bins=10)
    quantile = calibration_bins_v101(pred, "quantile", bins=10)
    assert sum(r["count"] for r in fixed) == 20
    assert sum(r["count"] for r in quantile) == 20
    assert {r["bin_type"] for r in fixed} == {"fixed_width"}
    assert {r["bin_type"] for r in quantile} == {"quantile"}


def test_calibration_quantile_handles_many_identical_values() -> None:
    pred = pl.DataFrame({
        "data_split": ["validation"] * 12,
        "actual": [0, 1] * 6,
        "pred_probability": [0.2] * 12,
    })
    rows = calibration_bins_v101(pred, "quantile", bins=10)
    assert sum(r["count"] for r in rows) == 12
    assert len(rows) <= 10
