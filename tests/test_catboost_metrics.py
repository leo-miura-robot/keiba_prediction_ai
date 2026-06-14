from __future__ import annotations

import numpy as np
import polars as pl

from src.models.catboost_metrics import probability_metrics, race_metrics, validate_probabilities


def test_probability_metrics_known_values() -> None:
    m = probability_metrics(np.array([0, 1]), np.array([0.25, 0.75]))
    assert round(m["brier"], 5) == 0.0625
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_race_metrics_win_and_place() -> None:
    pred = pl.DataFrame({
        "race_id": ["r1", "r1", "r1", "r2", "r2"],
        "entry_id": ["a", "b", "c", "d", "e"],
        "data_split": ["validation"] * 5,
        "actual": [0, 1, 0, 1, 0],
        "pred_probability": [0.1, 0.9, 0.2, 0.4, 0.3],
        "place_rank_limit": [2, 2, 2, 1, 1],
    })
    win = race_metrics(pred, "win")[0]
    assert win["top1_winner_accuracy"] == 1.0
    place = race_metrics(pred, "place")[0]
    assert place["precision_at_k"] == 0.75


def test_probability_range_validation() -> None:
    ok = pl.DataFrame({"pred_probability": [0.0, 0.5, 1.0]})
    bad = pl.DataFrame({"pred_probability": [-0.1, 1.1]})
    assert validate_probabilities(ok)
    assert not validate_probabilities(bad)
