from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.models.catboost_prediction_regeneration import compare_predictions, resolve_split_definition


def test_compare_predictions_all_rows_and_tolerance(tmp_path: Path) -> None:
    old_path = tmp_path / "old.parquet"
    pl.DataFrame({"entry_id": ["a", "b"], "pred_probability": [0.1, 0.2]}).write_parquet(old_path)
    new = pl.DataFrame({"entry_id": ["a", "b", "c"], "pred_probability": [0.1, 0.2000000002, 0.3]})
    stats = compare_predictions(old_path, new, 1e-10)
    assert stats["compared_rows"] == 2
    assert stats["missing_in_old"] == 1
    assert stats["missing_in_new"] == 0
    assert stats["mismatch_count"] == 1


def test_split_definition_detects_duplicate_year() -> None:
    config = {
        "splits": {
            "train": {"years": [2016, 2017]},
            "validation": {"years": [2017]},
            "test": {"years": [2018]},
            "latest_holdout": {"years": [2019]},
        }
    }
    with pytest.raises(ValueError):
        resolve_split_definition(config)


def test_split_definition_requires_all_splits() -> None:
    with pytest.raises(ValueError):
        resolve_split_definition({"splits": {"train": {"years": [2016]}}})
