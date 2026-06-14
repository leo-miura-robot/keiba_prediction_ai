from __future__ import annotations

import numpy as np
import polars as pl

from src.models.catboost_data import load_dataset, load_feature_sets, prepare_pandas, split_overlap_errors, validate_feature_set


def test_v2_1_1_feature_sets_exist_in_parquet() -> None:
    df = load_dataset([2016])
    feature_sets = load_feature_sets()
    assert set(feature_sets) == {"market_free", "market_history", "market_aware"}
    for name in feature_sets:
        assert validate_feature_set(df, name, feature_sets) == []


def test_splits_have_no_race_or_entry_overlap() -> None:
    assert split_overlap_errors(load_dataset([2016, 2024, 2025, 2026])) == []


def test_prepare_pandas_handles_missing_and_inf() -> None:
    df = pl.DataFrame({"num": [1.0, float("inf"), float("-inf")], "cat": ["", None, "A"]})
    pdf, cols = prepare_pandas(df, ["num"], ["cat"])
    assert np.isnan(pdf["num"].iloc[1])
    assert np.isnan(pdf["num"].iloc[2])
    assert pdf["cat"].tolist() == ["__MISSING__", "__MISSING__", "A"]


def test_market_free_has_no_market_columns() -> None:
    fs = load_feature_sets()
    cols = set(fs["market_free"]["numeric"] + fs["market_free"]["categorical"])
    assert "tan_odds" not in cols
    assert "horse_last3_avg_ninki" not in cols
