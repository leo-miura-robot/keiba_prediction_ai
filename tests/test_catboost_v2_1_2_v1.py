from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.analyze_catboost_baseline_v2_1_2_v1 import calibration_bins, complete_win_races
from scripts.train_catboost_baseline_v2_1_2_v1 import feature_columns_for, gpu_smoke_check
from src.models.catboost_config import load_yaml_config


def test_v2_1_2_config_uses_versioned_inputs_outputs() -> None:
    cfg = load_yaml_config(Path("config/catboost_baseline_v2_1_2_v1.yaml"))
    assert cfg["input_dataset_dir"] == "outputs/model_feature_dataset_v2_1_2"
    assert cfg["feature_set_yaml"] == "config/feature_sets_v2_1_2.yaml"
    assert cfg["output_root"].endswith("catboost_baseline_v2_1_2_v1")
    assert cfg["model_root"].endswith("catboost_baseline_v2_1_2_v1")
    assert "final NL_O1 odds" in cfg["market_aware_notice"]


def test_v2_1_2_feature_columns_are_loaded_from_v2_1_2_yaml() -> None:
    cols = feature_columns_for("market_aware", Path("config/feature_sets_v2_1_2.yaml"))
    assert "tan_odds" in cols["numeric"]
    assert "fuku_odds_low" in cols["numeric"]
    assert "fuku_odds_high" in cols["numeric"]


def test_gpu_smoke_rejects_cpu_fallback() -> None:
    try:
        gpu_smoke_check("CPU", None)
    except RuntimeError as exc:
        assert "CPU fallback is disabled" in str(exc)
    else:
        raise AssertionError("CPU fallback must be rejected")


def test_quantile_bins_do_not_split_identical_predictions() -> None:
    pred = pl.DataFrame({
        "data_split": ["validation"] * 20,
        "actual": [0, 1] * 10,
        "pred_probability": [0.1] * 7 + [0.2] * 6 + [0.9] * 7,
    })
    rows = calibration_bins(pred, "quantile", requested=4)
    seen: dict[float, set[int]] = {}
    for row in rows:
        # All bins contain a single constant or a contiguous range. Reconstructing
        # from bounds is enough for the equal-value test data.
        if row["lower_bound"] == row["upper_bound"]:
            seen.setdefault(row["lower_bound"], set()).add(row["bin_id"])
    assert all(len(v) == 1 for v in seen.values())


def test_complete_win_races_excludes_incomplete_odds() -> None:
    base = pl.DataFrame({
        "entry_id": ["a", "b", "c", "d"],
        "race_id": ["r1", "r1", "r2", "r2"],
        "data_split": ["validation"] * 4,
        "actual": [1, 0, 1, 0],
        "eligible": [True] * 4,
        "tan_odds": [2.0, 3.0, None, 5.0],
        "pred_probability": [0.6, 0.4, 0.7, 0.3],
    })
    preds = {
        "market_free": base,
        "market_history": base.with_columns((pl.col("pred_probability") + 0.01).alias("pred_probability")),
        "market_aware": base.with_columns((pl.col("pred_probability") + 0.02).alias("pred_probability")),
    }
    complete, excluded = complete_win_races(preds)
    assert complete["race_id"].unique().to_list() == ["r1"]
    assert excluded.filter(pl.col("race_id") == "r2")["exclusion_reason"][0] == "invalid_odds_runner"
