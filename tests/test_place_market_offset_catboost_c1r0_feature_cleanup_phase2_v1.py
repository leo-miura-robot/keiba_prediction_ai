from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1 import candidate_features, transform_params, transform_starts


def test_config_uses_new_outputs_and_fixed300() -> None:
    with Path("config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["output_root"] == "outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1"
    assert cfg["model_root"] == "models/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1"
    assert cfg["fixed_tree_count"] == 300
    assert cfg["fixed_calibration_method"] == "isotonic"


def test_monthday_ablation_changes_only_monthday_plus_working_base_drops() -> None:
    numeric = ["MonthDay", "trainer_past_starts", "jockey_past_starts", "Kyori"]
    cat = ["KisyuCode", "ChokyosiCode", "JyoCD"]
    n2, c2 = candidate_features(numeric, cat, ["KisyuCode", "ChokyosiCode", "MonthDay"])
    assert n2 == ["trainer_past_starts", "jockey_past_starts", "Kyori"]
    assert c2 == ["JyoCD"]


def test_cumulative_starts_transform_targets_only_two_columns() -> None:
    df = pd.DataFrame({
        "trainer_past_starts": [0, 10, 100],
        "jockey_past_starts": [1, 20, 200],
        "horse_surface_past_starts": [5, 6, 7],
    })
    out = transform_starts(df, ["trainer_past_starts", "jockey_past_starts"], "log1p")
    assert out["horse_surface_past_starts"].tolist() == [5, 6, 7]
    assert out["trainer_past_starts"].iloc[0] == 0
    assert out["jockey_past_starts"].iloc[2] < df["jockey_past_starts"].iloc[2]


def test_p99_params_are_computed_from_provided_train_frame_only() -> None:
    train = pd.DataFrame({"trainer_past_starts": [1, 2, 3, 100], "jockey_past_starts": [2, 4, 6, 200]})
    params, rows = transform_params(train, ["trainer_past_starts", "jockey_past_starts"], "clip_p99", {"fold_eval_year": 2024})
    assert set(params) == {"trainer_past_starts", "jockey_past_starts"}
    assert len(rows) == 2
    assert params["trainer_past_starts"] < 100
    assert params["jockey_past_starts"] < 200
