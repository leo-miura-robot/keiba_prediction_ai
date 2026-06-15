from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_c1r0_feature_cleanup_v1 import candidate_features
from scripts.run_place_market_offset_catboost_c1r0_tree_count_v1 import fixed_params


def test_config_uses_separate_outputs_and_fixed300() -> None:
    with Path("config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["output_root"] == "outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1"
    assert cfg["model_root"] == "models/place_market_offset_catboost_c1r0_feature_cleanup_v1"
    assert cfg["base_fixed300_output_root"] == "outputs/place_market_offset_catboost_c1r0_tree_count_v1"
    assert cfg["fixed_tree_count"] == 300
    assert cfg["fixed_calibration_method"] == "isotonic"


def test_candidate_features_drop_only_requested_features() -> None:
    numeric = ["BaTaijyu", "trainer_past_starts", "Kyori"]
    cat = ["KisyuCode", "ChokyosiCode", "JyoCD"]
    n2, c2 = candidate_features(numeric, cat, ["KisyuCode", "ChokyosiCode"])
    assert n2 == numeric
    assert c2 == ["JyoCD"]
    n3, c3 = candidate_features(numeric, cat, ["BaTaijyu"])
    assert n3 == ["trainer_past_starts", "Kyori"]
    assert c3 == cat


def test_fixed_params_keeps_hyperparameters_except_tree_count_and_early_stop() -> None:
    base = {
        "iterations": 3000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "loss_function": "Logloss",
        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": 42,
    }
    params = fixed_params(base, 300, gpu_ram_part=0.75)
    assert params["iterations"] == 300
    assert params["learning_rate"] == base["learning_rate"]
    assert params["depth"] == base["depth"]
    assert params["l2_leaf_reg"] == base["l2_leaf_reg"]
    assert params["gpu_ram_part"] == 0.75
    assert "od_type" not in params
    assert "od_wait" not in params


def test_ablation_candidates_are_expected_small_set() -> None:
    with Path("config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert set(cfg["ablation_candidates"]) == {
        "drop_person_codes",
        "drop_global_cumulative_starts",
        "drop_raw_body_weight",
        "drop_unadjusted_raw_time",
        "drop_meeting_admin",
    }
