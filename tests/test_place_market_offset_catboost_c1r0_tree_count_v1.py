from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_place_market_offset_catboost_c1r0_tree_count_v1 import fixed_params


def test_config_uses_separate_outputs() -> None:
    with Path("config/place_market_offset_catboost_c1r0_tree_count_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["output_root"] == "outputs/place_market_offset_catboost_c1r0_tree_count_v1"
    assert cfg["model_root"] == "models/place_market_offset_catboost_c1r0_tree_count_v1"
    assert cfg["base_c1r0_output_root"] == "outputs/place_market_offset_catboost_c1r0_v1"


def test_fixed_params_only_changes_iterations_and_disables_early_stop_for_fixed_count() -> None:
    base = {
        "iterations": 3000,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "Logloss",
        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": 42,
    }
    params = fixed_params(base, 350)
    assert params["iterations"] == 350
    assert params["learning_rate"] == base["learning_rate"]
    assert params["depth"] == base["depth"]
    assert params["loss_function"] == base["loss_function"]
    assert params["random_seed"] == base["random_seed"]
    assert "od_type" not in params
    assert "od_wait" not in params


def test_gpu_ram_part_is_execution_constraint_when_requested() -> None:
    base = {
        "iterations": 3000,
        "learning_rate": 0.05,
        "depth": 8,
        "loss_function": "Logloss",
        "od_type": "Iter",
        "od_wait": 200,
        "random_seed": 42,
    }
    params = fixed_params(base, 300, gpu_ram_part=0.75)
    assert params["iterations"] == 300
    assert params["gpu_ram_part"] == 0.75
    assert params["learning_rate"] == 0.05


def test_candidate_tree_counts_are_required_set() -> None:
    with Path("config/place_market_offset_catboost_c1r0_tree_count_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["candidate_tree_counts"] == [250, 300, 350, 400, 450]
