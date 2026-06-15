from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1 import candidate_features, choose_starts


def test_config_outputs_and_bootstrap_settings() -> None:
    with Path("config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["output_root"] == "outputs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1"
    assert cfg["model_root"] == "models/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1"
    assert cfg["fixed_tree_count"] == 300
    assert cfg["bootstrap"]["n_bootstrap"] == 5000
    assert cfg["fixed_calibration_method"] == "isotonic"


def test_monthday_is_not_in_phase3_drop_sets() -> None:
    with Path("config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for spec in cfg["additional_ablation_candidates"].values():
        assert "MonthDay" not in spec["drop_features"]


def test_additional_ablation_drops_person_codes_plus_only_requested_group() -> None:
    numeric = ["Kaiji", "Nichiji", "RaceNum", "horse_last3_avg_time", "horse_last5_avg_time", "BaTaijyu", "MonthDay", "Kyori"]
    cat = ["KisyuCode", "ChokyosiCode", "JyoCD"]
    n2, c2 = candidate_features(numeric, cat, ["KisyuCode", "ChokyosiCode", "Kaiji", "Nichiji", "RaceNum"])
    assert "MonthDay" in n2
    assert "Kaiji" not in n2
    assert c2 == ["JyoCD"]


def test_choose_starts_keeps_raw_when_ci_is_ambiguous() -> None:
    summary = pd.DataFrame([
        {"unit": "race_id", "metric": "delta_logloss", "ci_lower": -0.001, "ci_upper": 0.001, "point_estimate_delta": -0.0001},
        {"unit": "race_id", "metric": "delta_brier", "ci_lower": -0.001, "ci_upper": 0.001, "point_estimate_delta": -0.0001},
    ])
    by_year = pd.DataFrame({"metric": ["delta_logloss"] * 5, "point_estimate_delta": [-1, 1, -1, 1, -1]})
    decision = choose_starts(summary, by_year)
    assert decision["selected_starts"] == "raw"
