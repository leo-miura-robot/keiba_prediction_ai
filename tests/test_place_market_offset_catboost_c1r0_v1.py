from __future__ import annotations

from pathlib import Path

import yaml

from scripts.run_place_market_offset_catboost_c1r0_v1 import build_c1r0_features, c1r0_exclusion_reason


def load_cfg() -> dict:
    with Path("config/place_market_offset_catboost_c1r0_v1.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_forbidden_features_are_excluded() -> None:
    for feature in ["Year", "p_market", "market_logit", "tan_odds", "fuku_odds_low", "fuku_ninki", "race_id", "entry_id", "KettoNum", "target_place_paid", "fuku_pay"]:
        included, _, _ = c1r0_exclusion_reason(feature)
        assert included is False


def test_allowlist_excludes_market_and_ids() -> None:
    cfg = load_cfg()
    dataset_columns = {"Year", "JyoCD", "Kyori", "p_market", "market_logit", "tan_odds", "race_id", "entry_id", "target_place_paid"}
    numeric, cat, exclusion = build_c1r0_features(cfg, dataset_columns)
    features = set(numeric + cat)
    assert "Year" not in features
    assert "p_market" not in features
    assert "market_logit" not in features
    assert "tan_odds" not in features
    assert "race_id" not in features
    assert "entry_id" not in features
    assert "target_place_paid" not in features
    assert "JyoCD" in features
    assert "Kyori" in features
    assert set(exclusion.columns) == {"feature", "present_in_dataset", "present_in_c1", "included_in_c1r0", "reason", "category"}


def test_config_uses_separate_outputs() -> None:
    cfg = load_cfg()
    assert cfg["output_root"] == "outputs/place_market_offset_catboost_c1r0_v1"
    assert cfg["model_root"] == "models/place_market_offset_catboost_c1r0_v1"
    assert cfg["current_c1_output_dir"] == "outputs/place_market_offset_catboost_v1"
