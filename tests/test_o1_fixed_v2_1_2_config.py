from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import scripts.build_full_runner_dataset_o1_fixed as base_runner
import scripts.build_model_features_v2_1_2 as features_runner
from src.features.feature_sets_v2_1_2 import load_feature_set_yaml, validate_feature_sets_from_file


def test_o1_fixed_base_config_points_to_versioned_outputs() -> None:
    cfg = base_runner.load_simple_yaml(Path("config/base_runner_dataset_o1_fixed.yaml"))
    assert cfg["database"]["path"].endswith("new_jra_2016-2026_fixed/keiba.db")
    assert cfg["database"]["mode"] == "read_only"
    assert cfg["outputs"]["dataset_dir"] == "outputs/base_runner_dataset_o1_fixed"
    assert "base_runner_dataset_o1_fixed" in cfg["outputs"]["checkpoint"]


def test_v2_1_2_feature_config_points_to_versioned_outputs() -> None:
    cfg = features_runner.load_simple_yaml(Path("config/model_features_v2_1_2.yaml"))
    assert cfg["input"]["base_dataset_dir"] == "outputs/base_runner_dataset_o1_fixed"
    assert cfg["outputs"]["dataset_dir"] == "outputs/model_feature_dataset_v2_1_2"
    assert cfg["feature_sets"]["yaml"] == "config/feature_sets_v2_1_2.yaml"
    assert features_runner.split_by_year_from_config(cfg)[2026] == "latest_holdout"


def test_v2_1_2_feature_sets_validate_and_keep_market_boundaries() -> None:
    path = Path("config/feature_sets_v2_1_2.yaml")
    rows = validate_feature_sets_from_file(path)
    assert all(row["status"] == "pass" for row in rows)
    sets = load_feature_set_yaml(path)
    free = set(sets["market_free"]["numeric"] + sets["market_free"]["categorical"])
    history = set(sets["market_history"]["numeric"] + sets["market_history"]["categorical"])
    aware = set(sets["market_aware"]["numeric"] + sets["market_aware"]["categorical"])
    current_market = {"tan_odds", "tan_ninki", "fuku_odds_low", "fuku_odds_high", "fuku_ninki", "TanVote", "FukuVote"}
    assert not (free & current_market)
    assert not (history & current_market)
    assert current_market <= aware


def test_fixed_db_read_only_and_fingerprint_if_available() -> None:
    db_path = Path("D:/keiba/new_jra_2016-2026_fixed/keiba.db")
    if not db_path.exists():
        pytest.skip("fixed DB is not available")
    fp = base_runner.db_fingerprint(db_path)
    assert fp["size"] > 0
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        assert con.execute("SELECT name FROM sqlite_master WHERE name='NL_O1'").fetchone() is not None
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE codex_readonly_probe(x INTEGER)")
    finally:
        con.close()
