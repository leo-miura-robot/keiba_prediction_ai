from __future__ import annotations

from pathlib import Path

import pytest

from src.models.catboost_config import ConfigError, load_yaml_config, resolve_training_params, sha256_data


def test_v101_yaml_loads_and_resolves() -> None:
    config = load_yaml_config(Path("config/catboost_baseline_v1_0_1.yaml"))
    params = resolve_training_params(config)
    assert params["iterations"] == config["training_params"]["iterations"]
    assert params["learning_rate"] == config["training_params"]["learning_rate"]
    assert params["task_type"] == "GPU"
    assert sha256_data(params)


def test_yaml_values_reflect_in_real_params() -> None:
    config = load_yaml_config(Path("config/catboost_baseline_v1_0_1.yaml"))
    config["training_params"]["depth"] = 6
    config["training_params"]["devices"] = "1"
    params = resolve_training_params(config)
    assert params["depth"] == 6
    assert params["devices"] == "1"


def test_invalid_config_stops() -> None:
    config = load_yaml_config(Path("config/catboost_baseline_v1_0_1.yaml"))
    del config["training_params"]["iterations"]
    with pytest.raises(ConfigError):
        from src.models.catboost_config import validate_config

        validate_config(config)


def test_cli_override_restriction() -> None:
    config = load_yaml_config(Path("config/catboost_baseline_v1_0_1.yaml"))
    with pytest.raises(ConfigError):
        resolve_training_params(config, {"depth": 4})
    params = resolve_training_params(config, {"task_type": "CPU", "devices": "0"})
    assert params["task_type"] == "CPU"
    assert "devices" not in params


def test_smoke_overrides_are_applied() -> None:
    config = load_yaml_config(Path("config/catboost_baseline_v1_0_1.yaml"))
    params = resolve_training_params(config, smoke_test=True)
    assert params["iterations"] == config["smoke_overrides"]["iterations"]
    assert params["verbose"] is False
