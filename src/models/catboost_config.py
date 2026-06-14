from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_CLI_OVERRIDES = {"task_type", "devices", "smoke_test"}
REQUIRED_TOP_LEVEL = {
    "version",
    "input_dataset_dir",
    "feature_set_yaml",
    "output_root",
    "model_root",
    "random_seed",
    "splits",
    "phase1_goal",
    "training_params",
}
REQUIRED_TRAINING_PARAMS = {
    "iterations",
    "learning_rate",
    "depth",
    "loss_function",
    "eval_metric",
    "random_seed",
    "task_type",
    "allow_writing_files",
}


class ConfigError(ValueError):
    pass


def load_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("config must be a mapping")
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ConfigError(f"missing top-level keys: {missing}")
    params = config.get("training_params")
    if not isinstance(params, dict):
        raise ConfigError("training_params must be a mapping")
    missing_params = sorted(REQUIRED_TRAINING_PARAMS - set(params))
    if missing_params:
        raise ConfigError(f"missing training params: {missing_params}")
    if str(params["task_type"]).upper() not in {"GPU", "CPU"}:
        raise ConfigError("training_params.task_type must be GPU or CPU")
    if int(params["iterations"]) <= 0:
        raise ConfigError("iterations must be positive")
    if float(params["learning_rate"]) <= 0:
        raise ConfigError("learning_rate must be positive")
    splits = config["splits"]
    if not isinstance(splits, dict) or not splits.get("train") or not splits.get("validation"):
        raise ConfigError("splits must contain train and validation years")
    goal = config["phase1_goal"]
    if sorted(goal) != ["place_roi_min", "win_roi_min"]:
        raise ConfigError("phase1_goal must contain win_roi_min and place_roi_min")


def resolve_training_params(
    config: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    params = copy.deepcopy(config["training_params"])
    cli_overrides = cli_overrides or {}
    illegal = sorted(set(cli_overrides) - ALLOWED_CLI_OVERRIDES)
    if illegal:
        raise ConfigError(f"CLI override is not allowed: {illegal}")
    if "task_type" in cli_overrides and cli_overrides["task_type"]:
        params["task_type"] = cli_overrides["task_type"]
    if "devices" in cli_overrides and cli_overrides["devices"] is not None:
        params["devices"] = str(cli_overrides["devices"])
    if str(params.get("task_type", "")).upper() == "CPU":
        params.pop("devices", None)
    if smoke_test:
        params.update(copy.deepcopy(config.get("smoke_overrides", {})))
    return params


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_data(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def write_resolved_config(path: Path, params: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return sha256_data(params)
