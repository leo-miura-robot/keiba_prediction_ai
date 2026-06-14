from __future__ import annotations

from pathlib import Path

import polars as pl

from src.models.catboost_resume import artifact_paths, decide_resume, feature_columns_hash, upsert_csv
from src.models.model_manifest import write_json


def _make_artifacts(tmp_path: Path, expected: dict) -> dict:
    model_root = tmp_path / "models"
    output_root = tmp_path / "outputs"
    paths = artifact_paths(model_root, output_root, "win", "market_free")
    for key in ["model", "prediction", "training_config_resolved"]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        if key == "prediction":
            pl.DataFrame({"entry_id": ["a"], "pred_probability": [0.2]}).write_parquet(paths[key])
        else:
            paths[key].write_text(key, encoding="utf-8")
    write_json(paths["features"], {"numeric": ["x"], "categorical": ["c"]})
    write_json(paths["categorical"], {"categorical": ["c"]})
    write_json(paths["metrics"], {"ok": True})
    from src.models.catboost_resume import add_artifact_hashes

    metadata = add_artifact_hashes(dict(expected), paths)
    write_json(paths["metadata"], metadata)
    return paths


def test_resume_fingerprint_match_skips(tmp_path: Path) -> None:
    expected = {
        "target": "win",
        "feature_set": "market_free",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "input_dataset_fingerprint": "input",
        "feature_set_yaml_sha256": "feature",
        "training_config_resolved_sha256": "config",
        "code_bundle_sha256": "code",
    }
    paths = _make_artifacts(tmp_path, expected)
    decision = decide_resume(paths, expected, strict=True)
    assert decision.action == "skip"


def test_resume_detects_input_feature_config_code_and_artifacts(tmp_path: Path) -> None:
    expected = {
        "target": "win",
        "feature_set": "market_free",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "input_dataset_fingerprint": "input",
        "feature_set_yaml_sha256": "feature",
        "training_config_resolved_sha256": "config",
        "code_bundle_sha256": "code",
    }
    paths = _make_artifacts(tmp_path, expected)
    for key in ["input_dataset_fingerprint", "feature_set_yaml_sha256", "training_config_resolved_sha256", "code_bundle_sha256"]:
        changed = dict(expected)
        changed[key] = "changed"
        assert decide_resume(paths, changed, strict=False).action == "train"
        assert decide_resume(paths, changed, strict=True).action == "error"
    paths["model"].unlink()
    assert decide_resume(paths, expected, strict=True).action == "error"


def test_git_sha_only_is_not_part_of_resume_decision(tmp_path: Path) -> None:
    expected = {
        "target": "win",
        "feature_set": "market_free",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "input_dataset_fingerprint": "input",
        "feature_set_yaml_sha256": "feature",
        "training_config_resolved_sha256": "config",
        "code_bundle_sha256": "code",
        "git_commit_sha": "a",
    }
    paths = _make_artifacts(tmp_path, expected)
    changed = dict(expected)
    changed["git_commit_sha"] = "b"
    assert decide_resume(paths, changed, strict=True).action == "skip"


def test_missing_metrics_and_predictions_detected(tmp_path: Path) -> None:
    expected = {
        "target": "win",
        "feature_set": "market_free",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": 42,
        "input_dataset_fingerprint": "input",
        "feature_set_yaml_sha256": "feature",
        "training_config_resolved_sha256": "config",
        "code_bundle_sha256": "code",
    }
    paths = _make_artifacts(tmp_path, expected)
    paths["metrics"].unlink()
    assert decide_resume(paths, expected, strict=False).action == "train"
    paths = _make_artifacts(tmp_path / "b", expected)
    paths["prediction"].unlink()
    assert decide_resume(paths, expected, strict=True).action == "error"


def test_upsert_csv_is_idempotent_and_unique(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    rows = [{"target": "win", "feature_set": "market_free", "data_split": "validation", "logloss": 0.1}]
    upsert_csv(path, rows, ["target", "feature_set", "data_split"])
    upsert_csv(path, rows, ["target", "feature_set", "data_split"])
    df = pl.read_csv(path)
    assert df.height == 1
    assert df.group_by(["target", "feature_set", "data_split"]).len().filter(pl.col("len") > 1).height == 0


def test_feature_columns_hash_changes_on_mismatch() -> None:
    a = feature_columns_hash({"numeric": ["x"], "categorical": ["c"]})
    b = feature_columns_hash({"numeric": ["x", "z"], "categorical": ["c"]})
    assert a != b
