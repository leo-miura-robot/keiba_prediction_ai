from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.models.catboost_config import sha256_data
from src.models.model_manifest import sha256_file


FINGERPRINT_KEYS = [
    "target",
    "feature_set",
    "task_type",
    "devices",
    "random_seed",
    "input_dataset_fingerprint",
    "feature_set_yaml_sha256",
    "training_config_resolved_sha256",
    "split_definition_sha256",
    "code_bundle_sha256",
]
ARTIFACT_KEYS = [
    "model_file_sha256",
    "predictions_file_sha256",
    "metrics_file_sha256",
    "training_config_resolved_file_sha256",
]


@dataclass
class ResumeDecision:
    action: str
    reasons: list[str]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths(model_root: Path, output_root: Path, target: str, feature_set: str) -> dict[str, Path]:
    model_dir = model_root / target / feature_set
    return {
        "model": model_dir / "model.cbm",
        "metrics": model_dir / "metrics.json",
        "metadata": model_dir / "model_metadata.json",
        "features": model_dir / "feature_columns.json",
        "categorical": model_dir / "categorical_columns.json",
        "training_config_resolved": model_dir / "training_config_resolved.json",
        "prediction": output_root / "predictions" / f"{target}_{feature_set}.parquet",
    }


def complete_artifacts(paths: dict[str, Path]) -> list[str]:
    required = ["model", "metrics", "metadata", "features", "categorical", "training_config_resolved", "prediction"]
    return [name for name in required if not paths[name].exists()]


def add_artifact_hashes(fingerprint: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    out = dict(fingerprint)
    if paths["model"].exists():
        out["model_file_sha256"] = sha256_file(paths["model"])
    if paths["prediction"].exists():
        out["predictions_file_sha256"] = sha256_file(paths["prediction"])
    if paths["metrics"].exists():
        out["metrics_file_sha256"] = sha256_file(paths["metrics"])
    if paths["training_config_resolved"].exists():
        out["training_config_resolved_file_sha256"] = sha256_file(paths["training_config_resolved"])
    return out


def fingerprint_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches = []
    for key in FINGERPRINT_KEYS:
        if expected.get(key) != actual.get(key):
            mismatches.append(key)
    return mismatches


def artifact_hash_mismatches(actual: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    current = add_artifact_hashes({}, paths)
    mismatches = []
    for key in ARTIFACT_KEYS:
        if actual.get(key) != current.get(key):
            mismatches.append(key)
    return mismatches


def decide_resume(paths: dict[str, Path], expected: dict[str, Any], strict: bool = False) -> ResumeDecision:
    missing = complete_artifacts(paths)
    if missing:
        return ResumeDecision("error" if strict else "train", [f"missing artifacts: {missing}"])
    metadata = read_json(paths["metadata"])
    mismatches = fingerprint_mismatches(expected, metadata) + artifact_hash_mismatches(metadata, paths)
    if mismatches:
        return ResumeDecision("error" if strict else "train", [f"fingerprint mismatch: {sorted(mismatches)}"])
    return ResumeDecision("skip", ["fingerprint and artifacts match"])


def upsert_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old = pl.read_csv(path)
        combined = pl.concat([old, new], how="diagonal_relaxed")
    else:
        combined = new
    combined = combined.unique(subset=keys, keep="last", maintain_order=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.write_csv(tmp)
    tmp.replace(path)


def copy_artifacts(source_paths: dict[str, Path], dest_paths: dict[str, Path]) -> None:
    for key in ["model", "metrics", "metadata", "features", "categorical", "prediction"]:
        dest_paths[key].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths[key], dest_paths[key])


def feature_columns_hash(columns: dict[str, list[str]]) -> str:
    return sha256_data({"numeric": columns.get("numeric", []), "categorical": columns.get("categorical", [])})
