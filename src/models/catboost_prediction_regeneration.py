from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.models.catboost_config import sha256_data
from src.models.catboost_data import (
    filter_target,
    load_feature_sets,
    prepare_pandas,
    prediction_metadata,
    split_frame,
    validate_feature_set,
)
from src.models.catboost_metrics import validate_probabilities
from src.models.model_manifest import sha256_file


REQUIRED_SPLITS = ["train", "validation", "test", "latest_holdout"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_split_definition(config: dict[str, Any]) -> tuple[dict[str, list[int]], dict[int, str]]:
    resolved: dict[str, list[int]] = {}
    year_to_split: dict[int, str] = {}
    for split in REQUIRED_SPLITS:
        if split not in config["splits"]:
            raise ValueError(f"missing split: {split}")
        years = [int(y) for y in config["splits"][split]["years"]]
        if not years:
            raise ValueError(f"empty split: {split}")
        resolved[split] = years
        for year in years:
            if year in year_to_split:
                raise ValueError(f"year appears in multiple splits: {year}")
            year_to_split[year] = split
    return resolved, year_to_split


def load_dataset_with_split(input_dir: Path, split_by_year: dict[int, str]) -> pl.DataFrame:
    frames = []
    for year in sorted(split_by_year):
        path = input_dir / f"year={year}" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pl.read_parquet(path))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(pl.col("Year").map_elements(lambda y: split_by_year[int(y)], return_dtype=pl.String).alias("data_split"))


def split_validation_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in REQUIRED_SPLITS:
        part = df.filter(pl.col("data_split") == split)
        rows.append({
            "data_split": split,
            "rows": part.height,
            "races": part["race_id"].n_unique() if "race_id" in part.columns else None,
            "entry_ids": part["entry_id"].n_unique() if "entry_id" in part.columns else None,
        })
    for col in ["race_id", "entry_id"]:
        overlap = df.group_by(col).agg(pl.col("data_split").n_unique().alias("n")).filter(pl.col("n") > 1).height
        if overlap:
            raise ValueError(f"{col} overlaps across splits: {overlap}")
    return rows


def regenerate_predictions(
    model_path: Path,
    df: pl.DataFrame,
    target: str,
    feature_set: str,
    feature_columns: dict[str, list[str]],
) -> tuple[pl.DataFrame, bool]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    target_df = filter_target(df, target)
    groups = load_feature_sets()[feature_set]
    errors = validate_feature_set(df, feature_set, load_feature_sets())
    if errors:
        raise RuntimeError("; ".join(errors))
    if feature_columns != {"numeric": groups["numeric"], "categorical": groups["categorical"]}:
        raise RuntimeError("feature columns do not match current feature YAML")
    pred_frames = []
    sample_match = True
    for split, part in split_frame(target_df).items():
        if part.height == 0:
            continue
        x_part, _ = prepare_pandas(part, feature_columns["numeric"], feature_columns["categorical"])
        cat_features = [x_part.columns.get_loc(c) for c in feature_columns["categorical"]]
        probs = model.predict_proba(Pool(x_part, cat_features=cat_features))[:, 1]
        pred_frames.append(prediction_metadata(part, target, feature_set, probs))
        if split == "validation":
            sample_pool = Pool(x_part.head(min(100, len(x_part))), cat_features=cat_features)
            reloaded_probs = model.predict_proba(sample_pool)[:, 1]
            sample_match = bool(np.allclose(reloaded_probs, probs[: len(reloaded_probs)]))
    pred = pl.concat(pred_frames, how="diagonal_relaxed")
    if not validate_probabilities(pred):
        raise RuntimeError("prediction probability out of range")
    return pred, sample_match


def compare_predictions(old_path: Path, new_pred: pl.DataFrame, tolerance: float) -> dict[str, Any]:
    if not old_path.exists():
        return {"compared_rows": 0, "missing_in_old": new_pred.height, "missing_in_new": 0, "max_abs_diff": None, "mean_abs_diff": None, "p99_abs_diff": None, "mismatch_count": new_pred.height, "tolerance": tolerance}
    old = pl.read_parquet(old_path).select(["entry_id", pl.col("pred_probability").alias("old_pred_probability")])
    new = new_pred.select(["entry_id", pl.col("pred_probability").alias("new_pred_probability")])
    joined = old.join(new, on="entry_id", how="inner").with_columns((pl.col("old_pred_probability") - pl.col("new_pred_probability")).abs().alias("abs_diff"))
    missing_in_old = new.join(old, on="entry_id", how="anti").height
    missing_in_new = old.join(new, on="entry_id", how="anti").height
    if joined.height:
        vals = joined["abs_diff"].to_numpy()
        max_abs = float(vals.max())
        mean_abs = float(vals.mean())
        p99 = float(np.quantile(vals, 0.99))
        mismatch = int((vals > tolerance).sum())
    else:
        max_abs = mean_abs = p99 = None
        mismatch = 0
    return {
        "compared_rows": joined.height,
        "missing_in_old": missing_in_old,
        "missing_in_new": missing_in_new,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p99_abs_diff": p99,
        "mismatch_count": mismatch,
        "tolerance": tolerance,
    }


def source_paths(source: dict[str, Any], target: str, feature_set: str) -> dict[str, Path]:
    model_dir = Path(source["model_root"]) / target / feature_set
    output_root = Path(source["output_root"])
    return {
        "manifest": output_root / "run_manifest.json",
        "model": model_dir / "model.cbm",
        "metadata": model_dir / "model_metadata.json",
        "features": model_dir / "feature_columns.json",
        "categorical": model_dir / "categorical_columns.json",
        "prediction": output_root / "predictions" / f"{target}_{feature_set}.parquet",
    }


def verify_source(
    source: dict[str, Any],
    target: str,
    feature_set: str,
    current: dict[str, Any],
    params: dict[str, Any],
    split_sha: str,
) -> tuple[bool, list[str], dict[str, Path]]:
    paths = source_paths(source, target, feature_set)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return False, [f"missing source artifacts: {missing}"], paths
    reasons = []
    metadata = read_json(paths["metadata"])
    manifest = read_json(paths["manifest"])
    if metadata.get("target") != target or metadata.get("feature_set") != feature_set:
        reasons.append("target or feature_set mismatch")
    if metadata.get("input_dataset_fingerprint") != current["input_dataset_fingerprint"]:
        reasons.append("input fingerprint mismatch or missing")
    if metadata.get("feature_set_yaml_sha256") != current["feature_set_yaml_sha256"]:
        reasons.append("feature YAML hash mismatch or missing")
    if metadata.get("params") != params:
        reasons.append("training params mismatch")
    if read_json(paths["features"]) != current["feature_columns"]:
        reasons.append("feature columns mismatch")
    if read_json(paths["categorical"]).get("categorical") != current["feature_columns"]["categorical"]:
        reasons.append("categorical columns mismatch")
    if metadata.get("random_seed") != params.get("random_seed"):
        reasons.append("random seed mismatch")
    if metadata.get("task_type") != params.get("task_type"):
        reasons.append("task type mismatch")
    if source["name"] == "v1_0_1" and not manifest.get("training_config_resolved_sha256"):
        reasons.append("source manifest lacks resolved config hash")
    try:
        source_pred = pl.read_parquet(paths["prediction"], columns=["entry_id", "race_id", "data_split"])
        split_values = set(source_pred["data_split"].unique().to_list())
        if split_values != set(REQUIRED_SPLITS):
            reasons.append(f"source split values mismatch: {sorted(split_values)}")
        for col in ["entry_id", "race_id"]:
            overlap = source_pred.group_by(col).agg(pl.col("data_split").n_unique().alias("n")).filter(pl.col("n") > 1).height
            if overlap:
                reasons.append(f"source {col} overlaps across splits: {overlap}")
    except Exception as exc:
        reasons.append(f"source split proof unavailable: {exc}")
    if not paths["model"].exists():
        reasons.append("model file missing")
    try:
        CatBoostClassifier().load_model(paths["model"])
    except Exception as exc:
        reasons.append(f"model load failed: {exc}")
    return not reasons, reasons, paths


def copy_model_artifacts(source_paths_: dict[str, Path], dest_model_dir: Path) -> None:
    dest_model_dir.mkdir(parents=True, exist_ok=True)
    for key, name in [("model", "model.cbm"), ("features", "feature_columns.json"), ("categorical", "categorical_columns.json")]:
        shutil.copy2(source_paths_[key], dest_model_dir / name)
