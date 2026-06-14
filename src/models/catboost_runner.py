from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.models.catboost_data import (
    TARGETS,
    all_missing_numeric_columns,
    class_balance_rows,
    filter_target,
    load_dataset,
    load_feature_sets,
    prepare_pandas,
    prediction_metadata,
    split_frame,
    split_overlap_errors,
    validate_feature_set,
)
from src.models.catboost_metrics import (
    calibration_bins,
    metrics_by_split,
    probability_sum_diagnostics,
    race_metrics,
    validate_probabilities,
)
from src.models.model_manifest import write_json


OUTPUT_ROOT = Path("outputs/model_training/catboost_baseline_v1")
MODEL_ROOT = Path("models/catboost_baseline_v1")


DEFAULT_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "od_type": "Iter",
    "od_wait": 200,
    "random_seed": 42,
    "task_type": "GPU",
    "devices": "0",
    "allow_writing_files": False,
    "verbose": 100,
}


def model_name(target: str, feature_set: str) -> str:
    return f"catboost_{target}_{feature_set}_v1"


def model_dir(target: str, feature_set: str) -> Path:
    return MODEL_ROOT / target / feature_set


def output_paths(target: str, feature_set: str) -> dict[str, Path]:
    return {
        "prediction": OUTPUT_ROOT / "predictions" / f"{target}_{feature_set}.parquet",
        "metrics": model_dir(target, feature_set) / "metrics.json",
        "model": model_dir(target, feature_set) / "model.cbm",
    }


def completed(target: str, feature_set: str) -> bool:
    paths = output_paths(target, feature_set)
    return all(p.exists() for p in paths.values())


def make_params(task_type: str, smoke_test: bool = False) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params["task_type"] = task_type
    if task_type.upper() == "CPU":
        params.pop("devices", None)
    if smoke_test:
        params["iterations"] = 30
        params["od_wait"] = 10
        params["verbose"] = False
    return params


def train_one(target: str, feature_set: str, task_type: str = "GPU", smoke_test: bool = False, force: bool = False) -> dict[str, Any]:
    if not force and not smoke_test and completed(target, feature_set):
        return {"target": target, "feature_set": feature_set, "status": "skipped_complete"}
    feature_sets = load_feature_sets()
    df = load_dataset()
    errors = validate_feature_set(df, feature_set, feature_sets) + split_overlap_errors(df)
    if errors:
        raise RuntimeError("; ".join(errors))
    df = filter_target(df, target)
    if smoke_test:
        df = pl.concat([
            df.filter(pl.col("data_split") == "train").head(4000),
            df.filter(pl.col("data_split") == "validation").head(1000),
            df.filter(pl.col("data_split") == "test").head(1000),
            df.filter(pl.col("data_split") == "latest_holdout").head(1000),
        ], how="diagonal_relaxed")
    groups = feature_sets[feature_set]
    numeric_cols = groups["numeric"]
    categorical_cols = groups["categorical"]
    missing_all = all_missing_numeric_columns(df, numeric_cols)
    splits = split_frame(df)
    train_df = splits["train"]
    valid_df = splits["validation"]
    if train_df.height == 0 or valid_df.height == 0:
        raise RuntimeError("train/validation split is empty")
    x_train, _ = prepare_pandas(train_df, numeric_cols, categorical_cols)
    x_valid, _ = prepare_pandas(valid_df, numeric_cols, categorical_cols)
    y_train = train_df["__target__"].to_numpy()
    y_valid = valid_df["__target__"].to_numpy()
    cat_features = [x_train.columns.get_loc(c) for c in categorical_cols]
    train_pool = Pool(x_train, y_train, cat_features=cat_features)
    valid_pool = Pool(x_valid, y_valid, cat_features=cat_features)
    params = make_params(task_type, smoke_test)
    started = time.time()
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    training_seconds = time.time() - started

    pred_frames = []
    prediction_seconds = 0.0
    for split, part in splits.items():
        if part.height == 0:
            continue
        x_part, _ = prepare_pandas(part, numeric_cols, categorical_cols)
        pool = Pool(x_part, cat_features=cat_features)
        ps = time.time()
        probs = model.predict_proba(pool)[:, 1]
        prediction_seconds += time.time() - ps
        pred_frames.append(prediction_metadata(part, target, feature_set, probs))
    predictions = pl.concat(pred_frames, how="diagonal_relaxed")
    if not validate_probabilities(predictions):
        raise RuntimeError("prediction probability out of range")

    metrics_rows = []
    for row in metrics_by_split(predictions):
        row.update({"target": target, "feature_set": feature_set})
        metrics_rows.append(row)
    race_rows = []
    for row in race_metrics(predictions, target):
        row.update({"feature_set": feature_set})
        race_rows.append(row)
    prob_sum_rows = []
    for row in probability_sum_diagnostics(predictions):
        row.update({"target": target, "feature_set": feature_set})
        prob_sum_rows.append(row)
    cal_rows = []
    for row in calibration_bins(predictions):
        row.update({"target": target, "feature_set": feature_set})
        cal_rows.append(row)

    mdir = model_dir(target, feature_set)
    mdir.mkdir(parents=True, exist_ok=True)
    pdir = OUTPUT_ROOT / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    model_path = mdir / "model.cbm"
    model.save_model(model_path)
    reloaded = CatBoostClassifier()
    reloaded.load_model(model_path)
    sample_pool = Pool(x_valid.head(min(100, len(x_valid))), cat_features=cat_features)
    if not np.allclose(model.predict_proba(sample_pool)[:, 1], reloaded.predict_proba(sample_pool)[:, 1]):
        raise RuntimeError("model reload prediction mismatch")
    predictions.write_parquet(output_paths(target, feature_set)["prediction"], compression="zstd")
    importances = model.get_feature_importance(prettified=True)
    fi = pl.DataFrame({
        "target": [target] * len(importances),
        "feature_set": [feature_set] * len(importances),
        "feature": importances["Feature Id"].astype(str).tolist(),
        "importance": importances["Importances"].astype(float).tolist(),
    })
    summary = {
        "target": target,
        "feature_set": feature_set,
        "model_name": model_name(target, feature_set),
        "best_iteration": int(model.get_best_iteration() or 0),
        "params": params,
        "train_rows": train_df.height,
        "validation_rows": valid_df.height,
        "test_rows": splits["test"].height,
        "latest_holdout_rows": splits["latest_holdout"].height,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "model_path": str(model_path),
        "all_missing_numeric_columns": missing_all,
        "metrics_by_split": metrics_rows,
        "race_metrics": race_rows,
        "probability_sum_diagnostics": prob_sum_rows,
        "model_reload_prediction_match": True,
    }
    write_json(mdir / "metrics.json", summary)
    write_json(mdir / "model_metadata.json", summary)
    write_json(mdir / "feature_columns.json", {"numeric": numeric_cols, "categorical": categorical_cols})
    write_json(mdir / "categorical_columns.json", {"categorical": categorical_cols})
    write_json(mdir / "training_config.json", params)
    return {
        "summary": summary,
        "metrics_rows": metrics_rows,
        "race_rows": race_rows,
        "prob_sum_rows": prob_sum_rows,
        "calibration_rows": cal_rows,
        "feature_importance": fi,
        "class_balance_rows": class_balance_rows(df, target, feature_set),
        "predictions": predictions,
    }
