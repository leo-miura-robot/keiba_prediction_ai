from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.features.feature_sets_v2_1_2 import load_feature_set_yaml
from src.models.catboost_atomic_output import atomic_write_text
from src.models.catboost_config import load_yaml_config, resolve_training_params, sha256_data, write_resolved_config
from src.models.catboost_data import all_missing_numeric_columns, filter_target, prepare_pandas, prediction_metadata, split_frame
from src.models.catboost_metrics import metrics_by_split, probability_sum_diagnostics, race_metrics, validate_probabilities
from src.models.catboost_prediction_regeneration import load_dataset_with_split, resolve_split_definition, split_validation_rows
from src.models.catboost_resume import add_artifact_hashes, artifact_paths, decide_resume, feature_columns_hash
from src.models.model_manifest import dataset_fingerprints, git_info, hash_files, package_versions, sha256_file, write_json


CONFIG_PATH = Path("config/catboost_baseline_v2_1_2_v1.yaml")
LOG_PATH = Path("logs/train_catboost_baseline_v2_1_2_v1.log")
TARGETS = ["win", "place"]
FEATURE_SETS = ["market_free", "market_history", "market_aware"]
MODEL_COMBOS = [(target, fs) for target in TARGETS for fs in FEATURE_SETS]
CODE_FILES = [
    Path("scripts/train_catboost_baseline_v2_1_2_v1.py"),
    Path("scripts/analyze_catboost_baseline_v2_1_2_v1.py"),
    Path("src/features/feature_sets_v2_1_2.py"),
    Path("src/models/catboost_atomic_output.py"),
    Path("src/models/catboost_config.py"),
    Path("src/models/catboost_resume.py"),
    Path("src/models/catboost_data.py"),
    Path("src/models/catboost_metrics.py"),
    Path("src/models/catboost_prediction_regeneration.py"),
    Path("src/models/model_manifest.py"),
    CONFIG_PATH,
    Path("config/feature_sets_v2_1_2.yaml"),
]


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_catboost_baseline_v2_1_2_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def git_status_summary() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def gpu_name(devices: str | None) -> str:
    try:
        idx = (devices or "0").split(",")[0]
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={idx}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return out or "unknown"
    except Exception as exc:
        return f"unknown: {exc}"


def gpu_smoke_check(task_type: str, devices: str | None) -> dict[str, Any]:
    if task_type.upper() != "GPU":
        raise RuntimeError("GPU task_type is required; CPU fallback is disabled")
    from sklearn.datasets import make_classification

    x, y = make_classification(n_samples=256, n_features=12, random_state=42)
    model = CatBoostClassifier(
        iterations=4,
        task_type="GPU",
        devices=devices or "0",
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        random_seed=42,
    )
    model.fit(x, y)
    return {
        "task_type": "GPU",
        "gpu_required": True,
        "devices": devices or "0",
        "gpu_name": gpu_name(devices),
        "catboost_gpu_smoke": "ok",
        "cpu_fallback_used": False,
    }


def load_feature_sets(path: Path) -> dict[str, dict[str, list[str]]]:
    return load_feature_set_yaml(path)


def validate_feature_set(df: pl.DataFrame, feature_set_name: str, feature_sets: dict[str, dict[str, list[str]]]) -> list[str]:
    from src.features.feature_sets_v2_1_2 import CURRENT_MARKET_COLUMNS, LEAKAGE_COLUMNS, MARKET_HISTORY_COLUMNS, RAW_MARKET_COLUMNS

    errors: list[str] = []
    groups = feature_sets.get(feature_set_name)
    if groups is None:
        return [f"unknown feature_set={feature_set_name}"]
    columns = groups.get("numeric", []) + groups.get("categorical", [])
    missing = [c for c in columns if c not in df.columns]
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    forbidden = sorted(set(columns) & (LEAKAGE_COLUMNS | RAW_MARKET_COLUMNS | {
        "race_id", "entry_id", "Bamei", "KettoNum", "race_date",
        "win_training_exclusion_reason", "place_training_exclusion_reason", "ranking_training_exclusion_reason",
    }))
    if missing:
        errors.append(f"missing columns: {missing}")
    if duplicated:
        errors.append(f"duplicated columns: {duplicated}")
    if forbidden:
        errors.append(f"forbidden columns: {forbidden}")
    if feature_set_name == "market_free":
        bad = sorted(set(columns) & (CURRENT_MARKET_COLUMNS | MARKET_HISTORY_COLUMNS | RAW_MARKET_COLUMNS))
        if bad:
            errors.append(f"market_free contains market columns: {bad}")
    if feature_set_name == "market_history":
        bad = sorted(set(columns) & (CURRENT_MARKET_COLUMNS | RAW_MARKET_COLUMNS))
        if bad:
            errors.append(f"market_history contains current market columns: {bad}")
    if feature_set_name == "market_aware":
        history_cols = set(feature_sets["market_history"].get("numeric", []) + feature_sets["market_history"].get("categorical", []))
        if not history_cols <= set(columns):
            errors.append(f"market_aware missing market_history columns: {sorted(history_cols - set(columns))}")
    return errors


def feature_columns_for(feature_set: str, feature_set_path: Path) -> dict[str, list[str]]:
    groups = load_feature_sets(feature_set_path)[feature_set]
    return {"numeric": groups.get("numeric", []), "categorical": groups.get("categorical", [])}


def eligible_counts(df: pl.DataFrame, target: str, feature_set: str) -> dict[str, Any]:
    target_df = filter_target(df, target)
    out: dict[str, Any] = {"eligible_rows": target_df.height}
    if feature_set == "market_aware" and target == "win":
        complete = target_df.filter((pl.col("tan_odds").is_not_null()) & (pl.col("tan_odds") > 0) & (pl.col("tan_ninki").is_not_null()))
        out["market_complete_rows"] = complete.height
    elif feature_set == "market_aware" and target == "place":
        complete = target_df.filter(
            (pl.col("fuku_odds_low").is_not_null()) & (pl.col("fuku_odds_low") > 0)
            & (pl.col("fuku_odds_high").is_not_null()) & (pl.col("fuku_odds_high") > 0)
        )
        out["market_complete_rows"] = complete.height
    else:
        out["market_complete_rows"] = None
    return out


def train_one(
    df: pl.DataFrame,
    target: str,
    feature_set: str,
    feature_sets: dict[str, dict[str, list[str]]],
    params: dict[str, Any],
    model_dir: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    errors = validate_feature_set(df, feature_set, feature_sets)
    if errors:
        raise RuntimeError("; ".join(errors))
    target_df = filter_target(df, target)
    groups = feature_sets[feature_set]
    numeric_cols = groups["numeric"]
    categorical_cols = groups["categorical"]
    splits = split_frame(target_df)
    train_df = splits["train"]
    valid_df = splits["validation"]
    if train_df.height == 0 or valid_df.height == 0:
        raise RuntimeError(f"empty train/validation for {target} {feature_set}")
    x_train, _ = prepare_pandas(train_df, numeric_cols, categorical_cols)
    x_valid, _ = prepare_pandas(valid_df, numeric_cols, categorical_cols)
    cat_features = [x_train.columns.get_loc(c) for c in categorical_cols]
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(
        Pool(x_train, train_df["__target__"].to_numpy(), cat_features=cat_features),
        eval_set=Pool(x_valid, valid_df["__target__"].to_numpy(), cat_features=cat_features),
        use_best_model=True,
    )
    training_seconds = time.time() - started
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.cbm"
    model.save_model(model_path)
    write_json(model_dir / "feature_columns.json", {"numeric": numeric_cols, "categorical": categorical_cols})
    write_json(model_dir / "categorical_columns.json", {"categorical": categorical_cols})
    pred_frames = []
    prediction_seconds = 0.0
    for split, part in splits.items():
        x_part, _ = prepare_pandas(part, numeric_cols, categorical_cols)
        ps = time.time()
        probs = model.predict_proba(Pool(x_part, cat_features=cat_features))[:, 1]
        prediction_seconds += time.time() - ps
        pred_frames.append(prediction_metadata(part, target, feature_set, probs))
    pred = pl.concat(pred_frames, how="diagonal_relaxed")
    if not validate_probabilities(pred):
        raise RuntimeError("prediction probability out of range")
    return pred, {
        "best_iteration": int(model.get_best_iteration() or 0),
        "tree_count": int(model.tree_count_ or 0),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "all_missing_numeric_columns": all_missing_numeric_columns(target_df, numeric_cols),
        **eligible_counts(df, target, feature_set),
    }


def expected_fingerprint(
    target: str,
    feature_set: str,
    params: dict[str, Any],
    input_fp: list[dict[str, Any]],
    feature_set_path: Path,
    resolved_sha: str,
    split_sha: str,
    code_hash: str,
) -> dict[str, Any]:
    cols = feature_columns_for(feature_set, feature_set_path)
    target_definition = {"win": "target_win_paid", "place": "target_place_paid"}[target]
    return {
        "target": target,
        "feature_set": feature_set,
        "target_definition": target_definition,
        "task_type": params.get("task_type"),
        "devices": params.get("devices"),
        "random_seed": params.get("random_seed"),
        "input_dataset_fingerprint": sha256_data(input_fp),
        "feature_set_yaml_sha256": sha256_file(feature_set_path),
        "feature_columns_sha256": feature_columns_hash(cols),
        "training_config_resolved_sha256": resolved_sha,
        "split_definition_sha256": split_sha,
        "code_bundle_sha256": code_hash,
    }


def model_metrics_summary(pred: pl.DataFrame, target: str, feature_set: str) -> dict[str, Any]:
    return {
        "metrics_by_split": [{**r, "target": target, "feature_set": feature_set} for r in metrics_by_split(pred)],
        "race_metrics": [{**r, "feature_set": feature_set} for r in race_metrics(pred, target)],
        "probability_sum_diagnostics": [{**r, "target": target, "feature_set": feature_set} for r in probability_sum_diagnostics(pred)],
    }


def write_prediction(path: Path, pred: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pred.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def finalize_model_metadata(
    model_root: Path,
    output_root: Path,
    model_dir: Path,
    target: str,
    feature_set: str,
    params: dict[str, Any],
    pred: pl.DataFrame,
    train_meta: dict[str, Any],
    fingerprint: dict[str, Any],
    versions: dict[str, Any],
    git: dict[str, Any],
    phase1_goal: dict[str, Any],
    notice: str,
) -> dict[str, Any]:
    pred_path = output_root / "predictions" / f"{target}_{feature_set}.parquet"
    write_prediction(pred_path, pred)
    summary = {
        "target": target,
        "feature_set": feature_set,
        "model_name": f"catboost_{target}_{feature_set}_v2_1_2_v1",
        "model_origin": "newly_trained_v2_1_2",
        "old_model_reused": False,
        "params": params,
        "train_rows": pred.filter(pl.col("data_split") == "train").height,
        "validation_rows": pred.filter(pl.col("data_split") == "validation").height,
        "test_rows": pred.filter(pl.col("data_split") == "test").height,
        "latest_holdout_rows": pred.filter(pl.col("data_split") == "latest_holdout").height,
        "model_path": str(model_dir / "model.cbm"),
        "prediction_path": str(pred_path),
        "phase1_goal": phase1_goal,
        "market_aware_notice": notice,
        **train_meta,
        **model_metrics_summary(pred, target, feature_set),
    }
    write_json(model_dir / "metrics.json", summary)
    meta = {
        **summary,
        **fingerprint,
        "python_version": versions.get("python"),
        "catboost_version": versions.get("catboost"),
        "numpy_version": versions.get("numpy"),
        "pandas_version": versions.get("pandas"),
        "sklearn_version": versions.get("sklearn"),
        **git,
        "git_status_summary": git_status_summary(),
    }
    paths = artifact_paths(model_root, output_root, target, feature_set)
    meta = add_artifact_hashes(meta, paths)
    write_json(model_dir / "model_metadata.json", meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--feature-set", choices=FEATURE_SETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-type", choices=["GPU", "CPU"])
    parser.add_argument("--devices")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logger = setup_logger()
    config = load_yaml_config(CONFIG_PATH)
    suffix = "_smoke" if args.smoke_test else ""
    output_root = Path(config["output_root"] + suffix)
    model_root = Path(config["model_root"] + suffix)
    params = resolve_training_params(config, {"task_type": args.task_type, "devices": args.devices}, smoke_test=args.smoke_test)
    try:
        gpu_info = gpu_smoke_check(str(params["task_type"]), params.get("devices"))
        logger.info("device check: %s", gpu_info)
    except Exception:
        logger.exception("GPU smoke check failed; not falling back to CPU")
        return 2
    combos = MODEL_COMBOS if args.all else [(args.target, args.feature_set)]
    if not args.all and (not args.target or not args.feature_set):
        raise SystemExit("--target and --feature-set are required unless --all is used")
    split_def, split_by_year = resolve_split_definition(config)
    split_sha = sha256_data(split_def)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_root / "split_definition_resolved.json", json.dumps(split_def, ensure_ascii=False, indent=2, sort_keys=True))
    resolved_sha = write_resolved_config(output_root / "training_config_resolved.json", params)
    input_fp = dataset_fingerprints(Path(config["input_dataset_dir"]))
    code_hash = hash_files(CODE_FILES)
    versions = package_versions()
    git = git_info(ROOT)
    git["git_status_summary"] = git_status_summary()
    feature_set_path = Path(config["feature_set_yaml"])
    feature_sets = load_feature_sets(feature_set_path)
    df = load_dataset_with_split(Path(config["input_dataset_dir"]), split_by_year)
    if args.smoke_test:
        df = pl.concat([
            df.filter(pl.col("data_split") == "train").head(6000),
            df.filter(pl.col("data_split") == "validation").head(1500),
            df.filter(pl.col("data_split") == "test").head(1500),
            df.filter(pl.col("data_split") == "latest_holdout").head(1500),
        ], how="diagonal_relaxed")
    write_json(output_root / "split_summary_internal.json", {"rows": split_validation_rows(df)})
    manifest = {
        "version": config["version"] + suffix,
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase1_goal": config["phase1_goal"],
        "market_aware_notice": config.get("market_aware_notice"),
        "training_config_resolved_sha256": resolved_sha,
        "split_definition_sha256": split_sha,
        "input_dataset_fingerprint": sha256_data(input_fp),
        "feature_set_yaml_sha256": sha256_file(feature_set_path),
        "code_bundle_sha256": code_hash,
        "gpu_info": gpu_info,
        "versions": versions,
        **git,
        "models": [],
    }
    for target, feature_set in combos:
        logger.info("train target=%s feature_set=%s", target, feature_set)
        model_dir = model_root / target / feature_set
        paths = artifact_paths(model_root, output_root, target, feature_set)
        model_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(model_dir / "training_config_resolved.json", params)
        fingerprint = expected_fingerprint(target, feature_set, params, input_fp, feature_set_path, resolved_sha, split_sha, code_hash)
        if not args.force and (args.resume or args.strict_resume):
            decision = decide_resume(paths, fingerprint, strict=args.strict_resume)
            logger.info("resume decision %s %s %s %s", target, feature_set, decision.action, decision.reasons)
            if decision.action == "skip":
                meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
                manifest["models"].append({"target": target, "feature_set": feature_set, "status": "skipped_resume", "best_iteration": meta.get("best_iteration")})
                continue
            if decision.action == "error":
                write_json(output_root / "run_manifest.json", manifest)
                return 2
        pred, train_meta = train_one(df, target, feature_set, feature_sets, params, model_dir)
        meta = finalize_model_metadata(
            model_root,
            output_root,
            model_dir,
            target,
            feature_set,
            params,
            pred,
            train_meta,
            fingerprint,
            versions,
            git,
            config["phase1_goal"],
            config.get("market_aware_notice", ""),
        )
        manifest["models"].append({
            "target": target,
            "feature_set": feature_set,
            "status": "trained_new_v2_1_2",
            "predicted_rows": pred.height,
            "best_iteration": train_meta["best_iteration"],
            "training_seconds": train_meta["training_seconds"],
            "artifact_hashes": {k: meta.get(k) for k in meta if k.endswith("_sha256")},
        })
    manifest["run_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(output_root / "run_manifest.json", manifest)
    atomic_write_text(output_root / "run_summary.md", f"# CatBoost Baseline V2.1.2 V1 Run Summary\n\nModels handled: {len(manifest['models'])}\nOld models reused: false\nGit dirty: {manifest.get('git_is_dirty')}\n")
    logger.info("done models=%s", len(manifest["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
