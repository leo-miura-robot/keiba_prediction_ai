from __future__ import annotations

import argparse
import json
import logging
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

from src.models.catboost_atomic_output import atomic_write_text
from src.models.catboost_config import load_yaml_config, resolve_training_params, sha256_data, write_resolved_config
from src.models.catboost_data import all_missing_numeric_columns, class_balance_rows, filter_target, load_feature_sets, prepare_pandas, prediction_metadata, split_frame, validate_feature_set
from src.models.catboost_metrics import metrics_by_split, probability_sum_diagnostics, race_metrics, validate_probabilities
from src.models.catboost_prediction_regeneration import (
    compare_predictions,
    copy_model_artifacts,
    load_dataset_with_split,
    read_json,
    regenerate_predictions,
    resolve_split_definition,
    split_validation_rows,
    verify_source,
)
from src.models.catboost_resume import add_artifact_hashes, artifact_paths, complete_artifacts, decide_resume, feature_columns_hash
from src.models.model_manifest import dataset_fingerprints, git_info, hash_files, package_versions, sha256_file, write_json


CONFIG_PATH = Path("config/catboost_baseline_v1_0_2.yaml")
LOG_PATH = Path("logs/train_catboost_baseline_v1_0_2.log")
MODEL_COMBOS = [(t, fs) for t in ["win", "place"] for fs in ["market_free", "market_history", "market_aware"]]
CODE_FILES = [
    Path("scripts/train_catboost_baseline_v1_0_2.py"),
    Path("scripts/analyze_catboost_baseline_v1_0_2.py"),
    Path("src/models/catboost_market_comparison.py"),
    Path("src/models/catboost_prediction_regeneration.py"),
    Path("src/models/catboost_atomic_output.py"),
    Path("src/models/catboost_config.py"),
    Path("src/models/catboost_resume.py"),
    Path("src/models/catboost_data.py"),
    Path("src/models/catboost_metrics.py"),
    Path("src/models/model_manifest.py"),
    CONFIG_PATH,
    Path("config/feature_sets_v2_1_1.yaml"),
]


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("train_catboost_baseline_v1_0_2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def git_status_summary() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip()
    except Exception as exc:
        return f"git status unavailable: {exc}"


def gpu_smoke_check(task_type: str, devices: str | None) -> dict[str, Any]:
    if task_type.upper() == "CPU":
        return {"task_type": "CPU", "gpu_required": False}
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=128, n_features=8, random_state=42)
    model = CatBoostClassifier(iterations=2, task_type="GPU", devices=devices or "0", loss_function="Logloss", verbose=False, allow_writing_files=False)
    model.fit(x, y)
    return {"task_type": "GPU", "gpu_required": True, "devices": devices or "0", "catboost_gpu_smoke": "ok"}


def feature_columns_for(feature_set: str) -> dict[str, list[str]]:
    groups = load_feature_sets()[feature_set]
    return {"numeric": groups["numeric"], "categorical": groups["categorical"]}


def train_one(df: pl.DataFrame, target: str, feature_set: str, params: dict[str, Any], model_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    feature_sets = load_feature_sets()
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
    x_train, _ = prepare_pandas(train_df, numeric_cols, categorical_cols)
    x_valid, _ = prepare_pandas(valid_df, numeric_cols, categorical_cols)
    cat_features = [x_train.columns.get_loc(c) for c in categorical_cols]
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(Pool(x_train, train_df["__target__"].to_numpy(), cat_features=cat_features), eval_set=Pool(x_valid, valid_df["__target__"].to_numpy(), cat_features=cat_features), use_best_model=True)
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
    summary = {
        "best_iteration": int(model.get_best_iteration() or 0),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "all_missing_numeric_columns": all_missing_numeric_columns(target_df, numeric_cols),
        "model_reload_prediction_match": True,
    }
    return pred, summary


def expected_fingerprint(target: str, feature_set: str, params: dict[str, Any], input_fp: list[dict[str, Any]], resolved_sha: str, split_sha: str, code_hash: str) -> dict[str, Any]:
    cols = feature_columns_for(feature_set)
    return {
        "target": target,
        "feature_set": feature_set,
        "task_type": params.get("task_type"),
        "devices": params.get("devices"),
        "random_seed": params.get("random_seed"),
        "input_dataset_fingerprint": sha256_data(input_fp),
        "feature_set_yaml_sha256": sha256_file(Path("config/feature_sets_v2_1_1.yaml")),
        "feature_columns_sha256": feature_columns_hash(cols),
        "training_config_resolved_sha256": resolved_sha,
        "split_definition_sha256": split_sha,
        "code_bundle_sha256": code_hash,
    }


def model_metrics_summary(pred: pl.DataFrame, target: str, feature_set: str) -> dict[str, Any]:
    metrics_rows = [{**r, "target": target, "feature_set": feature_set} for r in metrics_by_split(pred)]
    race_rows = [{**r, "feature_set": feature_set} for r in race_metrics(pred, target)]
    prob_rows = [{**r, "target": target, "feature_set": feature_set} for r in probability_sum_diagnostics(pred)]
    return {"metrics_by_split": metrics_rows, "race_metrics": race_rows, "probability_sum_diagnostics": prob_rows}


def finalize_model_metadata(model_dir: Path, output_root: Path, target: str, feature_set: str, params: dict[str, Any], pred: pl.DataFrame, base_meta: dict[str, Any], fingerprint: dict[str, Any], versions: dict[str, Any], git: dict[str, Any], phase1_goal: dict[str, Any]) -> dict[str, Any]:
    pred_path = output_root / "predictions" / f"{target}_{feature_set}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(pred_path, compression="zstd")
    summary = {
        "target": target,
        "feature_set": feature_set,
        "model_name": f"catboost_{target}_{feature_set}_v1_0_2",
        "params": params,
        "train_rows": pred.filter(pl.col("data_split") == "train").height,
        "validation_rows": pred.filter(pl.col("data_split") == "validation").height,
        "test_rows": pred.filter(pl.col("data_split") == "test").height,
        "latest_holdout_rows": pred.filter(pl.col("data_split") == "latest_holdout").height,
        "model_path": str(model_dir / "model.cbm"),
        "prediction_path": str(pred_path),
        "predictions_origin": "regenerated_from_current_v2_1_1",
        "phase1_goal": phase1_goal,
        **base_meta,
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
        "pyarrow_version": versions.get("pyarrow"),
        **git,
        "git_status_summary": git_status_summary(),
    }
    paths = artifact_paths(model_dir.parents[1], output_root, target, feature_set)
    meta = add_artifact_hashes(meta, paths)
    write_json(model_dir / "model_metadata.json", meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["win", "place"])
    parser.add_argument("--feature-set", choices=["market_free", "market_history", "market_aware"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-type", choices=["GPU", "CPU"])
    parser.add_argument("--devices")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-compatible-models", action="store_true")
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
    current_base = {"input_dataset_fingerprint": sha256_data(input_fp), "feature_set_yaml_sha256": sha256_file(Path(config["feature_set_yaml"]))}
    code_hash = hash_files(CODE_FILES)
    versions = package_versions()
    git = git_info(ROOT)
    git["git_status_summary"] = git_status_summary()
    df = load_dataset_with_split(Path(config["input_dataset_dir"]), split_by_year)
    if args.smoke_test:
        df = pl.concat([
            df.filter(pl.col("data_split") == "train").head(4000),
            df.filter(pl.col("data_split") == "validation").head(1000),
            df.filter(pl.col("data_split") == "test").head(1000),
            df.filter(pl.col("data_split") == "latest_holdout").head(1000),
        ], how="diagonal_relaxed")
    split_rows = split_validation_rows(df)
    write_json(output_root / "split_summary_internal.json", {"rows": split_rows})
    manifest = {
        "version": config["version"],
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase1_goal": config["phase1_goal"],
        "training_config_resolved_sha256": resolved_sha,
        "split_definition_sha256": split_sha,
        "gpu_info": gpu_info,
        "versions": versions,
        **git,
        "models": [],
    }
    comparison_rows = []
    for target, feature_set in combos:
        logger.info("handle target=%s feature_set=%s", target, feature_set)
        model_dir = model_root / target / feature_set
        paths = artifact_paths(model_root, output_root, target, feature_set)
        write_resolved_config(model_dir / "training_config_resolved.json", params)
        fingerprint = expected_fingerprint(target, feature_set, params, input_fp, resolved_sha, split_sha, code_hash)
        if not args.force and (args.resume or args.strict_resume):
            decision = decide_resume(paths, fingerprint, strict=args.strict_resume)
            logger.info("resume decision %s %s %s %s", target, feature_set, decision.action, decision.reasons)
            if decision.action == "skip":
                meta = read_json(paths["metadata"])
                manifest["models"].append({"target": target, "feature_set": feature_set, "status": "skipped_resume", "model_origin": meta.get("model_origin"), "predictions_origin": meta.get("predictions_origin")})
                continue
            if decision.action == "error":
                write_json(output_root / "run_manifest.json", manifest)
                return 2
        source_used = None
        source_reasons = []
        pred = None
        train_meta: dict[str, Any] = {}
        if args.reuse_compatible_models and not args.force and not args.smoke_test:
            for source in config["source_candidates"]:
                current = {**current_base, "feature_columns": feature_columns_for(feature_set)}
                ok, reasons, spaths = verify_source(source, target, feature_set, current, params, split_sha)
                if ok:
                    copy_model_artifacts(spaths, model_dir)
                    pred, reload_match = regenerate_predictions(model_dir / "model.cbm", df, target, feature_set, feature_columns_for(feature_set))
                    cmp = compare_predictions(spaths["prediction"], pred, float(config["prediction_tolerance"]))
                    comparison_rows.append({"target": target, "feature_set": feature_set, "source": source["name"], **cmp})
                    if cmp["mismatch_count"] == 0 and cmp["missing_in_old"] == 0 and cmp["missing_in_new"] == 0:
                        source_used = (source, spaths, cmp, reload_match)
                        break
                    source_reasons.append(f"{source['name']}: prediction mismatch {cmp}")
                else:
                    source_reasons.append(f"{source['name']}: {reasons}")
        if source_used is not None and pred is not None:
            source, spaths, cmp, reload_match = source_used
            base_meta = {
                "model_origin": f"reused_from_{source['name']}",
                "source_model_path": str(spaths["model"]),
                "source_manifest_path": str(spaths["manifest"]),
                "full_prediction_comparison": cmp,
                "model_reload_prediction_match": reload_match,
            }
            status = base_meta["model_origin"]
        else:
            if args.strict_resume:
                logger.error("strict mode cannot reuse model: %s", source_reasons)
                write_json(output_root / "run_manifest.json", manifest)
                return 2
            pred, train_meta = train_one(df, target, feature_set, params, model_dir)
            cmp = {"compared_rows": 0, "missing_in_old": 0, "missing_in_new": 0, "max_abs_diff": None, "mean_abs_diff": None, "p99_abs_diff": None, "mismatch_count": None, "tolerance": config["prediction_tolerance"]}
            comparison_rows.append({"target": target, "feature_set": feature_set, "source": "retrained", **cmp})
            base_meta = {"model_origin": "retrained_v1_0_2", "source_reuse_rejection_reasons": source_reasons, "full_prediction_comparison": cmp, **train_meta}
            status = "retrained_v1_0_2"
        meta = finalize_model_metadata(model_dir, output_root, target, feature_set, params, pred, base_meta, fingerprint, versions, git, config["phase1_goal"])
        manifest["models"].append({"target": target, "feature_set": feature_set, "status": status, "predicted_rows": pred.height, "full_prediction_comparison": base_meta["full_prediction_comparison"], "artifact_hashes": {k: meta.get(k) for k in meta if k.endswith("_sha256")}})
    if comparison_rows:
        pl.DataFrame(comparison_rows).write_csv(output_root / "prediction_regeneration_comparison.csv")
    manifest["run_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(output_root / "run_manifest.json", manifest)
    atomic_write_text(output_root / "run_summary.md", f"# CatBoost Baseline V1.0.2 Run Summary\n\nModels: {len(manifest['models'])}\nGit dirty: {manifest.get('git_is_dirty')}\n")
    logger.info("done models=%s", len(manifest["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
