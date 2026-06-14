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

from src.models.catboost_config import load_yaml_config, resolve_training_params, sha256_data, write_resolved_config
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
from src.models.catboost_metrics import metrics_by_split, probability_sum_diagnostics, race_metrics, validate_probabilities
from src.models.catboost_resume import (
    add_artifact_hashes,
    artifact_paths,
    complete_artifacts,
    copy_artifacts,
    decide_resume,
    feature_columns_hash,
    read_json,
)
from src.models.model_manifest import dataset_fingerprints, git_info, hash_files, package_versions, sha256_file, write_json


CONFIG_PATH = Path("config/catboost_baseline_v1_0_1.yaml")
LOG_PATH = Path("logs/train_catboost_baseline_v1_0_1.log")
MODEL_COMBOS = [(target, fs) for target in ["win", "place"] for fs in ["market_free", "market_history", "market_aware"]]
CODE_FILES = [
    Path("scripts/train_catboost_baseline_v1_0_1.py"),
    Path("scripts/analyze_catboost_baseline_v1_0_1.py"),
    Path("src/models/catboost_config.py"),
    Path("src/models/catboost_resume.py"),
    Path("src/models/catboost_analysis.py"),
    Path("src/models/catboost_data.py"),
    Path("src/models/catboost_metrics.py"),
    Path("src/models/model_manifest.py"),
    CONFIG_PATH,
    Path("config/feature_sets_v2_1_1.yaml"),
]
_TARGET_DF_CACHE: dict[str, pl.DataFrame] = {}


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("train_catboost_baseline_v1_0_1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def gpu_smoke_check(task_type: str, devices: str | None) -> dict[str, Any]:
    if task_type.upper() == "CPU":
        return {"task_type": "CPU", "gpu_required": False}
    from sklearn.datasets import make_classification

    x, y = make_classification(n_samples=128, n_features=8, random_state=42)
    model = CatBoostClassifier(
        iterations=2,
        task_type="GPU",
        devices=devices or "0",
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    return {"task_type": "GPU", "gpu_required": True, "devices": devices or "0", "catboost_gpu_smoke": "ok"}


def model_name(target: str, feature_set: str) -> str:
    return f"catboost_{target}_{feature_set}_v1_0_1"


def feature_columns_for(feature_set: str) -> dict[str, list[str]]:
    groups = load_feature_sets()[feature_set]
    return {"numeric": groups["numeric"], "categorical": groups["categorical"]}


def expected_fingerprint(
    target: str,
    feature_set: str,
    params: dict[str, Any],
    resolved_config_sha: str,
    code_hash: str,
    input_fp: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "training_config_resolved_sha256": resolved_config_sha,
        "code_bundle_sha256": code_hash,
    }


def train_one(
    target: str,
    feature_set: str,
    params: dict[str, Any],
    model_root: Path,
    output_root: Path,
    smoke_test: bool,
) -> dict[str, Any]:
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

    model_dir = model_root / target / feature_set
    model_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.cbm"
    pred_path = output_root / "predictions" / f"{target}_{feature_set}.parquet"
    model.save_model(model_path)
    reloaded = CatBoostClassifier()
    reloaded.load_model(model_path)
    sample_pool = Pool(x_valid.head(min(100, len(x_valid))), cat_features=cat_features)
    reload_match = bool(np.allclose(model.predict_proba(sample_pool)[:, 1], reloaded.predict_proba(sample_pool)[:, 1]))
    if not reload_match:
        raise RuntimeError("model reload prediction mismatch")
    predictions.write_parquet(pred_path, compression="zstd")

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
        "prediction_path": str(pred_path),
        "all_missing_numeric_columns": missing_all,
        "metrics_by_split": metrics_rows,
        "race_metrics": race_rows,
        "probability_sum_diagnostics": prob_sum_rows,
        "model_reload_prediction_match": reload_match,
        "artifact_origin": "trained_v1_0_1",
    }
    write_json(model_dir / "metrics.json", summary)
    write_json(model_dir / "feature_columns.json", {"numeric": numeric_cols, "categorical": categorical_cols})
    write_json(model_dir / "categorical_columns.json", {"categorical": categorical_cols})
    return summary


def v1_paths(config: dict[str, Any], target: str, feature_set: str) -> dict[str, Path]:
    return artifact_paths(Path(config["source_v1_model_root"]), Path(config["source_v1_output_root"]), target, feature_set)


def can_reuse_v1(config: dict[str, Any], target: str, feature_set: str, params: dict[str, Any], logger: logging.Logger) -> tuple[bool, list[str]]:
    paths = v1_paths(config, target, feature_set)
    missing = [name for name in ["model", "metrics", "metadata", "features", "categorical", "prediction"] if not paths[name].exists()]
    if missing:
        return False, [f"v1 missing artifacts: {missing}"]
    reasons = []
    v1_params = read_json(paths["metrics"]).get("params")
    if v1_params != params:
        reasons.append("resolved params differ")
    if read_json(paths["features"]) != feature_columns_for(feature_set):
        reasons.append("feature columns differ")
    pred = pl.read_parquet(paths["prediction"], columns=["entry_id", "pred_probability"])
    if pred.height == 0:
        reasons.append("v1 prediction empty")
    if pred.filter((pl.col("pred_probability") < 0) | (pl.col("pred_probability") > 1) | pl.col("pred_probability").is_nan()).height:
        reasons.append("v1 prediction probability out of range")
    try:
        model = CatBoostClassifier()
        model.load_model(paths["model"])
        logger.info("v1 model reload ok target=%s feature_set=%s", target, feature_set)
        if target not in _TARGET_DF_CACHE:
            _TARGET_DF_CACHE[target] = filter_target(load_dataset(), target)
        df = _TARGET_DF_CACHE[target]
        expected_entries = df.select("entry_id")
        if pred.height != df.height:
            reasons.append(f"prediction row count differs: pred={pred.height} expected={df.height}")
        missing_from_pred = expected_entries.join(pred.select("entry_id"), on="entry_id", how="anti").height
        extra_in_pred = pred.select("entry_id").join(expected_entries, on="entry_id", how="anti").height
        if missing_from_pred or extra_in_pred:
            reasons.append(f"prediction entry_id mismatch missing={missing_from_pred} extra={extra_in_pred}")
        if not reasons:
            cols = feature_columns_for(feature_set)
            sample_pred = pred.head(200).select(["entry_id", "pred_probability"])
            sample_df = sample_pred.join(df, on="entry_id", how="inner")
            x_sample, _ = prepare_pandas(sample_df, cols["numeric"], cols["categorical"])
            cat_features = [x_sample.columns.get_loc(c) for c in cols["categorical"]]
            reloaded_probs = model.predict_proba(Pool(x_sample, cat_features=cat_features))[:, 1]
            if not np.allclose(sample_df["pred_probability"].to_numpy(), reloaded_probs):
                reasons.append("reloaded model predictions differ from stored v1 predictions")
    except Exception as exc:
        reasons.append(f"v1 model reload failed: {exc}")
    return not reasons, reasons


def finalize_metadata(
    target: str,
    feature_set: str,
    params: dict[str, Any],
    paths: dict[str, Path],
    fingerprint: dict[str, Any],
    versions: dict[str, Any],
    git: dict[str, Any],
    artifact_origin: str,
    source_model_path: str | None,
    phase1_goal: dict[str, Any],
) -> dict[str, Any]:
    metrics = read_json(paths["metrics"])
    metrics.update({
        "target": target,
        "feature_set": feature_set,
        "model_name": model_name(target, feature_set),
        "params": params,
        "artifact_origin": artifact_origin,
        "source_model_path": source_model_path,
        "phase1_goal": phase1_goal,
    })
    write_json(paths["metrics"], metrics)
    meta = {
        **metrics,
        **fingerprint,
        "python_version": versions.get("python"),
        "catboost_version": versions.get("catboost"),
        "numpy_version": versions.get("numpy"),
        "pandas_version": versions.get("pandas"),
        "sklearn_version": versions.get("sklearn"),
        "pyarrow_version": versions.get("pyarrow"),
        **git,
    }
    meta = add_artifact_hashes(meta, paths)
    write_json(paths["metadata"], meta)
    return meta


def write_docs(output_root: Path, manifest: dict[str, Any]) -> None:
    Path("docs/catboost_baseline_v1_0_1_design.md").write_text(
        "\n".join([
            "# CatBoost Baseline V1.0.1 Design",
            "",
            "V1.0.1 keeps the V2.1.1 feature dataset unchanged and adds YAML-resolved training configuration, artifact fingerprint resume, idempotent analysis outputs, fixed-width and quantile calibration diagnostics, and same-sample market comparison.",
            "",
            "Phase 1 future ROI goal is documented only: win ROI >= 0.90 and place ROI >= 0.90. ROI, EV, bet generation, bankroll allocation, probability calibration application, Ability, ANA, LightGBM Ranker, Optuna, and walk-forward redesign are not implemented in this phase.",
            "",
            "A future ROI pass must not be judged successful from a single high payout or one long-shot dependency; it must consider sufficient bet count, validation-only threshold decisions, no test tuning, odds-band ROI, period stability, and top-payout exclusion robustness.",
        ]) + "\n",
        encoding="utf-8",
    )
    Path("docs/catboost_baseline_v1_0_1_results.md").write_text(
        "# CatBoost Baseline V1.0.1 Results\n\n"
        f"Run manifest: `{output_root / 'run_manifest.json'}`\n\n"
        f"Models handled: {len(manifest.get('models', []))}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["win", "place"])
    parser.add_argument("--feature-set", choices=["market_free", "market_history", "market_aware"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-type", default=None, choices=["GPU", "CPU"])
    parser.add_argument("--devices", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-compatible-v1", action="store_true")
    args = parser.parse_args()

    logger = setup_logger()
    config = load_yaml_config(CONFIG_PATH)
    output_root = Path(config["output_root"] + ("_smoke" if args.smoke_test else ""))
    model_root = Path(config["model_root"] + ("_smoke" if args.smoke_test else ""))
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    combos = MODEL_COMBOS if args.all else [(args.target, args.feature_set)]
    if not args.all and (not args.target or not args.feature_set):
        raise SystemExit("--target and --feature-set are required unless --all is used")

    params = resolve_training_params(config, {"task_type": args.task_type, "devices": args.devices}, smoke_test=args.smoke_test)
    try:
        gpu_info = gpu_smoke_check(str(params["task_type"]), params.get("devices"))
        logger.info("device check: %s", gpu_info)
    except Exception:
        logger.exception("GPU smoke check failed; not falling back to CPU")
        return 2

    resolved_path = output_root / "training_config_resolved.json"
    resolved_sha = write_resolved_config(resolved_path, params)
    input_fp = dataset_fingerprints(Path(config["input_dataset_dir"]))
    code_hash = hash_files(CODE_FILES)
    versions = package_versions()
    git = git_info(ROOT)
    manifest = {
        "version": config["version"],
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(CONFIG_PATH),
        "training_config_resolved_sha256": resolved_sha,
        "phase1_goal": config["phase1_goal"],
        "gpu_info": gpu_info,
        "versions": versions,
        **git,
        "models": [],
    }

    for target, feature_set in combos:
        logger.info("handle target=%s feature_set=%s", target, feature_set)
        paths = artifact_paths(model_root, output_root, target, feature_set)
        paths["training_config_resolved"].parent.mkdir(parents=True, exist_ok=True)
        write_resolved_config(paths["training_config_resolved"], params)
        expected = expected_fingerprint(target, feature_set, params, resolved_sha, code_hash, input_fp)
        if args.force:
            logger.info("force retrain target=%s feature_set=%s overwrite=%s", target, feature_set, paths)
        elif args.resume or args.strict_resume:
            decision = decide_resume(paths, expected, strict=args.strict_resume)
            logger.info("resume decision target=%s feature_set=%s action=%s reasons=%s", target, feature_set, decision.action, decision.reasons)
            if decision.action == "skip":
                meta = read_json(paths["metadata"])
                manifest["models"].append({
                    "target": target,
                    "feature_set": feature_set,
                    "status": "skipped_resume",
                    "artifact_origin": meta.get("artifact_origin"),
                    "source_model_path": meta.get("source_model_path"),
                    "reasons": decision.reasons,
                    "artifact_hashes": {k: meta.get(k) for k in meta if k.endswith("_sha256")},
                })
                continue
            if decision.action == "error":
                write_json(output_root / "run_manifest.json", manifest)
                return 2
        if args.reuse_compatible_v1 and not args.force and not args.smoke_test:
            reusable, reasons = can_reuse_v1(config, target, feature_set, params, logger)
            if reusable:
                logger.info("reuse V1 target=%s feature_set=%s", target, feature_set)
                source = v1_paths(config, target, feature_set)
                copy_artifacts(source, paths)
                write_resolved_config(paths["training_config_resolved"], params)
                meta = finalize_metadata(target, feature_set, params, paths, expected, versions, git, "reused_from_v1", str(source["model"]), config["phase1_goal"])
                manifest["models"].append({"target": target, "feature_set": feature_set, "status": "reused_from_v1", "artifact_hashes": {k: meta.get(k) for k in meta if k.endswith("_sha256")}})
                continue
            logger.info("V1 reuse rejected target=%s feature_set=%s reasons=%s", target, feature_set, reasons)
        summary = train_one(target, feature_set, params, model_root, output_root, args.smoke_test)
        meta = finalize_metadata(target, feature_set, params, paths, expected, versions, git, "trained_v1_0_1", None, config["phase1_goal"])
        manifest["models"].append({"target": target, "feature_set": feature_set, "status": "trained", "best_iteration": summary.get("best_iteration"), "artifact_hashes": {k: meta.get(k) for k in meta if k.endswith("_sha256")}})
        missing = complete_artifacts(paths)
        if missing:
            raise RuntimeError(f"missing artifacts after train: {missing}")

    manifest["run_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["elapsed_seconds"] = time.time() - started
    write_json(output_root / "run_manifest.json", manifest)
    write_docs(output_root, manifest)
    logger.info("done elapsed=%.1fs models=%s", manifest["elapsed_seconds"], len(manifest["models"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
