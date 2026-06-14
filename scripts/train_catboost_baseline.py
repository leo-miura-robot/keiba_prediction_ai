from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from src.models.catboost_data import FEATURE_SET_PATH, load_dataset, load_feature_sets, validate_feature_set
from src.models.catboost_metrics import probability_metrics
from src.models.catboost_runner import OUTPUT_ROOT, train_one
from src.models.model_manifest import dataset_fingerprints, git_info, hash_files, package_versions, sha256_file, write_json


LOG_PATH = Path("logs/train_catboost_baseline_v1.log")
MODEL_COMBOS = [(t, fs) for t in ["win", "place"] for fs in ["market_free", "market_history", "market_aware"]]
CODE_FILES = [
    Path("scripts/train_catboost_baseline.py"),
    Path("src/models/catboost_data.py"),
    Path("src/models/catboost_metrics.py"),
    Path("src/models/catboost_runner.py"),
    Path("src/models/model_manifest.py"),
    Path("config/catboost_baseline_v1.yaml"),
    FEATURE_SET_PATH,
]


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("train_catboost_baseline_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def gpu_smoke_check(task_type: str) -> dict:
    if task_type.upper() == "CPU":
        return {"task_type": "CPU", "gpu_required": False}
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=128, n_features=8, random_state=42)
    model = CatBoostClassifier(iterations=2, task_type="GPU", devices="0", loss_function="Logloss", verbose=False, allow_writing_files=False)
    model.fit(x, y)
    return {"task_type": "GPU", "gpu_required": True, "catboost_gpu_smoke": "ok"}


def append_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old = pl.read_csv(path)
        new = pl.concat([old, new], how="diagonal_relaxed")
    new.write_csv(path)


def write_docs(results: list[dict], manifest: dict) -> None:
    Path("docs/catboost_baseline_v1_design.md").write_text(
        "# CatBoost Baseline V1 Design\n\nUses V2.1.1 dataset, CatBoost classification, train=2016-2023, validation=2024, test=2025, latest_holdout=2026. No calibration, ROI, EV, or betting strategy is applied.\n",
        encoding="utf-8",
    )
    lines = ["# CatBoost Baseline V1 Results", "", f"Run at: {manifest['run_started_at']}", ""]
    for result in results:
        s = result["summary"]
        lines.append(f"- {s['target']} / {s['feature_set']}: best_iteration={s['best_iteration']}, training_seconds={s['training_seconds']:.3f}")
    Path("docs/catboost_baseline_v1_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["win", "place"])
    parser.add_argument("--feature-set", choices=["market_free", "market_history", "market_aware"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task-type", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logger = setup_logger()
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        gpu_info = gpu_smoke_check(args.task_type)
        logger.info("device check: %s", gpu_info)
    except Exception as exc:
        logger.exception("GPU smoke check failed; not falling back to CPU")
        return 2
    combos = MODEL_COMBOS if args.all else [(args.target, args.feature_set)]
    if not args.all and (not args.target or not args.feature_set):
        raise SystemExit("--target and --feature-set are required unless --all is used")
    manifest = {
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_type": args.task_type,
        "smoke_test": args.smoke_test,
        "dataset_fingerprints": dataset_fingerprints(),
        "feature_set_yaml": str(FEATURE_SET_PATH),
        "feature_set_sha256": sha256_file(FEATURE_SET_PATH),
        "code_bundle_hash": hash_files(CODE_FILES),
        **git_info(ROOT),
        "versions": package_versions(),
        "gpu_info": gpu_info,
        "random_seed": 42,
    }
    results = []
    for target, feature_set in combos:
        logger.info("train target=%s feature_set=%s smoke=%s", target, feature_set, args.smoke_test)
        result = train_one(target, feature_set, task_type=args.task_type, smoke_test=args.smoke_test, force=args.force)
        if "summary" not in result:
            logger.info("skip complete target=%s feature_set=%s", target, feature_set)
            continue
        results.append(result)
        append_table(OUTPUT_ROOT / "metrics_by_split.csv", result["metrics_rows"])
        append_table(OUTPUT_ROOT / "race_metrics.csv", result["race_rows"])
        append_table(OUTPUT_ROOT / "race_probability_sum_diagnostics.csv", result["prob_sum_rows"])
        append_table(OUTPUT_ROOT / "calibration_bins.csv", result["calibration_rows"])
        append_table(OUTPUT_ROOT / "class_balance.csv", result["class_balance_rows"])
        fi_path = OUTPUT_ROOT / "feature_importance.csv"
        fi = result["feature_importance"]
        if fi_path.exists() and fi_path.stat().st_size > 0:
            fi = pl.concat([pl.read_csv(fi_path), fi], how="diagonal_relaxed")
        fi.write_csv(fi_path)
    comparison = []
    for result in results:
        s = result["summary"]
        metrics = {m["data_split"]: m for m in s["metrics_by_split"]}
        race = {m["data_split"]: m for m in s["race_metrics"]}
        comparison.append({
            "target": s["target"],
            "feature_set": s["feature_set"],
            "best_iteration": s["best_iteration"],
            "train_rows": s["train_rows"],
            "validation_rows": s["validation_rows"],
            "test_rows": s["test_rows"],
            "latest_holdout_rows": s["latest_holdout_rows"],
            "validation_logloss": metrics.get("validation", {}).get("logloss"),
            "test_logloss": metrics.get("test", {}).get("logloss"),
            "latest_holdout_logloss": metrics.get("latest_holdout", {}).get("logloss"),
            "validation_brier": metrics.get("validation", {}).get("brier"),
            "test_brier": metrics.get("test", {}).get("brier"),
            "validation_roc_auc": metrics.get("validation", {}).get("roc_auc"),
            "test_roc_auc": metrics.get("test", {}).get("roc_auc"),
            "validation_pr_auc": metrics.get("validation", {}).get("pr_auc"),
            "test_pr_auc": metrics.get("test", {}).get("pr_auc"),
            "validation_race_metric": race.get("validation", {}).get("top1_winner_accuracy") or race.get("validation", {}).get("precision_at_k"),
            "test_race_metric": race.get("test", {}).get("top1_winner_accuracy") or race.get("test", {}).get("precision_at_k"),
            "training_seconds": s["training_seconds"],
            "prediction_seconds": s["prediction_seconds"],
            "gpu_name": manifest["gpu_info"].get("gpu_name") or manifest["gpu_info"].get("task_type"),
            "model_path": s["model_path"],
        })
    if comparison:
        append_table(OUTPUT_ROOT / "model_comparison.csv", comparison)
    manifest["run_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["elapsed_seconds"] = time.time() - started
    manifest["models_run"] = [r["summary"]["model_name"] for r in results]
    write_json(OUTPUT_ROOT / "run_manifest.json", manifest)
    write_docs(results, manifest)
    logger.info("done elapsed=%.1fs models=%s", manifest["elapsed_seconds"], len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
