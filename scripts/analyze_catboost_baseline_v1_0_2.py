from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.models.catboost_atomic_output import atomic_write_csv, atomic_write_text, file_sha256
from src.models.catboost_config import load_yaml_config
from src.models.catboost_data import filter_target, load_feature_sets, prepare_pandas
from src.models.catboost_market_comparison import build_complete_market_frame, complete_market_metrics, exclusion_summary
from src.models.catboost_metrics import metrics_by_split, probability_sum_diagnostics, race_metrics
from src.models.catboost_prediction_regeneration import load_dataset_with_split, read_json, resolve_split_definition, split_validation_rows


CONFIG_PATH = Path("config/catboost_baseline_v1_0_2.yaml")
TARGETS = ["win", "place"]
FEATURE_SETS = ["market_free", "market_history", "market_aware"]


def calibration_bins(pred: pl.DataFrame, bin_type: str, requested: int = 10) -> list[dict]:
    rows = []
    for split in ["validation", "test", "latest_holdout"]:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        pdf = part.select(["actual", "pred_probability"]).to_pandas()
        if bin_type == "fixed_width":
            pdf["bin_id"] = np.minimum((pdf["pred_probability"].clip(0, 1) * requested).astype(int), requested - 1)
        else:
            try:
                bins = pd.qcut(pdf["pred_probability"], q=requested, duplicates="drop")
                codes = bins.cat.codes
                pdf["bin_id"] = codes
            except ValueError:
                pdf["bin_id"] = 0
        actual_bins = int(pd.Series(pdf["bin_id"]).nunique())
        for bin_id, grp in pdf.groupby("bin_id", sort=True):
            rows.append({
                "data_split": split,
                "bin_type": bin_type,
                "requested_bin_count": requested,
                "actual_bin_count": actual_bins,
                "bin_id": int(bin_id),
                "lower_bound": float(grp["pred_probability"].min()),
                "upper_bound": float(grp["pred_probability"].max()),
                "count": int(len(grp)),
                "mean_pred_probability": float(grp["pred_probability"].mean()),
                "actual_rate": float(grp["actual"].mean()),
                "calibration_gap": float(grp["actual"].mean() - grp["pred_probability"].mean()),
            })
    return rows


def feature_importance_rows(model_path: Path, target: str, feature_set: str) -> list[dict]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    values = model.get_feature_importance(prettified=True)
    return [{"target": target, "feature_set": feature_set, "importance_type": "prediction_values_change", "feature": str(row["Feature Id"]), "importance": float(row["Importances"])} for _, row in values.iterrows()]


def shap_rows(model_path: Path, df: pl.DataFrame, feature_columns: dict, target: str, feature_set: str, sample_size: int = 5000) -> list[dict]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    sample = filter_target(df, target).filter(pl.col("data_split") == "validation").head(sample_size)
    x, _ = prepare_pandas(sample, feature_columns["numeric"], feature_columns["categorical"])
    cat_features = [x.columns.get_loc(c) for c in feature_columns["categorical"]]
    vals = model.get_feature_importance(Pool(x, cat_features=cat_features), type="ShapValues")
    means = np.abs(vals[:, :-1]).mean(axis=0)
    return [{"target": target, "feature_set": feature_set, "importance_type": "mean_abs_shap", "feature": f, "importance": float(v), "sample_rows": sample.height} for f, v in zip(feature_columns["numeric"] + feature_columns["categorical"], means)]


def sorted_frame(rows, sort_cols: list[str]) -> pl.DataFrame:
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    float_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns([pl.col(c).round(12).alias(c) for c in float_cols])
    cols = [c for c in sort_cols if c in df.columns]
    return df.sort(cols) if cols and df.height else df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        raise SystemExit("--all is required for V1.0.2 analysis so CSVs are fully regenerated")
    config = load_yaml_config(CONFIG_PATH)
    output_root = Path(config["output_root"])
    model_root = Path(config["model_root"])
    split_def, split_by_year = resolve_split_definition(config)
    df = load_dataset_with_split(Path(config["input_dataset_dir"]), split_by_year)
    metrics_rows = []
    race_rows = []
    prob_rows = []
    cal_rows = []
    fi_rows = []
    shap_importance = []
    comparison_rows = []
    class_rows = []
    predictions = {}
    started = time.time()
    for target in TARGETS:
        for feature_set in FEATURE_SETS:
            pred_path = output_root / "predictions" / f"{target}_{feature_set}.parquet"
            model_dir = model_root / target / feature_set
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = pl.read_parquet(pred_path)
            predictions[(target, feature_set)] = pred
            for row in metrics_by_split(pred):
                row.update({"target": target, "feature_set": feature_set})
                metrics_rows.append(row)
            for row in race_metrics(pred, target):
                row.update({"feature_set": feature_set})
                race_rows.append(row)
            for row in probability_sum_diagnostics(pred):
                row.update({"target": target, "feature_set": feature_set})
                prob_rows.append(row)
            for typ in ["fixed_width", "quantile"]:
                for row in calibration_bins(pred, typ):
                    row.update({"target": target, "feature_set": feature_set})
                    cal_rows.append(row)
            fi_rows.extend(feature_importance_rows(model_dir / "model.cbm", target, feature_set))
            features = read_json(model_dir / "feature_columns.json")
            shap_importance.extend(shap_rows(model_dir / "model.cbm", df, features, target, feature_set))
            meta = read_json(model_dir / "metrics.json")
            by_split = {m["data_split"]: m for m in metrics_by_split(pred)}
            comparison_rows.append({
                "target": target,
                "feature_set": feature_set,
                "model_origin": meta.get("model_origin"),
                "validation_logloss": by_split["validation"]["logloss"],
                "test_logloss": by_split["test"]["logloss"],
                "latest_holdout_logloss": by_split["latest_holdout"]["logloss"],
            })
            for split in ["train", "validation", "test", "latest_holdout"]:
                part = pred.filter(pl.col("data_split") == split)
                class_rows.append({"target": target, "feature_set": feature_set, "data_split": split, "rows": part.height, "positive": int(part["actual"].sum()), "negative": part.height - int(part["actual"].sum()), "positive_rate": float(part["actual"].mean())})
    complete, race_status, excluded = build_complete_market_frame({fs: predictions[("win", fs)] for fs in FEATURE_SETS})
    market_rows = complete_market_metrics(complete)
    complete_summary = []
    for split in ["validation", "test", "latest_holdout"]:
        part = complete.filter(pl.col("data_split") == split)
        complete_summary.append({"data_split": split, "rows": part.height, "races": part["race_id"].n_unique(), "positive": int(part["actual"].sum()) if part.height else 0})
    hashes = {}
    outputs = {
        "model_comparison.csv": sorted_frame(comparison_rows, ["target", "feature_set"]),
        "split_summary.csv": sorted_frame(split_validation_rows(df), ["data_split"]),
        "class_balance.csv": sorted_frame(class_rows, ["target", "feature_set", "data_split"]),
        "metrics_by_split.csv": sorted_frame(metrics_rows, ["target", "feature_set", "data_split"]),
        "race_metrics.csv": sorted_frame(race_rows, ["target", "feature_set", "data_split"]),
        "race_probability_sum_diagnostics.csv": sorted_frame(prob_rows, ["target", "feature_set", "data_split"]),
        "calibration_bins.csv": sorted_frame(cal_rows, ["target", "feature_set", "data_split", "bin_type", "bin_id"]),
        "feature_importance.csv": sorted_frame(fi_rows, ["target", "feature_set", "importance_type", "feature"]),
        "shap_importance.csv": sorted_frame(shap_importance, ["target", "feature_set", "importance_type", "feature"]),
        "market_comparison_complete_races.csv": sorted_frame(market_rows, ["data_split", "model"]),
        "market_comparison_complete_race_summary.csv": sorted_frame(complete_summary, ["data_split"]),
        "market_comparison_excluded_races.csv": sorted_frame(excluded, ["data_split", "race_id"]),
        "market_comparison_exclusion_summary.csv": sorted_frame(exclusion_summary(excluded), ["data_split", "market_exclusion_reason"]),
    }
    for name, rows in outputs.items():
        hashes[name] = atomic_write_csv(output_root / name, rows)
    complete.sort(["data_split", "race_id", "entry_id"]).write_parquet(output_root / "market_comparison_complete_race_entries.parquet", compression="zstd")
    manifest = {"analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_seconds": time.time() - started, "csv_hashes": hashes}
    atomic_write_text(output_root / "analysis_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    atomic_write_text(output_root / "analysis_summary.md", f"# CatBoost Baseline V1.0.2 Analysis Summary\n\nComplete market rows: {complete.height}\nComplete market races: {complete['race_id'].n_unique()}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
