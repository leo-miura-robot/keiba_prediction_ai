from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

from src.models.catboost_atomic_output import atomic_write_csv, atomic_write_text, file_sha256
from src.models.catboost_config import load_yaml_config, sha256_data
from src.models.catboost_data import filter_target, prepare_pandas
from src.models.catboost_metrics import metrics_by_split, probability_metrics, probability_sum_diagnostics, race_metrics
from src.models.catboost_prediction_regeneration import load_dataset_with_split, read_json, resolve_split_definition, split_validation_rows


CONFIG_PATH = Path("config/catboost_baseline_v2_1_2_v1.yaml")
TARGETS = ["win", "place"]
FEATURE_SETS = ["market_free", "market_history", "market_aware"]
SPLITS = ["train", "validation", "test", "latest_holdout"]
EVAL_SPLITS = ["validation", "test", "latest_holdout"]
CSV_OUTPUTS = [
    "metrics_by_split.csv",
    "race_metrics.csv",
    "race_probability_sum_diagnostics.csv",
    "class_balance.csv",
    "split_summary.csv",
    "calibration_bins.csv",
    "calibration_summary.csv",
    "prediction_distribution_summary.csv",
    "prediction_distribution_by_year.csv",
    "feature_importance.csv",
    "shap_importance.csv",
    "complete_race_summary_win.csv",
    "complete_race_summary_place.csv",
    "excluded_races_win.csv",
    "excluded_races_place.csv",
    "market_comparison_win.csv",
    "place_odds_band_diagnostics.csv",
    "old_new_model_metric_comparison.csv",
    "old_new_prediction_comparison.csv",
    "old_new_prediction_diff_summary.csv",
    "analysis_hashes.csv",
]


def sorted_frame(rows: list[dict[str, Any]] | pl.DataFrame, sort_cols: list[str]) -> pl.DataFrame:
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    if df.height == 0:
        return df
    float_cols = [name for name, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns([pl.col(c).round(12).alias(c) for c in float_cols])
    cols = [c for c in sort_cols if c in df.columns]
    return df.sort(cols) if cols else df


def calibration_bins(pred: pl.DataFrame, bin_type: str, requested: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in EVAL_SPLITS:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        pdf = part.select(["actual", "pred_probability"]).to_pandas()
        if bin_type == "fixed_width":
            pdf["bin_id"] = np.minimum((pdf["pred_probability"].clip(0, 1) * requested).astype(int), requested - 1)
        else:
            unique = np.sort(pdf["pred_probability"].unique())
            if len(unique) <= requested:
                mapping = {v: i for i, v in enumerate(unique)}
            else:
                groups = np.array_split(unique, requested)
                mapping = {v: i for i, group in enumerate(groups) for v in group}
            pdf["bin_id"] = pdf["pred_probability"].map(mapping).astype(int)
        actual_bins = int(pdf["bin_id"].nunique())
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
                "absolute_calibration_error": float(abs(grp["actual"].mean() - grp["pred_probability"].mean())),
            })
    return rows


def calibration_summary(pred: pl.DataFrame, target: str, feature_set: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in EVAL_SPLITS:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        y = part["actual"].to_numpy().astype(int)
        p = np.clip(part["pred_probability"].to_numpy().astype(float), 1e-8, 1 - 1e-8)
        fixed = [r for r in calibration_bins(part, "fixed_width") if r["data_split"] == split]
        total = sum(r["count"] for r in fixed)
        ece = sum(r["count"] * r["absolute_calibration_error"] for r in fixed) / total if total else None
        mce = max((r["absolute_calibration_error"] for r in fixed), default=None)
        slope = intercept = None
        if len(np.unique(y)) == 2:
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            lr.fit(logit, y)
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
        rows.append({
            "target": target,
            "feature_set": feature_set,
            "data_split": split,
            "ece_fixed_width": ece,
            "mce_fixed_width": mce,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        })
    return rows


def feature_importance_rows(model_path: Path, target: str, feature_set: str) -> list[dict[str, Any]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    values = model.get_feature_importance(prettified=True)
    return [{"target": target, "feature_set": feature_set, "importance_type": "prediction_values_change", "feature": str(row["Feature Id"]), "importance": float(row["Importances"])} for _, row in values.iterrows()]


def shap_rows(model_path: Path, df: pl.DataFrame, feature_columns: dict[str, list[str]], target: str, feature_set: str, sample_size: int = 1000) -> list[dict[str, Any]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    sample = filter_target(df, target).filter(pl.col("data_split") == "validation").head(sample_size)
    if sample.height == 0:
        return []
    x, _ = prepare_pandas(sample, feature_columns["numeric"], feature_columns["categorical"])
    cat_features = [x.columns.get_loc(c) for c in feature_columns["categorical"]]
    vals = model.get_feature_importance(Pool(x, cat_features=cat_features), type="ShapValues")
    means = np.abs(vals[:, :-1]).mean(axis=0)
    return [{"target": target, "feature_set": feature_set, "importance_type": "mean_abs_shap", "feature": f, "importance": float(v), "sample_rows": sample.height} for f, v in zip(feature_columns["numeric"] + feature_columns["categorical"], means)]


def prediction_distribution(pred: pl.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    year_rows = []
    for split in SPLITS:
        part = pred.filter(pl.col("data_split") == split)
        if part.height:
            rows.append(dist_row(part, {"data_split": split}))
    for year in sorted(pred["Year"].unique().to_list()):
        part = pred.filter(pl.col("Year") == year)
        year_rows.append(dist_row(part, {"year": int(year), "data_split": str(part["data_split"][0]) if part.height else ""}))
    return rows, year_rows


def dist_row(df: pl.DataFrame, prefix: dict[str, Any]) -> dict[str, Any]:
    vals = df["pred_probability"].to_numpy()
    return {
        **prefix,
        "rows": int(len(vals)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p01": float(np.quantile(vals, 0.01)),
        "p05": float(np.quantile(vals, 0.05)),
        "p25": float(np.quantile(vals, 0.25)),
        "p50": float(np.quantile(vals, 0.50)),
        "p75": float(np.quantile(vals, 0.75)),
        "p95": float(np.quantile(vals, 0.95)),
        "p99": float(np.quantile(vals, 0.99)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def complete_win_races(predictions: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = predictions["market_free"].select(["entry_id", "race_id", "data_split", "actual", "tan_odds"])
    joined = base
    for fs in FEATURE_SETS:
        joined = joined.join(predictions[fs].select(["entry_id", pl.col("pred_probability").alias(f"catboost_{fs}")]), on="entry_id", how="left")
    joined = joined.with_columns([
        pl.col("actual").is_in([0, 1]).alias("__valid_actual"),
        (pl.col("tan_odds").is_not_null() & (pl.col("tan_odds") > 0)).alias("__valid_odds"),
        pl.all_horizontal([pl.col(f"catboost_{fs}").is_not_null() for fs in FEATURE_SETS]).alias("__has_preds"),
    ])
    race = joined.group_by("race_id").agg([
        pl.first("data_split").alias("data_split"),
        pl.len().alias("runner_rows"),
        pl.col("entry_id").n_unique().alias("entry_ids"),
        (~pl.col("__valid_actual")).sum().alias("invalid_actual"),
        (~pl.col("__valid_odds")).sum().alias("invalid_odds"),
        (~pl.col("__has_preds")).sum().alias("missing_prediction"),
        pl.col("actual").sum().alias("positive"),
    ]).with_columns([
        (pl.col("runner_rows") - pl.col("entry_ids")).alias("duplicate_entry_id"),
    ])
    race = race.with_columns(
        pl.when(pl.col("duplicate_entry_id") > 0).then(pl.lit("duplicate_entry_id"))
        .when(pl.col("invalid_actual") > 0).then(pl.lit("invalid_actual"))
        .when(pl.col("positive") < 1).then(pl.lit("missing_winner"))
        .when(pl.col("invalid_odds") > 0).then(pl.lit("invalid_odds_runner"))
        .when(pl.col("missing_prediction") > 0).then(pl.lit("missing_prediction"))
        .otherwise(pl.lit(""))
        .alias("exclusion_reason")
    )
    complete = joined.join(race.filter(pl.col("exclusion_reason") == "").select("race_id"), on="race_id", how="inner")
    complete = complete.with_columns((1.0 / pl.col("tan_odds")).alias("raw_market_probability"))
    complete = complete.with_columns((pl.col("raw_market_probability") / pl.col("raw_market_probability").sum().over("race_id")).alias("market_probability"))
    excluded = race.filter(pl.col("exclusion_reason") != "")
    return complete, excluded


def complete_place_races(predictions: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = predictions["market_free"].select(["entry_id", "race_id", "data_split", "actual", "fuku_odds_low", "fuku_odds_high", "place_rank_limit"])
    joined = base
    for fs in FEATURE_SETS:
        joined = joined.join(predictions[fs].select(["entry_id", pl.col("pred_probability").alias(f"catboost_{fs}")]), on="entry_id", how="left")
    joined = joined.with_columns([
        pl.col("actual").is_in([0, 1]).alias("__valid_actual"),
        (pl.col("fuku_odds_low").is_not_null() & (pl.col("fuku_odds_low") > 0) & pl.col("fuku_odds_high").is_not_null() & (pl.col("fuku_odds_high") > 0)).alias("__valid_odds"),
        pl.all_horizontal([pl.col(f"catboost_{fs}").is_not_null() for fs in FEATURE_SETS]).alias("__has_preds"),
    ])
    race = joined.group_by("race_id").agg([
        pl.first("data_split").alias("data_split"),
        pl.len().alias("runner_rows"),
        pl.col("entry_id").n_unique().alias("entry_ids"),
        (~pl.col("__valid_actual")).sum().alias("invalid_actual"),
        (~pl.col("__valid_odds")).sum().alias("invalid_odds"),
        (~pl.col("__has_preds")).sum().alias("missing_prediction"),
        pl.col("actual").sum().alias("positive"),
        pl.max("place_rank_limit").alias("place_rank_limit"),
    ]).with_columns((pl.col("runner_rows") - pl.col("entry_ids")).alias("duplicate_entry_id"))
    race = race.with_columns(
        pl.when(pl.col("duplicate_entry_id") > 0).then(pl.lit("duplicate_entry_id"))
        .when(pl.col("invalid_actual") > 0).then(pl.lit("invalid_actual"))
        .when(pl.col("positive") < 1).then(pl.lit("missing_place_paid"))
        .when(pl.col("invalid_odds") > 0).then(pl.lit("invalid_odds_runner"))
        .when(pl.col("missing_prediction") > 0).then(pl.lit("missing_prediction"))
        .otherwise(pl.lit(""))
        .alias("exclusion_reason")
    )
    complete = joined.join(race.filter(pl.col("exclusion_reason") == "").select("race_id"), on="race_id", how="inner")
    excluded = race.filter(pl.col("exclusion_reason") != "")
    return complete, excluded


def top1_for_score(df: pl.DataFrame, score_col: str) -> dict[str, Any]:
    hits = []
    for _, grp in df.group_by("race_id"):
        chosen = grp.sort(score_col, descending=True).head(1)
        hits.append(int(chosen["actual"][0] == 1))
    return {"race_level_top1_hit_rate": sum(hits) / len(hits) if hits else None}


def market_win_metrics(complete: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in EVAL_SPLITS:
        part = complete.filter(pl.col("data_split") == split)
        for model_name, col in [("market", "market_probability"), *[(fs, f"catboost_{fs}") for fs in FEATURE_SETS]]:
            if part.height == 0:
                continue
            rows.append({
                "data_split": split,
                "model": model_name,
                "rows": part.height,
                "races": part["race_id"].n_unique(),
                **probability_metrics(part["actual"].to_numpy(), part[col].to_numpy()),
                **top1_for_score(part, col),
            })
    return rows


def odds_band_rows(complete: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    bands = [0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 9999.0]
    labels = ["<=1.1", "1.1-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0-10.0", ">10.0"]
    for split in EVAL_SPLITS:
        part = complete.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        pdf = part.to_pandas()
        pdf["low_band"] = pd.cut(pdf["fuku_odds_low"], bins=bands, labels=labels, include_lowest=True)
        pdf["high_band"] = pd.cut(pdf["fuku_odds_high"], bins=bands, labels=labels, include_lowest=True)
        for band_col in ["low_band", "high_band"]:
            for fs in FEATURE_SETS:
                for band, grp in pdf.groupby(band_col, observed=True):
                    rows.append({
                        "data_split": split,
                        "feature_set": fs,
                        "band_column": band_col,
                        "odds_band": str(band),
                        "count": int(len(grp)),
                        "prediction_mean": float(grp[f"catboost_{fs}"].mean()),
                        "actual_place_rate": float(grp["actual"].mean()),
                    })
    return rows


def old_new_comparison(output_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], pl.DataFrame, list[dict[str, Any]]]:
    old_root = Path(config["old_v1_0_2_output_root"])
    metric_rows = []
    diff_frames = []
    diff_summary = []
    for target in TARGETS:
        for fs in FEATURE_SETS:
            old_path = old_root / "predictions" / f"{target}_{fs}.parquet"
            new_path = output_root / "predictions" / f"{target}_{fs}.parquet"
            if not old_path.exists() or not new_path.exists():
                continue
            old = pl.read_parquet(old_path).select(["entry_id", "data_split", "actual", pl.col("pred_probability").alias("old_prediction")])
            new = pl.read_parquet(new_path).select(["entry_id", "data_split", "actual", pl.col("pred_probability").alias("new_prediction")])
            same = old.join(new, on=["entry_id", "data_split", "actual"], how="inner").with_columns((pl.col("old_prediction") - pl.col("new_prediction")).abs().alias("absolute_difference"))
            same = same.with_columns([pl.lit(target).alias("target"), pl.lit(fs).alias("feature_set")])
            diff_frames.append(same)
            vals = same["absolute_difference"].to_numpy()
            diff_summary.append({
                "target": target,
                "feature_set": fs,
                "same_entry_rows": same.height,
                "old_rows": old.height,
                "new_rows": new.height,
                "max_abs_diff": float(vals.max()) if len(vals) else None,
                "mean_abs_diff": float(vals.mean()) if len(vals) else None,
                "p99_abs_diff": float(np.quantile(vals, 0.99)) if len(vals) else None,
            })
            for split in EVAL_SPLITS:
                part = same.filter(pl.col("data_split") == split)
                if part.height:
                    old_m = probability_metrics(part["actual"].to_numpy(), part["old_prediction"].to_numpy())
                    new_m = probability_metrics(part["actual"].to_numpy(), part["new_prediction"].to_numpy())
                    metric_rows.append({
                        "target": target,
                        "feature_set": fs,
                        "data_split": split,
                        "rows": part.height,
                        "old_logloss": old_m["logloss"],
                        "new_logloss": new_m["logloss"],
                        "delta_logloss_new_minus_old": (new_m["logloss"] - old_m["logloss"]) if old_m["logloss"] is not None and new_m["logloss"] is not None else None,
                        "old_brier": old_m["brier"],
                        "new_brier": new_m["brier"],
                        "delta_brier_new_minus_old": (new_m["brier"] - old_m["brier"]) if old_m["brier"] is not None and new_m["brier"] is not None else None,
                        "old_roc_auc": old_m["roc_auc"],
                        "new_roc_auc": new_m["roc_auc"],
                        "delta_roc_auc_new_minus_old": (new_m["roc_auc"] - old_m["roc_auc"]) if old_m["roc_auc"] is not None and new_m["roc_auc"] is not None else None,
                        "old_pr_auc": old_m["pr_auc"],
                        "new_pr_auc": new_m["pr_auc"],
                        "delta_pr_auc_new_minus_old": (new_m["pr_auc"] - old_m["pr_auc"]) if old_m["pr_auc"] is not None and new_m["pr_auc"] is not None else None,
                    })
    diff = pl.concat(diff_frames, how="diagonal_relaxed") if diff_frames else pl.DataFrame()
    return metric_rows, diff, diff_summary


def hash_csv_outputs(output_root: Path) -> list[dict[str, str]]:
    rows = []
    for name in CSV_OUTPUTS:
        path = output_root / name
        if path.exists():
            rows.append({"file_name": name, "sha256": file_sha256(path)})
    return sorted(rows, key=lambda r: r["file_name"])


def write_docs(output_root: Path, config: dict[str, Any], hashes: dict[str, str]) -> None:
    Path("docs").mkdir(exist_ok=True)
    atomic_write_text(Path("docs/catboost_baseline_v2_1_2_v1_design.md"), "\n".join([
        "# CatBoost Baseline V2.1.2 V1 Design",
        "",
        f"Input: `{config['input_dataset_dir']}`",
        f"Feature sets: `{config['feature_set_yaml']}`",
        "",
        "Six new CatBoost GPU binary classifiers are trained. V1.0.2 weights are not reused.",
        "",
        "The `market_aware` feature set uses final `NL_O1` odds for an ideal-condition final-odds model. It is not a pre-race live operation model.",
        "",
        "Phase 1 future ROI goals are documented only: win ROI >= 90%, place ROI >= 90%. ROI, EV, bet generation, bankroll allocation, calibration application, Ability, ANA, and Ranker are not implemented here.",
        "",
    ]))
    atomic_write_text(Path("docs/catboost_baseline_v2_1_2_v1_results.md"), "\n".join([
        "# CatBoost Baseline V2.1.2 V1 Results",
        "",
        f"Output: `{output_root}`",
        "",
        "Analysis CSV outputs are regenerated atomically. `analysis_hashes.csv` records content hashes.",
        "",
        f"CSV files: {len(hashes)}",
        "",
    ]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        raise SystemExit("--all is required so analysis outputs are fully regenerated")
    config = load_yaml_config(CONFIG_PATH)
    output_root = Path(config["output_root"])
    model_root = Path(config["model_root"])
    split_def, split_by_year = resolve_split_definition(config)
    df = load_dataset_with_split(Path(config["input_dataset_dir"]), split_by_year)
    metrics_rows = []
    race_rows = []
    prob_rows = []
    class_rows = []
    cal_rows = []
    cal_summary_rows = []
    dist_rows = []
    dist_year_rows = []
    fi_rows = []
    shap_importance = []
    predictions: dict[tuple[str, str], pl.DataFrame] = {}
    started = time.time()
    for target in TARGETS:
        for fs in FEATURE_SETS:
            pred_path = output_root / "predictions" / f"{target}_{fs}.parquet"
            model_dir = model_root / target / fs
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = pl.read_parquet(pred_path)
            predictions[(target, fs)] = pred
            for row in metrics_by_split(pred):
                row.update({"target": target, "feature_set": fs})
                metrics_rows.append(row)
            for row in race_metrics(pred, target):
                row.update({"feature_set": fs})
                race_rows.append(row)
            for row in probability_sum_diagnostics(pred):
                row.update({"target": target, "feature_set": fs})
                prob_rows.append(row)
            for typ in ["fixed_width", "quantile"]:
                for row in calibration_bins(pred, typ):
                    row.update({"target": target, "feature_set": fs})
                    cal_rows.append(row)
            cal_summary_rows.extend(calibration_summary(pred, target, fs))
            d, dy = prediction_distribution(pred)
            dist_rows.extend([{**r, "target": target, "feature_set": fs} for r in d])
            dist_year_rows.extend([{**r, "target": target, "feature_set": fs} for r in dy])
            fi_rows.extend(feature_importance_rows(model_dir / "model.cbm", target, fs))
            features = read_json(model_dir / "feature_columns.json")
            shap_importance.extend(shap_rows(model_dir / "model.cbm", df, features, target, fs))
            for split in SPLITS:
                part = pred.filter(pl.col("data_split") == split)
                class_rows.append({"target": target, "feature_set": fs, "data_split": split, "rows": part.height, "positive": int(part["actual"].sum()), "negative": part.height - int(part["actual"].sum()), "positive_rate": float(part["actual"].mean())})
    complete_win, excluded_win = complete_win_races({fs: predictions[("win", fs)] for fs in FEATURE_SETS})
    complete_place, excluded_place = complete_place_races({fs: predictions[("place", fs)] for fs in FEATURE_SETS})
    complete_win_summary = complete_win.group_by("data_split").agg(pl.len().alias("rows"), pl.col("race_id").n_unique().alias("races"), pl.col("actual").sum().alias("positive")).sort("data_split").to_dicts()
    complete_place_summary = complete_place.group_by("data_split").agg(pl.len().alias("rows"), pl.col("race_id").n_unique().alias("races"), pl.col("actual").sum().alias("positive")).sort("data_split").to_dicts()
    old_metric_rows, old_diff, old_diff_summary = old_new_comparison(output_root, config)
    outputs: dict[str, pl.DataFrame] = {
        "metrics_by_split.csv": sorted_frame(metrics_rows, ["target", "feature_set", "data_split"]),
        "race_metrics.csv": sorted_frame(race_rows, ["target", "feature_set", "data_split"]),
        "race_probability_sum_diagnostics.csv": sorted_frame(prob_rows, ["target", "feature_set", "data_split"]),
        "class_balance.csv": sorted_frame(class_rows, ["target", "feature_set", "data_split"]),
        "split_summary.csv": sorted_frame(split_validation_rows(df), ["data_split"]),
        "calibration_bins.csv": sorted_frame(cal_rows, ["target", "feature_set", "data_split", "bin_type", "bin_id"]),
        "calibration_summary.csv": sorted_frame(cal_summary_rows, ["target", "feature_set", "data_split"]),
        "prediction_distribution_summary.csv": sorted_frame(dist_rows, ["target", "feature_set", "data_split"]),
        "prediction_distribution_by_year.csv": sorted_frame(dist_year_rows, ["target", "feature_set", "year"]),
        "feature_importance.csv": sorted_frame(fi_rows, ["target", "feature_set", "importance_type", "feature"]),
        "shap_importance.csv": sorted_frame(shap_importance, ["target", "feature_set", "importance_type", "feature"]),
        "complete_race_summary_win.csv": sorted_frame(complete_win_summary, ["data_split"]),
        "complete_race_summary_place.csv": sorted_frame(complete_place_summary, ["data_split"]),
        "excluded_races_win.csv": sorted_frame(excluded_win, ["data_split", "race_id"]),
        "excluded_races_place.csv": sorted_frame(excluded_place, ["data_split", "race_id"]),
        "market_comparison_win.csv": sorted_frame(market_win_metrics(complete_win), ["data_split", "model"]),
        "place_odds_band_diagnostics.csv": sorted_frame(odds_band_rows(complete_place), ["data_split", "feature_set", "band_column", "odds_band"]),
        "old_new_model_metric_comparison.csv": sorted_frame(old_metric_rows, ["target", "feature_set", "data_split"]),
        "old_new_prediction_comparison.csv": sorted_frame(old_diff.select(["target", "feature_set", "data_split", "entry_id", "actual", "old_prediction", "new_prediction", "absolute_difference"]) if old_diff.height else old_diff, ["target", "feature_set", "data_split", "entry_id"]),
        "old_new_prediction_diff_summary.csv": sorted_frame(old_diff_summary, ["target", "feature_set"]),
    }
    hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        hashes[name] = atomic_write_csv(output_root / name, frame)
    # write hashes after other CSVs, then hash it too
    hash_rows = [{"file_name": name, "sha256": digest} for name, digest in sorted(hashes.items())]
    hashes["analysis_hashes.csv"] = atomic_write_csv(output_root / "analysis_hashes.csv", hash_rows)
    complete_win.sort(["data_split", "race_id", "entry_id"]).write_parquet(output_root / "complete_race_entries_win.parquet", compression="zstd")
    complete_place.sort(["data_split", "race_id", "entry_id"]).write_parquet(output_root / "complete_race_entries_place.parquet", compression="zstd")
    manifest = {
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "csv_hashes": hashes,
        "market_aware_notice": config.get("market_aware_notice"),
        "roi_ev_calculated": False,
        "calibration_applied": False,
    }
    atomic_write_text(output_root / "analysis_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    write_docs(output_root, config, hashes)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
