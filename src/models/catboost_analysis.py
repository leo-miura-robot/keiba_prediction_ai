from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool

from src.models.catboost_data import load_dataset, prepare_pandas, split_frame
from src.models.catboost_metrics import metrics_by_split, probability_metrics, probability_sum_diagnostics, race_metrics
from src.models.catboost_resume import upsert_csv


SUMMARY_KEYS = ["target", "feature_set"]
SPLITS = ["train", "validation", "test", "latest_holdout"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_bins_v101(pred: pl.DataFrame, bin_type: str, bins: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test", "latest_holdout"]:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        pdf = part.select(["actual", "pred_probability"]).to_pandas()
        if bin_type == "fixed_width":
            pdf["bin_id"] = np.minimum((pdf["pred_probability"].clip(0, 1) * bins).astype(int), bins - 1)
        elif bin_type == "quantile":
            ranks = pdf["pred_probability"].rank(method="first")
            pdf["bin_id"] = np.minimum(((ranks - 1) * bins / len(pdf)).astype(int), bins - 1)
        else:
            raise ValueError(f"unknown bin_type={bin_type}")
        for bin_id, grp in pdf.groupby("bin_id", sort=True):
            rows.append({
                "data_split": split,
                "bin_type": bin_type,
                "bin_id": int(bin_id),
                "lower_bound": float(grp["pred_probability"].min()),
                "upper_bound": float(grp["pred_probability"].max()),
                "count": int(len(grp)),
                "mean_pred_probability": float(grp["pred_probability"].mean()),
                "actual_rate": float(grp["actual"].mean()),
                "calibration_gap": float(grp["actual"].mean() - grp["pred_probability"].mean()),
            })
    return rows


def feature_importance_rows(model_path: Path, feature_set: str, target: str) -> list[dict[str, Any]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    values = model.get_feature_importance(prettified=True)
    return [{
        "target": target,
        "feature_set": feature_set,
        "importance_type": "prediction_values_change",
        "feature": str(row["Feature Id"]),
        "importance": float(row["Importances"]),
    } for _, row in values.iterrows()]


def shap_importance_rows(model_path: Path, df: pl.DataFrame, feature_columns: dict[str, list[str]], target: str, feature_set: str, sample_size: int = 5000) -> list[dict[str, Any]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    numeric = feature_columns["numeric"]
    categorical = feature_columns["categorical"]
    sample = df.filter(pl.col("data_split") == "validation").head(sample_size)
    x_sample, _ = prepare_pandas(sample, numeric, categorical)
    cat_features = [x_sample.columns.get_loc(c) for c in categorical]
    shap = model.get_feature_importance(Pool(x_sample, cat_features=cat_features), type="ShapValues")
    vals = np.abs(shap[:, :-1]).mean(axis=0)
    return [{
        "target": target,
        "feature_set": feature_set,
        "importance_type": "mean_abs_shap",
        "feature": feature,
        "importance": float(value),
        "sample_rows": int(sample.height),
    } for feature, value in zip(numeric + categorical, vals)]


def market_probability_frame(predictions: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict[str, Any]]:
    required = ["market_free", "market_history", "market_aware"]
    for key in required:
        if key not in predictions:
            raise FileNotFoundError(f"missing win prediction for {key}")
    base = predictions["market_free"].filter(
        (pl.col("eligible") == True)
        & (pl.col("tan_odds") > 0)
        & pl.col("actual").is_in([0, 1])
    ).select(["entry_id", "race_id", "data_split", "actual", "tan_odds"])
    joined = base
    for fs in required:
        p = predictions[fs].select(["entry_id", pl.col("pred_probability").alias(f"catboost_{fs}")])
        joined = joined.join(p, on="entry_id", how="inner")
    joined = joined.with_columns((1.0 / pl.col("tan_odds")).alias("raw_implied_probability"))
    joined = joined.with_columns(
        (pl.col("raw_implied_probability") / pl.col("raw_implied_probability").sum().over("race_id")).alias("market_probability")
    )
    race_counts = joined.group_by("race_id").agg(pl.len().alias("rows"), pl.col("market_probability").sum().alias("market_prob_sum"))
    summary = {
        "rows": joined.height,
        "races": race_counts.height,
        "entry_id_sets_equal": True,
        "market_probability_sum_min": float(race_counts["market_prob_sum"].min()) if race_counts.height else None,
        "market_probability_sum_max": float(race_counts["market_prob_sum"].max()) if race_counts.height else None,
    }
    return joined, summary


def top_race_metrics_for_scores(df: pl.DataFrame, score_col: str) -> dict[str, Any]:
    race_rows = []
    for _, grp in df.group_by("race_id"):
        top1 = grp.sort(score_col, descending=True).head(1)
        top3 = grp.sort(score_col, descending=True).head(3)
        race_rows.append({
            "top1": int(top1["actual"][0] == 1),
            "top3": int(top3["actual"].max() == 1),
        })
    n = len(race_rows)
    return {
        "top1_winner_accuracy": sum(r["top1"] for r in race_rows) / n if n else None,
        "top3_winner_inclusion_rate": sum(r["top3"] for r in race_rows) / n if n else None,
    }


def market_comparison_rows(same: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    score_cols = ["market_probability", "catboost_market_free", "catboost_market_history", "catboost_market_aware"]
    for split in ["validation", "test", "latest_holdout"]:
        part = same.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        for score_col in score_cols:
            metrics = probability_metrics(part["actual"].to_numpy(), part[score_col].to_numpy())
            race = top_race_metrics_for_scores(part, score_col)
            rows.append({
                "data_split": split,
                "model": score_col,
                "rows": metrics["rows"],
                "races": part["race_id"].n_unique(),
                "positive": metrics["positive"],
                "logloss": metrics["logloss"],
                "brier": metrics["brier"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                **race,
                "race_probability_sum_mean": float(part.group_by("race_id").agg(pl.col(score_col).sum().alias("s"))["s"].mean()),
            })
    return rows


def analyze_all(output_root: Path, model_root: Path, targets: list[str], feature_sets: list[str], write: bool = True) -> dict[str, Any]:
    predictions: dict[tuple[str, str], pl.DataFrame] = {}
    summaries = []
    metrics_rows = []
    race_rows = []
    prob_rows = []
    cal_rows = []
    fi_rows = []
    shap_rows = []
    class_rows = []
    split_rows = []

    full_df_by_target: dict[str, pl.DataFrame] = {}
    for target in targets:
        full_df_by_target[target] = None  # type: ignore[assignment]

    for target in targets:
        for feature_set in feature_sets:
            pred_path = output_root / "predictions" / f"{target}_{feature_set}.parquet"
            model_dir = model_root / target / feature_set
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            for required in ["model.cbm", "metrics.json", "feature_columns.json"]:
                if not (model_dir / required).exists():
                    raise FileNotFoundError(model_dir / required)
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
            for bin_type in ["fixed_width", "quantile"]:
                for row in calibration_bins_v101(pred, bin_type):
                    row.update({"target": target, "feature_set": feature_set})
                    cal_rows.append(row)
            fi_rows.extend(feature_importance_rows(model_dir / "model.cbm", feature_set, target))
            feature_columns = read_json(model_dir / "feature_columns.json")
            source_df = pred
            # SHAP needs original features; load once per target after all predictions are available.
            if full_df_by_target[target] is None:
                from src.models.catboost_data import filter_target
                raw = load_dataset()
                full_df_by_target[target] = filter_target(raw, target)
            shap_rows.extend(shap_importance_rows(model_dir / "model.cbm", full_df_by_target[target], feature_columns, target, feature_set))
            metrics = {m["data_split"]: m for m in metrics_rows if m.get("target") == target and m.get("feature_set") == feature_set}
            model_metrics = read_json(model_dir / "metrics.json")
            summaries.append({
                "target": target,
                "feature_set": feature_set,
                "best_iteration": model_metrics.get("best_iteration"),
                "validation_logloss": metrics.get("validation", {}).get("logloss"),
                "test_logloss": metrics.get("test", {}).get("logloss"),
                "latest_holdout_logloss": metrics.get("latest_holdout", {}).get("logloss"),
                "artifact_origin": model_metrics.get("artifact_origin", model_metrics.get("model_metadata", {}).get("artifact_origin", "trained_v1_0_1")),
            })
            for split in SPLITS:
                part = pred.filter(pl.col("data_split") == split)
                class_rows.append({
                    "target": target,
                    "feature_set": feature_set,
                    "data_split": split,
                    "rows": part.height,
                    "positive": int(part["actual"].sum()) if part.height else 0,
                    "negative": int(part.height - int(part["actual"].sum())) if part.height else 0,
                    "positive_rate": float(part["actual"].mean()) if part.height else None,
                })
    for split in SPLITS:
        any_pred = next(iter(predictions.values()))
        part = any_pred.filter(pl.col("data_split") == split)
        split_rows.append({"data_split": split, "rows": part.height, "races": part["race_id"].n_unique()})

    win_preds = {fs: predictions[("win", fs)] for fs in feature_sets if ("win", fs) in predictions}
    same, market_summary = market_probability_frame(win_preds)
    market_rows = market_comparison_rows(same)
    market_baseline = [r for r in market_rows if r["model"] == "market_probability"]

    if write:
        upsert_csv(output_root / "metrics_by_split.csv", metrics_rows, ["target", "feature_set", "data_split"])
        upsert_csv(output_root / "race_metrics.csv", race_rows, ["target", "feature_set", "data_split"])
        upsert_csv(output_root / "race_probability_sum_diagnostics.csv", prob_rows, ["target", "feature_set", "data_split"])
        upsert_csv(output_root / "calibration_bins.csv", cal_rows, ["target", "feature_set", "data_split", "bin_type", "bin_id"])
        upsert_csv(output_root / "feature_importance.csv", fi_rows, ["target", "feature_set", "importance_type", "feature"])
        upsert_csv(output_root / "shap_importance.csv", shap_rows, ["target", "feature_set", "importance_type", "feature"])
        upsert_csv(output_root / "model_comparison.csv", summaries, ["target", "feature_set"])
        upsert_csv(output_root / "class_balance.csv", class_rows, ["target", "feature_set", "data_split"])
        upsert_csv(output_root / "split_summary.csv", split_rows, ["data_split"])
        upsert_csv(output_root / "market_baseline_win.csv", market_baseline, ["data_split", "model"])
        upsert_csv(output_root / "market_comparison_same_sample.csv", market_rows, ["data_split", "model"])
        upsert_csv(output_root / "market_comparison_sample_summary.csv", [market_summary], ["entry_id_sets_equal"])
        same.write_parquet(output_root / "market_comparison_same_sample_entries.parquet", compression="zstd")

    return {
        "model_comparison": summaries,
        "market_summary": market_summary,
        "market_rows": market_rows,
        "calibration_rows": cal_rows,
        "feature_importance_rows": len(fi_rows),
        "shap_rows": len(shap_rows),
    }
