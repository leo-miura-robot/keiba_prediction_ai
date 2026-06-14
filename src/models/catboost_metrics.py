from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "pred_probability_mean": float(y_prob.mean()) if len(y_prob) else None,
        "logloss": None,
        "brier": None,
        "roc_auc": None,
        "pr_auc": None,
        "auc_reason": "",
    }
    if not len(y_true):
        out["auc_reason"] = "empty"
        return out
    out["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    if len(set(y_true.tolist())) < 2:
        out["auc_reason"] = "single_class"
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def metrics_by_split(pred: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "validation", "test", "latest_holdout"]:
        part = pred.filter(pl.col("data_split") == split)
        metrics = probability_metrics(part["actual"].to_numpy(), part["pred_probability"].to_numpy())
        metrics["data_split"] = split
        rows.append(metrics)
    return rows


def race_metrics(pred: pl.DataFrame, target: str) -> list[dict[str, Any]]:
    rows = []
    for split in ["validation", "test", "latest_holdout"]:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        if target == "win":
            race_rows = []
            tie_count = 0
            for race_id, grp in part.group_by("race_id"):
                maxp = grp["pred_probability"].max()
                top = grp.filter(pl.col("pred_probability") == maxp)
                if top.height > 1:
                    tie_count += 1
                chosen = top.sort("entry_id").head(1)
                race_rows.append({
                    "hit_top1": int(chosen["actual"][0] == 1),
                    "winner_in_top3": int(grp.sort("pred_probability", descending=True).head(3)["actual"].max() == 1),
                })
            n = len(race_rows)
            rows.append({
                "data_split": split,
                "target": target,
                "race_count": n,
                "top1_winner_accuracy": sum(r["hit_top1"] for r in race_rows) / n if n else None,
                "top3_contains_winner_rate": sum(r["winner_in_top3"] for r in race_rows) / n if n else None,
                "tie_max_probability_races": tie_count,
            })
        else:
            race_rows = []
            for race_id, grp in part.group_by("race_id"):
                k = int(grp["place_rank_limit"].max() or 0)
                if k <= 0:
                    continue
                top = grp.sort("pred_probability", descending=True).head(k)
                positives = int(grp["actual"].sum())
                hits = int(top["actual"].sum())
                race_rows.append({
                    "precision": hits / k,
                    "recall": hits / positives if positives else None,
                    "hit": int(hits > 0),
                })
            n = len(race_rows)
            rows.append({
                "data_split": split,
                "target": target,
                "race_count": n,
                "precision_at_k": sum(r["precision"] for r in race_rows) / n if n else None,
                "recall_at_k": sum(r["recall"] for r in race_rows if r["recall"] is not None) / n if n else None,
                "hit_race_rate": sum(r["hit"] for r in race_rows) / n if n else None,
            })
    return rows


def probability_sum_diagnostics(pred: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "validation", "test", "latest_holdout"]:
        sums = pred.filter(pl.col("data_split") == split).group_by("race_id").agg(pl.col("pred_probability").sum().alias("prob_sum"))
        if sums.height == 0:
            continue
        vals = sums["prob_sum"].to_numpy()
        rows.append({
            "data_split": split,
            "race_count": sums.height,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    return rows


def calibration_bins(pred: pl.DataFrame, bins: int = 10) -> list[dict[str, Any]]:
    rows = []
    for split in ["validation", "test", "latest_holdout"]:
        part = pred.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        pdf = part.select(["actual", "pred_probability"]).to_pandas()
        pdf["bin"] = np.minimum((pdf["pred_probability"] * bins).astype(int), bins - 1)
        for b, grp in pdf.groupby("bin"):
            rows.append({
                "data_split": split,
                "bin": int(b),
                "rows": int(len(grp)),
                "avg_pred_probability": float(grp["pred_probability"].mean()),
                "actual_rate": float(grp["actual"].mean()),
                "actual_minus_pred": float(grp["actual"].mean() - grp["pred_probability"].mean()),
            })
    return rows


def validate_probabilities(pred: pl.DataFrame) -> bool:
    return pred.filter((pl.col("pred_probability") < 0) | (pl.col("pred_probability") > 1) | pl.col("pred_probability").is_nan()).height == 0
