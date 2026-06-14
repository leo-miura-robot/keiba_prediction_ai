from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.models.catboost_metrics import probability_metrics


REASON_PRIORITY = [
    "duplicate_entry_id",
    "invalid_actual",
    "missing_winner",
    "missing_odds_runner",
    "invalid_odds_runner",
    "missing_model_prediction",
    "runner_count_mismatch",
]


def _reason(row: dict[str, Any]) -> str:
    reasons = []
    if row["duplicate_entry_id_count"] > 0:
        reasons.append("duplicate_entry_id")
    if row["invalid_actual_count"] > 0:
        reasons.append("invalid_actual")
    if row["actual_positive_count"] < 1:
        reasons.append("missing_winner")
    if row["missing_odds_count"] > 0:
        reasons.append("missing_odds_runner")
    if row["invalid_odds_count"] > 0:
        reasons.append("invalid_odds_runner")
    if row["missing_prediction_count"] > 0:
        reasons.append("missing_model_prediction")
    if row["valid_runner_count"] != row["expected_runner_count"]:
        reasons.append("runner_count_mismatch")
    if not reasons:
        return ""
    return next(r for r in REASON_PRIORITY if r in reasons)


def build_complete_market_frame(predictions: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    required = ["market_free", "market_history", "market_aware"]
    base = predictions["market_free"].select(["entry_id", "race_id", "data_split", "actual", "eligible", "tan_odds"])
    joined = base.rename({"pred_probability": "unused"}) if "pred_probability" in base.columns else base
    for fs in required:
        p = predictions[fs].select(["entry_id", pl.col("pred_probability").alias(f"catboost_{fs}")])
        joined = joined.join(p, on="entry_id", how="left")
    joined = joined.with_columns([
        pl.col("actual").is_in([0, 1]).alias("__valid_actual"),
        pl.col("tan_odds").is_not_null().alias("__has_odds"),
        (pl.col("tan_odds") > 0).fill_null(False).alias("__valid_odds"),
        pl.all_horizontal([pl.col(f"catboost_{fs}").is_not_null() for fs in required]).alias("__has_all_predictions"),
    ])
    race = joined.group_by("race_id").agg([
        pl.first("data_split").alias("data_split"),
        pl.len().alias("expected_runner_count"),
        pl.col("entry_id").n_unique().alias("unique_entry_id_count"),
        (~pl.col("__valid_actual")).sum().alias("invalid_actual_count"),
        (~pl.col("__has_odds")).sum().alias("missing_odds_count"),
        ((pl.col("__has_odds")) & (~pl.col("__valid_odds"))).sum().alias("invalid_odds_count"),
        (~pl.col("__has_all_predictions")).sum().alias("missing_prediction_count"),
        pl.col("actual").sum().alias("actual_positive_count"),
        (pl.col("__valid_actual") & pl.col("__valid_odds") & pl.col("__has_all_predictions")).sum().alias("valid_runner_count"),
    ]).with_columns([
        (pl.col("expected_runner_count") - pl.col("unique_entry_id_count")).alias("duplicate_entry_id_count"),
    ])
    reasons = [_reason(row) for row in race.to_dicts()]
    race = race.with_columns(pl.Series("market_exclusion_reason", reasons)).with_columns([
        (pl.col("market_exclusion_reason") == "").alias("is_complete_market_race"),
        pl.col("valid_runner_count").alias("valid_odds_runner_count"),
        pl.col("valid_runner_count").alias("valid_prediction_runner_count"),
    ])
    complete_races = race.filter(pl.col("is_complete_market_race")).select("race_id")
    complete = joined.join(complete_races, on="race_id", how="inner").with_columns([
        (1.0 / pl.col("tan_odds")).alias("raw_implied_probability"),
    ]).with_columns([
        (pl.col("raw_implied_probability") / pl.col("raw_implied_probability").sum().over("race_id")).alias("market_probability")
    ]).select([
        "entry_id", "race_id", "data_split", "actual", "tan_odds",
        "market_probability", "catboost_market_free", "catboost_market_history", "catboost_market_aware",
    ])
    excluded = race.filter(~pl.col("is_complete_market_race"))
    return complete, race, excluded


def top_race_metrics_for_scores(df: pl.DataFrame, score_col: str) -> dict[str, Any]:
    rows = []
    for _, grp in df.group_by("race_id"):
        top1 = grp.sort(score_col, descending=True).head(1)
        top3 = grp.sort(score_col, descending=True).head(3)
        rows.append({"top1": int(top1["actual"][0] == 1), "top3": int(top3["actual"].max() == 1)})
    n = len(rows)
    return {
        "top1_winner_accuracy": sum(r["top1"] for r in rows) / n if n else None,
        "top3_winner_inclusion_rate": sum(r["top3"] for r in rows) / n if n else None,
    }


def complete_market_metrics(complete: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    score_cols = ["market_probability", "catboost_market_free", "catboost_market_history", "catboost_market_aware"]
    for split in ["validation", "test", "latest_holdout"]:
        part = complete.filter(pl.col("data_split") == split)
        if part.height == 0:
            continue
        for score_col in score_cols:
            metrics = probability_metrics(part["actual"].to_numpy(), part[score_col].to_numpy())
            race = top_race_metrics_for_scores(part, score_col)
            sums = part.group_by("race_id").agg(pl.col(score_col).sum().alias("s"))
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
                "race_probability_sum_mean": float(sums["s"].mean()),
                "race_probability_sum_min": float(sums["s"].min()),
                "race_probability_sum_max": float(sums["s"].max()),
            })
    return rows


def exclusion_summary(excluded: pl.DataFrame) -> list[dict[str, Any]]:
    if excluded.height == 0:
        return []
    return excluded.group_by(["data_split", "market_exclusion_reason"]).agg(pl.len().alias("races")).sort(["data_split", "market_exclusion_reason"]).to_dicts()


def validate_same_entry_set(complete: pl.DataFrame) -> bool:
    return complete.select("entry_id").n_unique() == complete.height
