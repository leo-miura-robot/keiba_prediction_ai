from __future__ import annotations

from typing import Any

import polars as pl


def split_name(year: int) -> str:
    if 2016 <= year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    if year == 2025:
        return "test"
    if year == 2026:
        return "latest_holdout"
    return "out_of_scope"


def is_valid_id(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text == "" or text == "0":
        return False
    return any(ch != "0" for ch in text)


def is_valid_positive_int(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except Exception:
        return False


def is_positive(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0
    except Exception:
        return False


def exclusion_reason(row: dict[str, Any], kind: str) -> str:
    if not row.get("race_has_result"):
        return "no_result"
    if kind == "win" and not row.get("race_has_win_payout"):
        return "no_win_payout"
    if kind == "place" and not row.get("race_has_place_payout"):
        return "no_place_payout"
    if row.get("IJyoCD") != "0":
        return "abnormal_or_cancelled"
    if kind == "ranking" and not is_positive(row.get("KakuteiJyuni")):
        return "missing_rank"
    if not is_valid_positive_int(row.get("Umaban")):
        return "invalid_umaban"
    if not is_valid_id(row.get("KettoNum")):
        return "invalid_horse_id"
    return ""


def add_target_columns(df: pl.DataFrame) -> pl.DataFrame:
    race_flags = df.group_by("race_id").agg([
        (pl.col("KakuteiJyuni").fill_null(0).max() > 0).alias("race_has_result"),
        (pl.col("tan_pay").fill_null(0).max() > 0).alias("race_has_win_payout"),
        (pl.col("fuku_pay").fill_null(0).max() > 0).alias("race_has_place_payout"),
    ]).with_columns(
        (pl.col("race_has_result") & (pl.col("race_has_win_payout") | pl.col("race_has_place_payout"))).alias("race_is_finalized")
    )
    out = df.join(race_flags, on="race_id", how="left")
    out = out.with_columns([
        pl.when(pl.col("SyussoTosu") <= 4).then(0)
        .when(pl.col("SyussoTosu") <= 7).then(2)
        .otherwise(3)
        .alias("place_rank_limit"),
        (pl.col("SyussoTosu") >= 5).alias("place_bet_available_by_rule"),
        ((pl.col("IJyoCD") == "0") & (pl.col("KakuteiJyuni") > 0)).cast(pl.Int8).alias("is_normal_runner_strict"),
        pl.col("Year").map_elements(split_name, return_dtype=pl.String).alias("data_split"),
    ])
    out = out.with_columns([
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") == 1)).cast(pl.Int8).alias("target_win_rank"),
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") >= 1) & (pl.col("KakuteiJyuni") <= 2)).cast(pl.Int8).alias("target_ren_rank"),
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") >= 1) & (pl.col("KakuteiJyuni") <= 3)).cast(pl.Int8).alias("target_top3_rank"),
        pl.col("is_win_paid").fill_null(0).cast(pl.Int8).alias("target_win_paid"),
        pl.col("is_place_paid").fill_null(0).cast(pl.Int8).alias("target_place_paid"),
        (
            (pl.col("is_normal_runner_strict") == 1)
            & pl.col("place_bet_available_by_rule")
            & (pl.col("KakuteiJyuni") >= 1)
            & (pl.col("KakuteiJyuni") <= pl.col("place_rank_limit"))
        ).cast(pl.Int8).alias("target_place_by_rule"),
        pl.col("tan_odds").map_elements(is_positive, return_dtype=pl.Boolean).alias("win_odds_available"),
        (pl.col("fuku_odds_low").map_elements(is_positive, return_dtype=pl.Boolean) | pl.col("fuku_odds_high").map_elements(is_positive, return_dtype=pl.Boolean)).alias("place_odds_available"),
        pl.col("TanVote").map_elements(is_positive, return_dtype=pl.Boolean).alias("win_votes_available"),
        pl.col("FukuVote").map_elements(is_positive, return_dtype=pl.Boolean).alias("place_votes_available"),
    ])
    out = out.with_columns([
        pl.struct(pl.all()).map_elements(lambda row: exclusion_reason(row, "win"), return_dtype=pl.String).alias("win_training_exclusion_reason"),
        pl.struct(pl.all()).map_elements(lambda row: exclusion_reason(row, "place"), return_dtype=pl.String).alias("place_training_exclusion_reason"),
        pl.struct(pl.all()).map_elements(lambda row: exclusion_reason(row, "ranking"), return_dtype=pl.String).alias("ranking_training_exclusion_reason"),
    ])
    out = out.with_columns([
        (pl.col("win_training_exclusion_reason") == "").alias("eligible_for_win_training"),
        (pl.col("place_training_exclusion_reason") == "").alias("eligible_for_place_training"),
        (pl.col("ranking_training_exclusion_reason") == "").alias("eligible_for_ranking_training"),
    ])
    return out

