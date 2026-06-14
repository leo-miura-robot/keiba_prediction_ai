from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from src.features.feature_sets_v2_1_1 import LEAKAGE_COLUMNS, CURRENT_MARKET_COLUMNS, MARKET_HISTORY_COLUMNS, load_feature_set_yaml


DATA_DIR = Path("outputs/model_feature_dataset_v2_1_1")
FEATURE_SET_PATH = Path("config/feature_sets_v2_1_1.yaml")
YEARS = list(range(2016, 2027))
TARGETS = {
    "win": {"target": "target_win_paid", "eligible": "eligible_for_win_training"},
    "place": {"target": "target_place_paid", "eligible": "eligible_for_place_training"},
}
SPLIT_BY_YEAR = {
    **{year: "train" for year in range(2016, 2024)},
    2024: "validation",
    2025: "test",
    2026: "latest_holdout",
}
FORBIDDEN_MODEL_COLUMNS = LEAKAGE_COLUMNS | {
    "race_id", "entry_id", "Bamei", "KettoNum", "race_date",
    "win_training_exclusion_reason", "place_training_exclusion_reason", "ranking_training_exclusion_reason",
}


def load_dataset(years: list[int] | None = None) -> pl.DataFrame:
    frames = []
    for year in years or YEARS:
        path = DATA_DIR / f"year={year}" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pl.read_parquet(path))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.with_columns(pl.col("Year").map_elements(lambda y: SPLIT_BY_YEAR.get(int(y), "out_of_scope"), return_dtype=pl.String).alias("data_split"))


def load_feature_sets(path: Path = FEATURE_SET_PATH) -> dict[str, dict[str, list[str]]]:
    return load_feature_set_yaml(path)


def validate_feature_set(df: pl.DataFrame, feature_set_name: str, feature_sets: dict[str, dict[str, list[str]]]) -> list[str]:
    errors: list[str] = []
    if feature_set_name not in feature_sets:
        return [f"unknown feature_set={feature_set_name}"]
    groups = feature_sets[feature_set_name]
    columns = groups.get("numeric", []) + groups.get("categorical", [])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    duplicated = sorted({c for c in columns if columns.count(c) > 1})
    if duplicated:
        errors.append(f"duplicated columns: {duplicated}")
    forbidden = sorted(set(columns) & FORBIDDEN_MODEL_COLUMNS)
    if forbidden:
        errors.append(f"forbidden columns: {forbidden}")
    if feature_set_name == "market_free":
        market_cols = sorted(set(columns) & (CURRENT_MARKET_COLUMNS | MARKET_HISTORY_COLUMNS))
        if market_cols:
            errors.append(f"market_free contains market columns: {market_cols}")
    if feature_set_name == "market_history":
        current_market = sorted(set(columns) & CURRENT_MARKET_COLUMNS)
        if current_market:
            errors.append(f"market_history contains current market columns: {current_market}")
    if feature_set_name == "market_aware":
        history_cols = set(feature_sets["market_history"]["numeric"] + feature_sets["market_history"]["categorical"])
        if not history_cols <= set(columns):
            errors.append(f"market_aware missing market_history columns: {sorted(history_cols - set(columns))}")
    return errors


def split_overlap_errors(df: pl.DataFrame) -> list[str]:
    errors = []
    for col in ["race_id", "entry_id"]:
        overlap = (
            df.group_by(col)
            .agg(pl.col("data_split").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        )
        if overlap:
            errors.append(f"{col} overlaps across splits: {overlap}")
    return errors


def filter_target(df: pl.DataFrame, target_name: str) -> pl.DataFrame:
    meta = TARGETS[target_name]
    return df.filter(pl.col(meta["eligible"]) == True).with_columns(pl.col(meta["target"]).cast(pl.Int8).alias("__target__"))


def prepare_pandas(df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    pdf = df.select(numeric_cols + categorical_cols).to_pandas()
    for col in numeric_cols:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        pdf[col] = pdf[col].replace([np.inf, -np.inf], np.nan)
    for col in categorical_cols:
        series = pdf[col].astype("object")
        series = series.where(pd.notna(series), "__MISSING__").astype(str)
        series = series.replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
        pdf[col] = series
    return pdf, {"numeric": numeric_cols, "categorical": categorical_cols}


def split_frame(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {name: df.filter(pl.col("data_split") == name) for name in ["train", "validation", "test", "latest_holdout"]}


def class_balance_rows(df: pl.DataFrame, target_name: str, feature_set_name: str) -> list[dict[str, Any]]:
    rows = []
    for split, part in split_frame(df).items():
        n = part.height
        pos = int(part["__target__"].sum()) if n else 0
        rows.append({
            "target": target_name,
            "feature_set": feature_set_name,
            "data_split": split,
            "rows": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": pos / n if n else None,
        })
    return rows


def all_missing_numeric_columns(df: pl.DataFrame, numeric_cols: list[str]) -> list[str]:
    return [c for c in numeric_cols if df[c].null_count() == df.height]


def prediction_metadata(df: pl.DataFrame, target_name: str, feature_set_name: str, probs: np.ndarray) -> pl.DataFrame:
    target_col = TARGETS[target_name]["target"]
    eligible_col = TARGETS[target_name]["eligible"]
    keep = ["entry_id", "race_id", "race_date", "Year", "Umaban", "KettoNum", "data_split", target_col, eligible_col, "tan_odds", "fuku_odds_low", "fuku_odds_high", "place_rank_limit"]
    out = df.select([c for c in keep if c in df.columns]).with_columns([
        pl.lit(target_name).alias("target"),
        pl.lit(feature_set_name).alias("feature_set"),
        pl.col(target_col).alias("actual"),
        pl.col(eligible_col).alias("eligible"),
        pl.Series("pred_probability", probs.astype(float)),
    ])
    return out
