from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl


BASE_DIR = Path("outputs/base_runner_dataset")
OUT_DIR = Path("outputs/model_feature_dataset")
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "build_model_features.log"
CHECKPOINT_PATH = Path("outputs/model_feature_dataset_checkpoint.json")

SAMPLE_CSV = Path("outputs/model_feature_dataset_sample.csv")
ELIGIBILITY_CSV = Path("outputs/training_eligibility_summary.csv")
LABEL_MISMATCH_CSV = Path("outputs/label_mismatch_cases.csv")
HIST_QUALITY_CSV = Path("outputs/historical_feature_quality.csv")
LEAKAGE_SAMPLE_CSV = Path("outputs/time_leakage_validation_samples.csv")

DESIGN_DOC = Path("docs/model_feature_design.md")
TARGET_DOC = Path("docs/target_definition.md")
LEAKAGE_DOC = Path("docs/time_leakage_validation.md")

YEARS_ALL = list(range(2016, 2027))
HISTORY_WINDOWS = (1, 3, 5)

HIST_FEATURE_COLUMNS = [
    "horse_days_since_last",
    "horse_past_starts",
    "horse_last1_avg_finish", "horse_last3_avg_finish", "horse_last5_avg_finish",
    "horse_last3_win_rate", "horse_last5_win_rate",
    "horse_last3_ren_rate", "horse_last5_ren_rate",
    "horse_last3_place_rate", "horse_last5_place_rate",
    "horse_last3_avg_ninki", "horse_last5_avg_ninki",
    "horse_last3_avg_tan_odds", "horse_last5_avg_tan_odds",
    "horse_last3_avg_haron_l3", "horse_last5_avg_haron_l3",
    "horse_last3_avg_time", "horse_last5_avg_time",
    "horse_distance_diff_last", "horse_futan_diff_last", "horse_body_weight_diff_last",
    "horse_jyo_past_starts", "horse_jyo_win_rate", "horse_jyo_place_rate",
    "horse_surface_past_starts", "horse_surface_win_rate", "horse_surface_place_rate",
    "horse_dist_band_past_starts", "horse_dist_band_win_rate", "horse_dist_band_place_rate",
    "horse_baba_past_starts", "horse_baba_win_rate", "horse_baba_place_rate",
    "jockey_past_starts", "jockey_win_rate", "jockey_ren_rate", "jockey_place_rate",
    "trainer_past_starts", "trainer_win_rate", "trainer_ren_rate", "trainer_place_rate",
    "jockey_jyo_past_starts", "jockey_jyo_win_rate", "jockey_jyo_place_rate",
    "jockey_dist_band_past_starts", "jockey_dist_band_win_rate", "jockey_dist_band_place_rate",
    "horse_jockey_past_starts", "horse_jockey_win_rate", "horse_jockey_place_rate",
]

MARKET_AWARE_COLUMNS = [
    "TanOdds", "TanNinki", "TanVote", "FukuOddsLow", "FukuOddsHigh", "FukuNinki", "FukuVote",
    "tan_odds", "tan_ninki", "fuku_odds_low", "fuku_odds_high", "fuku_ninki",
    "market_odds_available", "market_votes_available",
]

LEAKAGE_COLUMNS = [
    "NyusenJyuni", "KakuteiJyuni", "Time", "ChakusaCD", "HaronTimeL3",
    "Jyuni1c", "Jyuni2c", "Jyuni3c", "Jyuni4c",
    "tan_pay", "fuku_pay", "is_win_paid", "is_place_paid",
    "target_win_rank", "target_ren_rank", "target_top3_rank",
    "target_win_paid", "target_place_paid", "target_place_by_rule",
]


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("build_model_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_years(value: str | None) -> list[int]:
    if not value:
        return YEARS_ALL
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return sorted(years)


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"years": {}}


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def read_base() -> pl.DataFrame:
    frames = []
    for year in YEARS_ALL:
        path = BASE_DIR / f"year={year}" / "data.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        raise RuntimeError(f"No base parquet files under {BASE_DIR}")
    return pl.concat(frames, how="diagonal_relaxed")


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


def date_from_str(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def avg(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def rate(values: list[int | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def distance_band(kyori: Any) -> int | None:
    value = to_float(kyori)
    if value is None:
        return None
    return int(value // 200 * 200)


def surface_code(track_cd: Any) -> str | None:
    if is_empty(track_cd):
        return None
    text = str(track_cd)
    return text[0]


def baba_code(row: dict[str, Any]) -> str | None:
    track = str(row.get("TrackCD") or "")
    if track.startswith("1"):
        return row.get("SibaBabaCD")
    if track.startswith("2"):
        return row.get("DirtBabaCD")
    return row.get("SibaBabaCD") or row.get("DirtBabaCD")


def stats_empty() -> dict[str, int]:
    return {"starts": 0, "wins": 0, "rens": 0, "places": 0}


def stats_features(stats: dict[str, int], prefix: str, include_ren: bool = False) -> dict[str, Any]:
    starts = stats["starts"]
    out = {
        f"{prefix}_past_starts": starts,
        f"{prefix}_win_rate": stats["wins"] / starts if starts else None,
        f"{prefix}_place_rate": stats["places"] / starts if starts else None,
    }
    if include_ren:
        out[f"{prefix}_ren_rate"] = stats["rens"] / starts if starts else None
    return out


def update_stats(stats: dict[str, int], row: dict[str, Any]) -> None:
    stats["starts"] += 1
    stats["wins"] += int(row["target_win_rank"] == 1)
    stats["rens"] += int(row["target_ren_rank"] == 1)
    stats["places"] += int(row["target_top3_rank"] == 1)


def add_labels(df: pl.DataFrame) -> pl.DataFrame:
    race_flags = df.group_by("race_id").agg([
        (pl.col("KakuteiJyuni").fill_null(0).max() > 0).alias("race_has_result"),
        (pl.col("tan_pay").fill_null(0).max() > 0).alias("race_has_win_payout"),
        (pl.col("fuku_pay").fill_null(0).max() > 0).alias("race_has_place_payout"),
    ]).with_columns(
        (pl.col("race_has_result") & (pl.col("race_has_win_payout") | pl.col("race_has_place_payout"))).alias("race_is_finalized")
    )
    out = df.join(race_flags, on="race_id", how="left")
    out = out.with_columns([
        pl.when(pl.col("SyussoTosu") >= 8).then(3).otherwise(2).alias("place_rank_limit"),
        ((pl.col("IJyoCD") == "0") & (pl.col("KakuteiJyuni") > 0)).cast(pl.Int8).alias("is_normal_runner_strict"),
    ])
    out = out.with_columns([
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") == 1)).cast(pl.Int8).alias("target_win_rank"),
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") >= 1) & (pl.col("KakuteiJyuni") <= 2)).cast(pl.Int8).alias("target_ren_rank"),
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") >= 1) & (pl.col("KakuteiJyuni") <= 3)).cast(pl.Int8).alias("target_top3_rank"),
        pl.col("is_win_paid").cast(pl.Int8).alias("target_win_paid"),
        pl.col("is_place_paid").cast(pl.Int8).alias("target_place_paid"),
        ((pl.col("is_normal_runner_strict") == 1) & (pl.col("KakuteiJyuni") >= 1) & (pl.col("KakuteiJyuni") <= pl.col("place_rank_limit"))).cast(pl.Int8).alias("target_place_by_rule"),
        ((pl.col("tan_odds").is_not_null()) | (pl.col("fuku_odds_low").is_not_null())).cast(pl.Int8).alias("market_odds_available"),
        ((pl.col("TanVote").is_not_null()) | (pl.col("FukuVote").is_not_null())).cast(pl.Int8).alias("market_votes_available"),
        pl.col("Year").map_elements(split_name, return_dtype=pl.String).alias("data_split"),
    ])
    out = out.with_columns([
        (
            pl.col("race_is_finalized")
            & (pl.col("is_normal_runner_strict") == 1)
            & (pl.col("Umaban").is_not_null())
            & (pl.col("Umaban") > 0)
            & (pl.col("KettoNum").is_not_null())
            & (pl.col("KettoNum").cast(pl.String).str.len_chars() > 0)
        ).alias("eligible_for_win_training"),
        (
            pl.col("race_is_finalized")
            & (pl.col("is_normal_runner_strict") == 1)
            & (pl.col("Umaban").is_not_null())
            & (pl.col("Umaban") > 0)
            & (pl.col("KettoNum").is_not_null())
            & (pl.col("KettoNum").cast(pl.String).str.len_chars() > 0)
        ).alias("eligible_for_place_training"),
        (
            pl.col("race_is_finalized")
            & (pl.col("is_normal_runner_strict") == 1)
            & (pl.col("Umaban").is_not_null())
            & (pl.col("Umaban") > 0)
            & (pl.col("KettoNum").is_not_null())
            & (pl.col("KettoNum").cast(pl.String).str.len_chars() > 0)
        ).alias("eligible_for_ranking_training"),
    ])
    return out


def build_history_features(df: pl.DataFrame, logger: logging.Logger) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    sort_cols = ["race_date", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban", "entry_id"]
    sorted_df = df.sort(sort_cols)
    rows = sorted_df.to_dicts()

    horse_hist: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    stats = defaultdict(stats_empty)
    features: list[dict[str, Any]] = []
    leakage_samples: list[dict[str, Any]] = []

    started = time.time()
    for idx, row in enumerate(rows, start=1):
        if idx % 100000 == 0:
            logger.info("history progress rows=%s elapsed=%.1fs", idx, time.time() - started)

        entry_id = row["entry_id"]
        horse = str(row.get("KettoNum") or "")
        jockey = str(row.get("KisyuCode") or "")
        trainer = str(row.get("ChokyosiCode") or "")
        jyo = str(row.get("JyoCD") or "")
        surf = surface_code(row.get("TrackCD"))
        dist = distance_band(row.get("Kyori"))
        baba = baba_code(row)
        current_date = date_from_str(row["race_date"])

        hist = list(horse_hist[horse])
        previous = hist[-1] if hist else None
        feature = {"entry_id": entry_id}
        feature["horse_past_starts"] = len(hist)
        feature["horse_days_since_last"] = (current_date - previous["race_date_obj"]).days if previous else None
        feature["horse_distance_diff_last"] = to_float(row.get("Kyori")) - to_float(previous.get("Kyori")) if previous and to_float(row.get("Kyori")) is not None and to_float(previous.get("Kyori")) is not None else None
        feature["horse_futan_diff_last"] = to_float(row.get("Futan")) - to_float(previous.get("Futan")) if previous and to_float(row.get("Futan")) is not None and to_float(previous.get("Futan")) is not None else None
        feature["horse_body_weight_diff_last"] = to_float(row.get("BaTaijyu")) - to_float(previous.get("BaTaijyu")) if previous and to_float(row.get("BaTaijyu")) is not None and to_float(previous.get("BaTaijyu")) is not None else None

        for n in HISTORY_WINDOWS:
            recent = hist[-n:]
            feature[f"horse_last{n}_avg_finish"] = avg([to_float(r.get("KakuteiJyuni")) for r in recent])
            if n in (3, 5):
                feature[f"horse_last{n}_win_rate"] = rate([r.get("target_win_rank") for r in recent])
                feature[f"horse_last{n}_ren_rate"] = rate([r.get("target_ren_rank") for r in recent])
                feature[f"horse_last{n}_place_rate"] = rate([r.get("target_top3_rank") for r in recent])
                feature[f"horse_last{n}_avg_ninki"] = avg([to_float(r.get("Ninki")) for r in recent])
                feature[f"horse_last{n}_avg_tan_odds"] = avg([to_float(r.get("tan_odds")) for r in recent])
                feature[f"horse_last{n}_avg_haron_l3"] = avg([to_float(r.get("HaronTimeL3")) for r in recent])
                feature[f"horse_last{n}_avg_time"] = avg([to_float(r.get("Time")) for r in recent])

        feature.update(stats_features(stats[("horse_jyo", horse, jyo)], "horse_jyo"))
        feature.update(stats_features(stats[("horse_surface", horse, surf)], "horse_surface"))
        feature.update(stats_features(stats[("horse_dist_band", horse, dist)], "horse_dist_band"))
        feature.update(stats_features(stats[("horse_baba", horse, baba)], "horse_baba"))
        feature.update(stats_features(stats[("jockey", jockey)], "jockey", include_ren=True))
        feature.update(stats_features(stats[("trainer", trainer)], "trainer", include_ren=True))
        feature.update(stats_features(stats[("jockey_jyo", jockey, jyo)], "jockey_jyo"))
        feature.update(stats_features(stats[("jockey_dist_band", jockey, dist)], "jockey_dist_band"))
        feature.update(stats_features(stats[("horse_jockey", horse, jockey)], "horse_jockey"))
        features.append(feature)

        if previous and len(leakage_samples) < 200:
            leakage_samples.append({
                "entry_id": entry_id,
                "race_date": row["race_date"],
                "horse_id": horse,
                "previous_entry_id": previous["entry_id"],
                "previous_race_date": previous["race_date"],
                "previous_is_before_current": previous["race_date_obj"] < current_date or previous["race_sort_key"] < row["race_sort_key"],
            })

        update_ok = bool(row.get("race_is_finalized")) and row.get("is_normal_runner_strict") == 1 and (row.get("KakuteiJyuni") or 0) > 0
        if update_ok and horse:
            record = dict(row)
            record["race_date_obj"] = current_date
            record["race_sort_key"] = row["race_sort_key"]
            horse_hist[horse].append(record)
            if len(horse_hist[horse]) > 20:
                horse_hist[horse].popleft()
            for key in [
                ("horse_jyo", horse, jyo),
                ("horse_surface", horse, surf),
                ("horse_dist_band", horse, dist),
                ("horse_baba", horse, baba),
                ("jockey", jockey),
                ("trainer", trainer),
                ("jockey_jyo", jockey, jyo),
                ("jockey_dist_band", jockey, dist),
                ("horse_jockey", horse, jockey),
            ]:
                update_stats(stats[key], row)

    features_df = pl.DataFrame(features, infer_schema_length=10000)
    out = sorted_df.join(features_df, on="entry_id", how="left")
    return out, leakage_samples


def add_sort_key(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("Year") * 10000000000
            + pl.col("MonthDay") * 1000000
            + pl.col("JyoCD").cast(pl.Int64) * 10000
            + pl.col("Kaiji") * 1000
            + pl.col("Nichiji") * 100
            + pl.col("RaceNum")
        ).alias("race_sort_key")
    )


def write_year(df: pl.DataFrame, year: int, force: bool, resume: bool, checkpoint: dict[str, Any], logger: logging.Logger) -> dict[str, Any] | None:
    year_key = str(year)
    out = OUT_DIR / f"year={year}" / "data.parquet"
    done = checkpoint.get("years", {}).get(year_key, {}).get("status") == "complete"
    if resume and not force and done and out.exists():
        logger.info("year=%s skip complete model feature output", year)
        return None

    year_df = df.filter(pl.col("Year") == year).drop("race_sort_key")
    if year_df.height == 0:
        raise RuntimeError(f"No rows for year={year}")

    OUT_DIR.joinpath(f"year={year}").mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    started = time.time()
    logger.info("year=%s write feature parquet rows=%s", year, year_df.height)
    year_df.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    elapsed = time.time() - started

    checkpoint.setdefault("years", {})[year_key] = {
        "status": "complete",
        "rows": year_df.height,
        "output": str(out),
        "elapsed_sec": round(elapsed, 3),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_checkpoint(checkpoint)
    return summarize_year(year_df, year, elapsed)


def summarize_requested_years(df: pl.DataFrame, years: list[int], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for year in years:
        year_df = df.filter(pl.col("Year") == year).drop("race_sort_key")
        if year_df.height == 0:
            continue
        elapsed = checkpoint.get("years", {}).get(str(year), {}).get("elapsed_sec", 0.0)
        summaries.append(summarize_year(year_df, year, float(elapsed or 0.0)))
    return summaries


def summarize_year(df: pl.DataFrame, year: int, elapsed: float) -> dict[str, Any]:
    race_total = df["race_id"].n_unique()
    finalized_races = df.filter(pl.col("race_is_finalized")).select(pl.col("race_id").n_unique()).item()
    unfinalized_races = race_total - finalized_races
    return {
        "year": year,
        "rows": df.height,
        "races": race_total,
        "unfinalized_races": unfinalized_races,
        "finalized_races": finalized_races,
        "win_training_rows": int(df["eligible_for_win_training"].sum()),
        "place_training_rows": int(df["eligible_for_place_training"].sum()),
        "ranking_training_rows": int(df["eligible_for_ranking_training"].sum()),
        "target_win_rank_paid_mismatch": int((df["target_win_rank"] != df["target_win_paid"]).sum()),
        "target_top3_rank_place_paid_mismatch": int((df["target_top3_rank"] != df["target_place_paid"]).sum()),
        "target_place_by_rule_paid_mismatch": int((df["target_place_by_rule"] != df["target_place_paid"]).sum()),
        "new_horse_rows": int((df["horse_past_starts"] == 0).sum()),
        "entry_id_duplicates": df.height - df["entry_id"].n_unique(),
        "elapsed_sec": round(elapsed, 3),
    }


def quality_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for year in sorted(df["Year"].unique().to_list()):
        part = df.filter(pl.col("Year") == year)
        for col in HIST_FEATURE_COLUMNS:
            if col in part.columns:
                rows.append({
                    "year": year,
                    "column_name": col,
                    "null_count": int(part[col].null_count()),
                    "missing_rate": round(part[col].null_count() / part.height, 8),
                    "unique_count": int(part[col].n_unique()),
                })
    return rows


def label_mismatches(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("target_win_rank") != pl.col("target_win_paid"))
        | (pl.col("target_top3_rank") != pl.col("target_place_paid"))
        | (pl.col("target_place_by_rule") != pl.col("target_place_paid"))
    ).select([
        "race_id", "entry_id", "race_date", "Year", "Umaban", "Bamei", "KettoNum",
        "IJyoCD", "SyussoTosu", "NyusenJyuni", "KakuteiJyuni", "place_rank_limit",
        "target_win_rank", "target_win_paid", "target_top3_rank", "target_place_by_rule",
        "target_place_paid", "tan_pay", "fuku_pay", "is_abnormal_result",
    ])


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_docs(summaries: list[dict[str, Any]]) -> None:
    total_rows = sum(s["rows"] for s in summaries)
    total_races = sum(s["races"] for s in summaries)
    unfinalized_races = sum(s["unfinalized_races"] for s in summaries)
    win_rows = sum(s["win_training_rows"] for s in summaries)
    place_rows = sum(s["place_training_rows"] for s in summaries)
    rank_rows = sum(s["ranking_training_rows"] for s in summaries)
    win_mismatch = sum(s["target_win_rank_paid_mismatch"] for s in summaries)
    top3_mismatch = sum(s["target_top3_rank_place_paid_mismatch"] for s in summaries)
    rule_mismatch = sum(s["target_place_by_rule_paid_mismatch"] for s in summaries)
    DESIGN_DOC.write_text("\n".join([
        "# Model Feature Design",
        "",
        "入力は `outputs/base_runner_dataset/year=YYYY/data.parquet` の年別Parquetです。出力は `outputs/model_feature_dataset/year=YYYY/data.parquet` に保存します。",
        "",
        "出力は 1行=1出走馬を維持し、レース確定判定、修正ターゲット、学習対象フラグ、データ分割、市場データ取得フラグ、時系列特徴量を追加します。",
        "",
        f"- 総行数: {total_rows:,}",
        f"- 総レース数: {total_races:,}",
        f"- 未確定扱いレース数: {unfinalized_races:,}",
        f"- 単勝学習対象行: {win_rows:,}",
        f"- 複勝学習対象行: {place_rows:,}",
        f"- ランキング学習対象行: {rank_rows:,}",
        "",
        "時系列特徴量はグローバルなレース順で作成します。各行の特徴量を作成してから、その行の結果を馬、騎手、調教師、条件別の履歴に追加します。",
        "",
        "主な時系列特徴量:",
        "",
        "- 馬の近走: `horse_days_since_last`, `horse_past_starts`, `horse_last1_avg_finish`, `horse_last3_avg_finish`, `horse_last5_avg_finish`",
        "- 馬の近走率: `horse_last3_win_rate`, `horse_last5_win_rate`, `horse_last3_ren_rate`, `horse_last5_ren_rate`, `horse_last3_place_rate`, `horse_last5_place_rate`",
        "- 近走平均: `horse_last3_avg_ninki`, `horse_last5_avg_ninki`, `horse_last3_avg_tan_odds`, `horse_last5_avg_tan_odds`, `horse_last3_avg_haron_l3`, `horse_last5_avg_haron_l3`, `horse_last3_avg_time`, `horse_last5_avg_time`",
        "- 条件別: `horse_jyo_*`, `horse_surface_*`, `horse_dist_band_*`, `horse_baba_*`",
        "- 騎手・調教師: `jockey_*`, `trainer_*`, `jockey_jyo_*`, `jockey_dist_band_*`, `horse_jockey_*`",
        "",
        "`market_free` はオッズ、人気、票数を除外します。`market_aware` は `TanOdds`, `TanNinki`, `FukuOddsLow`, `FukuOddsHigh`, `FukuNinki`, `TanVote`, `FukuVote` と取得フラグを含めます。",
        "",
    ]), encoding="utf-8-sig")

    TARGET_DOC.write_text("\n".join([
        "# Target Definition",
        "",
        "- `race_has_result`: 同一 `race_id` 内に `KakuteiJyuni > 0` が存在する。",
        "- `race_has_win_payout`: 同一 `race_id` 内に `tan_pay > 0` が存在する。",
        "- `race_has_place_payout`: 同一 `race_id` 内に `fuku_pay > 0` が存在する。",
        "- `race_is_finalized`: `race_has_result` かつ、単勝または複勝払戻が確認できる。",
        "",
        "- `target_win_rank`: 通常出走馬かつ `KakuteiJyuni = 1`。",
        "- `target_ren_rank`: 通常出走馬かつ `KakuteiJyuni <= 2`。",
        "- `target_top3_rank`: 通常出走馬かつ `KakuteiJyuni <= 3`。",
        "- `target_win_paid`: `is_win_paid`。",
        "- `target_place_paid`: `is_place_paid`。回収率最大化の正式な複勝ターゲット。",
        "- `place_rank_limit`: `SyussoTosu >= 8` なら3、それ以外は2。",
        "- `target_place_by_rule`: 通常出走馬かつ `KakuteiJyuni <= place_rank_limit`。",
        "",
        f"- `target_win_rank` と `target_win_paid` の不一致: {win_mismatch:,}行",
        f"- `target_top3_rank` と `target_place_paid` の不一致: {top3_mismatch:,}行",
        f"- `target_place_by_rule` と `target_place_paid` の不一致: {rule_mismatch:,}行",
        "",
        "不一致行は削除せず `outputs/label_mismatch_cases.csv` に出力します。",
        "",
    ]), encoding="utf-8-sig")

    LEAKAGE_DOC.write_text("\n".join([
        "# Time Leakage Validation",
        "",
        "時系列特徴量は one-pass の時系列アルゴリズムで作成します。現在行は特徴量を作成した後に履歴へ追加するため、当該レース自身を含みません。",
        "",
        "検証サンプル `outputs/time_leakage_validation_samples.csv` には現在レース日と参照元過去走日を出力します。`previous_is_before_current` は全件 true である必要があります。",
        "",
        "モデル特徴量に使わない列: " + ", ".join(f"`{c}`" for c in LEAKAGE_COLUMNS) + ".",
        "",
        "オッズ、人気、票数は取得時点に依存するため `market_free` と `market_aware` を分けて扱います。",
        "",
        f"Rows processed in latest run summary: `{total_rows}`.",
        "",
    ]), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", help="Comma-separated years or ranges, e.g. 2016,2017 or 2016-2026")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    years = parse_years(args.years)
    logger.info("start years=%s resume=%s force=%s", years, args.resume, args.force)
    checkpoint = load_checkpoint()
    checkpoint["last_started_at"] = datetime.now().isoformat(timespec="seconds")
    save_checkpoint(checkpoint)

    try:
        base = read_base()
        logger.info("base loaded rows=%s cols=%s", base.height, len(base.columns))
        base = add_sort_key(add_labels(base))
        logger.info("labels added")
        feature_df, leakage_samples = build_history_features(base, logger)
        logger.info("history features done rows=%s cols=%s", feature_df.height, len(feature_df.columns))

        for year in years:
            write_year(feature_df, year, args.force, args.resume, checkpoint, logger)

        summaries = summarize_requested_years(feature_df, years, checkpoint)
        if summaries:
            write_csv_rows(ELIGIBILITY_CSV, summaries)
            quality = quality_rows(feature_df.filter(pl.col("Year").is_in(years)))
            write_csv_rows(HIST_QUALITY_CSV, quality)
            mismatches = label_mismatches(feature_df.filter(pl.col("Year").is_in(years)))
            mismatches.write_csv(LABEL_MISMATCH_CSV)
            feature_df.filter(pl.col("Year").is_in(years)).head(200).drop("race_sort_key").write_csv(SAMPLE_CSV)
            write_csv_rows(LEAKAGE_SAMPLE_CSV, leakage_samples)
            write_docs(summaries)
        checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_checkpoint(checkpoint)
        logger.info("done")
        return 0
    except Exception as exc:
        logger.error("failed: %s", exc)
        logger.error(traceback.format_exc())
        checkpoint["last_error"] = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_checkpoint(checkpoint)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
