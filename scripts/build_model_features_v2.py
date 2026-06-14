from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_sets import feature_sets, validate_feature_sets, write_feature_set_yaml
from src.features.history_builder import HIST_FEATURE_COLUMNS, build_pre_day_history_features, load_state, new_state, save_state
from src.features.target_builder import add_target_columns


BASE_DIR = Path("outputs/base_runner_dataset")
OUT_DIR = Path("outputs/model_feature_dataset_v2")
CHECKPOINT_DIR = Path("outputs/model_feature_dataset_v2_checkpoint")
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "build_model_features_v2.log"

SAMPLE_CSV = Path("outputs/model_feature_dataset_v2_sample.csv")
ELIGIBILITY_CSV = Path("outputs/training_eligibility_summary_v2.csv")
LABEL_MISMATCH_CSV = Path("outputs/label_mismatch_cases_v2.csv")
LEAKAGE_CSV = Path("outputs/history_leakage_validation_v2.csv")
FEATURE_INVENTORY_CSV = Path("outputs/feature_inventory_v2.csv")
COMPARISON_CSV = Path("outputs/model_feature_v1_v2_comparison.csv")
CHECKPOINT_JSON = CHECKPOINT_DIR / "checkpoint.json"

DESIGN_DOC = Path("docs/model_feature_design_v2.md")
TARGET_DOC = Path("docs/target_definition_v2.md")
LEAKAGE_DOC = Path("docs/time_leakage_validation_v2.md")
FEATURE_SET_DOC = Path("docs/feature_set_design.md")
FEATURE_SET_YAML = Path("config/feature_sets.yaml")

YEARS_ALL = list(range(2016, 2027))
SCRIPT_VERSION = "v2_pre_day_phase1"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("build_model_features_v2")
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
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    return {"script_version": SCRIPT_VERSION, "years": {}}


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_JSON)


def base_path(year: int) -> Path:
    return BASE_DIR / f"year={year}" / "data.parquet"


def out_path(year: int) -> Path:
    return OUT_DIR / f"year={year}" / "data.parquet"


def state_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"history_state_after_{year}.pkl"


def read_base_year(year: int) -> pl.DataFrame:
    path = base_path(year)
    if not path.exists():
        raise RuntimeError(f"Missing base parquet: {path}")
    return pl.read_parquet(path)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8-sig")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_year(df: pl.DataFrame, year: int) -> float:
    year_out = out_path(year)
    year_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = year_out.with_suffix(".tmp")
    started = time.time()
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(year_out)
    return time.time() - started


def summarize_year(df: pl.DataFrame, year: int, elapsed_sec: float) -> dict[str, Any]:
    race_total = df["race_id"].n_unique()
    finalized_races = df.filter(pl.col("race_is_finalized")).select(pl.col("race_id").n_unique()).item()
    return {
        "year": year,
        "rows": df.height,
        "columns": len(df.columns),
        "races": race_total,
        "finalized_races": finalized_races,
        "unfinalized_races": race_total - finalized_races,
        "win_training_rows": int(df["eligible_for_win_training"].sum()),
        "place_training_rows": int(df["eligible_for_place_training"].sum()),
        "ranking_training_rows": int(df["eligible_for_ranking_training"].sum()),
        "target_win_rank_paid_mismatch": int((df["target_win_rank"] != df["target_win_paid"]).sum()),
        "target_top3_rank_place_paid_mismatch": int((df["target_top3_rank"] != df["target_place_paid"]).sum()),
        "target_place_by_rule_paid_mismatch": int((df["target_place_by_rule"] != df["target_place_paid"]).sum()),
        "entry_id_duplicates": df.height - df["entry_id"].n_unique(),
        "horse_past_starts_max": int(df["horse_past_starts"].max() or 0),
        "new_horse_rows": int((df["horse_past_starts"] == 0).sum()),
        "win_no_result_rows": int((df["win_training_exclusion_reason"] == "no_result").sum()),
        "place_no_result_rows": int((df["place_training_exclusion_reason"] == "no_result").sum()),
        "ranking_no_result_rows": int((df["ranking_training_exclusion_reason"] == "no_result").sum()),
        "win_invalid_horse_id_rows": int((df["win_training_exclusion_reason"] == "invalid_horse_id").sum()),
        "place_invalid_horse_id_rows": int((df["place_training_exclusion_reason"] == "invalid_horse_id").sum()),
        "ranking_invalid_horse_id_rows": int((df["ranking_training_exclusion_reason"] == "invalid_horse_id").sum()),
        "elapsed_sec": round(elapsed_sec, 3),
    }


def label_mismatches(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("target_win_rank") != pl.col("target_win_paid"))
        | (pl.col("target_top3_rank") != pl.col("target_place_paid"))
        | (pl.col("target_place_by_rule") != pl.col("target_place_paid"))
    ).select([
        "race_id", "entry_id", "race_date", "Year", "Umaban", "Bamei", "KettoNum", "IJyoCD",
        "SyussoTosu", "NyusenJyuni", "KakuteiJyuni", "place_bet_available_by_rule",
        "place_rank_limit", "target_win_rank", "target_win_paid", "target_top3_rank",
        "target_place_by_rule", "target_place_paid", "tan_pay", "fuku_pay", "is_abnormal_result",
    ])


def feature_inventory(df: pl.DataFrame) -> list[dict[str, Any]]:
    sets = feature_sets()
    inventory: list[dict[str, Any]] = []
    for set_name, groups in sets.items():
        for kind, columns in groups.items():
            for column in columns:
                inventory.append({
                    "feature_set": set_name,
                    "kind": kind,
                    "column_name": column,
                    "exists_in_dataset": column in df.columns,
                    "null_rate": round(df[column].null_count() / df.height, 8) if column in df.columns and df.height else None,
                })
    return inventory


def compare_v1_v2(years: list[int]) -> list[dict[str, Any]]:
    rows = []
    for year in years:
        v1 = Path("outputs/model_feature_dataset") / f"year={year}" / "data.parquet"
        v2 = out_path(year)
        v1_rows = v1_cols = None
        v2_rows = v2_cols = None
        if v1.exists():
            df1 = pl.read_parquet(v1)
            v1_rows, v1_cols = df1.height, len(df1.columns)
        if v2.exists():
            df2 = pl.read_parquet(v2)
            v2_rows, v2_cols = df2.height, len(df2.columns)
        rows.append({
            "year": year,
            "v1_rows": v1_rows,
            "v2_rows": v2_rows,
            "row_diff": (v2_rows - v1_rows) if v1_rows is not None and v2_rows is not None else None,
            "v1_columns": v1_cols,
            "v2_columns": v2_cols,
        })
    return rows


def latest_completed_year(checkpoint: dict[str, Any]) -> int | None:
    complete = [int(y) for y, meta in checkpoint.get("years", {}).items() if meta.get("status") == "complete" and state_path(int(y)).exists() and out_path(int(y)).exists()]
    return max(complete) if complete else None


def process_year(year: int, state: dict[str, Any], logger: logging.Logger) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]], float]:
    started = time.time()
    logger.info("year=%s read base", year)
    base = read_base_year(year)
    logger.info("year=%s base rows=%s cols=%s", year, base.height, len(base.columns))
    labeled = add_target_columns(base)
    logger.info("year=%s labels added", year)
    features, next_state, leakage_samples = build_pre_day_history_features(labeled, logger, state)
    elapsed = time.time() - started
    logger.info("year=%s features built rows=%s cols=%s elapsed=%.1fs", year, features.height, len(features.columns), elapsed)
    return features, next_state, leakage_samples, elapsed


def write_docs(summaries: list[dict[str, Any]], leakage_rows: list[dict[str, Any]], feature_inventory_rows: list[dict[str, Any]]) -> None:
    total_rows = sum(r["rows"] for r in summaries)
    total_races = sum(r["races"] for r in summaries)
    total_unfinalized = sum(r["unfinalized_races"] for r in summaries)
    total_win = sum(r["win_training_rows"] for r in summaries)
    total_place = sum(r["place_training_rows"] for r in summaries)
    total_rank = sum(r["ranking_training_rows"] for r in summaries)
    leakage_bad = sum(1 for r in leakage_rows if not r.get("source_before_current"))
    same_day = sum(1 for r in leakage_rows if r.get("same_day_reference"))
    market_free_num = sum(1 for r in feature_inventory_rows if r["feature_set"] == "market_free" and r["kind"] == "numeric")
    market_free_cat = sum(1 for r in feature_inventory_rows if r["feature_set"] == "market_free" and r["kind"] == "categorical")
    market_aware_num = sum(1 for r in feature_inventory_rows if r["feature_set"] == "market_aware" and r["kind"] == "numeric")
    market_aware_cat = sum(1 for r in feature_inventory_rows if r["feature_set"] == "market_aware" and r["kind"] == "categorical")

    DESIGN_DOC.write_text("\n".join([
        "# Model Feature Design V2",
        "",
        "V2 is a Phase 1 pre-day dataset. It does not use time-series odds, model training, backtesting, Optuna, or betting strategy optimization.",
        "",
        "Outputs are written separately from V1 under `outputs/model_feature_dataset_v2/year=YYYY/data.parquet`.",
        "",
        f"- rows: {total_rows:,}",
        f"- races: {total_races:,}",
        f"- unfinalized races: {total_unfinalized:,}",
        f"- win training rows: {total_win:,}",
        f"- place training rows: {total_place:,}",
        f"- ranking training rows: {total_rank:,}",
        "",
        "History snapshot policy:",
        "",
        "- `feature_snapshot_mode = pre_day`",
        "- `historical_source_race_date < current_race_date`",
        "- same-day results are not used, even if race number is earlier",
        "- current race rows are fully scored before any result from that date is added to history",
        "",
        "Resume stores yearly history state files under `outputs/model_feature_dataset_v2_checkpoint/history_state_after_YYYY.pkl`.",
    ]), encoding="utf-8")

    TARGET_DOC.write_text("\n".join([
        "# Target Definition V2",
        "",
        "- `target_win_rank`: normal runner and `KakuteiJyuni = 1`",
        "- `target_ren_rank`: normal runner and `KakuteiJyuni <= 2`",
        "- `target_top3_rank`: normal runner and `KakuteiJyuni <= 3`",
        "- `target_win_paid`: `is_win_paid`",
        "- `target_place_paid`: `is_place_paid`; formal target for place ROI modeling",
        "",
        "Place rule diagnostics:",
        "",
        "- `SyussoTosu <= 4`: `place_bet_available_by_rule=False`, `place_rank_limit=0`",
        "- `SyussoTosu 5..7`: `place_bet_available_by_rule=True`, `place_rank_limit=2`",
        "- `SyussoTosu >= 8`: `place_bet_available_by_rule=True`, `place_rank_limit=3`",
        "",
        "Eligibility flags are separated for win, place, and ranking. Exclusion reasons are stored in `*_training_exclusion_reason` columns.",
    ]), encoding="utf-8")

    LEAKAGE_DOC.write_text("\n".join([
        "# Time Leakage Validation V2",
        "",
        "V2 uses pre-day history. The same date is never available as history for any race on that date.",
        "",
        f"- validation sample rows: {len(leakage_rows):,}",
        f"- future or same-day reference violations: {leakage_bad:,}",
        f"- same-day references: {same_day:,}",
        "",
        "Leakage sample output: `outputs/history_leakage_validation_v2.csv`.",
    ]), encoding="utf-8")

    FEATURE_SET_DOC.write_text("\n".join([
        "# Feature Set Design",
        "",
        "`config/feature_sets.yaml` is generated from an allow-list. Leakage columns, target columns, payout columns, split columns, and raw result columns are not included.",
        "",
        f"- market_free numeric: {market_free_num}",
        f"- market_free categorical: {market_free_cat}",
        f"- market_aware numeric: {market_aware_num}",
        f"- market_aware categorical: {market_aware_cat}",
        "",
        "Phase 1 does not include time-series odds features. `market_aware` only adds current normalized market columns and market availability flags.",
    ]), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", help="Comma-separated years or ranges, e.g. 2016-2017")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    years = parse_years(args.years)
    logger.info("start v2 years=%s resume=%s force=%s", years, args.resume, args.force)
    feature_errors = validate_feature_sets()
    if feature_errors:
        for err in feature_errors:
            logger.error("feature set error: %s", err)
        return 1
    write_feature_set_yaml(FEATURE_SET_YAML)

    checkpoint = load_checkpoint()
    if checkpoint.get("script_version") != SCRIPT_VERSION and args.resume and not args.force:
        logger.error("checkpoint script version mismatch: %s", checkpoint.get("script_version"))
        return 1
    checkpoint["script_version"] = SCRIPT_VERSION
    checkpoint["last_started_at"] = datetime.now().isoformat(timespec="seconds")
    save_checkpoint(checkpoint)

    try:
        process_years = years
        state = new_state()
        if args.resume and not args.force:
            latest = latest_completed_year(checkpoint)
            if latest is not None:
                state = load_state(state_path(latest))
                process_years = [y for y in years if y > latest]
                logger.info("resume from year=%s remaining=%s", latest, process_years)
        if args.force:
            logger.info("force mode starts with empty history state for requested years")

        all_leakage_samples: list[dict[str, Any]] = []
        latest_feature_df: pl.DataFrame | None = None
        for year in process_years:
            if args.resume and not args.force and checkpoint.get("years", {}).get(str(year), {}).get("status") == "complete" and out_path(year).exists():
                logger.info("year=%s skip complete", year)
                continue
            feature_df, state, leakage_samples, elapsed = process_year(year, state, logger)
            write_elapsed = write_year(feature_df, year)
            save_state(state_path(year), state)
            checkpoint.setdefault("years", {})[str(year)] = {
                "status": "complete",
                "rows": feature_df.height,
                "columns": len(feature_df.columns),
                "output": str(out_path(year)),
                "state": str(state_path(year)),
                "input": str(base_path(year)),
                "elapsed_sec": round(elapsed + write_elapsed, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_checkpoint(checkpoint)
            all_leakage_samples.extend(leakage_samples)
            latest_feature_df = feature_df
            logger.info("year=%s complete write_elapsed=%.3fs", year, write_elapsed)

        existing_years = [y for y in years if out_path(y).exists()]
        frames = [pl.read_parquet(out_path(y)) for y in existing_years]
        if not frames:
            raise RuntimeError("No V2 output frames were produced or available")
        combined = pl.concat(frames, how="diagonal_relaxed")
        summaries = [
            summarize_year(
                combined.filter(pl.col("Year") == y),
                y,
                checkpoint.get("years", {}).get(str(y), {}).get("elapsed_sec", 0.0),
            )
            for y in existing_years
        ]
        write_csv_rows(ELIGIBILITY_CSV, summaries)
        mismatch_df = label_mismatches(combined)
        mismatch_df.write_csv(LABEL_MISMATCH_CSV)
        leakage_df = pl.DataFrame(all_leakage_samples) if all_leakage_samples else pl.DataFrame({"entry_id": [], "source_before_current": []})
        leakage_df.write_csv(LEAKAGE_CSV)
        combined.head(200).write_csv(SAMPLE_CSV)
        inventory_rows = feature_inventory(combined)
        write_csv_rows(FEATURE_INVENTORY_CSV, inventory_rows)
        write_csv_rows(COMPARISON_CSV, compare_v1_v2(existing_years))
        write_docs(summaries, all_leakage_samples, inventory_rows)

        checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_checkpoint(checkpoint)
        logger.info("done v2 rows=%s years=%s latest_df_rows=%s", combined.height, existing_years, latest_feature_df.height if latest_feature_df is not None else 0)
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
