from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
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

from src.features.feature_sets_v2_1 import feature_sets, validate_feature_sets, write_feature_set_yaml
from src.features.history_builder_v2_1 import STORE_NAMES, build_pre_day_history_features_v2_1, load_state, new_state, save_state
from src.features.target_builder import add_target_columns


BASE_DIR = Path("outputs/base_runner_dataset")
OUT_DIR = Path("outputs/model_feature_dataset_v2_1")
CHECKPOINT_DIR = Path("outputs/model_feature_dataset_v2_1_checkpoint")
CHECKPOINT_JSON = CHECKPOINT_DIR / "checkpoint.json"
LOG_PATH = Path("logs/build_model_features_v2_1.log")
FEATURE_SET_YAML = Path("config/feature_sets.yaml")

SAMPLE_CSV = Path("outputs/model_feature_dataset_v2_1_sample.csv")
ELIGIBILITY_CSV = Path("outputs/training_eligibility_summary_v2_1.csv")
LABEL_MISMATCH_CSV = Path("outputs/label_mismatch_cases_v2_1.csv")
LEAKAGE_CSV = Path("outputs/history_leakage_validation_v2_1.csv")
LEAKAGE_BY_STORE_CSV = Path("outputs/history_leakage_validation_by_store_v2_1.csv")
LEAKAGE_SAMPLES_CSV = Path("outputs/history_leakage_validation_samples_v2_1.csv")
FEATURE_INVENTORY_CSV = Path("outputs/feature_inventory_v2_1.csv")
FEATURE_SET_VALIDATION_CSV = Path("outputs/feature_set_validation_v2_1.csv")
COMPARISON_CSV = Path("outputs/model_feature_v2_v2_1_comparison.csv")

DESIGN_DOC = Path("docs/model_feature_design_v2_1.md")
LEAKAGE_DOC = Path("docs/time_leakage_validation_v2_1.md")
FEATURE_SET_DOC = Path("docs/feature_set_design_v2_1.md")
RESUME_DOC = Path("docs/resume_design_v2_1.md")

YEARS_ALL = list(range(2016, 2027))
SCRIPT_VERSION = "v2_1_pre_day_audit_20260614"
STATE_VERSION = "v2_1_pre_day_audit"


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("build_model_features_v2_1")
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def config_hash() -> str:
    if not FEATURE_SET_YAML.exists():
        return ""
    return sha256_file(FEATURE_SET_YAML)


def base_path(year: int) -> Path:
    return BASE_DIR / f"year={year}" / "data.parquet"


def out_path(year: int) -> Path:
    return OUT_DIR / f"year={year}" / "data.parquet"


def state_path(year: int) -> Path:
    return CHECKPOINT_DIR / f"history_state_after_{year}.pkl"


def load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8-sig"))
    return {"script_version": SCRIPT_VERSION, "state_version": STATE_VERSION, "years": {}}


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_JSON)


def invalidate_from_year(checkpoint: dict[str, Any], year: int) -> None:
    for key in list(checkpoint.get("years", {}).keys()):
        if int(key) >= year:
            checkpoint["years"][key]["status"] = "invalidated"


def resume_mismatches(checkpoint: dict[str, Any], years: list[int], current_config_hash: str) -> list[dict[str, Any]]:
    mismatches = []
    for year in years:
        meta = checkpoint.get("years", {}).get(str(year))
        if not meta or meta.get("status") != "complete":
            continue
        path = base_path(year)
        if not path.exists():
            mismatches.append({"year": year, "field": "input_exists", "expected": True, "actual": False})
            continue
        current_fp = file_fingerprint(path)
        for field in ["size", "mtime_ns", "sha256"]:
            expected = meta.get("input", {}).get(field)
            actual = current_fp[field]
            if expected != actual:
                mismatches.append({"year": year, "field": f"input_{field}", "expected": expected, "actual": actual})
        if meta.get("feature_set_hash") != current_config_hash:
            mismatches.append({"year": year, "field": "feature_set_hash", "expected": meta.get("feature_set_hash"), "actual": current_config_hash})
        if meta.get("script_version") != SCRIPT_VERSION:
            mismatches.append({"year": year, "field": "script_version", "expected": meta.get("script_version"), "actual": SCRIPT_VERSION})
        if meta.get("state_version") != STATE_VERSION:
            mismatches.append({"year": year, "field": "state_version", "expected": meta.get("state_version"), "actual": STATE_VERSION})
    return mismatches


def latest_completed_year(checkpoint: dict[str, Any], years: list[int]) -> int | None:
    completed = []
    for year in years:
        meta = checkpoint.get("years", {}).get(str(year), {})
        if meta.get("status") == "complete" and out_path(year).exists() and state_path(year).exists():
            completed.append(year)
    return max(completed) if completed else None


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_year(df: pl.DataFrame, year: int) -> float:
    out = out_path(year)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    started = time.time()
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    return time.time() - started


def leakage_summary_rows(audit_counts: dict[str, dict[str, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_store = []
    total = {"validation_rows": 0, "ok": 0, "no_history": 0, "same_race": 0, "same_day": 0, "future": 0}
    for store in STORE_NAMES:
        counts = audit_counts.get(store, {})
        row = {
            "store_name": store,
            "validation_rows": counts.get("validation_rows", 0),
            "ok_past_reference_count": counts.get("ok", 0),
            "no_history_count": counts.get("no_history", 0),
            "same_race_reference_count": counts.get("same_race", 0),
            "same_day_reference_count": counts.get("same_day", 0),
            "future_reference_count": counts.get("future", 0),
        }
        by_store.append(row)
        total["validation_rows"] += row["validation_rows"]
        total["ok"] += row["ok_past_reference_count"]
        total["no_history"] += row["no_history_count"]
        total["same_race"] += row["same_race_reference_count"]
        total["same_day"] += row["same_day_reference_count"]
        total["future"] += row["future_reference_count"]
    overall = [{
        "validation_rows": total["validation_rows"],
        "ok_past_reference_count": total["ok"],
        "no_history_count": total["no_history"],
        "same_race_reference_count": total["same_race"],
        "same_day_reference_count": total["same_day"],
        "future_reference_count": total["future"],
        "violation_count": total["same_race"] + total["same_day"] + total["future"],
    }]
    return overall, by_store


def aggregate_checkpoint_audit(checkpoint: dict[str, Any], years: list[int]) -> dict[str, dict[str, int]]:
    total = {store: {"validation_rows": 0, "ok": 0, "no_history": 0, "same_race": 0, "same_day": 0, "future": 0} for store in STORE_NAMES}
    for year in years:
        audit = checkpoint.get("years", {}).get(str(year), {}).get("audit_counts", {})
        for store in STORE_NAMES:
            for key in total[store]:
                total[store][key] += int(audit.get(store, {}).get(key, 0))
    return total


def summarize_year(df: pl.DataFrame, year: int, elapsed: float) -> dict[str, Any]:
    race_total = df["race_id"].n_unique()
    finalized = df.filter(pl.col("race_is_finalized")).select(pl.col("race_id").n_unique()).item()
    cutoff_violation = df.filter(pl.col("history_cutoff_date").is_not_null() & (pl.col("history_cutoff_date") >= pl.col("race_date"))).height
    return {
        "year": year,
        "rows": df.height,
        "columns": len(df.columns),
        "races": race_total,
        "finalized_races": finalized,
        "unfinalized_races": race_total - finalized,
        "win_training_rows": int(df["eligible_for_win_training"].sum()),
        "place_training_rows": int(df["eligible_for_place_training"].sum()),
        "ranking_training_rows": int(df["eligible_for_ranking_training"].sum()),
        "entry_id_duplicates": df.height - df["entry_id"].n_unique(),
        "history_cutoff_date_violation_count": cutoff_violation,
        "horse_past_starts_max": int(df["horse_past_starts"].max() or 0),
        "elapsed_sec": round(elapsed, 3),
    }


def label_mismatches(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("target_win_rank") != pl.col("target_win_paid"))
        | (pl.col("target_top3_rank") != pl.col("target_place_paid"))
        | (pl.col("target_place_by_rule") != pl.col("target_place_paid"))
    ).select([
        "race_id", "entry_id", "race_date", "Year", "Umaban", "Bamei", "KettoNum", "IJyoCD",
        "SyussoTosu", "NyusenJyuni", "KakuteiJyuni", "place_rank_limit", "target_win_rank",
        "target_win_paid", "target_top3_rank", "target_place_by_rule", "target_place_paid",
    ])


def feature_inventory(df: pl.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for set_name, groups in feature_sets().items():
        for kind, cols in groups.items():
            for col in cols:
                rows.append({
                    "feature_set": set_name,
                    "kind": kind,
                    "column_name": col,
                    "exists_in_dataset": col in df.columns,
                    "null_rate": round(df[col].null_count() / df.height, 8) if col in df.columns and df.height else None,
                })
    return rows


def compare_v2_v2_1(years: list[int]) -> list[dict[str, Any]]:
    rows = []
    for year in years:
        p2 = Path("outputs/model_feature_dataset_v2") / f"year={year}" / "data.parquet"
        p21 = out_path(year)
        v2_rows = v2_cols = v21_rows = v21_cols = None
        if p2.exists():
            d2 = pl.read_parquet(p2)
            v2_rows, v2_cols = d2.height, len(d2.columns)
        if p21.exists():
            d21 = pl.read_parquet(p21)
            v21_rows, v21_cols = d21.height, len(d21.columns)
        rows.append({"year": year, "v2_rows": v2_rows, "v2_1_rows": v21_rows, "row_diff": (v21_rows - v2_rows) if v2_rows is not None and v21_rows is not None else None, "v2_columns": v2_cols, "v2_1_columns": v21_cols})
    return rows


def process_year(year: int, state: dict[str, Any], logger: logging.Logger) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, int]], list[dict[str, Any]], float]:
    started = time.time()
    logger.info("year=%s read base", year)
    base = pl.read_parquet(base_path(year))
    logger.info("year=%s base rows=%s cols=%s", year, base.height, len(base.columns))
    labeled = add_target_columns(base)
    features, next_state, audit_counts, audit_samples = build_pre_day_history_features_v2_1(labeled, logger, state)
    elapsed = time.time() - started
    logger.info("year=%s built rows=%s cols=%s elapsed=%.1fs", year, features.height, len(features.columns), elapsed)
    return features, next_state, audit_counts, audit_samples, elapsed


def write_docs(summary: list[dict[str, Any]], leakage: list[dict[str, Any]], feature_validation: list[dict[str, str]], elapsed_total: float) -> None:
    rows = sum(r["rows"] for r in summary)
    cutoff_bad = sum(r["history_cutoff_date_violation_count"] for r in summary)
    violation = leakage[0]["violation_count"] if leakage else 0
    fs = feature_sets()
    DESIGN_DOC.write_text(f"# Model Feature Design V2.1\n\nRows: {rows:,}\n\nV2.1 keeps V1/V2 outputs intact and writes to `outputs/model_feature_dataset_v2_1`.\n\n`history_cutoff_date` is the last completed history date, not the current race date. Missing history uses NULL.\n\nElapsed seconds: {elapsed_total:.3f}\n", encoding="utf-8")
    LEAKAGE_DOC.write_text(f"# Time Leakage Validation V2.1\n\nhistory_cutoff_date violations: {cutoff_bad}\n\nLeakage violations: {violation}\n\nOutputs: `outputs/history_leakage_validation_v2_1.csv`, `outputs/history_leakage_validation_by_store_v2_1.csv`, `outputs/history_leakage_validation_samples_v2_1.csv`.\n", encoding="utf-8")
    FEATURE_SET_DOC.write_text("\n".join([
        "# Feature Set Design V2.1",
        "",
        f"- market_free: numeric {len(fs['market_free']['numeric'])}, categorical {len(fs['market_free']['categorical'])}",
        f"- market_history: numeric {len(fs['market_history']['numeric'])}, categorical {len(fs['market_history']['categorical'])}",
        f"- market_aware: numeric {len(fs['market_aware']['numeric'])}, categorical {len(fs['market_aware']['categorical'])}",
        "",
        "Validation:",
        *[f"- {r['check_name']}: {r['status']} {r['details']}" for r in feature_validation],
    ]), encoding="utf-8")
    RESUME_DOC.write_text("# Resume Design V2.1\n\nStrict resume validates input file size, mtime, SHA-256, feature set hash, script version, state version, and state file existence. `--rebuild-from-year` invalidates that year and later.\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-from-year", type=int)
    args = parser.parse_args()
    logger = setup_logging()
    years = parse_years(args.years)
    started_all = time.time()
    logger.info("start v2.1 years=%s resume=%s strict=%s force=%s rebuild_from=%s", years, args.resume, args.strict_resume, args.force, args.rebuild_from_year)

    if not (args.resume and args.strict_resume):
        write_feature_set_yaml(FEATURE_SET_YAML)
    feature_validation = validate_feature_sets()
    write_csv_rows(FEATURE_SET_VALIDATION_CSV, feature_validation)
    if any(r["status"] == "fail" for r in feature_validation):
        logger.error("feature set validation failed")
        return 1
    current_config_hash = config_hash()
    checkpoint = load_checkpoint()
    checkpoint["script_version"] = SCRIPT_VERSION
    checkpoint["state_version"] = STATE_VERSION
    checkpoint["git_sha"] = git_sha()
    checkpoint["last_started_at"] = datetime.now().isoformat(timespec="seconds")
    if args.rebuild_from_year:
        invalidate_from_year(checkpoint, args.rebuild_from_year)
    mismatches = resume_mismatches(checkpoint, years, current_config_hash) if args.resume else []
    if mismatches:
        first_year = min(int(m["year"]) for m in mismatches)
        if args.strict_resume:
            logger.error("strict resume mismatch: %s", mismatches[:5])
            save_checkpoint(checkpoint)
            return 2
        invalidate_from_year(checkpoint, first_year)
        logger.warning("resume mismatch invalidated from year=%s", first_year)
    save_checkpoint(checkpoint)

    try:
        state = new_state()
        process_years = years
        if args.resume and not args.force and not args.rebuild_from_year:
            latest = latest_completed_year(checkpoint, years)
            if latest is not None:
                state = load_state(state_path(latest))
                process_years = [y for y in years if y > latest]
                logger.info("resume from year=%s remaining=%s", latest, process_years)
        elif args.resume and args.rebuild_from_year:
            prev = args.rebuild_from_year - 1
            if prev in years and state_path(prev).exists():
                state = load_state(state_path(prev))
            process_years = [y for y in years if y >= args.rebuild_from_year]
            logger.info("rebuild from year=%s remaining=%s", args.rebuild_from_year, process_years)

        all_samples: list[dict[str, Any]] = []
        total_audit = {store: {"validation_rows": 0, "ok": 0, "no_history": 0, "same_race": 0, "same_day": 0, "future": 0} for store in STORE_NAMES}
        for year in process_years:
            features, state, audit_counts, samples, elapsed = process_year(year, state, logger)
            write_elapsed = write_year(features, year)
            save_state(state_path(year), state)
            for store in STORE_NAMES:
                for key in total_audit[store]:
                    total_audit[store][key] += audit_counts[store][key]
            all_samples.extend(samples[:1000])
            checkpoint.setdefault("years", {})[str(year)] = {
                "status": "complete",
                "input": file_fingerprint(base_path(year)),
                "feature_set_hash": current_config_hash,
                "script_version": SCRIPT_VERSION,
                "state_version": STATE_VERSION,
                "git_sha": checkpoint["git_sha"],
                "state": str(state_path(year)),
                "output": str(out_path(year)),
                "rows": features.height,
                "columns": len(features.columns),
                "audit_counts": audit_counts,
                "elapsed_sec": round(elapsed + write_elapsed, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_checkpoint(checkpoint)

        existing_years = [y for y in years if out_path(y).exists()]
        frames = [pl.read_parquet(out_path(y)) for y in existing_years]
        combined = pl.concat(frames, how="diagonal_relaxed")
        summaries = [summarize_year(combined.filter(pl.col("Year") == y), y, checkpoint.get("years", {}).get(str(y), {}).get("elapsed_sec", 0.0)) for y in existing_years]
        write_csv_rows(ELIGIBILITY_CSV, summaries)
        label_mismatches(combined).write_csv(LABEL_MISMATCH_CSV)
        write_csv_rows(FEATURE_INVENTORY_CSV, feature_inventory(combined))
        write_csv_rows(COMPARISON_CSV, compare_v2_v2_1(existing_years))
        combined.head(200).write_csv(SAMPLE_CSV)
        total_audit = aggregate_checkpoint_audit(checkpoint, existing_years)
        overall, by_store = leakage_summary_rows(total_audit)
        write_csv_rows(LEAKAGE_CSV, overall)
        write_csv_rows(LEAKAGE_BY_STORE_CSV, by_store)
        write_csv_rows(LEAKAGE_SAMPLES_CSV, all_samples[:5000])
        elapsed_total = time.time() - started_all
        write_docs(summaries, overall, feature_validation, elapsed_total)
        cutoff_bad = sum(r["history_cutoff_date_violation_count"] for r in summaries)
        if overall[0]["violation_count"] or cutoff_bad:
            logger.error("leakage validation failed violation=%s cutoff_bad=%s", overall[0]["violation_count"], cutoff_bad)
            return 3
        checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
        checkpoint["elapsed_sec_last_run"] = round(elapsed_total, 3)
        save_checkpoint(checkpoint)
        logger.info("done v2.1 rows=%s years=%s elapsed=%.1fs", combined.height, existing_years, elapsed_total)
        return 0
    except Exception as exc:
        logger.error("failed: %s", exc)
        logger.error(traceback.format_exc())
        checkpoint["last_error"] = {"at": datetime.now().isoformat(timespec="seconds"), "error": str(exc), "traceback": traceback.format_exc()}
        save_checkpoint(checkpoint)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
