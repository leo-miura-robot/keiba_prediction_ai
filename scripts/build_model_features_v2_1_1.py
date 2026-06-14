from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import shutil
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

from src.features.feature_sets_v2_1_1 import (
    canonical_feature_set_text,
    feature_inventory_rows,
    load_feature_set_yaml,
    validate_feature_sets_from_file,
    write_feature_set_yaml,
)
from src.features.history_builder_v2_1_1 import (
    STATE_VERSION,
    STORE_NAMES,
    build_pre_day_history_features_v2_1,
    load_state,
    mark_completed_year,
    new_state,
    save_state,
)
from src.features.target_builder import add_target_columns


BASE_DIR = Path("outputs/base_runner_dataset")
OUT_DIR = Path("outputs/model_feature_dataset_v2_1_1")
CHECKPOINT_DIR = Path("outputs/model_feature_dataset_v2_1_1_checkpoint")
CHECKPOINT_JSON = CHECKPOINT_DIR / "checkpoint.json"
LOG_PATH = Path("logs/build_model_features_v2_1_1.log")
FEATURE_SET_YAML = Path("config/feature_sets_v2_1_1.yaml")

SAMPLE_CSV = Path("outputs/model_feature_dataset_v2_1_1_sample.csv")
ELIGIBILITY_CSV = Path("outputs/training_eligibility_summary_v2_1_1.csv")
LABEL_MISMATCH_CSV = Path("outputs/label_mismatch_cases_v2_1_1.csv")
LEAKAGE_CSV = Path("outputs/history_leakage_validation_v2_1_1.csv")
LEAKAGE_BY_STORE_CSV = Path("outputs/history_leakage_validation_by_store_v2_1_1.csv")
LEAKAGE_SAMPLES_CSV = Path("outputs/history_leakage_validation_samples_v2_1_1.csv")
FEATURE_INVENTORY_CSV = Path("outputs/feature_inventory_v2_1_1.csv")
FEATURE_SET_VALIDATION_CSV = Path("outputs/feature_set_validation_v2_1_1.csv")
RESUME_VALIDATION_CSV = Path("outputs/resume_validation_v2_1_1.csv")
COMPARISON_CSV = Path("outputs/model_feature_v2_1_v2_1_1_comparison.csv")

DESIGN_DOC = Path("docs/model_feature_design_v2_1_1.md")
RESUME_DOC = Path("docs/resume_design_v2_1_1.md")
FEATURE_SET_DOC = Path("docs/feature_set_design_v2_1_1.md")

YEARS_ALL = list(range(2016, 2027))
SCRIPT_VERSION = "v2_1_1_resume_safety_20260614"
CODE_FILES = [
    Path("scripts/build_model_features_v2_1_1.py"),
    Path("src/features/history_builder_v2_1_1.py"),
    Path("src/features/feature_sets_v2_1_1.py"),
    Path("src/features/target_builder.py"),
    FEATURE_SET_YAML,
]


class ResumeValidationError(RuntimeError):
    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        first = issues[0] if issues else {}
        year = first.get("year", "")
        field = first.get("field", "unknown")
        super().__init__(f"Resume validation failed at year {year}: {field}")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("build_model_features_v2_1_1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def feature_set_hash() -> str:
    return sha256_text(canonical_feature_set_text(FEATURE_SET_YAML))


def code_bundle_hash() -> str:
    h = hashlib.sha256()
    for rel in CODE_FILES:
        path = FEATURE_SET_YAML if rel == Path("config/feature_sets_v2_1_1.yaml") else ROOT / rel
        h.update(str(rel).replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception:
        return {"git_commit_sha": "unknown", "git_is_dirty": None}


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


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def issue(year: int | str, field: str, expected: Any, actual: Any, message: str) -> dict[str, Any]:
    cmd_year = year if isinstance(year, int) else 2016
    return {
        "year": year,
        "field": field,
        "expected": expected,
        "actual": actual,
        "message": message,
        "recommended_command": f"python scripts/build_model_features_v2_1_1.py --resume --rebuild-from-year {cmd_year}",
    }


def read_state_checked(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def complete_years_contiguous(checkpoint: dict[str, Any], start_year: int = 2016) -> tuple[list[int], list[dict[str, Any]]]:
    years_meta = checkpoint.get("years", {})
    complete = {int(y) for y, meta in years_meta.items() if meta.get("status") == "complete"}
    contiguous: list[int] = []
    issues: list[dict[str, Any]] = []
    y = start_year
    while y in complete:
        contiguous.append(y)
        y += 1
    later = sorted(complete - set(contiguous))
    if later:
        issues.append(issue(later[0], "non_contiguous_complete_year", f"complete years contiguous from {start_year}", later, "Complete year exists after a gap."))
    return contiguous, issues


def validate_completed_year(checkpoint: dict[str, Any], year: int, expected_feature_hash: str, expected_code_hash: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    meta = checkpoint.get("years", {}).get(str(year), {})
    if meta.get("status") != "complete":
        return [issue(year, "status", "complete", meta.get("status"), "Year is not complete.")]
    sp = state_path(year)
    op = out_path(year)
    if not sp.exists():
        issues.append(issue(year, "state_exists", True, False, "State pickle is missing."))
    else:
        try:
            state = read_state_checked(sp)
            if state.get("history_state_version") != STATE_VERSION:
                issues.append(issue(year, "state_version", STATE_VERSION, state.get("history_state_version"), "State version mismatch."))
            if state.get("completed_year") != year:
                issues.append(issue(year, "state_completed_year", year, state.get("completed_year"), "State completed year mismatch."))
        except Exception as exc:
            issues.append(issue(year, "state_readable", True, repr(exc), "State pickle is unreadable."))
    if not op.exists():
        issues.append(issue(year, "output_exists", True, False, "Output parquet is missing."))
    else:
        try:
            rows = pl.read_parquet(op).height
            if rows != meta.get("rows"):
                issues.append(issue(year, "output_rows", meta.get("rows"), rows, "Output row count mismatch."))
        except Exception as exc:
            issues.append(issue(year, "output_readable", True, repr(exc), "Output parquet is unreadable."))
    inp = base_path(year)
    if not inp.exists():
        issues.append(issue(year, "input_exists", True, False, "Input parquet is missing."))
    else:
        current = file_fingerprint(inp)
        for field in ["size", "mtime_ns", "sha256"]:
            expected = meta.get("input", {}).get(field)
            if expected != current[field]:
                issues.append(issue(year, f"input_{field}", expected, current[field], "Input fingerprint mismatch."))
    if meta.get("feature_set_hash") != expected_feature_hash:
        issues.append(issue(year, "feature_set_hash", meta.get("feature_set_hash"), expected_feature_hash, "Feature set hash mismatch."))
    if meta.get("code_bundle_hash") != expected_code_hash:
        issues.append(issue(year, "code_bundle_hash", meta.get("code_bundle_hash"), expected_code_hash, "Code bundle hash mismatch."))
    if meta.get("state_version") != STATE_VERSION:
        issues.append(issue(year, "checkpoint_state_version", STATE_VERSION, meta.get("state_version"), "Checkpoint state version mismatch."))
    return issues


def validate_resume(checkpoint: dict[str, Any], expected_feature_hash: str, expected_code_hash: str) -> tuple[list[int], list[dict[str, Any]]]:
    contiguous, issues = complete_years_contiguous(checkpoint)
    for year in contiguous:
        issues.extend(validate_completed_year(checkpoint, year, expected_feature_hash, expected_code_hash))
    return contiguous, issues


def remove_year_outputs(year: int, logger: logging.Logger) -> None:
    targets = [state_path(year), out_path(year)]
    for target in targets:
        if target.exists():
            logger.info("remove rebuild target %s", target)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


def invalidate_from_year(checkpoint: dict[str, Any], year: int, logger: logging.Logger) -> None:
    for key in list(checkpoint.get("years", {}).keys()):
        if int(key) >= year:
            logger.info("invalidate checkpoint year=%s", key)
            checkpoint["years"][key]["status"] = "invalidated"
            remove_year_outputs(int(key), logger)


def load_rebuild_state(year: int) -> dict[str, Any]:
    if year == 2016:
        return new_state()
    prev = year - 1
    sp = state_path(prev)
    if not sp.exists():
        raise ResumeValidationError([issue(prev, "previous_state_exists", True, False, f"Previous year state is required to rebuild from {year}.")])
    state = load_state(sp)
    if state.get("history_state_version") != STATE_VERSION or state.get("completed_year") != prev:
        raise ResumeValidationError([issue(prev, "previous_state_valid", f"{STATE_VERSION}/{prev}", f"{state.get('history_state_version')}/{state.get('completed_year')}", "Previous state is not valid for rebuild.")])
    return state


def write_year(df: pl.DataFrame, year: int) -> float:
    op = out_path(year)
    op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_suffix(".tmp")
    started = time.time()
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(op)
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
    return [{
        "validation_rows": total["validation_rows"],
        "ok_past_reference_count": total["ok"],
        "no_history_count": total["no_history"],
        "same_race_reference_count": total["same_race"],
        "same_day_reference_count": total["same_day"],
        "future_reference_count": total["future"],
        "violation_count": total["same_race"] + total["same_day"] + total["future"],
    }], by_store


def aggregate_checkpoint_audit(checkpoint: dict[str, Any], years: list[int]) -> dict[str, dict[str, int]]:
    total = {store: {"validation_rows": 0, "ok": 0, "no_history": 0, "same_race": 0, "same_day": 0, "future": 0} for store in STORE_NAMES}
    for year in years:
        audit = checkpoint.get("years", {}).get(str(year), {}).get("audit_counts", {})
        for store in STORE_NAMES:
            for key in total[store]:
                total[store][key] += int(audit.get(store, {}).get(key, 0))
    return total


def summarize_year(df: pl.DataFrame, year: int, elapsed: float) -> dict[str, Any]:
    races = df["race_id"].n_unique()
    finalized = df.filter(pl.col("race_is_finalized")).select(pl.col("race_id").n_unique()).item()
    cutoff_bad = df.filter(pl.col("history_cutoff_date").is_not_null() & (pl.col("history_cutoff_date") >= pl.col("race_date"))).height
    return {
        "year": year,
        "rows": df.height,
        "columns": len(df.columns),
        "races": races,
        "finalized_races": finalized,
        "unfinalized_races": races - finalized,
        "entry_id_duplicates": df.height - df["entry_id"].n_unique(),
        "history_cutoff_date_violation_count": cutoff_bad,
        "win_training_rows": int(df["eligible_for_win_training"].sum()),
        "place_training_rows": int(df["eligible_for_place_training"].sum()),
        "ranking_training_rows": int(df["eligible_for_ranking_training"].sum()),
        "elapsed_sec": round(float(elapsed), 3),
    }


def label_mismatches(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("target_win_rank") != pl.col("target_win_paid"))
        | (pl.col("target_top3_rank") != pl.col("target_place_paid"))
        | (pl.col("target_place_by_rule") != pl.col("target_place_paid"))
    ).select(["race_id", "entry_id", "race_date", "Year", "Umaban", "Bamei", "KettoNum", "IJyoCD", "SyussoTosu", "KakuteiJyuni", "target_win_rank", "target_win_paid", "target_top3_rank", "target_place_by_rule", "target_place_paid"])


def compare_v2_1_v2_1_1(years: list[int]) -> list[dict[str, Any]]:
    rows = []
    for year in years:
        p_old = Path("outputs/model_feature_dataset_v2_1") / f"year={year}" / "data.parquet"
        p_new = out_path(year)
        old_rows = old_cols = new_rows = new_cols = None
        if p_old.exists():
            old = pl.read_parquet(p_old)
            old_rows, old_cols = old.height, len(old.columns)
        if p_new.exists():
            new = pl.read_parquet(p_new)
            new_rows, new_cols = new.height, len(new.columns)
        rows.append({"year": year, "v2_1_rows": old_rows, "v2_1_1_rows": new_rows, "row_diff": (new_rows - old_rows) if old_rows is not None and new_rows is not None else None, "v2_1_columns": old_cols, "v2_1_1_columns": new_cols})
    return rows


def write_docs(summaries: list[dict[str, Any]], leakage: list[dict[str, Any]], elapsed: float, validation_rows: list[dict[str, str]]) -> None:
    total_rows = sum(r["rows"] for r in summaries)
    DESIGN_DOC.write_text(f"# Model Feature Design V2.1.1\n\nRows: {total_rows:,}\n\nV2.1.1 is a separate resume-safety build. It writes to `outputs/model_feature_dataset_v2_1_1` and does not overwrite V1/V2/V2.1.\n\nElapsed seconds: {elapsed:.3f}\n", encoding="utf-8")
    RESUME_DOC.write_text("# Resume Design V2.1.1\n\nStrict resume validates contiguous completion from 2016, state pickle, output parquet, row count, state version, completed year, input fingerprint, feature set hash, and code bundle hash. Git SHA is recorded but not used as a resume-failure condition.\n\nUse `--rebuild-from-year YYYY` to regenerate YYYY and later. For YYYY > 2016, the previous year state is required even if the previous year is not included in `--years`.\n", encoding="utf-8")
    FEATURE_SET_DOC.write_text("\n".join(["# Feature Set Design V2.1.1", "", "Dedicated config: `config/feature_sets_v2_1_1.yaml`.", "", *[f"- {r['check_name']}: {r['status']} {r['details']}" for r in validation_rows]]), encoding="utf-8")


def process_year(year: int, state: dict[str, Any], logger: logging.Logger) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, int]], list[dict[str, Any]], float]:
    started = time.time()
    base = pl.read_parquet(base_path(year))
    logger.info("year=%s base rows=%s cols=%s", year, base.height, len(base.columns))
    labeled = add_target_columns(base)
    features, next_state, audit_counts, audit_samples = build_pre_day_history_features_v2_1(labeled, logger, state)
    mark_completed_year(next_state, year)
    return features, next_state, audit_counts, audit_samples, time.time() - started


def finalize_outputs(years: list[int], checkpoint: dict[str, Any], started: float, validation_rows: list[dict[str, str]]) -> int:
    existing = [y for y in years if out_path(y).exists()]
    frames = [pl.read_parquet(out_path(y)) for y in existing]
    combined = pl.concat(frames, how="diagonal_relaxed")
    summaries = [summarize_year(combined.filter(pl.col("Year") == y), y, checkpoint.get("years", {}).get(str(y), {}).get("elapsed_sec", 0.0)) for y in existing]
    write_csv_rows(ELIGIBILITY_CSV, summaries)
    label_mismatches(combined).write_csv(LABEL_MISMATCH_CSV)
    combined.head(200).write_csv(SAMPLE_CSV)
    null_rates = {c: round(combined[c].null_count() / combined.height, 8) for c in combined.columns}
    write_csv_rows(Path("outputs/feature_inventory_v2_1_1.csv"), feature_inventory_rows(FEATURE_SET_YAML, set(combined.columns), null_rates))
    write_csv_rows(COMPARISON_CSV, compare_v2_1_v2_1_1(existing))
    audit_total = aggregate_checkpoint_audit(checkpoint, existing)
    overall, by_store = leakage_summary_rows(audit_total)
    write_csv_rows(LEAKAGE_CSV, overall)
    write_csv_rows(LEAKAGE_BY_STORE_CSV, by_store)
    write_csv_rows(FEATURE_SET_VALIDATION_CSV, validation_rows)
    elapsed = time.time() - started
    write_docs(summaries, overall, elapsed, validation_rows)
    cutoff_bad = sum(r["history_cutoff_date_violation_count"] for r in summaries)
    if cutoff_bad or overall[0]["violation_count"]:
        return 3
    checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
    checkpoint["elapsed_sec_last_run"] = round(elapsed, 3)
    save_checkpoint(checkpoint)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-from-year", type=int)
    args = parser.parse_args()
    logger = setup_logging()
    started = time.time()
    years = parse_years(args.years)
    logger.info("start v2.1.1 years=%s resume=%s strict=%s force=%s rebuild_from=%s", years, args.resume, args.strict_resume, args.force, args.rebuild_from_year)

    if not FEATURE_SET_YAML.exists():
        write_feature_set_yaml(FEATURE_SET_YAML)
    load_feature_set_yaml(FEATURE_SET_YAML)
    validation_rows = validate_feature_sets_from_file(FEATURE_SET_YAML)
    write_csv_rows(FEATURE_SET_VALIDATION_CSV, validation_rows)
    if any(r["status"] == "fail" for r in validation_rows):
        logger.error("feature set validation failed")
        return 1
    fhash = feature_set_hash()
    chash = code_bundle_hash()
    ginfo = git_info()
    checkpoint = load_checkpoint()
    checkpoint.update({"script_version": SCRIPT_VERSION, "state_version": STATE_VERSION, "feature_set_path": str(FEATURE_SET_YAML), "last_started_at": datetime.now().isoformat(timespec="seconds"), **ginfo})
    try:
        if args.rebuild_from_year:
            invalidate_from_year(checkpoint, args.rebuild_from_year, logger)
            state = load_rebuild_state(args.rebuild_from_year)
            process_years = [y for y in years if y >= args.rebuild_from_year]
        elif args.resume:
            contiguous, issues = validate_resume(checkpoint, fhash, chash)
            write_csv_rows(RESUME_VALIDATION_CSV, issues or [{"year": "", "field": "resume_validation", "expected": "valid", "actual": "valid", "message": "pass", "recommended_command": ""}])
            if issues:
                raise ResumeValidationError(issues)
            latest = max(contiguous) if contiguous else None
            if latest:
                state = load_state(state_path(latest))
                process_years = [y for y in years if y > latest]
                logger.info("resume from year=%s remaining=%s", latest, process_years)
            else:
                state = new_state()
                process_years = years
        else:
            state = new_state()
            process_years = years
        if args.force:
            state = new_state()
            process_years = years

        all_samples: list[dict[str, Any]] = []
        for year in process_years:
            features, state, audit_counts, audit_samples, elapsed = process_year(year, state, logger)
            write_elapsed = write_year(features, year)
            save_state(state_path(year), state)
            meta = {
                "status": "complete",
                "input": file_fingerprint(base_path(year)),
                "feature_set_path": str(FEATURE_SET_YAML),
                "feature_set_hash": fhash,
                "code_bundle_hash": chash,
                "script_version": SCRIPT_VERSION,
                "state_version": STATE_VERSION,
                "git_commit_sha": ginfo["git_commit_sha"],
                "git_is_dirty": ginfo["git_is_dirty"],
                "state": str(state_path(year)),
                "output": str(out_path(year)),
                "rows": features.height,
                "columns": len(features.columns),
                "audit_counts": audit_counts,
                "elapsed_sec": round(elapsed + write_elapsed, 3),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            checkpoint.setdefault("years", {})[str(year)] = meta
            save_checkpoint(checkpoint)
            all_samples.extend(audit_samples[:100])
            logger.info("year=%s complete rows=%s", year, features.height)
        write_csv_rows(LEAKAGE_SAMPLES_CSV, all_samples[:5000])
        rc = finalize_outputs(years, checkpoint, started, validation_rows)
        if rc == 0:
            logger.info("done v2.1.1 elapsed=%.1fs", time.time() - started)
        return rc
    except ResumeValidationError as exc:
        logger.error("%s", exc)
        for row in exc.issues[:20]:
            logger.error("resume issue: %s", row)
        write_csv_rows(RESUME_VALIDATION_CSV, exc.issues)
        save_checkpoint(checkpoint)
        return 2
    except Exception as exc:
        logger.error("failed: %s", exc)
        logger.error(traceback.format_exc())
        checkpoint["last_error"] = {"at": datetime.now().isoformat(timespec="seconds"), "error": str(exc), "traceback": traceback.format_exc()}
        save_checkpoint(checkpoint)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
