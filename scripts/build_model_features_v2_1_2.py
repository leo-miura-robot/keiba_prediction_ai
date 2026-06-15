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

from src.features.feature_sets_v2_1_2 import (
    canonical_feature_set_text,
    feature_inventory_rows,
    load_feature_set_yaml,
    validate_feature_sets_from_file,
    write_feature_set_yaml,
)
from src.features.history_builder_v2_1_2 import (
    STATE_VERSION,
    STORE_NAMES,
    build_pre_day_history_features_v2_1,
    load_state,
    mark_completed_year,
    new_state,
    save_state,
)
from src.database.db_validation_cache import DatabaseValidationError, db_validation_fingerprint, validate_or_require_full
from src.features.target_builder import add_target_columns


CONFIG_PATH = Path("config/model_features_v2_1_2.yaml")
BASE_DIR = Path("outputs/base_runner_dataset_o1_fixed")
OUT_DIR = Path("outputs/model_feature_dataset_v2_1_2")
CHECKPOINT_DIR = Path("outputs/model_feature_dataset_v2_1_2_checkpoint")
CHECKPOINT_JSON = CHECKPOINT_DIR / "checkpoint.json"
LOG_PATH = Path("logs/build_model_features_v2_1_2.log")
FEATURE_SET_YAML = Path("config/feature_sets_v2_1_2.yaml")

SAMPLE_CSV = OUT_DIR / "model_feature_dataset_v2_1_2_sample.csv"
ELIGIBILITY_CSV = OUT_DIR / "training_eligibility_summary.csv"
LABEL_MISMATCH_CSV = OUT_DIR / "label_mismatch_cases.csv"
LEAKAGE_CSV = OUT_DIR / "history_leakage_validation.csv"
LEAKAGE_BY_STORE_CSV = OUT_DIR / "history_leakage_validation_by_store.csv"
LEAKAGE_SAMPLES_CSV = OUT_DIR / "history_leakage_validation_samples.csv"
FEATURE_INVENTORY_CSV = OUT_DIR / "feature_inventory.csv"
FEATURE_SET_VALIDATION_CSV = OUT_DIR / "feature_set_validation.csv"
RESUME_VALIDATION_CSV = OUT_DIR / "resume_validation.csv"
COMPARISON_CSV = OUT_DIR / "old_new_dataset_comparison.csv"
MARKET_COVERAGE_COMPARISON_CSV = OUT_DIR / "market_feature_coverage_comparison.csv"
MANIFEST_JSON = OUT_DIR / "manifest.json"

DESIGN_DOC = Path("docs/model_features_v2_1_2_results.md")
RESUME_DOC = Path("docs/resume_design_v2_1_2.md")
FEATURE_SET_DOC = Path("docs/feature_set_design_v2_1_2.md")

YEARS_ALL = list(range(2016, 2027))
SCRIPT_VERSION = "v2_1_2_o1_fixed_20260614"
ACTIVE_CONFIG: dict[str, Any] = {}
CODE_FILES = [
    Path("scripts/build_model_features_v2_1_2.py"),
    Path("src/features/history_builder_v2_1_2.py"),
    Path("src/features/feature_sets_v2_1_2.py"),
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


def parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.startswith("[") and value.endswith("]"):
        inside = value[1:-1].strip()
        if not inside:
            return []
        out: list[Any] = []
        for item in inside.split(","):
            item = item.strip().strip('"').strip("'")
            try:
                out.append(int(item))
            except ValueError:
                out.append(item)
        return out
    try:
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            result[section] = {}
            continue
        if line.startswith("  ") and ":" in line and section:
            key, value = line.strip().split(":", 1)
            result[section][key.strip()] = parse_scalar(value.strip())
    return result


def apply_config(path: Path) -> dict[str, Any]:
    global BASE_DIR, OUT_DIR, CHECKPOINT_DIR, CHECKPOINT_JSON, LOG_PATH, FEATURE_SET_YAML
    global SAMPLE_CSV, ELIGIBILITY_CSV, LABEL_MISMATCH_CSV, LEAKAGE_CSV, LEAKAGE_BY_STORE_CSV
    global LEAKAGE_SAMPLES_CSV, FEATURE_INVENTORY_CSV, FEATURE_SET_VALIDATION_CSV, RESUME_VALIDATION_CSV
    global COMPARISON_CSV, MARKET_COVERAGE_COMPARISON_CSV, MANIFEST_JSON
    cfg = load_simple_yaml(path)
    BASE_DIR = Path(str(cfg["input"]["base_dataset_dir"]))
    OUT_DIR = Path(str(cfg["outputs"]["dataset_dir"]))
    CHECKPOINT_DIR = Path(str(cfg["outputs"]["checkpoint_dir"]))
    CHECKPOINT_JSON = CHECKPOINT_DIR / "checkpoint.json"
    LOG_PATH = Path(str(cfg["outputs"]["log"]))
    FEATURE_SET_YAML = Path(str(cfg["feature_sets"]["yaml"]))
    SAMPLE_CSV = OUT_DIR / "model_feature_dataset_v2_1_2_sample.csv"
    ELIGIBILITY_CSV = OUT_DIR / "training_eligibility_summary.csv"
    LABEL_MISMATCH_CSV = OUT_DIR / "label_mismatch_cases.csv"
    LEAKAGE_CSV = OUT_DIR / "history_leakage_validation.csv"
    LEAKAGE_BY_STORE_CSV = OUT_DIR / "history_leakage_validation_by_store.csv"
    LEAKAGE_SAMPLES_CSV = OUT_DIR / "history_leakage_validation_samples.csv"
    FEATURE_INVENTORY_CSV = OUT_DIR / "feature_inventory.csv"
    FEATURE_SET_VALIDATION_CSV = OUT_DIR / "feature_set_validation.csv"
    RESUME_VALIDATION_CSV = OUT_DIR / "resume_validation.csv"
    COMPARISON_CSV = OUT_DIR / "old_new_dataset_comparison.csv"
    MARKET_COVERAGE_COMPARISON_CSV = OUT_DIR / "market_feature_coverage_comparison.csv"
    MANIFEST_JSON = OUT_DIR / "manifest.json"
    return cfg


def split_by_year_from_config(config: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for split, years in config.get("splits", {}).items():
        for year in years:
            mapping[int(year)] = split
    return mapping


def split_hash(config: dict[str, Any]) -> str:
    return sha256_text(json.dumps(config.get("splits", {}), ensure_ascii=False, sort_keys=True))


def apply_split_from_config(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    mapping = split_by_year_from_config(config)
    return df.with_columns(
        pl.col("Year").map_elements(lambda y: mapping.get(int(y), "out_of_scope"), return_dtype=pl.String).alias("data_split")
    )


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("build_model_features_v2_1_2")
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
        path = FEATURE_SET_YAML if rel == FEATURE_SET_YAML else ROOT / rel
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
        "recommended_command": f"python scripts/build_model_features_v2_1_2.py --resume --rebuild-from-year {cmd_year}",
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


def coverage_counts(df: pl.DataFrame) -> dict[str, int]:
    return {
        "tan_odds_non_null": int((df["tan_odds"].is_not_null() & (df["tan_odds"] > 0)).sum()) if "tan_odds" in df.columns else 0,
        "fuku_odds_low_non_null": int((df["fuku_odds_low"].is_not_null() & (df["fuku_odds_low"] > 0)).sum()) if "fuku_odds_low" in df.columns else 0,
        "fuku_odds_high_non_null": int((df["fuku_odds_high"].is_not_null() & (df["fuku_odds_high"] > 0)).sum()) if "fuku_odds_high" in df.columns else 0,
        "tan_ninki_non_null": int((df["tan_ninki"].is_not_null() & (df["tan_ninki"] > 0)).sum()) if "tan_ninki" in df.columns else 0,
        "fuku_ninki_non_null": int((df["fuku_ninki"].is_not_null() & (df["fuku_ninki"] > 0)).sum()) if "fuku_ninki" in df.columns else 0,
    }


def compare_v2_1_1_v2_1_2(years: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    market_rows = []
    old_root = Path(str(ACTIVE_CONFIG.get("comparison", {}).get("old_v2_1_1_dir", "outputs/model_feature_dataset_v2_1_1")))
    for year in years:
        p_old = old_root / f"year={year}" / "data.parquet"
        p_new = out_path(year)
        old_rows = old_cols = new_rows = new_cols = None
        old = new = None
        if p_old.exists():
            old = pl.read_parquet(p_old)
            old_rows, old_cols = old.height, len(old.columns)
        if p_new.exists():
            new = pl.read_parquet(p_new)
            new_rows, new_cols = new.height, len(new.columns)
        entry_agreement = actual_agreement = None
        if old is not None and new is not None:
            entry_agreement = old.select("entry_id").join(new.select("entry_id"), on="entry_id", how="inner").height
            actual_cols = [c for c in ["target_win_paid", "target_place_paid", "target_win_rank", "target_top3_rank"] if c in old.columns and c in new.columns]
            if actual_cols:
                joined = old.select(["entry_id", *actual_cols]).join(
                    new.select(["entry_id", *actual_cols]).rename({c: f"{c}_new" for c in actual_cols}),
                    on="entry_id",
                    how="inner",
                )
                expr = pl.lit(True)
                for col in actual_cols:
                    expr = expr & (pl.col(col) == pl.col(f"{col}_new"))
                actual_agreement = int(joined.filter(expr).height)
            old_cov = coverage_counts(old)
            new_cov = coverage_counts(new)
            for col in old_cov:
                market_rows.append({"year": year, "column_name": col, "v2_1_1_valid": old_cov[col], "v2_1_2_valid": new_cov[col], "improvement": new_cov[col] - old_cov[col]})
        rows.append({
            "year": year,
            "v2_1_1_rows": old_rows,
            "v2_1_2_rows": new_rows,
            "row_diff": (new_rows - old_rows) if old_rows is not None and new_rows is not None else None,
            "v2_1_1_columns": old_cols,
            "v2_1_2_columns": new_cols,
            "entry_id_intersection": entry_agreement,
            "actual_agreement_rows": actual_agreement,
        })
    return rows, market_rows


def write_docs(summaries: list[dict[str, Any]], leakage: list[dict[str, Any]], elapsed: float, validation_rows: list[dict[str, str]]) -> None:
    total_rows = sum(r["rows"] for r in summaries)
    DESIGN_DOC.write_text(f"# Model Features V2.1.2 Results\n\nRows: {total_rows:,}\n\nV2.1.2 keeps V2.1.1 feature logic and rebuilds from the O1-fixed base dataset. It writes to `outputs/model_feature_dataset_v2_1_2` and does not overwrite V2.1.1.\n\n`market_aware` uses final `NL_O1` odds from the fixed DB. This is an ideal-condition final-odds dataset, not a pre-race live operation dataset.\n\nElapsed seconds: {elapsed:.3f}\n", encoding="utf-8")
    RESUME_DOC.write_text("# Resume Design V2.1.2\n\nStrict resume validates contiguous completion from 2016, state pickle, output parquet, row count, state version, completed year, input fingerprint, feature set hash, and code bundle hash. Git SHA is recorded but not used as a resume-failure condition.\n\nUse `--rebuild-from-year YYYY` to regenerate YYYY and later. For YYYY > 2016, the previous year state is required even if the previous year is not included in `--years`.\n", encoding="utf-8")
    FEATURE_SET_DOC.write_text("\n".join(["# Feature Set Design V2.1.2", "", "Dedicated config: `config/feature_sets_v2_1_2.yaml`.", "", "The three feature sets remain `market_free`, `market_history`, and `market_aware`.", "", *[f"- {r['check_name']}: {r['status']} {r['details']}" for r in validation_rows]]), encoding="utf-8")


def process_year(year: int, state: dict[str, Any], logger: logging.Logger) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, int]], list[dict[str, Any]], float]:
    started = time.time()
    base = pl.read_parquet(base_path(year))
    logger.info("year=%s base rows=%s cols=%s", year, base.height, len(base.columns))
    labeled = add_target_columns(base)
    labeled = apply_split_from_config(labeled, ACTIVE_CONFIG)
    features, next_state, audit_counts, audit_samples = build_pre_day_history_features_v2_1(labeled, logger, state)
    features = apply_split_from_config(features, ACTIVE_CONFIG)
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
    write_csv_rows(FEATURE_INVENTORY_CSV, feature_inventory_rows(FEATURE_SET_YAML, set(combined.columns), null_rates))
    comparison_rows, market_rows = compare_v2_1_1_v2_1_2(existing)
    write_csv_rows(COMPARISON_CSV, comparison_rows)
    write_csv_rows(MARKET_COVERAGE_COMPARISON_CSV, market_rows)
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
    output_files = [out_path(y) for y in existing]
    manifest = {
        "version": "v2_1_2_o1_fixed",
        "input_base_dir": str(BASE_DIR),
        "output_dir": str(OUT_DIR),
        "feature_set_yaml": str(FEATURE_SET_YAML),
        "feature_set_hash": feature_set_hash(),
        "code_bundle_hash": code_bundle_hash(),
        "split_definition": ACTIVE_CONFIG.get("splits", {}),
        "split_hash": split_hash(ACTIVE_CONFIG),
        "rows": combined.height,
        "columns": len(combined.columns),
        "years": existing,
        "market_free_columns": len(load_feature_set_yaml(FEATURE_SET_YAML)["market_free"].get("numeric", [])) + len(load_feature_set_yaml(FEATURE_SET_YAML)["market_free"].get("categorical", [])),
        "market_history_columns": len(load_feature_set_yaml(FEATURE_SET_YAML)["market_history"].get("numeric", [])) + len(load_feature_set_yaml(FEATURE_SET_YAML)["market_history"].get("categorical", [])),
        "market_aware_columns": len(load_feature_set_yaml(FEATURE_SET_YAML)["market_aware"].get("numeric", [])) + len(load_feature_set_yaml(FEATURE_SET_YAML)["market_aware"].get("categorical", [])),
        "leakage_audit": overall[0],
        "source_db_path": ACTIVE_CONFIG.get("input", {}).get("source_db_path"),
        "ideal_condition_notice": "market_aware uses final NL_O1 odds; this is not a pre-race live operation dataset.",
        "output_files": [{"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in output_files],
        "python_version": sys.version,
        "polars_version": pl.__version__,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started, 3),
        **git_info(),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
    checkpoint["elapsed_sec_last_run"] = round(elapsed, 3)
    save_checkpoint(checkpoint)
    return 0


def main() -> int:
    global CONFIG_PATH, ACTIVE_CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--years")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-from-year", type=int)
    parser.add_argument("--force-integrity-check", action="store_true")
    parser.add_argument("--skip-db-validation", action="store_true")
    parser.add_argument("--db-validation-config", default="config/database_validation.yaml")
    args = parser.parse_args()
    CONFIG_PATH = Path(args.config)
    ACTIVE_CONFIG = apply_config(CONFIG_PATH)
    logger = setup_logging()
    started = time.time()
    years = parse_years(args.years)
    logger.info("start v2.1.2 years=%s resume=%s strict=%s force=%s rebuild_from=%s config=%s", years, args.resume, args.strict_resume, args.force, args.rebuild_from_year, CONFIG_PATH)
    db_validation_result = {"status": "not_configured"}
    source_db = ACTIVE_CONFIG.get("input", {}).get("source_db_path")
    if source_db:
        try:
            db_validation_result = validate_or_require_full(
                Path(source_db),
                args.db_validation_config,
                force_integrity_check=args.force_integrity_check,
                skip=args.skip_db_validation,
            )
        except DatabaseValidationError as exc:
            logger.error("DB validation failed: %s", exc)
            return 2

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
    checkpoint.update({
        "script_version": SCRIPT_VERSION,
        "state_version": STATE_VERSION,
        "feature_set_path": str(FEATURE_SET_YAML),
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "split_hash": split_hash(ACTIVE_CONFIG),
        "last_started_at": datetime.now().isoformat(timespec="seconds"),
        "db_validation": db_validation_result if args.skip_db_validation else db_validation_fingerprint(Path(source_db), args.db_validation_config) if source_db else None,
        **ginfo,
    })
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
            logger.info("done v2.1.2 elapsed=%.1fs", time.time() - started)
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
