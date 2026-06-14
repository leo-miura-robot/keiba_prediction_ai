from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl


CONFIG_PATH = Path("config/base_runner_dataset_o1_fixed.yaml")
DB_PATH = Path(r"D:\keiba\new_jra_2016-2026_fixed\keiba.db")
OLD_DB_PATH = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
OUT_DIR = Path("outputs/base_runner_dataset_o1_fixed")
DATASET_DIR = OUT_DIR
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "build_full_runner_dataset_o1_fixed.log"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.json"

FULL_SUMMARY_CSV = OUT_DIR / "summary.csv"
COLUMN_QUALITY_CSV = OUT_DIR / "column_quality_summary.csv"
SPECIAL_CASES_CSV = OUT_DIR / "special_result_cases.csv"
SAMPLE_CSV = OUT_DIR / "base_runner_dataset_sample.csv"
O1_QUALITY_CSV = OUT_DIR / "o1_quality_summary.csv"
OLD_NEW_COMPARISON_CSV = OUT_DIR / "base_dataset_old_new_comparison.csv"
MANIFEST_JSON = OUT_DIR / "manifest.json"

PREFLIGHT_DIR = Path("outputs/o1_fixed_preflight")
PREFLIGHT_DB_SUMMARY_JSON = PREFLIGHT_DIR / "db_summary.json"
PREFLIGHT_COVERAGE_CSV = PREFLIGHT_DIR / "o1_coverage_summary.csv"
PREFLIGHT_RACE_CSV = PREFLIGHT_DIR / "o1_race_completeness.csv"
PREFLIGHT_SE_O1_CSV = PREFLIGHT_DIR / "se_o1_comparison.csv"

DESIGN_DOC = Path("docs/o1_fixed_ai_data_migration.md")
QUALITY_DOC = Path("docs/base_runner_dataset_o1_fixed_quality.md")

YEARS_ALL = list(range(2016, 2027))
JRA_JYOCD = [f"{i:02d}" for i in range(1, 11)]

RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
ENTRY_KEY = RACE_KEY + ["Umaban"]

DIRECT_FEATURES = [
    "race_date", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "YoubiCD",
    "GradeCD", "SyubetuCD", "JyokenCD1", "JyokenCD2", "JyokenCD3",
    "JyokenCD4", "JyokenCD5", "JyokenName", "Kyori", "TrackCD",
    "CourseKubunCD", "HassoTime", "TorokuTosu", "SyussoTosu",
    "TenkoCD", "SibaBabaCD", "DirtBabaCD", "Wakuban", "Umaban",
    "KettoNum", "SexCD", "Barei", "ChokyosiCode", "ChokyosiRyakusyo",
    "KisyuCode", "KisyuRyakusyo", "Futan",
]

CONDITIONAL_FEATURES = [
    "tan_odds", "tan_ninki", "TanVote", "fuku_odds_low", "fuku_odds_high",
    "fuku_ninki", "FukuVote", "BaTaijyu", "ZogenFugo", "ZogenSa",
    "Ninki",
]

DERIVED_FEATURE_IDEAS = [
    "days_since_last_race", "past_3_finish_mean", "past_5_finish_mean",
    "past_3_place_rate", "past_5_place_rate", "same_course_record",
    "same_distance_record", "surface_record", "jockey_recent_win_rate",
    "trainer_recent_win_rate", "distance_change", "class_change",
]

LEAKAGE_COLUMNS = [
    "NyusenJyuni", "KakuteiJyuni", "Time", "ChakusaCD", "HaronTimeL3",
    "Jyuni1c", "Jyuni2c", "Jyuni3c", "Jyuni4c", "tan_pay", "fuku_pay",
    "is_win_paid", "is_place_paid", "target_win", "target_ren", "target_place",
]


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
    global DB_PATH, OLD_DB_PATH, OUT_DIR, DATASET_DIR, LOG_PATH, CHECKPOINT_PATH
    global FULL_SUMMARY_CSV, COLUMN_QUALITY_CSV, SPECIAL_CASES_CSV, SAMPLE_CSV, O1_QUALITY_CSV
    global OLD_NEW_COMPARISON_CSV, MANIFEST_JSON

    cfg = load_simple_yaml(path)
    DB_PATH = Path(str(cfg["database"]["path"]))
    OLD_DB_PATH = Path(str(cfg.get("comparison_database", {}).get("old_path", OLD_DB_PATH)))
    OUT_DIR = Path(str(cfg["outputs"]["output_root"]))
    DATASET_DIR = Path(str(cfg["outputs"]["dataset_dir"]))
    LOG_PATH = Path(str(cfg["outputs"]["log"]))
    CHECKPOINT_PATH = Path(str(cfg["outputs"]["checkpoint"]))
    FULL_SUMMARY_CSV = OUT_DIR / "summary.csv"
    COLUMN_QUALITY_CSV = OUT_DIR / "column_quality_summary.csv"
    SPECIAL_CASES_CSV = OUT_DIR / "special_result_cases.csv"
    SAMPLE_CSV = OUT_DIR / "base_runner_dataset_sample.csv"
    O1_QUALITY_CSV = OUT_DIR / "o1_quality_summary.csv"
    OLD_NEW_COMPARISON_CSV = OUT_DIR / "base_dataset_old_new_comparison.csv"
    MANIFEST_JSON = OUT_DIR / "manifest.json"
    return cfg


def sha256_file(path: Path, limit_mb: int | None = None) -> str:
    h = hashlib.sha256()
    remaining = None if limit_mb is None else limit_mb * 1024 * 1024
    with path.open("rb") as f:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = f.read(size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_hash(config: dict[str, Any]) -> str:
    return sha256_text(json.dumps(config.get("splits", {}), sort_keys=True, ensure_ascii=False))


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception:
        return {"git_commit_sha": "unknown", "git_is_dirty": None}


def db_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "mtime_ns": stat.st_mtime_ns,
        "sha256_first_256mb": sha256_file(path, limit_mb=256),
    }


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("build_full_runner_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def connect_ro() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def connect_ro_path(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


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


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"db_path": str(DB_PATH), "db_fingerprint": db_fingerprint(DB_PATH), "years": {}}


def save_checkpoint(checkpoint: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_sql(year: int) -> str:
    se_cols = [
        "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "Wakuban", "Umaban", "KettoNum", "Bamei", "SexCD", "Barei",
        "ChokyosiCode", "ChokyosiRyakusyo", "KisyuCode", "KisyuRyakusyo",
        "Futan", "BaTaijyu", "ZogenFugo", "ZogenSa", "IJyoCD",
        "NyusenJyuni", "KakuteiJyuni", "Odds", "Ninki", "Time",
        "ChakusaCD", "HaronTimeL3", "Jyuni1c", "Jyuni2c", "Jyuni3c", "Jyuni4c",
    ]
    ra_cols = [
        "YoubiCD", "GradeCD", "SyubetuCD", "JyokenCD1", "JyokenCD2",
        "JyokenCD3", "JyokenCD4", "JyokenCD5", "JyokenName", "Kyori",
        "TrackCD", "CourseKubunCD", "HassoTime", "TorokuTosu", "SyussoTosu",
        "TenkoCD", "SibaBabaCD", "DirtBabaCD",
    ]
    o1_cols = ["TanOdds", "TanNinki", "TanVote", "FukuOddsLow", "FukuOddsHigh", "FukuNinki", "FukuVote"]

    se_select = ",\n            ".join(f"se.{qident(c)} AS {qident(c)}" for c in se_cols)
    ra_select = ",\n            ".join(f"ra.{qident(c)} AS {qident(c)}" for c in ra_cols)
    o1_select = ",\n            ".join(f"o1.{qident(c)} AS {qident(c)}" for c in o1_cols)

    race_join_ra = " AND ".join(f"ra.{c} = se.{c}" for c in RACE_KEY)
    race_join_o1 = " AND ".join(f"o1.{c} = se.{c}" for c in RACE_KEY) + " AND o1.Umaban = se.Umaban"
    race_join_tan = " AND ".join(f"tan.{c} = se.{c}" for c in RACE_KEY) + " AND tan.Umaban = se.Umaban"
    race_join_fuku = " AND ".join(f"fuku.{c} = se.{c}" for c in RACE_KEY) + " AND fuku.Umaban = se.Umaban"

    return f"""
        WITH tan_payouts AS (
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(TanUmaban AS INTEGER) AS Umaban, TanPay AS tan_pay, TanNinki AS tan_pay_ninki, 1 AS tan_slot
            FROM NL_HR WHERE Year = ? AND TanUmaban IS NOT NULL AND TRIM(TanUmaban) <> '' AND TanPay IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(TanUmaban2 AS INTEGER), TanPay2, TanNinki2, 2
            FROM NL_HR WHERE Year = ? AND TanUmaban2 IS NOT NULL AND TRIM(TanUmaban2) <> '' AND TanPay2 IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(TanUmaban3 AS INTEGER), TanPay3, TanNinki3, 3
            FROM NL_HR WHERE Year = ? AND TanUmaban3 IS NOT NULL AND TRIM(TanUmaban3) <> '' AND TanPay3 IS NOT NULL
        ),
        fuku_payouts AS (
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(FukuUmaban AS INTEGER) AS Umaban, FukuPay AS fuku_pay, FukuNinki AS fuku_pay_ninki, 1 AS fuku_slot
            FROM NL_HR WHERE Year = ? AND FukuUmaban IS NOT NULL AND TRIM(FukuUmaban) <> '' AND FukuPay IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(FukuUmaban2 AS INTEGER), FukuPay2, FukuNinki2, 2
            FROM NL_HR WHERE Year = ? AND FukuUmaban2 IS NOT NULL AND TRIM(FukuUmaban2) <> '' AND FukuPay2 IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(FukuUmaban3 AS INTEGER), FukuPay3, FukuNinki3, 3
            FROM NL_HR WHERE Year = ? AND FukuUmaban3 IS NOT NULL AND TRIM(FukuUmaban3) <> '' AND FukuPay3 IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(FukuUmaban4 AS INTEGER), FukuPay4, FukuNinki4, 4
            FROM NL_HR WHERE Year = ? AND FukuUmaban4 IS NOT NULL AND TRIM(FukuUmaban4) <> '' AND FukuPay4 IS NOT NULL
            UNION ALL
            SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, CAST(FukuUmaban5 AS INTEGER), FukuPay5, FukuNinki5, 5
            FROM NL_HR WHERE Year = ? AND FukuUmaban5 IS NOT NULL AND TRIM(FukuUmaban5) <> '' AND FukuPay5 IS NOT NULL
        )
        SELECT
            printf('%04d%04d%s%02d%02d%02d', se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum) AS race_id,
            printf('%04d%04d%s%02d%02d%02d%02d', se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum, se.Umaban) AS entry_id,
            printf('%04d-%02d-%02d', se.Year, se.MonthDay / 100, se.MonthDay % 100) AS race_date,
            {se_select},
            {ra_select},
            {o1_select},
            o1.TanOdds AS tan_odds,
            o1.TanNinki AS tan_ninki,
            o1.FukuOddsLow AS fuku_odds_low,
            o1.FukuOddsHigh AS fuku_odds_high,
            o1.FukuNinki AS fuku_ninki,
            CASE WHEN ra.Year IS NULL THEN 0 ELSE 1 END AS ra_join_found,
            CASE WHEN o1.Year IS NULL THEN 0 ELSE 1 END AS o1_join_found,
            COALESCE(tan.tan_pay, 0) AS tan_pay,
            tan.tan_pay_ninki,
            tan.tan_slot,
            CASE WHEN tan.tan_pay IS NULL THEN 0 ELSE 1 END AS tan_payout_record_found,
            COALESCE(fuku.fuku_pay, 0) AS fuku_pay,
            fuku.fuku_pay_ninki,
            fuku.fuku_slot,
            CASE WHEN fuku.fuku_pay IS NULL THEN 0 ELSE 1 END AS fuku_payout_record_found,
            CASE WHEN se.IJyoCD = '0' AND se.KakuteiJyuni > 0 THEN 1 ELSE 0 END AS is_normal_runner,
            CASE WHEN se.IJyoCD <> '0' OR se.KakuteiJyuni = 0 THEN 1 ELSE 0 END AS is_abnormal_result,
            CASE WHEN se.IJyoCD = '0' AND se.KakuteiJyuni = 1 THEN 1 ELSE 0 END AS target_win,
            CASE WHEN se.IJyoCD = '0' AND se.KakuteiJyuni BETWEEN 1 AND 2 THEN 1 ELSE 0 END AS target_ren,
            CASE WHEN se.IJyoCD = '0' AND se.KakuteiJyuni BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS target_place,
            CASE WHEN COALESCE(tan.tan_pay, 0) > 0 THEN 1 ELSE 0 END AS is_win_paid,
            CASE WHEN COALESCE(fuku.fuku_pay, 0) > 0 THEN 1 ELSE 0 END AS is_place_paid
        FROM NL_SE se
        LEFT JOIN NL_RA ra ON {race_join_ra}
        LEFT JOIN NL_O1 o1 ON {race_join_o1}
        LEFT JOIN tan_payouts tan ON {race_join_tan}
        LEFT JOIN fuku_payouts fuku ON {race_join_fuku}
        WHERE se.Year = ?
          AND se.JyoCD IN ('01','02','03','04','05','06','07','08','09','10')
        ORDER BY se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum, se.Umaban
    """


def fetch_year(year: int, logger: logging.Logger) -> pl.DataFrame:
    sql = build_sql(year)
    params = [year, year, year, year, year, year, year, year, year]
    started = time.time()
    logger.info("year=%s query start", year)
    with connect_ro() as con:
        rows = con.execute(sql, params).fetchall()
    elapsed = time.time() - started
    logger.info("year=%s query done rows=%s elapsed=%.3fs", year, len(rows), elapsed)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([dict(row) for row in rows], infer_schema_length=10000)


def quality_for_year(df: pl.DataFrame, year: int, elapsed: float) -> dict:
    row_count = df.height
    race_count = df.select(pl.col("race_id").n_unique()).item() if row_count else 0
    entry_dups = row_count - (df.select(pl.col("entry_id").n_unique()).item() if row_count else 0)
    ra_rate = float(df["ra_join_found"].mean()) if row_count else 0.0
    o1_rate = float(df["o1_join_found"].mean()) if row_count else 0.0
    tan_odds_available = int((df["tan_odds"].is_not_null() & (df["tan_odds"] > 0)).sum()) if row_count else 0
    fuku_odds_available = int((df["fuku_odds_low"].is_not_null() & (df["fuku_odds_low"] > 0) & df["fuku_odds_high"].is_not_null() & (df["fuku_odds_high"] > 0)).sum()) if row_count else 0
    tan_pay_count = int((df["tan_pay"] > 0).sum()) if row_count else 0
    fuku_pay_count = int((df["fuku_pay"] > 0).sum()) if row_count else 0
    target_win_count = int((df["target_win"] == 1).sum()) if row_count else 0
    target_ren_count = int((df["target_ren"] == 1).sum()) if row_count else 0
    target_place_count = int((df["target_place"] == 1).sum()) if row_count else 0
    win_mismatch = int(((df["target_win"] != df["is_win_paid"]) & (df["is_abnormal_result"] == 0)).sum()) if row_count else 0
    place_mismatch = int(((df["target_place"] != df["is_place_paid"]) & (df["is_abnormal_result"] == 0)).sum()) if row_count else 0
    tan_odds_bad = int((df["tan_odds"].is_not_null() & (df["tan_odds"] <= 0)).sum()) if row_count else 0
    fuku_odds_bad = int((df["fuku_odds_low"].is_not_null() & df["fuku_odds_high"].is_not_null() & (df["fuku_odds_low"] > df["fuku_odds_high"])).sum()) if row_count else 0

    return {
        "year": year,
        "rows": row_count,
        "races": race_count,
        "elapsed_sec": round(elapsed, 3),
        "entry_id_duplicates": entry_dups,
        "ra_join_success_rate": round(ra_rate, 8),
        "o1_join_success_rate": round(o1_rate, 8),
        "tan_odds_available": tan_odds_available,
        "fuku_odds_available": fuku_odds_available,
        "tan_pay_count": tan_pay_count,
        "fuku_pay_count": fuku_pay_count,
        "target_win_count": target_win_count,
        "target_ren_count": target_ren_count,
            "target_place_count": target_place_count,
            "is_win_paid_count": int((df["is_win_paid"] == 1).sum()) if row_count else 0,
            "is_place_paid_count": int((df["is_place_paid"] == 1).sum()) if row_count else 0,
        "target_win_is_win_paid_mismatch_normal": win_mismatch,
        "target_place_is_place_paid_mismatch_normal": place_mismatch,
        "ijyocd_counts": json.dumps(counts_dict(df, "IJyoCD"), ensure_ascii=False),
        "tan_odds_non_positive_count": tan_odds_bad,
        "fuku_odds_low_gt_high_count": fuku_odds_bad,
    }


def counts_dict(df: pl.DataFrame, col: str) -> dict:
    if df.height == 0 or col not in df.columns:
        return {}
    data = df.group_by(col).len().sort(col).to_dicts()
    return {str(row[col]): int(row["len"]) for row in data}


def column_quality(df: pl.DataFrame, year: int) -> list[dict]:
    rows = []
    row_count = df.height
    for col in df.columns:
        null_count = int(df[col].null_count())
        missing_rate = null_count / row_count if row_count else 0.0
        unique_count = None
        try:
            unique_count = int(df[col].n_unique())
        except Exception:
            unique_count = None
        rows.append({
            "year": year,
            "column_name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "missing_rate": round(missing_rate, 8),
            "unique_count": unique_count,
        })
    return rows


def special_cases(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame()
    out = df.filter(
        (pl.col("is_abnormal_result") == 1)
        | (pl.col("target_win") != pl.col("is_win_paid"))
        | (pl.col("target_place") != pl.col("is_place_paid"))
    ).select([
        "race_id", "entry_id", "Umaban", "Bamei", "IJyoCD", "NyusenJyuni", "KakuteiJyuni",
        "target_win", "target_ren", "target_place", "tan_pay", "fuku_pay",
        "is_win_paid", "is_place_paid",
    ])
    if out.height:
        out = out.with_columns(
            pl.when(pl.col("IJyoCD") != "0")
            .then(pl.lit("abnormal_result_or_refund_case"))
            .when(pl.col("target_win") != pl.col("is_win_paid"))
            .then(pl.lit("target_win_vs_paid_mismatch"))
            .when(pl.col("target_place") != pl.col("is_place_paid"))
            .then(pl.lit("target_place_vs_paid_mismatch"))
            .otherwise(pl.lit(""))
            .alias("notes")
        )
    return out


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def run_preflight(config: dict[str, Any], logger: logging.Logger, full_integrity_check: bool = False) -> dict[str, Any]:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    required = ["NL_RA", "NL_SE", "NL_HR", "NL_H1", "NL_H6", "NL_O1", "NL_O2", "NL_O3", "NL_O4", "NL_O5", "NL_O6"]
    fp = db_fingerprint(DB_PATH)
    started = time.time()
    logger.info("preflight start db=%s", DB_PATH)
    with connect_ro() as con:
        integrity_pragma = "integrity_check" if full_integrity_check else "quick_check"
        logger.info("preflight %s start", integrity_pragma)
        integrity = con.execute(f"PRAGMA {integrity_pragma}").fetchone()[0]
        logger.info("preflight %s done result=%s", integrity_pragma, integrity)
        missing = [t for t in required if not table_exists(con, t)]
        if missing:
            raise RuntimeError(f"missing required tables: {missing}")
        table_counts = {t: int(con.execute(f"SELECT COUNT(*) FROM {qident(t)}").fetchone()[0]) for t in required}
        coverage = dict(con.execute(
            """
            SELECT
              COUNT(*) AS se_runner_rows,
              SUM(CASE WHEN o1.Year IS NOT NULL THEN 1 ELSE 0 END) AS o1_matched_rows,
              SUM(CASE WHEN o1.TanOdds IS NOT NULL AND o1.TanOdds > 0 THEN 1 ELSE 0 END) AS valid_tan_odds_rows,
              SUM(CASE WHEN o1.FukuOddsLow IS NOT NULL AND o1.FukuOddsLow > 0 THEN 1 ELSE 0 END) AS valid_fuku_odds_low_rows,
              SUM(CASE WHEN o1.FukuOddsHigh IS NOT NULL AND o1.FukuOddsHigh > 0 THEN 1 ELSE 0 END) AS valid_fuku_odds_high_rows,
              SUM(CASE WHEN o1.TanNinki IS NOT NULL AND o1.TanNinki > 0 THEN 1 ELSE 0 END) AS valid_tan_ninki_rows,
              SUM(CASE WHEN o1.FukuNinki IS NOT NULL AND o1.FukuNinki > 0 THEN 1 ELSE 0 END) AS valid_fuku_ninki_rows
            FROM NL_SE se
            LEFT JOIN NL_O1 o1
              ON o1.Year = se.Year AND o1.MonthDay = se.MonthDay AND o1.JyoCD = se.JyoCD
             AND o1.Kaiji = se.Kaiji AND o1.Nichiji = se.Nichiji AND o1.RaceNum = se.RaceNum
             AND o1.Umaban = se.Umaban
            WHERE se.Year BETWEEN 2016 AND 2026
              AND se.JyoCD IN ('01','02','03','04','05','06','07','08','09','10')
            """
        ).fetchone())
        se_rows = int(coverage["se_runner_rows"])
        for key in ["o1_matched_rows", "valid_tan_odds_rows", "valid_fuku_odds_low_rows", "valid_fuku_odds_high_rows", "valid_tan_ninki_rows", "valid_fuku_ninki_rows"]:
            coverage[key] = int(coverage[key] or 0)
        coverage.update({
            "tan_odds_coverage": coverage["valid_tan_odds_rows"] / se_rows if se_rows else 0.0,
            "fuku_odds_low_coverage": coverage["valid_fuku_odds_low_rows"] / se_rows if se_rows else 0.0,
            "fuku_odds_high_coverage": coverage["valid_fuku_odds_high_rows"] / se_rows if se_rows else 0.0,
        })
        race_rows = [dict(row) for row in con.execute(
            """
            WITH race_o1 AS (
              SELECT
                se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum,
                COUNT(*) AS runner_rows,
                SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS missing_o1_rows,
                SUM(CASE WHEN o1.TanOdds IS NOT NULL AND o1.TanOdds > 0
                          AND o1.FukuOddsLow IS NOT NULL AND o1.FukuOddsLow > 0
                          AND o1.FukuOddsHigh IS NOT NULL AND o1.FukuOddsHigh > 0 THEN 1 ELSE 0 END) AS valid_rows
              FROM NL_SE se
              LEFT JOIN NL_O1 o1
                ON o1.Year = se.Year AND o1.MonthDay = se.MonthDay AND o1.JyoCD = se.JyoCD
               AND o1.Kaiji = se.Kaiji AND o1.Nichiji = se.Nichiji AND o1.RaceNum = se.RaceNum
               AND o1.Umaban = se.Umaban
              WHERE se.Year BETWEEN 2016 AND 2026
                AND se.JyoCD IN ('01','02','03','04','05','06','07','08','09','10')
              GROUP BY se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum
            )
            SELECT
              CASE
                WHEN missing_o1_rows = runner_rows THEN 'missing_o1_rows'
                WHEN valid_rows = runner_rows THEN 'all_valid'
                WHEN valid_rows = 0 THEN 'all_null'
                ELSE 'partially_valid'
              END AS race_o1_pattern,
              COUNT(*) AS races,
              SUM(runner_rows) AS runner_rows
            FROM race_o1
            GROUP BY race_o1_pattern
            ORDER BY race_o1_pattern
            """
        ).fetchall()]
        comparison = dict(con.execute(
            """
            SELECT
              COUNT(*) AS compared_rows,
              SUM(CASE WHEN CAST(se.Odds AS REAL) = CAST(o1.TanOdds AS REAL) THEN 1 ELSE 0 END) AS exact_match_count,
              MAX(ABS(CAST(se.Odds AS REAL) - CAST(o1.TanOdds AS REAL))) AS max_difference
            FROM NL_SE se
            JOIN NL_O1 o1
              ON o1.Year = se.Year AND o1.MonthDay = se.MonthDay AND o1.JyoCD = se.JyoCD
             AND o1.Kaiji = se.Kaiji AND o1.Nichiji = se.Nichiji AND o1.RaceNum = se.RaceNum
             AND o1.Umaban = se.Umaban
            WHERE se.Year BETWEEN 2016 AND 2026
              AND se.JyoCD IN ('01','02','03','04','05','06','07','08','09','10')
              AND se.Odds IS NOT NULL AND CAST(se.Odds AS REAL) > 0
              AND o1.TanOdds IS NOT NULL AND CAST(o1.TanOdds AS REAL) > 0
            """
        ).fetchone())
    compared = int(comparison["compared_rows"] or 0)
    comparison["exact_match_count"] = int(comparison["exact_match_count"] or 0)
    comparison["exact_match_rate"] = comparison["exact_match_count"] / compared if compared else 0.0
    comparison["max_difference"] = float(comparison["max_difference"] or 0.0)
    db_summary = {
        **fp,
        "read_only_connection": True,
        "integrity_pragma": integrity_pragma,
        "integrity_check": integrity,
        "required_tables_missing": missing,
        "table_counts": table_counts,
        "config_path": str(CONFIG_PATH),
        "split_hash": split_hash(config),
        "elapsed_sec": round(time.time() - started, 3),
    }
    PREFLIGHT_DB_SUMMARY_JSON.write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_rows(PREFLIGHT_COVERAGE_CSV, [coverage])
    write_csv_rows(PREFLIGHT_RACE_CSV, race_rows)
    write_csv_rows(PREFLIGHT_SE_O1_CSV, [comparison])
    logger.info("preflight done %s=%s coverage=%.6f exact=%.6f", integrity_pragma, integrity, coverage["tan_odds_coverage"], comparison["exact_match_rate"])
    if integrity != "ok":
        raise RuntimeError(f"integrity_check failed: {integrity}")
    return {"db_summary": db_summary, "coverage": coverage, "race_completeness": race_rows, "se_o1": comparison}


def append_or_write_csv(path: Path, df: pl.DataFrame) -> None:
    if df.height == 0:
        return
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        existing = pl.read_csv(path, infer_schema_length=10000)
        combined = pl.concat([existing, df], how="diagonal_relaxed")
        combined.write_csv(path)
    else:
        df.write_csv(path)


def output_year(df: pl.DataFrame, year: int, logger: logging.Logger) -> Path:
    year_dir = DATASET_DIR / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    out = year_dir / "data.parquet"
    tmp = year_dir / "data.parquet.tmp"
    logger.info("year=%s write parquet start path=%s", year, out)
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    logger.info("year=%s write parquet done bytes=%s", year, out.stat().st_size)
    return out


def write_docs(summaries: list[dict], quality_rows: list[dict]) -> None:
    total_rows = sum(int(s["rows"]) for s in summaries)
    total_races = sum(int(s["races"]) for s in summaries)
    total_tan_pay = sum(int(s["tan_pay_count"]) for s in summaries)
    total_fuku_pay = sum(int(s["fuku_pay_count"]) for s in summaries)
    total_tan_odds = sum(int(s["tan_odds_available"]) for s in summaries)
    total_fuku_odds = sum(int(s["fuku_odds_available"]) for s in summaries)

    DESIGN_DOC.write_text("\n".join([
        "# O1 Fixed AI Data Migration",
        "",
        f"Source DB: `{DB_PATH}`",
        "",
        "This migration reads the fixed O1 DB in read-only mode and writes a separate base runner dataset under `outputs/base_runner_dataset_o1_fixed`.",
        "",
        "Base table is `NL_SE`, one row per runner. Race metadata joins from `NL_RA` by race key. Odds joins from `NL_O1` by race key plus `Umaban`. Win and place payouts are expanded from `NL_HR` payout slots and joined by race key plus `Umaban`.",
        "",
        "`tan_odds`, `tan_ninki`, `fuku_odds_low`, `fuku_odds_high`, and `fuku_ninki` come from `NL_O1`. `COALESCE(O1, SE)` is not used.",
        "",
        "`market_aware` downstream is an ideal-condition final-odds dataset. It is not a pre-race live operation dataset.",
        "",
        "The dataset is limited to JRA central racecourse codes `01` through `10`. Other `JyoCD` values in `NL_SE` do not have matching JRA odds/payout records in `NL_O1/NL_HR` and are excluded from this base JRA runner dataset.",
        "",
        "`race_id` is `YYYYMMDDJyoCDKaijiNichijiRaceNum` with zero padding. `entry_id` appends zero-padded `Umaban`. `race_date` is derived from `Year` and `MonthDay`.",
        "",
        "Outputs are partitioned by year under `outputs/base_runner_dataset_o1_fixed/year=YYYY/data.parquet`.",
        "",
        "## Split",
        "",
        "Fixed split is read from YAML: train 2016-2023, validation 2024, test 2025, latest_holdout 2026.",
        "",
    ]), encoding="utf-8")

    QUALITY_DOC.write_text("\n".join([
        "# Base Runner Dataset O1 Fixed Quality",
        "",
        f"Total rows: `{total_rows}`",
        f"Total races: `{total_races}`",
        f"TanOdds available rows: `{total_tan_odds}`",
        f"Fuku odds available rows: `{total_fuku_odds}`",
        f"Win payout rows: `{total_tan_pay}`",
        f"Place payout rows: `{total_fuku_pay}`",
        "",
        "## Year Summary",
        "",
        "| year | rows | races | RA join | O1 join | tan_pay | fuku_pay | win mismatch | place mismatch | elapsed sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {s['year']} | {s['rows']} | {s['races']} | {s['ra_join_success_rate']} | {s['o1_join_success_rate']} | {s['tan_pay_count']} | {s['fuku_pay_count']} | {s['target_win_is_win_paid_mismatch_normal']} | {s['target_place_is_place_paid_mismatch_normal']} | {s['elapsed_sec']} |"
            for s in summaries
        ],
        "",
        "See `outputs/base_runner_dataset_o1_fixed/column_quality_summary.csv` for null rates and unique counts by column.",
        "See `outputs/base_runner_dataset_o1_fixed/special_result_cases.csv` for abnormal and payout/target mismatch rows.",
        "",
    ]), encoding="utf-8")


def read_existing_summaries() -> list[dict]:
    if FULL_SUMMARY_CSV.exists():
        with FULL_SUMMARY_CSV.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return []


def rebuild_docs_from_outputs() -> None:
    summaries = read_existing_summaries()
    quality_rows = []
    if COLUMN_QUALITY_CSV.exists():
        with COLUMN_QUALITY_CSV.open(encoding="utf-8-sig", newline="") as f:
            quality_rows = list(csv.DictReader(f))
    if summaries:
        write_docs(summaries, quality_rows)


def load_year_dataset(dataset_dir: Path, year: int) -> pl.DataFrame | None:
    path = dataset_dir / f"year={year}" / "data.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def compare_old_new_base(years: list[int]) -> list[dict[str, Any]]:
    old_dir = Path("outputs/base_runner_dataset")
    rows: list[dict[str, Any]] = []
    for year in years:
        old = load_year_dataset(old_dir, year)
        new = load_year_dataset(DATASET_DIR, year)
        if new is None:
            continue
        row: dict[str, Any] = {
            "year": year,
            "old_rows": old.height if old is not None else None,
            "new_rows": new.height,
            "old_races": old["race_id"].n_unique() if old is not None else None,
            "new_races": new["race_id"].n_unique(),
            "old_tan_odds_valid": int((old["tan_odds"].is_not_null() & (old["tan_odds"] > 0)).sum()) if old is not None and "tan_odds" in old.columns else None,
            "new_tan_odds_valid": int((new["tan_odds"].is_not_null() & (new["tan_odds"] > 0)).sum()),
            "old_fuku_odds_valid": int((old["fuku_odds_low"].is_not_null() & (old["fuku_odds_low"] > 0) & old["fuku_odds_high"].is_not_null() & (old["fuku_odds_high"] > 0)).sum()) if old is not None and "fuku_odds_low" in old.columns else None,
            "new_fuku_odds_valid": int((new["fuku_odds_low"].is_not_null() & (new["fuku_odds_low"] > 0) & new["fuku_odds_high"].is_not_null() & (new["fuku_odds_high"] > 0)).sum()),
        }
        if old is not None:
            old_small = old.select(["entry_id", "tan_odds", "fuku_odds_low", "fuku_odds_high"]).rename({
                "tan_odds": "old_tan_odds",
                "fuku_odds_low": "old_fuku_odds_low",
                "fuku_odds_high": "old_fuku_odds_high",
            })
            new_small = new.select(["entry_id", "tan_odds", "fuku_odds_low", "fuku_odds_high"]).rename({
                "tan_odds": "new_tan_odds",
                "fuku_odds_low": "new_fuku_odds_low",
                "fuku_odds_high": "new_fuku_odds_high",
            })
            joined = old_small.join(new_small, on="entry_id", how="inner")
            old_tan_valid = joined["old_tan_odds"].is_not_null() & (joined["old_tan_odds"] > 0)
            new_tan_valid = joined["new_tan_odds"].is_not_null() & (joined["new_tan_odds"] > 0)
            old_fuku_valid = joined["old_fuku_odds_low"].is_not_null() & (joined["old_fuku_odds_low"] > 0) & joined["old_fuku_odds_high"].is_not_null() & (joined["old_fuku_odds_high"] > 0)
            new_fuku_valid = joined["new_fuku_odds_low"].is_not_null() & (joined["new_fuku_odds_low"] > 0) & joined["new_fuku_odds_high"].is_not_null() & (joined["new_fuku_odds_high"] > 0)
            row.update({
                "entry_id_intersection": joined.height,
                "tan_old_null_new_valid": int((~old_tan_valid & new_tan_valid).sum()),
                "tan_old_valid_new_null": int((old_tan_valid & ~new_tan_valid).sum()),
                "fuku_old_null_new_valid": int((~old_fuku_valid & new_fuku_valid).sum()),
                "fuku_old_valid_new_null": int((old_fuku_valid & ~new_fuku_valid).sum()),
            })
        rows.append(row)
    return rows


def write_manifest(config: dict[str, Any], years: list[int], summaries: list[dict[str, Any]], started_at: float) -> None:
    paths = [DATASET_DIR / f"year={year}" / "data.parquet" for year in years if (DATASET_DIR / f"year={year}" / "data.parquet").exists()]
    manifest = {
        "version": "base_runner_dataset_o1_fixed",
        "source_db": db_fingerprint(DB_PATH),
        "old_db_compare_only": db_fingerprint(OLD_DB_PATH) if OLD_DB_PATH.exists() else None,
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "split_definition": config.get("splits", {}),
        "split_hash": split_hash(config),
        "output_dir": str(DATASET_DIR),
        "output_files": [{"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p)} for p in paths],
        "rows": sum(int(r["rows"]) for r in summaries),
        "races": sum(int(r["races"]) for r in summaries),
        "columns": len(pl.read_parquet(paths[0]).columns) if paths else 0,
        "years": years,
        "python_version": sys.version,
        "polars_version": pl.__version__,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started_at, 3),
        **git_info(),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def process_year(year: int, args, checkpoint: dict, logger: logging.Logger) -> dict | None:
    year_key = str(year)
    out = DATASET_DIR / f"year={year}" / "data.parquet"
    existing = checkpoint.get("years", {}).get(year_key, {})
    if args.resume and not args.force and existing.get("status") == "complete" and out.exists():
        logger.info("year=%s skip complete checkpoint rows=%s", year, existing.get("rows"))
        return None
    if out.exists() and not args.force and not args.resume:
        logger.info("year=%s output exists; skip without --force", year)
        return None

    started = time.time()
    checkpoint.setdefault("years", {})[year_key] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
    }
    save_checkpoint(checkpoint)

    df = fetch_year(year, logger)
    if df.height == 0:
        raise RuntimeError(f"year={year} produced zero rows")

    output_year(df, year, logger)
    elapsed = time.time() - started
    summary = quality_for_year(df, year, elapsed)

    checkpoint["years"][year_key] = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "rows": summary["rows"],
        "races": summary["races"],
        "elapsed_sec": summary["elapsed_sec"],
        "output": str(DATASET_DIR / f"year={year}" / "data.parquet"),
    }
    save_checkpoint(checkpoint)

    logger.info("year=%s quality=%s", year, json.dumps(summary, ensure_ascii=False))
    return {"df": df, "summary": summary, "column_quality": column_quality(df, year), "special": special_cases(df)}


def replace_year_rows(existing: list[dict], new_rows: list[dict], year: int) -> list[dict]:
    return [r for r in existing if str(r.get("year")) != str(year)] + new_rows


def update_aggregate_outputs(results: list[dict], logger: logging.Logger) -> None:
    if not results:
        rebuild_docs_from_outputs()
        return

    years = {str(r["summary"]["year"]) for r in results}

    summaries = read_existing_summaries()
    summaries = [s for s in summaries if str(s.get("year")) not in years]
    summaries.extend(r["summary"] for r in results)
    summaries = sorted(summaries, key=lambda x: int(x["year"]))
    write_csv_rows(FULL_SUMMARY_CSV, summaries)

    quality_existing = []
    if COLUMN_QUALITY_CSV.exists():
        with COLUMN_QUALITY_CSV.open(encoding="utf-8-sig", newline="") as f:
            quality_existing = list(csv.DictReader(f))
    quality_existing = [r for r in quality_existing if str(r.get("year")) not in years]
    quality_new = [row for result in results for row in result["column_quality"]]
    write_csv_rows(COLUMN_QUALITY_CSV, quality_existing + quality_new)

    special_existing = pl.DataFrame()
    if SPECIAL_CASES_CSV.exists():
        try:
            special_existing = pl.read_csv(SPECIAL_CASES_CSV, infer_schema_length=10000)
            special_existing = special_existing.filter(~pl.col("race_id").str.slice(0, 4).is_in(list(years)))
        except Exception:
            special_existing = pl.DataFrame()
    special_new = [r["special"] for r in results if r["special"].height]
    if special_new:
        special_df = pl.concat(([special_existing] if special_existing.height else []) + special_new, how="diagonal_relaxed")
        special_df.write_csv(SPECIAL_CASES_CSV)

    first_df = results[0]["df"].head(200)
    first_df.write_csv(SAMPLE_CSV)

    available_years = sorted(int(s["year"]) for s in summaries)
    frames = [pl.read_parquet(DATASET_DIR / f"year={year}" / "data.parquet") for year in available_years if (DATASET_DIR / f"year={year}" / "data.parquet").exists()]
    if frames:
        combined = pl.concat(frames, how="diagonal_relaxed")
        combined.write_parquet(DATASET_DIR / "base_runner_dataset.parquet", compression="zstd")
        o1_quality = [{
            "rows": combined.height,
            "races": combined["race_id"].n_unique(),
            "tan_odds_valid": int((combined["tan_odds"].is_not_null() & (combined["tan_odds"] > 0)).sum()),
            "fuku_odds_low_valid": int((combined["fuku_odds_low"].is_not_null() & (combined["fuku_odds_low"] > 0)).sum()),
            "fuku_odds_high_valid": int((combined["fuku_odds_high"].is_not_null() & (combined["fuku_odds_high"] > 0)).sum()),
            "tan_ninki_valid": int((combined["tan_ninki"].is_not_null() & (combined["tan_ninki"] > 0)).sum()),
            "fuku_ninki_valid": int((combined["fuku_ninki"].is_not_null() & (combined["fuku_ninki"] > 0)).sum()),
        }]
        for row in o1_quality:
            row["tan_odds_coverage"] = row["tan_odds_valid"] / row["rows"] if row["rows"] else 0.0
            row["fuku_odds_low_coverage"] = row["fuku_odds_low_valid"] / row["rows"] if row["rows"] else 0.0
            row["fuku_odds_high_coverage"] = row["fuku_odds_high_valid"] / row["rows"] if row["rows"] else 0.0
        write_csv_rows(O1_QUALITY_CSV, o1_quality)
        write_csv_rows(OLD_NEW_COMPARISON_CSV, compare_old_new_base(available_years))

    write_docs(summaries, quality_existing + quality_new)
    logger.info("aggregate outputs updated years=%s", ",".join(sorted(years)))


def main() -> int:
    global CONFIG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--years", help="Comma-separated years or ranges, e.g. 2016 or 2016-2026")
    parser.add_argument("--resume", action="store_true", help="Skip completed checkpoint years.")
    parser.add_argument("--force", action="store_true", help="Rebuild requested years even if complete.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full-integrity-check", action="store_true", help="Use PRAGMA integrity_check instead of quick_check.")
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)
    config = apply_config(CONFIG_PATH)
    logger = setup_logging()
    started_at = time.time()
    logger.info("start db=%s years=%s resume=%s force=%s config=%s", DB_PATH, args.years or "2016-2026", args.resume, args.force, CONFIG_PATH)
    if not DB_PATH.exists():
        logger.error("DB not found: %s", DB_PATH)
        return 2

    try:
        preflight = run_preflight(config, logger, full_integrity_check=args.full_integrity_check)
    except Exception as exc:
        logger.error("preflight failed: %s", exc)
        logger.error(traceback.format_exc())
        return 2
    if args.preflight_only:
        return 0

    years = parse_years(args.years)
    checkpoint = load_checkpoint()
    current_fp = db_fingerprint(DB_PATH)
    if args.resume and checkpoint.get("db_fingerprint") != current_fp:
        logger.error("resume rejected: DB fingerprint mismatch checkpoint=%s current=%s", checkpoint.get("db_fingerprint"), current_fp)
        return 2
    checkpoint["db_path"] = str(DB_PATH)
    checkpoint["db_fingerprint"] = current_fp
    checkpoint["config_path"] = str(CONFIG_PATH)
    checkpoint["config_sha256"] = sha256_file(CONFIG_PATH)
    checkpoint["split_hash"] = split_hash(config)
    checkpoint["preflight"] = preflight
    checkpoint["last_started_at"] = datetime.now().isoformat(timespec="seconds")
    save_checkpoint(checkpoint)

    results = []
    try:
        for year in years:
            result = process_year(year, args, checkpoint, logger)
            if result:
                results.append(result)
        update_aggregate_outputs(results, logger)
        checkpoint["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
        save_checkpoint(checkpoint)
        summaries = read_existing_summaries()
        write_manifest(config, years, summaries, started_at)
        logger.info("done years=%s", ",".join(str(y) for y in years))
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
