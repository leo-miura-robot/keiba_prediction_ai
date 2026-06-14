from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl


DB_PATH = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
OUT_DIR = Path("outputs")
DATASET_DIR = OUT_DIR / "base_runner_dataset"
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "build_full_dataset.log"
CHECKPOINT_PATH = OUT_DIR / "base_runner_dataset_checkpoint.json"

FULL_SUMMARY_CSV = OUT_DIR / "full_dataset_summary.csv"
COLUMN_QUALITY_CSV = OUT_DIR / "column_quality_summary.csv"
SPECIAL_CASES_CSV = OUT_DIR / "special_result_cases.csv"
SAMPLE_CSV = OUT_DIR / "base_runner_dataset_sample.csv"

DESIGN_DOC = Path("docs/full_dataset_design.md")
FEATURE_DOC = Path("docs/feature_inventory_full.md")
LEAKAGE_DOC = Path("docs/leakage_check_full.md")
QUALITY_DOC = Path("docs/full_dataset_quality_report.md")

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
    return {"db_path": str(DB_PATH), "years": {}}


def save_checkpoint(checkpoint: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
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
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

    DESIGN_DOC.write_text("\n".join([
        "# Full Runner Dataset Design",
        "",
        f"Source DB: `{DB_PATH}`",
        "",
        "Base table is `NL_SE`, one row per runner. Race metadata joins from `NL_RA` by race key. Odds joins from `NL_O1` by race key plus `Umaban`. Win and place payouts are expanded from `NL_HR` payout slots and joined by race key plus `Umaban`.",
        "",
        "The dataset is limited to JRA central racecourse codes `01` through `10`. Other `JyoCD` values in `NL_SE` do not have matching JRA odds/payout records in `NL_O1/NL_HR` and are excluded from this base JRA runner dataset.",
        "",
        "`race_id` is `YYYYMMDDJyoCDKaijiNichijiRaceNum` with zero padding. `entry_id` appends zero-padded `Umaban`. `race_date` is derived from `Year` and `MonthDay`.",
        "",
        "Outputs are partitioned by year under `outputs/base_runner_dataset/year=YYYY/data.parquet`.",
        "",
    ]), encoding="utf-8-sig")

    FEATURE_DOC.write_text("\n".join([
        "# Feature Inventory Full",
        "",
        "## Directly Usable",
        "",
        "\n".join(f"- `{c}`" for c in DIRECT_FEATURES),
        "",
        "## Conditionally Usable",
        "",
        "These depend on prediction timing and should be used only if available before the betting decision.",
        "",
        "\n".join(f"- `{c}`" for c in CONDITIONAL_FEATURES),
        "",
        "## Derived Later From Past Races",
        "",
        "\n".join(f"- `{c}`" for c in DERIVED_FEATURE_IDEAS),
        "",
    ]), encoding="utf-8-sig")

    LEAKAGE_DOC.write_text("\n".join([
        "# Leakage Check Full",
        "",
        "## Leakage Or Evaluation-Only Columns",
        "",
        "\n".join(f"- `{c}`" for c in LEAKAGE_COLUMNS),
        "",
        "Result columns, race-time performance, corner positions, payouts, and target columns must not be model features.",
        "",
        "Odds, popularity, votes, body weight, weather, and going are conditionally usable only when the intended prediction timestamp can reproduce their availability.",
        "",
    ]), encoding="utf-8-sig")

    QUALITY_DOC.write_text("\n".join([
        "# Full Dataset Quality Report",
        "",
        f"Total rows: `{total_rows}`",
        f"Total races: `{total_races}`",
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
        "See `outputs/column_quality_summary.csv` for null rates and unique counts by column.",
        "See `outputs/special_result_cases.csv` for abnormal and payout/target mismatch rows.",
        "",
    ]), encoding="utf-8-sig")


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

    write_docs(summaries, quality_existing + quality_new)
    logger.info("aggregate outputs updated years=%s", ",".join(sorted(years)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", help="Comma-separated years or ranges, e.g. 2016 or 2016-2026")
    parser.add_argument("--resume", action="store_true", help="Skip completed checkpoint years.")
    parser.add_argument("--force", action="store_true", help="Rebuild requested years even if complete.")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("start db=%s years=%s resume=%s force=%s", DB_PATH, args.years or "2016-2026", args.resume, args.force)
    if not DB_PATH.exists():
        logger.error("DB not found: %s", DB_PATH)
        return 2

    years = parse_years(args.years)
    checkpoint = load_checkpoint()
    checkpoint["db_path"] = str(DB_PATH)
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
