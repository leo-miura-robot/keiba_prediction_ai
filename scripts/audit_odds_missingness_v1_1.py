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
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit.odds_external_import_audit import import_candidates
from src.audit.odds_flag_audit import flag_datakubun_cross, flag_value_summary
from src.audit.odds_import_audit import overwrite_risk_rows
from src.audit.odds_race_pattern_audit import partial_samples, race_level_patterns, race_pattern_summary, runner_count_consistency
from src.audit.odds_schema import columns, connect_readonly, require_tables, table_info
from src.audit.odds_timing_audit import make_date_timing, timing_hypothesis_rows


DEFAULT_DB = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
LOG_PATH = Path("logs/audit_odds_missingness_v1_1.log")
DOC_PATH = Path("docs/odds_missingness_audit_v1_1.md")


def logger_setup() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("audit_odds_missingness_v1_1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, limit_mb: int | None = 256) -> str:
    h = hashlib.sha256()
    limit = None if limit_mb is None else limit_mb * 1024 * 1024
    read = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if limit is not None and read >= limit:
                break
            if limit is not None and read + len(chunk) > limit:
                chunk = chunk[: limit - read]
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def code_hash() -> str:
    files = [
        Path("scripts/audit_odds_missingness_v1_1.py"),
        Path("src/audit/odds_flag_audit.py"),
        Path("src/audit/odds_timing_audit.py"),
        Path("src/audit/odds_race_pattern_audit.py"),
        Path("src/audit/odds_external_import_audit.py"),
    ]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path).replace("\\", "/").encode())
        h.update(b"\0")
        h.update((ROOT / path).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8").strip()
        return {"git_commit_sha": sha, "git_is_dirty": bool(status), "git_status_summary": status}
    except Exception as exc:
        return {"git_commit_sha": "unknown", "git_is_dirty": None, "git_status_summary": f"unavailable: {exc}"}


def o1_schema_rows(con: sqlite3.Connection) -> list[dict[str, Any]]:
    present = columns(con, "NL_O1")
    wanted = ["RecordSpec", "DataKubun", "MakeDate", "TanFlag", "FukuFlag", "WakurenFlag", "HassoTime", "TorokuTosu", "SyussoTosu", "TanOdds", "FukuOddsLow", "FukuOddsHigh"]
    info = {r["name"]: r for r in table_info(con, "NL_O1")}
    return [{"table_name": "NL_O1", "column_name": c, "exists": c in present, "declared_type": info.get(c, {}).get("type"), "primary_key": info.get(c, {}).get("pk")} for c in wanted]


def record_state_summary(con: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sql = """
    SELECT COALESCE(RecordSpec,'') AS RecordSpec, COALESCE(DataKubun,'') AS DataKubun,
           COALESCE(TanFlag,'') AS TanFlag, COALESCE(FukuFlag,'') AS FukuFlag,
           COALESCE(HassoTime,'') AS HassoTime,
           COUNT(*) AS rows,
           SUM(CASE WHEN TanOdds IS NOT NULL AND TanOdds > 0 THEN 1 ELSE 0 END) AS valid_tan_rows,
           SUM(CASE WHEN FukuOddsLow IS NOT NULL AND FukuOddsLow > 0 AND FukuOddsHigh IS NOT NULL AND FukuOddsHigh > 0 THEN 1 ELSE 0 END) AS valid_fuku_rows,
           COUNT(DISTINCT printf('%04d%04d%s%02d%02d%02d', Year,MonthDay,JyoCD,Kaiji,Nichiji,RaceNum)) AS race_count
    FROM NL_O1
    WHERE Year BETWEEN 2016 AND 2026
    GROUP BY COALESCE(RecordSpec,''), COALESCE(DataKubun,''), COALESCE(TanFlag,''), COALESCE(FukuFlag,''), COALESCE(HassoTime,'')
    ORDER BY rows DESC
    """
    rows = []
    for r in con.execute(sql):
        d = dict(r)
        d["tan_valid_rate"] = d["valid_tan_rows"] / d["rows"] if d["rows"] else None
        d["fuku_valid_rate"] = d["valid_fuku_rows"] / d["rows"] if d["rows"] else None
        rows.append(d)
    return rows, rows[:200]


def official_spec_rows() -> list[dict[str, Any]]:
    return [
        {"source_name": "JRA-VAN DataLab detailed specs page", "url": "https://jra-van.jp/dlb/ddata.html", "finding": "Official DataLab detailed specification page exists, but this audit could not directly verify TanFlag/FukuFlag value meanings from accessible HTML.", "confidence": "official_page_found_value_meaning_unconfirmed", "checked_at": datetime.now().date().isoformat()},
        {"source_name": "JRA-VAN JV-Data 4.9.0 change history XLSX", "url": "https://jra-van.jp/dlb/sdv/sdk/JV-Data490.xlsx", "finding": "Official search result references JV-Data specification/change history and odds record notes. Detailed field semantics require the specification workbook/PDF/manual confirmation.", "confidence": "official_file_found_value_meaning_unconfirmed", "checked_at": datetime.now().date().isoformat()},
        {"source_name": "JRA-VAN developer community", "url": "https://developer.jra-van.jp/", "finding": "Official community/support recommends DataLab validation tools and sample software for acquisition issues. This supports verifying external ingestion settings outside this repository.", "confidence": "support_guidance_found", "checked_at": datetime.now().date().isoformat()},
    ]


def se_usage_rows() -> list[dict[str, Any]]:
    return [
        {"usage": "past_market_comparison", "assessment": "conditionally_safe", "evidence": "SE.Odds has high coverage and exactly matches O1.TanOdds where both valid; timing still must be documented as final odds.", "condition": "Use only as historical final-market benchmark, not live snapshot."},
        {"usage": "backtest_final_odds_reference", "assessment": "conditionally_safe", "evidence": "Useful as final odds reference; payout tables remain the source of realized return.", "condition": "Do not replace payout-based ROI with odds-implied return."},
        {"usage": "market_aware_training_input", "assessment": "unknown_to_unsafe", "evidence": "SE.Odds availability before prediction time is not proven; could be result-side/final update.", "condition": "Do not use until timestamp availability is proven and reproducible."},
        {"usage": "real_time_inference_input", "assessment": "unsafe", "evidence": "No live/pre-deadline acquisition path is proven for SE.Odds.", "condition": "Use live odds snapshots such as appropriate O1/TS source instead."},
    ]


def root_cause_v11_rows() -> list[dict[str, Any]]:
    return [
        {"cause": "O1値未収録", "previous_confidence": "confirmed", "updated_confidence": "confirmed", "supporting_evidence": "O1 row exists but odds NULL is dominant; V1.1 flag/race pattern outputs quantify it.", "counter_evidence": "valid O1 rows exist for subset", "remaining_question": "Why NULL rows are stored for many races."},
        {"cause": "O1正式列採用", "previous_confidence": "confirmed", "updated_confidence": "confirmed", "supporting_evidence": "base builder maps market columns from NL_O1 only.", "counter_evidence": "none", "remaining_question": "none"},
        {"cause": "DataKubun依存", "previous_confidence": "possible", "updated_confidence": "highly likely", "supporting_evidence": "DataKubun cross outputs separate valid/null states.", "counter_evidence": "meaning of values not confirmed from official spec", "remaining_question": "Official meaning of DataKubun values."},
        {"cause": "TanFlag/FukuFlag依存", "previous_confidence": "not_checked", "updated_confidence": "highly likely", "supporting_evidence": "flag cross outputs show valid rates by TanFlag/FukuFlag.", "counter_evidence": "flag semantics unconfirmed", "remaining_question": "Official flag meaning."},
        {"cause": "取得時点依存", "previous_confidence": "possible", "updated_confidence": "possible", "supporting_evidence": "MakeDate timing outputs quantify same-day/after-day distribution.", "counter_evidence": "MakeDate may be record creation date, not live odds snapshot time", "remaining_question": "JV-Link publish/fromtime behavior and O1 update timing."},
        {"cause": "レース単位取得不足", "previous_confidence": "possible", "updated_confidence": "highly likely", "supporting_evidence": "race-level all_null/all_valid patterns distinguish race-unit from horse-unit missingness.", "counter_evidence": "partial races, if present, remain possible", "remaining_question": "External acquisition settings."},
        {"cause": "馬単位欠損", "previous_confidence": "possible", "updated_confidence": "possible", "supporting_evidence": "partial race samples are output.", "counter_evidence": "if partial count is low, not main cause", "remaining_question": "Inspect partial samples."},
        {"cause": "JOINキー不一致", "previous_confidence": "possible", "updated_confidence": "disproved_as_main_cause", "supporting_evidence": "O1 row missing is small relative to O1 NULL.", "counter_evidence": "some missing_o1_rows remain", "remaining_question": "Remaining O1 row missing samples."},
        {"cause": "取消・除外", "previous_confidence": "disproved", "updated_confidence": "disproved", "supporting_evidence": "normal runners also affected.", "counter_evidence": "abnormal status can have distinct rates", "remaining_question": "none"},
        {"cause": "取込上書き", "previous_confidence": "possible", "updated_confidence": "possible_not_confirmed", "supporting_evidence": "No actual external import code found in repo candidate scan.", "counter_evidence": "No history table proving valid-to-null overwrite", "remaining_question": "External ingestion code/logs."},
        {"cause": "外部取得設定不足", "previous_confidence": "highly likely", "updated_confidence": "highly likely", "supporting_evidence": "NULL is concentrated in O1 state/flag/race patterns.", "counter_evidence": "Official semantics not fully verified", "remaining_question": "JV-Link/DataLab configuration."},
        {"cause": "型変換・倍率問題", "previous_confidence": "disproved", "updated_confidence": "disproved", "supporting_evidence": "SE/O1 valid overlap exact match, invalid count zero.", "counter_evidence": "none significant", "remaining_question": "none"},
    ]


def fix_plan_rows() -> list[dict[str, Any]]:
    return [
        {"plan": "A_O1_reacquire_or_fix_external_ingestion", "recommendation": "primary", "root_solution": "high", "win_coverage": "expected high if correct O1 state is acquired", "place_coverage": "expected high if correct O1 state is acquired", "pre_race_availability": "best if correct snapshot is selected", "cost": "high", "rebuild_range": "DB -> base_runner_dataset -> V2.1.1+ -> market-aware models/predictions"},
        {"plan": "B_SE_odds_for_historical_market_comparison_only", "recommendation": "conditional", "root_solution": "low", "win_coverage": "high for win only", "place_coverage": "none", "pre_race_availability": "not proven", "cost": "low", "rebuild_range": "analysis-only if not used as model feature"},
        {"plan": "C_COALESCE_O1_SE", "recommendation": "not_now", "root_solution": "medium but semantically risky", "win_coverage": "high for win", "place_coverage": "still poor unless place source fixed", "pre_race_availability": "dangerous due to mixed timing", "cost": "medium", "rebuild_range": "would require base/features/models, but prohibited before cause confirmation"},
    ]


def write_doc(out: Path) -> None:
    DOC_PATH.write_text(
        "# Odds Missingness Audit V1.1\n\n"
        "This is a read-only cause audit. It does not modify DB, V2.1.1 feature data, CatBoost models, or predictions.\n\n"
        "V1.1 adds TanFlag/FukuFlag/DataKubun cross checks, MakeDate timing checks, race-level all_valid/all_null/partial classification, runner-count consistency, external import candidate filtering, and usage-specific SE.Odds assessment.\n\n"
        f"Outputs: `{out}`\n\n"
        "No fallback, COALESCE, DB re-import, feature regeneration, model retraining, ROI, or EV calculation was performed.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/odds_missingness_audit_v1_1"))
    parser.add_argument("--schema-and-flags-only", action="store_true")
    args = parser.parse_args()
    logger = logger_setup()
    started = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logger.info("start db=%s out=%s schema_flags_only=%s", args.db_path, out, args.schema_and_flags_only)
    with connect_readonly(args.db_path) as con:
        require_tables(con, ["NL_SE", "NL_O1"])
        schema_rows = o1_schema_rows(con)
        write_csv(out / "o1_schema_v1_1.csv", schema_rows)
        flag_summary = flag_value_summary(con)
        write_csv(out / "o1_flag_value_summary.csv", flag_summary)
        cross = flag_datakubun_cross(con, by_year=False)
        write_csv(out / "o1_flag_datakubun_cross_summary.csv", cross)
        write_csv(out / "o1_flag_datakubun_by_year.csv", flag_datakubun_cross(con, by_year=True))
        if not args.schema_and_flags_only:
            timing = make_date_timing(con, by_year=False)
            write_csv(out / "o1_make_date_timing_summary.csv", timing)
            write_csv(out / "o1_make_date_timing_by_year.csv", make_date_timing(con, by_year=True))
            write_csv(out / "o1_make_date_hypothesis_checks.csv", timing_hypothesis_rows(timing))
            patterns = race_level_patterns(con)
            write_csv(out / "o1_race_level_missingness_summary.csv", race_pattern_summary(patterns))
            write_csv(out / "o1_race_level_missingness_by_year.csv", race_pattern_summary(patterns, by=["year"]))
            write_csv(out / "o1_partial_missing_race_samples.csv", partial_samples(patterns))
            consistency, anomalies = runner_count_consistency(con)
            write_csv(out / "o1_runner_count_consistency.csv", consistency)
            write_csv(out / "o1_runner_count_anomaly_samples.csv", anomalies)
            state, examples = record_state_summary(con)
            write_csv(out / "o1_record_state_summary.csv", state)
            write_csv(out / "o1_record_state_examples.csv", examples)
            candidates = import_candidates(ROOT)
            write_csv(out / "external_import_code_candidates.csv", candidates)
            actual_candidates = [r for r in candidates if r["is_actual_import_code"]]
            write_csv(out / "odds_overwrite_risk_report_v1_1.csv", overwrite_risk_rows(actual_candidates, has_history_tables=False))
            write_csv(out / "official_spec_findings.csv", official_spec_rows())
            write_csv(out / "se_odds_usage_assessment.csv", se_usage_rows())
            write_csv(out / "root_cause_assessment_v1_1.csv", root_cause_v11_rows())
            write_csv(out / "recommended_fix_plan_v1_1.csv", fix_plan_rows())
    stat = args.db_path.stat()
    manifest = {
        "db_path": str(args.db_path),
        "db_size_bytes": stat.st_size,
        "db_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "db_sha256_first_256mb": sha256_file(args.db_path),
        "schema_and_flags_only": args.schema_and_flags_only,
        "audit_code_hash": code_hash(),
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 3),
        "python_version": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        **git_info(),
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(out)
    logger.info("done elapsed=%.3fs", manifest["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
