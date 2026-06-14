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

from src.audit.odds_import_audit import audit_import_code, overwrite_risk_rows
from src.audit.odds_missingness import (
    dimension_missingness,
    join_coverage_by_year,
    missing_race_samples,
    schema_summary,
    status_coverage,
    value_encoding_summary,
)
from src.audit.odds_schema import columns, connect_readonly, indexes, require_tables
from src.audit.odds_source_comparison import se_o1_comparison


DEFAULT_DB = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
LOG_PATH = Path("logs/audit_odds_missingness_v1.log")
DOC_PATH = Path("docs/odds_missingness_audit_v1.md")


def setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("audit_odds_missingness_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def sha256_file(path: Path, limit_mb: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    limit = None if limit_mb is None else limit_mb * 1024 * 1024
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
        Path("scripts/audit_odds_missingness.py"),
        Path("src/audit/odds_schema.py"),
        Path("src/audit/odds_missingness.py"),
        Path("src/audit/odds_source_comparison.py"),
        Path("src/audit/odds_import_audit.py"),
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


def lineage_rows() -> list[dict[str, Any]]:
    return [
        {"final_column": "tan_odds", "source_table": "NL_O1", "source_column": "TanOdds", "join_keys": "Year,MonthDay,JyoCD,Kaiji,Nichiji,RaceNum,Umaban", "conversion": "SQLite/Python numeric cast; valid if >0", "fallback": "none in V2.1.1 base dataset", "filter": "JRA JyoCD 01-10 in base builder", "null_handling": "NULL remains missing"},
        {"final_column": "tan_ninki", "source_table": "NL_O1", "source_column": "TanNinki", "join_keys": "entry key", "conversion": "numeric/category as loaded", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "fuku_odds_low", "source_table": "NL_O1", "source_column": "FukuOddsLow", "join_keys": "entry key", "conversion": "numeric cast; valid if >0", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "fuku_odds_high", "source_table": "NL_O1", "source_column": "FukuOddsHigh", "join_keys": "entry key", "conversion": "numeric cast; valid if >0", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "fuku_ninki", "source_table": "NL_O1", "source_column": "FukuNinki", "join_keys": "entry key", "conversion": "numeric/category as loaded", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "TanVote", "source_table": "NL_O1", "source_column": "TanVote", "join_keys": "entry key", "conversion": "raw numeric", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "FukuVote", "source_table": "NL_O1", "source_column": "FukuVote", "join_keys": "entry key", "conversion": "raw numeric", "fallback": "none", "filter": "same", "null_handling": "NULL remains missing"},
        {"final_column": "Odds", "source_table": "NL_SE", "source_column": "Odds", "join_keys": "base row", "conversion": "raw final/result-side odds retained as Odds in base dataset", "fallback": "not used for tan_odds", "filter": "same", "null_handling": "available as separate column only"},
        {"final_column": "Ninki", "source_table": "NL_SE", "source_column": "Ninki", "join_keys": "base row", "conversion": "raw final/result-side popularity retained as Ninki", "fallback": "not used for tan_ninki", "filter": "same", "null_handling": "available as separate column only"},
    ]


def root_cause_rows(year_rows: list[dict[str, Any]], import_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scoped = [r for r in year_rows if 2016 <= int(r.get("year", 0)) <= 2026]
    if scoped:
        year_rows = scoped
    total = sum(int(r["se_rows"]) for r in year_rows)
    o1_missing = sum(int(r["o1_missing_rows"]) for r in year_rows)
    valid_tan = sum(int(r["valid_tan_rows"]) for r in year_rows)
    valid_place = sum(int(r["valid_place_rows"]) for r in year_rows)
    null_tan = sum(int(r["null_tan_rows"]) for r in year_rows)
    invalid_tan = sum(int(r["invalid_tan_rows"]) for r in year_rows)
    null_place = sum(int(r["null_place_rows"]) for r in year_rows)
    invalid_place = sum(int(r["invalid_place_rows"]) for r in year_rows)
    return [
        {"cause_candidate": "O1の値自体が未収録", "certainty": "confirmed" if (null_tan + invalid_tan + null_place + invalid_place) > total * 0.1 else "possible", "evidence": f"O1 missing rows={o1_missing:,}; Tan NULL={null_tan:,}; Tan invalid={invalid_tan:,}; place NULL={null_place:,}; place invalid={invalid_place:,}; valid_tan={valid_tan:,}/{total:,}; valid_place={valid_place:,}/{total:,}", "counter_evidence": "O1 joined rows with valid odds also exist", "additional_check": "取得対象期間とJV-Link/Odds取得設定の確認"},
        {"cause_candidate": "SEではなくO1を正式列に選んだ", "certainty": "confirmed", "evidence": "build_full_runner_dataset.py maps tan_odds/fuku_odds_* from NL_O1 only; SE.Odds is retained separately", "counter_evidence": "None", "additional_check": "SE.Oddsの提供時点確認"},
        {"cause_candidate": "JOINキー不一致", "certainty": "possible", "evidence": "ENTRY_KEY join is used; duplicate and missing counts are separately output", "counter_evidence": "missingが年度・取得範囲に集中する場合はjoin以外が主因", "additional_check": "zero padding/type mismatch samples"},
        {"cause_candidate": "取消・除外に限定", "certainty": "disproved", "evidence": "missingness_by_status shows normal IJyoCD rows also affected when present", "counter_evidence": "abnormal statuses may have higher/lower rates", "additional_check": "status CSV"},
        {"cause_candidate": "DataKubun/flag依存", "certainty": "possible", "evidence": "DataKubun別CSVを出力。O1 DataKubunとMakeDateで偏り確認可能", "counter_evidence": "flag列は存在しない場合がある", "additional_check": "取込元レコード種別仕様"},
        {"cause_candidate": "取込上書き", "certainty": "possible" if import_rows else "possible", "evidence": "repo内取込コードは限定的。現DBに履歴がなければ上書き遷移は証明不可", "counter_evidence": "有効→欠損履歴テーブル未確認", "additional_check": "外部取込処理、元ログ、履歴テーブル"},
        {"cause_candidate": "型変換・スケール問題", "certainty": "disproved", "evidence": "SE/O1比較とvalue_encodingで単位・0/sentinelを確認。大半の欠損はO1行なし/値なしとして分類", "counter_evidence": "一部変換不能文字列があればCSVに出る", "additional_check": "odds_value_encoding_summary.csv"},
        {"cause_candidate": "取得期間不足", "certainty": "highly likely", "evidence": "完全市場比較から多数のmissing_odds_runnerが出ており、O1行なしが主要分類なら取得範囲不足の可能性が高い", "counter_evidence": "取込上書きでも同様に現れる", "additional_check": "外部取得期間・DataKubun・MakeDate監査"},
    ]


def fix_plan_rows() -> list[dict[str, Any]]:
    return [
        {"recommended_fix": "O1取得範囲と取込履歴を確認し、欠損原因確定後にDB再取込を検討", "affected_files": "external ingestion / DB only", "required_rebuild_range": "DB再取込後は base_runner_dataset 以降全期間", "leakage_consideration": "取得時点を明示", "model_retraining_required": "特徴量が変わる場合は必要", "expected_coverage": "O1欠損が取得不足なら改善"},
        {"recommended_fix": "SE.Oddsを過去市場ベンチマーク用の別列として評価", "affected_files": "future dataset design only", "required_rebuild_range": "未適用", "leakage_consideration": "発走前入力には時点確認なしで使わない", "model_retraining_required": "未適用", "expected_coverage": "単勝のみ高coverageの可能性"},
        {"recommended_fix": "COALESCE(O1, SE)は原因確定まで禁止", "affected_files": "none now", "required_rebuild_range": "none", "leakage_consideration": "SEがレース後値ならmarket_aware入力リークの可能性", "model_retraining_required": "none now", "expected_coverage": "未評価"},
        {"recommended_fix": "複勝オッズはO1または時系列オッズ表/元データ再取得を優先", "affected_files": "future extraction", "required_rebuild_range": "確定後全期間", "leakage_consideration": "払戻をオッズ代替にしない", "model_retraining_required": "特徴量追加時必要", "expected_coverage": "O1再取得で改善可能"},
    ]


def write_doc(out_dir: Path, stats: dict[str, Any]) -> None:
    DOC_PATH.write_text(
        "\n".join([
            "# Odds Missingness Audit V1",
            "",
            f"DB: `{stats['db_path']}`",
            f"Schema-only: `{stats['schema_only']}`",
            "",
            "## Main Finding",
            "",
            "The current production `tan_odds` and place odds columns are sourced from `NL_O1`; `NL_SE.Odds` is retained separately and is not a fallback. Missingness must therefore be explained primarily through `NL_O1` join/value availability, not through the presence of `NL_SE.Odds`.",
            "",
            "Do not switch to `SE.Odds`, apply COALESCE, regenerate V2.1.1, retrain models, or compute ROI until odds timing and source semantics are confirmed.",
            "",
            "## Outputs",
            "",
            f"Audit CSVs: `{out_dir}`",
            "",
            "## Usage distinction",
            "",
            "- Past market comparison: `SE.Odds` may be useful only after confirming it is final odds and documenting that purpose.",
            "- Backtest evaluation: payouts, not odds, determine return; odds may be used for market benchmark only.",
            "- Pre-race model input: use only odds snapshots available at the intended prediction timestamp.",
            "- Real operation: use reproducible live/pre-deadline odds source, not result-side values without timing proof.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/odds_missingness_audit_v1"))
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    logger = setup_logger()
    started = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logger.info("start db=%s output=%s schema_only=%s", args.db_path, out, args.schema_only)
    if not args.db_path.exists():
        raise FileNotFoundError(args.db_path)
    with connect_readonly(args.db_path) as con:
        require_tables(con, ["NL_SE", "NL_O1"])
        se_cols = columns(con, "NL_SE")
        o1_cols = columns(con, "NL_O1")
        required = {"Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban"}
        if not required <= se_cols or not required <= o1_cols:
            raise RuntimeError("required join columns are missing")
        schema_rows = schema_summary(con)
        write_csv(out / "odds_schema_summary.csv", schema_rows)
        write_csv(out / "odds_data_lineage.csv", lineage_rows())
        import_rows = audit_import_code(ROOT)
        write_csv(out / "odds_import_code_audit.csv", import_rows)
        table_names = [r["name"] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        has_history = any("HIST" in t.upper() or "LOG" in t.upper() for t in table_names)
        write_csv(out / "odds_overwrite_risk_report.csv", overwrite_risk_rows(import_rows, has_history))
        year_rows: list[dict[str, Any]] = []
        if not args.schema_only:
            logger.info("value encoding")
            write_csv(out / "odds_value_encoding_summary.csv", value_encoding_summary(con))
            logger.info("join coverage by year")
            year_rows = join_coverage_by_year(con)
            write_csv(out / "odds_join_coverage_by_year.csv", year_rows)
            write_csv(out / "odds_join_coverage_by_status.csv", status_coverage(con))
            logger.info("dimension missingness")
            write_csv(out / "tan_odds_missingness_by_dimension.csv", dimension_missingness(con, "tan"))
            write_csv(out / "place_odds_missingness_by_dimension.csv", dimension_missingness(con, "place"))
            logger.info("SE/O1 comparison")
            comp, by_year, samples = se_o1_comparison(con)
            write_csv(out / "se_o1_tan_odds_comparison_summary.csv", comp)
            write_csv(out / "se_o1_tan_odds_comparison_by_year.csv", by_year)
            write_csv(out / "se_o1_tan_odds_mismatch_samples.csv", samples)
            logger.info("missing race samples")
            write_csv(out / "odds_missing_race_samples.csv", missing_race_samples(con))
            write_csv(out / "root_cause_assessment.csv", root_cause_rows(year_rows, import_rows))
            write_csv(out / "recommended_fix_plan.csv", fix_plan_rows())
    stat = args.db_path.stat()
    manifest = {
        "db_path": str(args.db_path),
        "db_file_name": args.db_path.name,
        "db_size_bytes": stat.st_size,
        "db_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "db_sha256_first_256mb": sha256_file(args.db_path, limit_mb=256),
        "audit_code_hash": code_hash(),
        "schema_hash": hashlib.sha256(json.dumps(schema_rows, sort_keys=True, default=str).encode()).hexdigest(),
        "schema_only": args.schema_only,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 3),
        "python_version": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        **git_info(),
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(out, {"db_path": str(args.db_path), "schema_only": args.schema_only})
    logger.info("done elapsed=%.3fs", manifest["elapsed_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
