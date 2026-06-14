from __future__ import annotations

import sqlite3
from typing import Any

from src.audit.odds_missingness import numeric_valid_expr
from src.audit.odds_schema import jra_where, key_join_condition


def date_diff_expr(make_col: str = "o1.MakeDate") -> str:
    race = "date(se.Year || '-' || substr('00' || CAST(CAST(se.MonthDay / 100 AS INTEGER) AS TEXT), -2) || '-' || substr('00' || CAST(se.MonthDay % 100 AS TEXT), -2))"
    make = f"date(substr({make_col},1,4) || '-' || substr({make_col},5,2) || '-' || substr({make_col},7,2))"
    return f"CAST(julianday({make}) - julianday({race}) AS INTEGER)"


def timing_bucket_expr(diff: str = "make_date_minus_race_date_days") -> str:
    return f"""CASE
      WHEN {diff} IS NULL THEN 'unknown'
      WHEN {diff} < 0 THEN 'before_race'
      WHEN {diff} = 0 THEN 'same_day'
      WHEN {diff} = 1 THEN 'day_after'
      WHEN {diff} BETWEEN 2 AND 7 THEN '2_to_7_days_after'
      WHEN {diff} > 7 THEN 'more_than_7_days_after'
      ELSE 'unknown' END"""


def make_date_timing(con: sqlite3.Connection, by_year: bool = False) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    fuku_valid = f"({numeric_valid_expr('o1','FukuOddsLow')} AND {numeric_valid_expr('o1','FukuOddsHigh')})"
    diff = date_diff_expr()
    year_select = "year," if by_year else ""
    year_group = "year," if by_year else ""
    sql = f"""
    WITH joined AS (
      SELECT se.Year AS year, o1.MakeDate AS MakeDate, {diff} AS make_date_minus_race_date_days,
             CASE WHEN {tan_valid} THEN 1 ELSE 0 END AS tan_valid,
             CASE WHEN {fuku_valid} THEN 1 ELSE 0 END AS fuku_valid
      FROM NL_SE se LEFT JOIN NL_O1 o1 ON {join}
      WHERE {jra_where('se')} AND se.Year BETWEEN 2016 AND 2026
    )
    SELECT {year_select}
      {timing_bucket_expr()} AS timing_bucket,
      COUNT(*) AS rows,
      SUM(CASE WHEN MakeDate IS NULL OR TRIM(CAST(MakeDate AS TEXT))='' THEN 1 ELSE 0 END) AS missing_make_date_rows,
      MIN(make_date_minus_race_date_days) AS min_days,
      MAX(make_date_minus_race_date_days) AS max_days,
      SUM(tan_valid) AS valid_tan_rows,
      SUM(fuku_valid) AS valid_fuku_rows
    FROM joined
    GROUP BY {year_group} timing_bucket
    ORDER BY {year_group} rows DESC
    """
    rows = []
    for row in con.execute(sql):
        d = dict(row)
        d["tan_valid_rate"] = d["valid_tan_rows"] / d["rows"] if d["rows"] else None
        d["fuku_valid_rate"] = d["valid_fuku_rows"] / d["rows"] if d["rows"] else None
        rows.append(d)
    return rows


def timing_hypothesis_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket = {r["timing_bucket"]: r for r in summary if "year" not in r}
    same = by_bucket.get("same_day", {})
    after = sum(int(by_bucket.get(k, {}).get("valid_tan_rows", 0)) for k in ["day_after", "2_to_7_days_after", "more_than_7_days_after"])
    return [
        {"hypothesis": "H1_pre_or_same_day_initial_records_are_null", "supporting_rows": same.get("rows", 0) - same.get("valid_tan_rows", 0), "counterexample_rows": same.get("valid_tan_rows", 0), "assessment": "mixed"},
        {"hypothesis": "H2_only_post_race_records_have_odds", "supporting_rows": after, "counterexample_rows": same.get("valid_tan_rows", 0), "assessment": "disproved_if_same_day_valid_exists"},
        {"hypothesis": "H3_SE_odds_filled_after_result_update", "supporting_rows": None, "counterexample_rows": None, "assessment": "unknown_from_O1_timing_only"},
        {"hypothesis": "H4_O1_only_some_snapshots_are_saved", "supporting_rows": same.get("rows", 0), "counterexample_rows": None, "assessment": "possible"},
    ]
