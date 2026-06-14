from __future__ import annotations

import sqlite3
from typing import Any

from src.audit.odds_missingness import numeric_valid_expr, null_or_empty_expr, invalid_expr
from src.audit.odds_schema import jra_where, key_join_condition


def flag_datakubun_cross(con: sqlite3.Connection, by_year: bool = False) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    tan_null = null_or_empty_expr("o1", "TanOdds")
    tan_invalid = invalid_expr("o1", "TanOdds")
    fuku_valid = f"({numeric_valid_expr('o1','FukuOddsLow')} AND {numeric_valid_expr('o1','FukuOddsHigh')})"
    fuku_null = f"({null_or_empty_expr('o1','FukuOddsLow')} OR {null_or_empty_expr('o1','FukuOddsHigh')})"
    fuku_invalid = f"({invalid_expr('o1','FukuOddsLow')} OR {invalid_expr('o1','FukuOddsHigh')})"
    year_select = "se.Year AS year," if by_year else ""
    year_group = "se.Year," if by_year else ""
    sql = f"""
    SELECT {year_select}
      COALESCE(o1.DataKubun, 'missing_row') AS DataKubun,
      COALESCE(o1.TanFlag, 'missing_row') AS TanFlag,
      COALESCE(o1.FukuFlag, 'missing_row') AS FukuFlag,
      CASE
        WHEN o1.Year IS NULL THEN 'missing_row'
        WHEN {tan_valid} THEN 'valid'
        WHEN {tan_null} THEN 'null'
        WHEN {tan_invalid} THEN 'zero_or_invalid'
        ELSE 'conversion_unknown'
      END AS tan_odds_status,
      CASE
        WHEN o1.Year IS NULL THEN 'missing_row'
        WHEN {fuku_valid} THEN 'valid'
        WHEN {fuku_null} THEN 'null'
        WHEN {fuku_invalid} THEN 'zero_or_invalid'
        ELSE 'conversion_unknown'
      END AS fuku_odds_status,
      COUNT(*) AS rows,
      COUNT(DISTINCT printf('%04d%04d%s%02d%02d%02d', se.Year,se.MonthDay,se.JyoCD,se.Kaiji,se.Nichiji,se.RaceNum)) AS race_count,
      SUM(CASE WHEN {tan_valid} THEN 1 ELSE 0 END) AS valid_tan_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_null} THEN 1 ELSE 0 END) AS null_tan_rows,
      SUM(CASE WHEN {fuku_valid} THEN 1 ELSE 0 END) AS valid_fuku_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_null} THEN 1 ELSE 0 END) AS null_fuku_rows
    FROM NL_SE se
    LEFT JOIN NL_O1 o1 ON {join}
    WHERE {jra_where('se')} AND se.Year BETWEEN 2016 AND 2026
    GROUP BY {year_group} COALESCE(o1.DataKubun, 'missing_row'), COALESCE(o1.TanFlag, 'missing_row'), COALESCE(o1.FukuFlag, 'missing_row'), tan_odds_status, fuku_odds_status
    ORDER BY {year_group} rows DESC
    """
    rows = []
    for row in con.execute(sql):
        d = dict(row)
        d["tan_valid_rate"] = d["valid_tan_rows"] / d["rows"] if d["rows"] else None
        d["fuku_valid_rate"] = d["valid_fuku_rows"] / d["rows"] if d["rows"] else None
        d["complete_race_count"] = None
        rows.append(d)
    return rows


def flag_value_summary(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for col in ["TanFlag", "FukuFlag", "WakurenFlag", "DataKubun"]:
        for row in con.execute(f"""
            SELECT '{col}' AS column_name, COALESCE(CAST({col} AS TEXT), '') AS value,
                   COUNT(*) AS o1_rows,
                   SUM(CASE WHEN {numeric_valid_expr('NL_O1','TanOdds')} THEN 1 ELSE 0 END) AS valid_tan_rows,
                   SUM(CASE WHEN {numeric_valid_expr('NL_O1','FukuOddsLow')} AND {numeric_valid_expr('NL_O1','FukuOddsHigh')} THEN 1 ELSE 0 END) AS valid_fuku_rows
            FROM NL_O1
            WHERE Year BETWEEN 2016 AND 2026
            GROUP BY COALESCE(CAST({col} AS TEXT), '')
            ORDER BY o1_rows DESC
        """):
            d = dict(row)
            d["tan_valid_rate"] = d["valid_tan_rows"] / d["o1_rows"] if d["o1_rows"] else None
            d["fuku_valid_rate"] = d["valid_fuku_rows"] / d["o1_rows"] if d["o1_rows"] else None
            rows.append(d)
    return rows
