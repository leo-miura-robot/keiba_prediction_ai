from __future__ import annotations

import sqlite3
from typing import Any

from src.audit.odds_schema import ENTRY_KEY, RACE_KEY, duplicate_key_count, jra_where, key_join_condition, q, row_count, table_info


def numeric_valid_expr(alias: str, col: str) -> str:
    x = f"{alias}.{q(col)}"
    return f"({x} IS NOT NULL AND TRIM(CAST({x} AS TEXT)) <> '' AND CAST({x} AS REAL) > 0 AND CAST({x} AS REAL) NOT IN (9999, 9999.0, 999.9, 99999))"


def null_or_empty_expr(alias: str, col: str) -> str:
    x = f"{alias}.{q(col)}"
    return f"({x} IS NULL OR TRIM(CAST({x} AS TEXT)) = '')"


def invalid_expr(alias: str, col: str) -> str:
    x = f"{alias}.{q(col)}"
    return f"({x} IS NOT NULL AND TRIM(CAST({x} AS TEXT)) <> '' AND (CAST({x} AS REAL) <= 0 OR CAST({x} AS REAL) IN (9999, 9999.0, 999.9, 99999)))"


def schema_summary(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for table in ["NL_SE", "NL_O1"]:
        dup = duplicate_key_count(con, table, ENTRY_KEY)
        count = row_count(con, table)
        for col in table_info(con, table):
            rows.append({
                "table_name": table,
                "column_name": col["name"],
                "declared_type": col["type"],
                "notnull": col["notnull"],
                "default_value": col["dflt_value"],
                "primary_key": col["pk"],
                "table_row_count": count,
                "duplicate_entry_key_count": dup,
            })
    return rows


def join_coverage_by_year(con: sqlite3.Connection) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    tan_null = null_or_empty_expr("o1", "TanOdds")
    tan_invalid = invalid_expr("o1", "TanOdds")
    fuku_valid = f"({numeric_valid_expr('o1', 'FukuOddsLow')} AND {numeric_valid_expr('o1', 'FukuOddsHigh')})"
    fuku_null = f"({null_or_empty_expr('o1', 'FukuOddsLow')} OR {null_or_empty_expr('o1', 'FukuOddsHigh')})"
    fuku_invalid = f"(({invalid_expr('o1', 'FukuOddsLow')}) OR ({invalid_expr('o1', 'FukuOddsHigh')}))"
    sql = f"""
    SELECT
      se.Year AS year,
      COUNT(*) AS se_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL THEN 1 ELSE 0 END) AS o1_matched_rows,
      SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS o1_missing_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_valid} THEN 1 ELSE 0 END) AS valid_tan_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_null} THEN 1 ELSE 0 END) AS null_tan_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_invalid} THEN 1 ELSE 0 END) AS invalid_tan_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_valid} THEN 1 ELSE 0 END) AS valid_place_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_null} THEN 1 ELSE 0 END) AS null_place_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_invalid} THEN 1 ELSE 0 END) AS invalid_place_rows,
      SUM(CASE WHEN se.Odds IS NOT NULL AND TRIM(CAST(se.Odds AS TEXT)) <> '' AND CAST(se.Odds AS REAL) > 0 THEN 1 ELSE 0 END) AS valid_se_odds_rows
    FROM NL_SE se
    LEFT JOIN NL_O1 o1 ON {join}
    WHERE {jra_where('se')}
    GROUP BY se.Year
    ORDER BY se.Year
    """
    out = []
    for row in con.execute(sql):
        d = dict(row)
        d["tan_coverage_rate"] = d["valid_tan_rows"] / d["se_rows"] if d["se_rows"] else None
        d["place_coverage_rate"] = d["valid_place_rows"] / d["se_rows"] if d["se_rows"] else None
        d["o1_match_rate"] = d["o1_matched_rows"] / d["se_rows"] if d["se_rows"] else None
        d["se_odds_coverage_rate"] = d["valid_se_odds_rows"] / d["se_rows"] if d["se_rows"] else None
        out.append(d)
    return out


def status_coverage(con: sqlite3.Connection) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    fuku_valid = f"({numeric_valid_expr('o1', 'FukuOddsLow')} AND {numeric_valid_expr('o1', 'FukuOddsHigh')})"
    sql = f"""
    SELECT
      COALESCE(se.IJyoCD, '') AS runner_status,
      COUNT(*) AS rows,
      SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS o1_missing_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_valid} THEN 1 ELSE 0 END) AS valid_tan_rows,
      SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_valid} THEN 1 ELSE 0 END) AS valid_place_rows
    FROM NL_SE se
    LEFT JOIN NL_O1 o1 ON {join}
    WHERE {jra_where('se')}
    GROUP BY COALESCE(se.IJyoCD, '')
    ORDER BY runner_status
    """
    return [dict(row) for row in con.execute(sql)]


def dimension_missingness(con: sqlite3.Connection, odds_type: str) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    valid = numeric_valid_expr("o1", "TanOdds") if odds_type == "tan" else f"({numeric_valid_expr('o1', 'FukuOddsLow')} AND {numeric_valid_expr('o1', 'FukuOddsHigh')})"
    dimensions = [
        ("Year", "se.Year"),
        ("Month", "CAST(se.MonthDay / 100 AS INTEGER)"),
        ("JyoCD", "se.JyoCD"),
        ("RaceNum", "se.RaceNum"),
        ("SE_DataKubun", "se.DataKubun"),
        ("O1_DataKubun", "o1.DataKubun"),
        ("IJyoCD", "se.IJyoCD"),
        ("MakeDate", "o1.MakeDate"),
    ]
    rows = []
    for name, expr in dimensions:
        sql = f"""
        SELECT '{name}' AS dimension, CAST({expr} AS TEXT) AS value,
               COUNT(*) AS rows,
               SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS o1_missing_rows,
               SUM(CASE WHEN o1.Year IS NOT NULL AND {valid} THEN 1 ELSE 0 END) AS valid_rows
        FROM NL_SE se
        LEFT JOIN NL_O1 o1 ON {join}
        WHERE {jra_where('se')}
        GROUP BY {expr}
        ORDER BY rows DESC
        LIMIT 200
        """
        for row in con.execute(sql):
            d = dict(row)
            d["odds_type"] = odds_type
            d["valid_rate"] = d["valid_rows"] / d["rows"] if d["rows"] else None
            rows.append(d)
    return rows


def value_encoding_summary(con: sqlite3.Connection) -> list[dict[str, Any]]:
    specs = [("NL_SE", "Odds"), ("NL_SE", "Ninki"), ("NL_O1", "TanOdds"), ("NL_O1", "TanNinki"), ("NL_O1", "FukuOddsLow"), ("NL_O1", "FukuOddsHigh"), ("NL_O1", "FukuNinki")]
    rows = []
    for table, col in specs:
        sql = f"""
        SELECT '{table}' AS table_name, '{col}' AS column_name,
          COUNT(*) AS rows,
          SUM(CASE WHEN {q(col)} IS NULL THEN 1 ELSE 0 END) AS null_count,
          SUM(CASE WHEN {q(col)} IS NOT NULL AND TRIM(CAST({q(col)} AS TEXT)) = '' THEN 1 ELSE 0 END) AS empty_count,
          SUM(CASE WHEN {q(col)} IS NOT NULL AND TRIM(CAST({q(col)} AS TEXT)) <> '' AND CAST({q(col)} AS REAL) = 0 THEN 1 ELSE 0 END) AS zero_count,
          SUM(CASE WHEN {q(col)} IS NOT NULL AND CAST({q(col)} AS REAL) < 0 THEN 1 ELSE 0 END) AS negative_count,
          SUM(CASE WHEN CAST({q(col)} AS REAL) IN (9999, 999.9, 99999) THEN 1 ELSE 0 END) AS sentinel_count,
          MIN(CAST({q(col)} AS REAL)) AS min_value,
          MAX(CAST({q(col)} AS REAL)) AS max_value
        FROM {q(table)}
        """
        rows.append(dict(con.execute(sql).fetchone()))
    return rows


def missing_race_samples(con: sqlite3.Connection, limit_races: int = 50) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    sql = f"""
    WITH race_missing AS (
      SELECT se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum,
             SUM(CASE WHEN o1.Year IS NULL OR NOT ({tan_valid}) THEN 1 ELSE 0 END) AS missing_rows,
             COUNT(*) AS runners
      FROM NL_SE se LEFT JOIN NL_O1 o1 ON {join}
      WHERE {jra_where('se')}
      GROUP BY se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum
      HAVING missing_rows > 0
      ORDER BY se.Year, se.JyoCD, se.MonthDay, se.RaceNum
      LIMIT {int(limit_races)}
    )
    SELECT se.Year,se.MonthDay,se.JyoCD,se.Kaiji,se.Nichiji,se.RaceNum,se.Umaban,se.KettoNum,se.Bamei,
           se.IJyoCD,se.Odds AS se_odds,se.Ninki AS se_ninki,
           o1.TanOdds,o1.TanNinki,o1.FukuOddsLow,o1.FukuOddsHigh,o1.FukuNinki,o1.DataKubun AS o1_DataKubun,o1.MakeDate,
           CASE WHEN o1.Year IS NULL THEN 0 ELSE 1 END AS o1_join_found
    FROM race_missing r
    JOIN NL_SE se ON {" AND ".join("se."+q(c)+" = r."+q(c) for c in RACE_KEY)}
    LEFT JOIN NL_O1 o1 ON {join}
    ORDER BY se.Year,se.MonthDay,se.JyoCD,se.RaceNum,se.Umaban
    """
    return [dict(row) for row in con.execute(sql)]
