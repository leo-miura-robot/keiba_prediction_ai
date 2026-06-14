from __future__ import annotations

import sqlite3
from typing import Any

from src.audit.odds_missingness import numeric_valid_expr, null_or_empty_expr
from src.audit.odds_schema import RACE_KEY, jra_where, key_join_condition, q


RACE_ID_EXPR = "printf('%04d%04d%s%02d%02d%02d', se.Year,se.MonthDay,se.JyoCD,se.Kaiji,se.Nichiji,se.RaceNum)"


def race_level_patterns(con: sqlite3.Connection) -> list[dict[str, Any]]:
    join = key_join_condition("se", "o1")
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    fuku_valid = f"({numeric_valid_expr('o1','FukuOddsLow')} AND {numeric_valid_expr('o1','FukuOddsHigh')})"
    tan_null = null_or_empty_expr("o1", "TanOdds")
    fuku_null = f"({null_or_empty_expr('o1','FukuOddsLow')} OR {null_or_empty_expr('o1','FukuOddsHigh')})"
    sql = f"""
    WITH race AS (
      SELECT {RACE_ID_EXPR} AS race_id, se.Year AS year, CAST(se.MonthDay/100 AS INTEGER) AS month, se.JyoCD AS JyoCD,
             COALESCE(o1.DataKubun,'missing_row') AS DataKubun,
             COUNT(*) AS runner_count,
             SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS missing_o1_count,
             SUM(CASE WHEN {tan_valid} THEN 1 ELSE 0 END) AS tan_valid_count,
             SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_null} THEN 1 ELSE 0 END) AS tan_null_count,
             SUM(CASE WHEN {fuku_valid} THEN 1 ELSE 0 END) AS fuku_valid_count,
             SUM(CASE WHEN o1.Year IS NOT NULL AND {fuku_null} THEN 1 ELSE 0 END) AS fuku_null_count
      FROM NL_SE se LEFT JOIN NL_O1 o1 ON {join}
      WHERE {jra_where('se')} AND se.Year BETWEEN 2016 AND 2026
      GROUP BY race_id, se.Year, month, se.JyoCD, COALESCE(o1.DataKubun,'missing_row')
    )
    SELECT *,
      CASE
        WHEN missing_o1_count > 0 THEN 'missing_o1_rows'
        WHEN tan_valid_count = runner_count THEN 'all_valid'
        WHEN tan_valid_count = 0 THEN 'all_null'
        ELSE 'partially_valid'
      END AS tan_race_pattern,
      CASE
        WHEN missing_o1_count > 0 THEN 'missing_o1_rows'
        WHEN fuku_valid_count = runner_count THEN 'all_valid'
        WHEN fuku_valid_count = 0 THEN 'all_null'
        ELSE 'partially_valid'
      END AS fuku_race_pattern,
      CASE WHEN tan_valid_count = fuku_valid_count AND tan_null_count = fuku_null_count THEN 1 ELSE 0 END AS tan_fuku_pattern_same
    FROM race
    """
    return [dict(row) for row in con.execute(sql)]


def race_pattern_summary(patterns: list[dict[str, Any]], by: list[str] | None = None) -> list[dict[str, Any]]:
    by = by or []
    groups: dict[tuple, dict[str, Any]] = {}
    for r in patterns:
        key = tuple(r[k] for k in by) + (r["tan_race_pattern"], r["fuku_race_pattern"])
        g = groups.setdefault(key, {k: r[k] for k in by} | {"tan_race_pattern": r["tan_race_pattern"], "fuku_race_pattern": r["fuku_race_pattern"], "races": 0, "runner_rows": 0, "tan_fuku_same_races": 0})
        g["races"] += 1
        g["runner_rows"] += int(r["runner_count"])
        g["tan_fuku_same_races"] += int(r["tan_fuku_pattern_same"])
    return sorted(groups.values(), key=lambda x: tuple(str(x.get(k, "")) for k in by) + (x["tan_race_pattern"], x["fuku_race_pattern"]))


def partial_samples(patterns: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    return [r for r in patterns if r["tan_race_pattern"] == "partially_valid" or r["fuku_race_pattern"] == "partially_valid"][:limit]


def runner_count_consistency(con: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    join = " AND ".join(f"ra.{q(c)} = se.{q(c)}" for c in RACE_KEY)
    o1_race_join = " AND ".join(f"o1.{q(c)} = se.{q(c)}" for c in RACE_KEY)
    tan_valid = numeric_valid_expr("o1", "TanOdds")
    sql = f"""
    WITH race AS (
      SELECT {RACE_ID_EXPR} AS race_id, se.Year AS year, se.JyoCD AS JyoCD,
             COUNT(DISTINCT se.Umaban) AS se_runner_rows,
             COUNT(DISTINCT o1.Umaban) AS o1_runner_rows,
             MAX(ra.TorokuTosu) AS ra_toroku_tosu,
             MAX(ra.SyussoTosu) AS ra_syusso_tosu,
             MAX(o1.TorokuTosu) AS o1_toroku_tosu,
             MAX(o1.SyussoTosu) AS o1_syusso_tosu,
             SUM(CASE WHEN o1.Year IS NULL THEN 1 ELSE 0 END) AS missing_o1_rows,
             SUM(CASE WHEN o1.Year IS NOT NULL AND {tan_valid} THEN 1 ELSE 0 END) AS valid_tan_rows
      FROM NL_SE se
      LEFT JOIN NL_RA ra ON {join}
      LEFT JOIN NL_O1 o1 ON {o1_race_join} AND o1.Umaban = se.Umaban
      WHERE {jra_where('se')} AND se.Year BETWEEN 2016 AND 2026
      GROUP BY race_id, se.Year, se.JyoCD
    )
    SELECT *,
      CASE WHEN o1_runner_rows <> se_runner_rows OR missing_o1_rows > 0 OR (ra_syusso_tosu IS NOT NULL AND ra_syusso_tosu <> se_runner_rows) THEN 1 ELSE 0 END AS has_count_anomaly
    FROM race
    """
    rows = [dict(row) for row in con.execute(sql)]
    return rows, [r for r in rows if r["has_count_anomaly"]][:500]
