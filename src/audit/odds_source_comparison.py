from __future__ import annotations

import sqlite3
from statistics import mean, median
from typing import Any

from src.audit.odds_missingness import numeric_valid_expr
from src.audit.odds_schema import jra_where, key_join_condition


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
    return xs[idx]


def se_o1_comparison(con: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    join = key_join_condition("se", "o1")
    sql = f"""
    SELECT se.Year,se.MonthDay,se.JyoCD,se.Kaiji,se.Nichiji,se.RaceNum,se.Umaban,
           CAST(se.Odds AS REAL) AS se_odds,
           CAST(o1.TanOdds AS REAL) AS o1_tan_odds,
           CAST(se.Ninki AS REAL) AS se_ninki,
           CAST(o1.TanNinki AS REAL) AS o1_tan_ninki,
           o1.DataKubun AS o1_DataKubun,o1.MakeDate
    FROM NL_SE se JOIN NL_O1 o1 ON {join}
    WHERE {jra_where('se')} AND {numeric_valid_expr('se', 'Odds')} AND {numeric_valid_expr('o1', 'TanOdds')}
    """
    rows = [dict(r) for r in con.execute(sql)]
    candidates = [
        ("se_vs_o1", lambda r: r["se_odds"], lambda r: r["o1_tan_odds"]),
        ("se_div10_vs_o1", lambda r: r["se_odds"] / 10, lambda r: r["o1_tan_odds"]),
        ("se_vs_o1_div10", lambda r: r["se_odds"], lambda r: r["o1_tan_odds"] / 10),
        ("se_div100_vs_o1", lambda r: r["se_odds"] / 100, lambda r: r["o1_tan_odds"]),
        ("se_vs_o1_div100", lambda r: r["se_odds"], lambda r: r["o1_tan_odds"] / 100),
    ]
    summary = []
    for name, lf, rf in candidates:
        diffs = [abs(float(lf(r)) - float(rf(r))) for r in rows]
        exact = sum(1 for d in diffs if d == 0)
        tol = sum(1 for d in diffs if d <= 1e-9)
        summary.append({
            "comparison": name,
            "compared_rows": len(rows),
            "exact_match_count": exact,
            "exact_match_rate": exact / len(rows) if rows else None,
            "tolerance_match_count": tol,
            "tolerance_match_rate": tol / len(rows) if rows else None,
            "mean_abs_diff": mean(diffs) if diffs else None,
            "median_abs_diff": median(diffs) if diffs else None,
            "p95_abs_diff": percentile(diffs, 0.95),
            "p99_abs_diff": percentile(diffs, 0.99),
            "max_abs_diff": max(diffs) if diffs else None,
        })
    by_year = []
    for year in sorted({r["Year"] for r in rows}):
        yr = [r for r in rows if r["Year"] == year]
        diffs = [abs(r["se_odds"] - r["o1_tan_odds"]) for r in yr]
        rank_match = sum(1 for r in yr if r["se_ninki"] == r["o1_tan_ninki"])
        by_year.append({
            "year": year,
            "compared_rows": len(yr),
            "exact_match_rate": sum(1 for d in diffs if d == 0) / len(yr) if yr else None,
            "mean_abs_diff": mean(diffs) if diffs else None,
            "ranking_exact_match_rate": rank_match / len(yr) if yr else None,
        })
    mismatch = []
    for r in rows:
        diff = abs(r["se_odds"] - r["o1_tan_odds"])
        if diff > 1e-9 or r["se_ninki"] != r["o1_tan_ninki"]:
            out = dict(r)
            out["abs_diff"] = diff
            out["ninki_match"] = r["se_ninki"] == r["o1_tan_ninki"]
            mismatch.append(out)
        if len(mismatch) >= 500:
            break
    return summary, by_year, mismatch
