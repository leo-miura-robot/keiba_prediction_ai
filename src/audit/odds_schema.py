from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
ENTRY_KEY = RACE_KEY + ["Umaban"]
JRA_JYOCD = [f"{i:02d}" for i in range(1, 11)]


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def table_info(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [dict(row) for row in con.execute(f"PRAGMA table_info({q(table)})")]


def columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in table_info(con, table)}


def indexes(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = []
    for idx in con.execute(f"PRAGMA index_list({q(table)})"):
        item = dict(idx)
        item["columns"] = [r["name"] for r in con.execute(f"PRAGMA index_info({q(item['name'])})")]
        rows.append(item)
    return rows


def require_tables(con: sqlite3.Connection, tables: list[str]) -> None:
    missing = [t for t in tables if not table_exists(con, t)]
    if missing:
        raise RuntimeError(f"missing required tables: {missing}")


def row_count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) AS n FROM {q(table)}").fetchone()["n"])


def duplicate_key_count(con: sqlite3.Connection, table: str, key_cols: list[str]) -> int:
    expr = ", ".join(q(c) for c in key_cols)
    sql = f"SELECT COUNT(*) AS n FROM (SELECT {expr}, COUNT(*) c FROM {q(table)} GROUP BY {expr} HAVING c > 1)"
    return int(con.execute(sql).fetchone()["n"])


def key_join_condition(left_alias: str = "se", right_alias: str = "o1", keys: list[str] | None = None) -> str:
    return " AND ".join(f"{left_alias}.{q(c)} = {right_alias}.{q(c)}" for c in (keys or ENTRY_KEY))


def jra_where(alias: str = "se") -> str:
    vals = ",".join(f"'{v}'" for v in JRA_JYOCD)
    return f"{alias}.JyoCD IN ({vals})"
