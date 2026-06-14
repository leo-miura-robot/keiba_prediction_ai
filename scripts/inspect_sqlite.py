import argparse
import csv
import sqlite3
from datetime import datetime
from pathlib import Path


DB_ROOT = Path(r"D:\keiba\new_jra_2016-2026")
OUT_DIR = Path("outputs")
DOC_DIR = Path("docs")
TABLE_COLUMNS_CSV = OUT_DIR / "table_columns.csv"
SCHEMA_DOC = DOC_DIR / "db_schema_summary.md"

RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
ENTRY_KEY = RACE_KEY + ["Umaban"]
SAMPLE_TABLE_NAMES = {
    "NL_RA", "RT_RA",
    "NL_SE", "RT_SE",
    "NL_O1", "RT_O1", "TS_O1", "TS_SOKUHO_O1",
    "NL_HR", "RT_HR",
    "NL_HA", "RT_HA",
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def find_sqlite_files(root: Path) -> list[Path]:
    exts = {".db", ".sqlite", ".sqlite3"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def fetch_tables(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [{"name": name, "sql": sql or ""} for name, sql in rows]


def fetch_columns(conn: sqlite3.Connection, table_name: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({qident(table_name)})").fetchall()
    return [
        {
            "cid": cid,
            "name": name,
            "type": data_type,
            "notnull": notnull,
            "default_value": default_value,
            "primary_key": primary_key,
        }
        for cid, name, data_type, notnull, default_value, primary_key in rows
    ]


def sample_rows(conn: sqlite3.Connection, table_name: str, limit: int = 5) -> tuple[list[str], list[tuple]]:
    columns = [row["name"] for row in fetch_columns(conn, table_name)]
    rows = conn.execute(f"SELECT * FROM {qident(table_name)} LIMIT {int(limit)}").fetchall()
    return columns, rows


def write_table_columns_header() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with TABLE_COLUMNS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "table_name",
                "column_name",
                "data_type",
                "notnull",
                "default_value",
                "primary_key",
            ],
        )
        writer.writeheader()


def append_table_columns(table_name: str, columns: list[dict]) -> None:
    with TABLE_COLUMNS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "table_name",
                "column_name",
                "data_type",
                "notnull",
                "default_value",
                "primary_key",
            ],
        )
        for col in columns:
            writer.writerow(
                {
                    "table_name": table_name,
                    "column_name": col["name"],
                    "data_type": col["type"],
                    "notnull": col["notnull"],
                    "default_value": "" if col["default_value"] is None else col["default_value"],
                    "primary_key": col["primary_key"],
                }
            )


def key_summary(columns: list[dict]) -> dict:
    names = {c["name"] for c in columns}
    pk = [c["name"] for c in sorted((c for c in columns if c["primary_key"]), key=lambda x: x["primary_key"])]
    return {
        "pk": pk,
        "has_race_key": all(k in names for k in RACE_KEY),
        "has_entry_key": all(k in names for k in ENTRY_KEY),
        "has_horse_id": any(k in names for k in ["KettoNum", "UmaID", "HansyokuNum"]),
        "has_date": any(k in names for k in ["Year", "MonthDay", "KaisaiDate", "MakeDate", "BirthDate"]),
    }


def format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(out)


def compact_value(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > 60:
        return text[:57] + "..."
    return text


def build_doc(db_summaries: list[dict], table_summaries: list[dict], samples: dict) -> str:
    lines = [
        "# DB Schema Summary",
        "",
        "This report was generated in lightweight schema-only mode. It does not run COUNT(*), COUNT(DISTINCT), NULL counts, or full-table scans.",
        "",
        "## SQLite files",
        "",
        format_table(
            [["path", "size_gb", "last_write_time", "table_count"]]
            + [[d["path"], d["size_gb"], d["last_write_time"], d["table_count"]] for d in db_summaries]
        ),
        "",
        "## Tables",
        "",
        format_table(
            [["table", "columns", "primary_key", "race_key", "entry_key", "horse_id", "date"]]
            + [
                [
                    t["table_name"],
                    t["column_count"],
                    ", ".join(t["primary_key"]),
                    t["has_race_key"],
                    t["has_entry_key"],
                    t["has_horse_id"],
                    t["has_date"],
                ]
                for t in table_summaries
            ]
        ),
        "",
        "## Main data structure",
        "",
        "- Race key is generally `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum`.",
        "- Entry key is generally race key plus `Umaban`; horse identifier is mainly `KettoNum`.",
        "- Race-level tables: `NL_RA` / `RT_RA` contain race metadata such as course, distance, class and race name.",
        "- Entry-level tables: `NL_SE` / `RT_SE` contain runners, horse, jockey, trainer, weight, body weight and result columns.",
        "- Odds tables: `NL_O1` / `RT_O1` / `TS_O1` / `TS_SOKUHO_O1` contain win/place odds fields such as `TanOdds`, `FukuOddsLow`, `FukuOddsHigh`.",
        "- Payout tables: `NL_HR` / `RT_HR` and `NL_HA` / `RT_HA` appear to contain payout/refund data. Detailed payout decoding should be done in a separate lightweight step.",
        "",
        "## Candidate join for one row per runner",
        "",
        "Use `RT_SE` or `NL_SE` as the base runner table. Join `RT_RA`/`NL_RA` by race key, and join `RT_O1`/`NL_O1` or time-series `TS_O1` by race key plus `Umaban`. Payout tables join by race key and then require bet-type/combination decoding for win/place returns.",
        "",
        "## Sample rows from main tables",
        "",
    ]

    for table_name, sample in samples.items():
        lines.extend([f"### {table_name}", ""])
        cols = sample["columns"]
        rows = sample["rows"]
        if not rows:
            lines.extend(["No rows returned by `LIMIT 5`.", ""])
            continue
        preview_cols = cols[:12]
        preview = [["column"] + [f"row{i + 1}" for i in range(len(rows))]]
        for col_idx, col_name in enumerate(preview_cols):
            preview.append([col_name] + [compact_value(row[col_idx]) for row in rows])
        lines.extend([
            "First 12 columns shown for readability; query used was `SELECT * FROM table LIMIT 5`.",
            "",
            format_table(preview),
            "",
        ])

    lines.extend([
        "## Notes",
        "",
        "- `outputs/table_columns.csv` contains all table/column definitions obtained from `PRAGMA table_info`.",
        "- No row counts or distinct counts are included in this lightweight pass.",
        "- Feature inventory and leakage classification should be generated after confirming table meanings and payout encoding.",
        "",
    ])
    return "\n".join(lines)


def inspect_schema(db_root: Path) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(exist_ok=True)
    write_table_columns_header()

    db_files = find_sqlite_files(db_root)
    db_summaries = []
    table_summaries = []
    samples = {}

    for db_path in db_files:
        stat = db_path.stat()
        print(f"[db] {db_path}", flush=True)
        with connect_readonly(db_path) as conn:
            tables = fetch_tables(conn)
            db_summaries.append(
                {
                    "path": str(db_path),
                    "size_gb": round(stat.st_size / 1024**3, 2),
                    "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "table_count": len(tables),
                }
            )

            for index, table in enumerate(tables, start=1):
                table_name = table["name"]
                print(f"[table {index}/{len(tables)}] {table_name}", flush=True)
                columns = fetch_columns(conn, table_name)
                append_table_columns(table_name, columns)

                summary = key_summary(columns)
                table_summaries.append(
                    {
                        "table_name": table_name,
                        "column_count": len(columns),
                        "primary_key": summary["pk"],
                        "has_race_key": summary["has_race_key"],
                        "has_entry_key": summary["has_entry_key"],
                        "has_horse_id": summary["has_horse_id"],
                        "has_date": summary["has_date"],
                    }
                )

                if table_name in SAMPLE_TABLE_NAMES:
                    print(f"[sample] {table_name} LIMIT 5", flush=True)
                    sample_columns, sample = sample_rows(conn, table_name, 5)
                    samples[table_name] = {"columns": sample_columns, "rows": sample}

    SCHEMA_DOC.write_text(build_doc(db_summaries, table_summaries, samples), encoding="utf-8")
    print(f"[done] wrote {TABLE_COLUMNS_CSV}", flush=True)
    print(f"[done] wrote {SCHEMA_DOC}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight SQLite schema inspection for JRA DB.")
    parser.add_argument("--db-root", default=str(DB_ROOT), help="Directory containing SQLite files.")
    parser.add_argument(
        "--mode",
        default="schema_only",
        choices=["schema_only"],
        help="Only schema_only is supported in the safe lightweight version.",
    )
    args = parser.parse_args()
    inspect_schema(Path(args.db_root))


if __name__ == "__main__":
    main()
