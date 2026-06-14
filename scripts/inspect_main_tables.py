import csv
import sqlite3
from pathlib import Path


DB_PATH = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
OUT_DIR = Path("outputs")
DOC_DIR = Path("docs")
SAMPLES_CSV = OUT_DIR / "main_table_samples.csv"
JOIN_CSV = OUT_DIR / "join_check_sample.csv"
DETAIL_DOC = DOC_DIR / "main_table_detail.md"
DESIGN_DOC = DOC_DIR / "dataset_design.md"

RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]

TABLE_COLUMNS = {
    "NL_SE": [
        "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "Wakuban", "Umaban", "KettoNum", "Bamei",
        "SexCD", "Barei", "ChokyosiCode", "ChokyosiRyakusyo",
        "KisyuCode", "KisyuRyakusyo",
        "Futan", "BaTaijyu", "ZogenFugo", "ZogenSa",
        "IJyoCD", "NyusenJyuni", "KakuteiJyuni",
        "Odds", "Ninki",
        "Time", "HaronTimeL3",
    ],
    "NL_RA": [
        "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "YoubiCD", "GradeCD", "SyubetuCD",
        "JyokenCD1", "JyokenCD2", "JyokenCD3", "JyokenCD4", "JyokenCD5",
        "JyokenName", "Kyori", "TrackCD", "CourseKubunCD",
        "HassoTime", "TorokuTosu", "SyussoTosu",
        "TenkoCD", "SibaBabaCD", "DirtBabaCD",
    ],
    "NL_O1": [
        "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "HassoTime", "Umaban",
        "TanOdds", "TanNinki",
        "FukuUmaban", "FukuOddsLow", "FukuOddsHigh", "FukuNinki",
        "TanVote", "FukuVote",
    ],
    "NL_HR": [
        "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "TanUmaban", "TanPay", "TanNinki",
        "FukuUmaban", "FukuPay", "FukuNinki",
    ],
}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect_readonly() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def select_columns(table: str, columns: list[str], limit: int = 20) -> tuple[list[str], list[sqlite3.Row]]:
    col_sql = ", ".join(qident(c) for c in columns)
    sql = f"SELECT {col_sql} FROM {qident(table)} LIMIT {int(limit)}"
    return columns, CONN.execute(sql).fetchall()


def write_samples(samples: dict[str, tuple[list[str], list[sqlite3.Row]]]) -> None:
    with SAMPLES_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["table_name", "row_no", "column_name", "value"])
        writer.writeheader()
        for table, (columns, rows) in samples.items():
            for row_no, row in enumerate(rows, start=1):
                for column, value in zip(columns, row):
                    writer.writerow({
                        "table_name": table,
                        "row_no": row_no,
                        "column_name": column,
                        "value": "" if value is None else value,
                    })


def make_race_id(prefix: str) -> str:
    formats = {
        "Year": "%04d",
        "MonthDay": "%04d",
        "Kaiji": "%02d",
        "Nichiji": "%02d",
        "RaceNum": "%02d",
    }
    parts = []
    for column in RACE_KEY:
        if column == "JyoCD":
            parts.append(f"{prefix}.{column}")
        else:
            parts.append(f"printf('{formats[column]}', {prefix}.{column})")
    return " || '-' || ".join(parts)


def join_check_sample() -> list[dict]:
    race_join = " AND ".join(f"se.{c} = ra.{c}" for c in RACE_KEY)
    o1_join = " AND ".join(f"se.{c} = o1.{c}" for c in RACE_KEY) + " AND se.Umaban = o1.Umaban"
    hr_join = " AND ".join(f"se.{c} = hr.{c}" for c in RACE_KEY)
    sql = f"""
        SELECT
            {make_race_id("se")} AS race_id,
            ({make_race_id("se")} || '-' || printf('%02d', se.Umaban)) AS entry_id,
            se.Year, se.MonthDay, se.JyoCD, se.Kaiji, se.Nichiji, se.RaceNum,
            se.Umaban, se.KettoNum, se.Bamei,
            se.KakuteiJyuni, se.IJyoCD,
            se.Odds AS se_odds, se.Ninki AS se_ninki,
            o1.TanOdds AS o1_tan_odds, o1.TanNinki AS o1_tan_ninki,
            o1.FukuOddsLow, o1.FukuOddsHigh, o1.FukuNinki,
            hr.TanUmaban, hr.TanPay, hr.TanNinki AS hr_tan_ninki,
            hr.FukuUmaban, hr.FukuPay, hr.FukuNinki AS hr_fuku_ninki,
            ra.Kyori, ra.TrackCD, ra.TenkoCD, ra.SibaBabaCD, ra.DirtBabaCD,
            CASE WHEN ra.Year IS NULL THEN 0 ELSE 1 END AS joined_ra,
            CASE WHEN o1.Year IS NULL THEN 0 ELSE 1 END AS joined_o1,
            CASE WHEN hr.Year IS NULL THEN 0 ELSE 1 END AS joined_hr,
            CASE WHEN se.KakuteiJyuni = 1 AND CAST(hr.TanUmaban AS INTEGER) = se.Umaban THEN 1 ELSE 0 END AS win_matches_tan
        FROM NL_SE se
        LEFT JOIN NL_RA ra ON {race_join}
        LEFT JOIN NL_O1 o1 ON {o1_join}
        LEFT JOIN NL_HR hr ON {hr_join}
        LIMIT 20
    """
    rows = CONN.execute(sql).fetchall()
    names = [d[0] for d in CONN.execute(sql).description]
    output = []
    for row in rows:
        item = dict(zip(names, row))
        fuku_umaban = str(item.get("FukuUmaban") or "")
        item["place_in_fuku_umaban"] = (
            1
            if item.get("KakuteiJyuni") is not None
            and item["KakuteiJyuni"] <= 3
            and f"{int(item['Umaban']):02d}" in fuku_umaban
            else 0
        )
        if item.get("se_odds") in (None, 0) or item.get("o1_tan_odds") in (None, 0):
            item["odds_relation"] = "missing_or_zero"
        elif abs(float(item["se_odds"]) - float(item["o1_tan_odds"])) < 0.0001:
            item["odds_relation"] = "same"
        elif abs(float(item["se_odds"]) / 10 - float(item["o1_tan_odds"])) < 0.0001:
            item["odds_relation"] = "se_odds_10x"
        elif abs(float(item["se_odds"]) - float(item["o1_tan_odds"]) / 10) < 0.0001:
            item["odds_relation"] = "o1_tan_odds_10x"
        else:
            item["odds_relation"] = "different"
        output.append(item)
    return output


def write_join(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with JOIN_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def limited_ijyo_values(limit: int = 10000) -> list[str]:
    sql = f"SELECT DISTINCT IJyoCD FROM (SELECT IJyoCD FROM NL_SE LIMIT {int(limit)}) ORDER BY IJyoCD"
    return ["" if row[0] is None else str(row[0]) for row in CONN.execute(sql).fetchall()]


def format_table(rows: list[list]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(lines)


def compact(value) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text[:57] + "..." if len(text) > 60 else text


def build_detail_doc(samples: dict, join_rows: list[dict], ijyo_values: list[str]) -> str:
    lines = [
        "# Main Table Detail",
        "",
        "This detail check only used targeted `SELECT ... LIMIT 20`, indexed joins from limited base rows, and a limited `IJyoCD` check over the first 10000 `NL_SE` rows. It does not run full-table counts or all-column aggregation.",
        "",
        "## Target tables",
        "",
        "- `NL_SE`: runner-level base table.",
        "- `NL_RA`: race-level metadata.",
        "- `NL_O1`: win/place odds by race key plus `Umaban`.",
        "- `NL_HR`: race-level payouts for win/place and other bet types.",
        "- `NL_UM`: horse master, joinable by `KettoNum`.",
        "- `NL_KS`: jockey master, joinable by `KisyuCode`.",
        "- `NL_CH`: trainer master, joinable by `ChokyosiCode`.",
        "",
        "## Limited samples",
        "",
    ]
    for table, (columns, rows) in samples.items():
        lines.extend([f"### {table}", ""])
        preview_cols = columns[:12]
        preview = [["column"] + [f"row{i + 1}" for i in range(min(len(rows), 5))]]
        for idx, col in enumerate(preview_cols):
            preview.append([col] + [compact(row[idx]) for row in rows[:5]])
        lines.extend([format_table(preview), ""])

    lines.extend([
        "## Join check sample",
        "",
        format_table(
            [["race_id", "Umaban", "Bamei", "KakuteiJyuni", "joined_ra", "joined_o1", "joined_hr", "se_odds", "o1_tan_odds", "odds_relation", "TanUmaban", "TanPay", "FukuUmaban", "FukuPay", "place_in_fuku_umaban"]]
            + [
                [
                    r["race_id"], r["Umaban"], r["Bamei"], r["KakuteiJyuni"],
                    r["joined_ra"], r["joined_o1"], r["joined_hr"],
                    r["se_odds"], r["o1_tan_odds"], r["odds_relation"],
                    r["TanUmaban"], r["TanPay"], r["FukuUmaban"], r["FukuPay"],
                    r["place_in_fuku_umaban"],
                ]
                for r in join_rows[:20]
            ]
        ),
        "",
        "## Limited IJyoCD values",
        "",
        "`IJyoCD` values observed in the first 10000 `NL_SE` rows: " + ", ".join(f"`{v}`" for v in ijyo_values),
        "",
        "## Odds relation",
        "",
        "In the limited join sample, `NL_SE.Odds` and `NL_O1.TanOdds` are compared in `outputs/join_check_sample.csv` with `odds_relation`. Use this as a structural check only; a later validation should sample multiple race dates intentionally without full scans.",
        "",
        "## Payout relation",
        "",
        "In the limited sample, `NL_HR.TanUmaban` matches the `KakuteiJyuni = 1` runner, so win payout linkage is straightforward by race key plus winning `Umaban`.",
        "",
        "`NL_HR.FukuUmaban/FukuPay` did not expose all top-3 place payouts in the first sampled race. For example, the first race has top-3 `Umaban` 12, 11, and 13 in `NL_SE`, but `NL_HR.FukuUmaban` is only `12`. This is enough to confirm a winner-place payout, but not enough to safely evaluate every `target_place` row. The payout parser/schema or another payout source must be confirmed before a full place-return backtest.",
        "",
    ])
    return "\n".join(lines)


def build_design_doc(ijyo_values: list[str]) -> str:
    return "\n".join([
        "# Dataset Design",
        "",
        "## One Row Per Runner",
        "",
        "Use `NL_SE` as the base table. It is already one row per runner, keyed by `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban`.",
        "",
        "Join race metadata from `NL_RA` on `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum`.",
        "",
        "Join win/place odds from `NL_O1` on `Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, Umaban`.",
        "",
        "Join payouts from `NL_HR` on race key only. `TanUmaban/TanPay` identify the winning horse and win return. `FukuUmaban/FukuPay` are race-level place payout fields, but the limited sample did not expose all top-3 place payouts, so they require parser/schema confirmation before being used for full place-return evaluation.",
        "",
        "Optional master joins: `NL_UM` by `KettoNum`, `NL_KS` by `KisyuCode`, and `NL_CH` by `ChokyosiCode`. These should be treated carefully because master tables may contain cumulative or latest records rather than strictly pre-race state.",
        "",
        "## IDs",
        "",
        "- `race_id`: concatenate zero-padded `Year`, `MonthDay`, `JyoCD`, `Kaiji`, `Nichiji`, `RaceNum`, for example `2016-0105-06-01-01-01`.",
        "- `entry_id`: `race_id` plus zero-padded `Umaban`.",
        "- `horse_id`: `NL_SE.KettoNum`.",
        "",
        "## Odds And Payout Columns",
        "",
        "- Win odds: prefer `NL_O1.TanOdds` for the canonical odds table. `NL_SE.Odds` appears to be final win odds in runner results and should be treated as post-result/evaluation unless the timing is confirmed.",
        "- Place odds: use `NL_O1.FukuOddsLow` and `NL_O1.FukuOddsHigh`.",
        "- Win payout: use `NL_HR.TanPay`, matched by race key and `NL_HR.TanUmaban`.",
        "- Place payout: `NL_HR.FukuPay` is only safe for rows where `NL_HR.FukuUmaban` matches the target `Umaban`. In the limited sample it did not cover every top-3 horse, so full place payout extraction is unresolved.",
        "",
        "## Targets",
        "",
        "- `target_win`: `1` when `KakuteiJyuni = 1` and normal starter, else `0`.",
        "- `target_ren`: `1` when `KakuteiJyuni <= 2` and normal starter, else `0`.",
        "- `target_place`: `1` when `KakuteiJyuni <= 3` and normal starter, else `0`. For races with fewer than standard starters, place payout rules may differ and should be handled separately in evaluation.",
        "- Exclude or separately flag abnormal rows using `IJyoCD`. Limited observed values: " + ", ".join(f"`{v}`" for v in ijyo_values) + ".",
        "",
        "## Predictable Features",
        "",
        "`NL_RA`: `Year`, `MonthDay`, `JyoCD`, `YoubiCD`, `GradeCD`, `SyubetuCD`, `JyokenCD1-5`, `JyokenName`, `Kyori`, `TrackCD`, `CourseKubunCD`, `HassoTime`, `TorokuTosu`, `SyussoTosu`, and pre-race weather/going if confirmed available before prediction.",
        "",
        "`NL_SE`: `Wakuban`, `Umaban`, `KettoNum`, `SexCD`, `Barei`, `ChokyosiCode`, `KisyuCode`, `Futan`, `BaTaijyu`, `ZogenFugo`, `ZogenSa`, `Blinker`, `MinaraiCD` if known before the bet.",
        "",
        "`NL_O1`: `TanOdds`, `TanNinki`, `FukuOddsLow`, `FukuOddsHigh`, `FukuNinki`, and vote columns only when using a clearly defined pre-deadline snapshot. For final odds modeling, keep them as market features but document the prediction timing.",
        "",
        "## Leakage Columns",
        "",
        "Do not use `KakuteiJyuni`, `NyusenJyuni`, `Time`, `ChakusaCD`, `Jyuni1c-4c`, `HaronTimeL3/L4`, `TimeDiff`, `DMTime`, `DMJyuni`, `KyakusituKubun`, prize columns from the completed race, or any `NL_HR` payout fields as model features.",
        "",
        "Avoid using `NL_RA` race-result fields such as `LapTime`, `Haron3F`, `Haron4F`, `Haron3L`, `Haron4L`, `Corner`, and `TsukaJyuni` as features.",
        "",
        "## Ambiguous Features",
        "",
        "`NL_SE.Odds/Ninki` and `NL_O1.TanOdds/TanNinki` may be final odds. They are acceptable only if the intended prediction point is immediately before betting close and the same timing can be reproduced. For stricter no-leakage prediction, use `TS_O1` or `TS_SOKUHO_O1` snapshots filtered by time.",
        "",
        "`BaTaijyu` and `ZogenSa` are usually announced before the race, but the operational timing should be confirmed. `TenkoCD`, `SibaBabaCD`, and `DirtBabaCD` can change during the day, so use only the latest value available before prediction.",
        "",
        "`NL_UM`, `NL_KS`, and `NL_CH` contain master and cumulative fields. Use stable identity/profile fields directly; cumulative performance fields need as-of-date reconstruction to avoid using future records.",
        "",
        "## Recommended Next Design",
        "",
        "Build a thin dataset extraction script with explicit selected columns, an as-of date filter, and no broad aggregation. Start from `NL_SE`, join `NL_RA` and `NL_O1`, create targets from `KakuteiJyuni`, and create win evaluation returns from `NL_HR`. Before modeling place ROI, resolve the complete source for all place payouts. After that, add historical rolling features using only races before each target race.",
        "",
    ])


def main() -> None:
    global CONN
    OUT_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(exist_ok=True)

    print(f"[db] {DB_PATH}", flush=True)
    CONN = connect_readonly()
    try:
        samples = {}
        for table, columns in TABLE_COLUMNS.items():
            print(f"[sample] {table} selected columns LIMIT 20", flush=True)
            samples[table] = select_columns(table, columns, 20)

        print("[join] NL_SE + NL_RA + NL_O1 + NL_HR LIMIT 20", flush=True)
        join_rows = join_check_sample()

        print("[limited] IJyoCD from first 10000 NL_SE rows", flush=True)
        ijyo_values = limited_ijyo_values(10000)

        write_samples(samples)
        write_join(join_rows)
        DETAIL_DOC.write_text(build_detail_doc(samples, join_rows, ijyo_values), encoding="utf-8-sig")
        DESIGN_DOC.write_text(build_design_doc(ijyo_values), encoding="utf-8-sig")
    finally:
        CONN.close()

    print(f"[done] wrote {SAMPLES_CSV}", flush=True)
    print(f"[done] wrote {JOIN_CSV}", flush=True)
    print(f"[done] wrote {DETAIL_DOC}", flush=True)
    print(f"[done] wrote {DESIGN_DOC}", flush=True)


if __name__ == "__main__":
    main()
