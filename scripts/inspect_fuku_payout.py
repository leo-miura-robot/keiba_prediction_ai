import csv
import sqlite3
from pathlib import Path


DB_PATH = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
OUT_DIR = Path("outputs")
DOC_DIR = Path("docs")
PAYOUT_COLUMNS_CSV = OUT_DIR / "payout_columns.csv"
FUKU_SAMPLE_CSV = OUT_DIR / "fuku_payout_sample.csv"
PAYOUT_COLUMNS_DOC = DOC_DIR / "payout_columns.md"
FUKU_DESIGN_DOC = DOC_DIR / "fuku_payout_design.md"

RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
PAYOUT_HINTS = [
    "tan", "fuku", "pay", "ninki", "umaban", "kumi", "haraimodoshi",
    "haraimodoshikingaku", "bet", "refund",
]


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect_readonly() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def race_where(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return " AND ".join(f"{prefix}{qident(k)} = ?" for k in RACE_KEY)


def race_params(race: dict) -> tuple:
    return tuple(race[k] for k in RACE_KEY)


def race_id(race: dict) -> str:
    return (
        f"{int(race['Year']):04d}"
        f"{int(race['MonthDay']):04d}"
        f"{race['JyoCD']}"
        f"{int(race['Kaiji']):02d}"
        f"{int(race['Nichiji']):02d}"
        f"{int(race['RaceNum']):02d}"
    )


def fetch_columns(conn: sqlite3.Connection, table: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({qident(table)})").fetchall()
    return [
        {
            "table_name": table,
            "cid": cid,
            "column_name": name,
            "data_type": data_type,
            "notnull": notnull,
            "default_value": "" if default is None else default,
            "primary_key": pk,
        }
        for cid, name, data_type, notnull, default, pk in rows
    ]


def infer_role(table: str, column: str) -> str:
    lower = column.lower()
    if "fuku" in lower and "pay" in lower:
        return "place payout amount candidate"
    if "fuku" in lower and "umaban" in lower:
        return "place payout horse-number candidate"
    if "tan" in lower and "pay" in lower:
        return "win payout amount candidate"
    if "tan" in lower and "umaban" in lower:
        return "win payout horse-number candidate"
    if lower == "bettype":
        return "bet type discriminator"
    if lower == "kumi":
        return "combination or horse-number key"
    if lower in {"hyo", "vote"}:
        return "vote count, not payout"
    if "ninki" in lower:
        return "popularity/rank"
    if "pay" in lower or "haraimodoshi" in lower or "refund" in lower:
        return "payout/refund candidate"
    if "kumi" in lower or "umaban" in lower:
        return "combination/horse-number candidate"
    return "context"


def payout_related_columns(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    for table in ["NL_HR", "NL_H1"]:
        for col in fetch_columns(conn, table):
            lower = col["column_name"].lower()
            if any(h in lower for h in PAYOUT_HINTS):
                col["inferred_role"] = infer_role(table, col["column_name"])
                rows.append(col)
    return rows


def select_races(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    sql = """
        SELECT Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum
        FROM NL_HR
        LIMIT ?
    """
    races = []
    for row in conn.execute(sql, (limit,)).fetchall():
        races.append(dict(zip(RACE_KEY, row)))
    return races


def se_top3(conn: sqlite3.Connection, race: dict) -> list[dict]:
    sql = f"""
        SELECT Umaban, Bamei, KakuteiJyuni, IJyoCD
        FROM NL_SE
        WHERE {race_where()}
          AND KakuteiJyuni <= 3
        ORDER BY KakuteiJyuni, Umaban
        LIMIT 10
    """
    cols = ["Umaban", "Bamei", "KakuteiJyuni", "IJyoCD"]
    return [dict(zip(cols, row)) for row in conn.execute(sql, race_params(race)).fetchall()]


def hr_payout(conn: sqlite3.Connection, race: dict) -> dict | None:
    cols = [
        "TanUmaban", "TanPay", "TanNinki",
        "FukuUmaban", "FukuPay", "FukuNinki",
        "WakuKumi", "WakuPay", "UmarenKumi", "UmarenPay",
        "WideKumi", "WidePay", "UmatanKumi", "UmatanPay",
        "SanrenfukuKumi", "SanrenfukuPay", "SanrentanKumi", "SanrentanPay",
    ]
    sql = f"SELECT {', '.join(qident(c) for c in cols)} FROM NL_HR WHERE {race_where()} LIMIT 1"
    row = conn.execute(sql, race_params(race)).fetchone()
    return dict(zip(cols, row)) if row else None


def h1_rows(conn: sqlite3.Connection, race: dict, limit: int = 80) -> list[dict]:
    cols = ["BetType", "Kumi", "Hyo", "Ninki"]
    sql = f"""
        SELECT BetType, Kumi, Hyo, Ninki
        FROM NL_H1
        WHERE {race_where()}
          AND BetType IN ('Tansyo', 'Fukusyo')
        ORDER BY BetType, Kumi
        LIMIT {int(limit)}
    """
    return [dict(zip(cols, row)) for row in conn.execute(sql, race_params(race)).fetchall()]


def h1_limit50(conn: sqlite3.Connection) -> list[dict]:
    cols = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "BetType", "Kumi", "Hyo", "Ninki"]
    sql = f"SELECT {', '.join(qident(c) for c in cols)} FROM NL_H1 LIMIT 50"
    return [dict(zip(cols, row)) for row in conn.execute(sql).fetchall()]


def h1_bettype_sample(conn: sqlite3.Connection, race: dict) -> list[dict]:
    cols = ["BetType", "Kumi", "Hyo", "Ninki"]
    sql = f"""
        SELECT BetType, Kumi, Hyo, Ninki
        FROM NL_H1
        WHERE {race_where()}
        LIMIT 500
    """
    return [dict(zip(cols, row)) for row in conn.execute(sql, race_params(race)).fetchall()]


def fuku_rows_from_hr(race: dict, top3: list[dict], hr: dict | None) -> list[dict]:
    output = []
    if not hr:
        return output
    fuku_umaban = str(hr.get("FukuUmaban") or "")
    for horse in top3:
        umaban2 = f"{int(horse['Umaban']):02d}"
        matched = fuku_umaban == umaban2
        output.append({
            "race_id": race_id(race),
            "Year": race["Year"],
            "MonthDay": race["MonthDay"],
            "JyoCD": race["JyoCD"],
            "Kaiji": race["Kaiji"],
            "Nichiji": race["Nichiji"],
            "RaceNum": race["RaceNum"],
            "Umaban": horse["Umaban"],
            "KakuteiJyuni": horse["KakuteiJyuni"],
            "source_table": "NL_HR",
            "source_column_umaban": "FukuUmaban",
            "source_column_pay": "FukuPay",
            "source_umaban_value": fuku_umaban,
            "source_pay_value": hr.get("FukuPay"),
            "fuku_pay": hr.get("FukuPay") if matched else "",
            "matched": 1 if matched else 0,
            "notes": "matched exact FukuUmaban" if matched else "not represented by sampled NL_HR FukuUmaban",
        })
    return output


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[list]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(row))) + " |")
        if i == 0:
            out.append("| " + " | ".join("-" * widths[j] for j in range(len(row))) + " |")
    return "\n".join(out)


def compact(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    return text[:77] + "..." if len(text) > 80 else text


def build_payout_columns_doc(payout_cols: list[dict], h1_cols: list[dict], h1_first50: list[dict], h1_race_sample: list[dict]) -> str:
    h1_types = []
    seen = set()
    for row in h1_race_sample:
        if row["BetType"] not in seen:
            seen.add(row["BetType"])
            h1_types.append(row["BetType"])

    return "\n".join([
        "# Payout Columns",
        "",
        "This check uses `PRAGMA table_info` and limited samples only. It does not run full-table counts or distinct scans.",
        "",
        "## Payout-related columns",
        "",
        format_table(
            [["table", "column", "type", "pk", "inferred_role"]]
            + [[r["table_name"], r["column_name"], r["data_type"], r["primary_key"], r["inferred_role"]] for r in payout_cols]
        ),
        "",
        "## NL_H1 columns",
        "",
        format_table(
            [["column", "type", "pk"]]
            + [[r["column_name"], r["data_type"], r["primary_key"]] for r in h1_cols]
        ),
        "",
        "## NL_H1 limited sample",
        "",
        "First 50 `NL_H1` rows were inspected. The payout-relevant fields visible in this table are `BetType`, `Kumi`, `Hyo`, and `Ninki`.",
        "",
        format_table(
            [["race", "BetType", "Kumi", "Hyo", "Ninki"]]
            + [
                [
                    race_id(row),
                    row["BetType"],
                    row["Kumi"],
                    row["Hyo"],
                    row["Ninki"],
                ]
                for row in h1_first50[:20]
            ]
        ),
        "",
        "## NL_H1 BetType inference",
        "",
        "For the first sampled race, observed `BetType` values in a `LIMIT 500` sample were: "
        + ", ".join(f"`{value}`" for value in h1_types)
        + ".",
        "",
        "`Fukusyo` rows use `Kumi` as the horse number and `Hyo` as vote count. No payout amount column was found in `NL_H1`; `Hyo` is not a refund amount.",
        "",
        "## Other schema candidates",
        "",
        "`NL_HA` has `PayKumi1-3`, `PayAmount1-3`, `TotalPay`, and `PayoutCount`, which look like normalized payout slots. However, `SELECT * FROM NL_HA LIMIT 5` returned no rows in this DB, so it cannot currently be used as the place payout source.",
        "",
        "`NL_WF` has `Kumi` and `PayJyushosiki`, but sampled rows look like WIN5-style aggregate data rather than horse-level place payout rows.",
        "",
    ])


def build_fuku_design_doc(race_reports: list[dict], fuku_rows: list[dict]) -> str:
    lines = [
        "# Fuku Payout Design",
        "",
        "## Conclusion",
        "",
        "複勝払戻の全対象馬を `NL_HR` だけから確定することは、今回の少数サンプルではできませんでした。`NL_HR.FukuUmaban/FukuPay` は存在しますが、最初のサンプルレースでは3着以内の全馬を表していません。",
        "",
        "`NL_H1` には `BetType = Fukusyo` と `Kumi` があり、馬番別の行はあります。ただし `Hyo` と `Ninki` は票数・人気であり、払戻額ではありません。したがって `NL_H1` は複勝の馬番候補や投票情報の確認には使えますが、複勝払戻額の取得元としては不足しています。",
        "",
        "現時点で単勝払戻は `NL_HR.TanUmaban/TanPay` で扱えますが、複勝回収率を正しく計算するには、全複勝払戻を持つ別テーブル、未展開カラム、または元データパーサの確認が必要です。",
        "",
        "列名上の別候補として `NL_HA.PayKumi1-3/PayAmount1-3` がありますが、このDBでは `LIMIT 5` で行が返らず、利用できる実データは確認できませんでした。`NL_WF.PayJyushosiki` はWIN5系の集計に見えるため、出走馬単位の複勝払戻ソースではないと判断しています。",
        "",
        "## Race Samples",
        "",
    ]
    for report in race_reports:
        lines.extend([
            f"### {report['race_id']}",
            "",
            f"- 1着馬番: {report['first_umaban']}",
            f"- 2着馬番: {report['second_umaban']}",
            f"- 3着馬番: {report['third_umaban']}",
            f"- NL_HR上の複勝馬番候補: `{report['hr_fuku_umaban']}`",
            f"- NL_HR上の複勝払戻候補: `{report['hr_fuku_pay']}`",
            f"- NL_H1上の複勝候補: {report['h1_fukusyo_candidates']}",
            f"- 結論: {report['conclusion']}",
            "",
        ])

    lines.extend([
        "## Vertical Format Feasibility",
        "",
        "Target format is:",
        "",
        "```text",
        "race_id,Umaban,fuku_pay",
        "2016010506010112,12,120",
        "2016010506010111,11,",
        "2016010506010113,13,",
        "```",
        "",
        "The format itself can be produced, and `race_id + Umaban` can be joined back to `NL_SE`. However, using only `NL_HR.FukuUmaban/FukuPay`, only rows whose `Umaban` equals `FukuUmaban` can receive a payout. Other placed horses remain unresolved in the current sample.",
        "",
        "## Candidate Vertical Rows From Sample",
        "",
        format_table(
            [["race_id", "Umaban", "KakuteiJyuni", "fuku_pay", "matched", "notes"]]
            + [[r["race_id"], r["Umaban"], r["KakuteiJyuni"], r["fuku_pay"], r["matched"], r["notes"]] for r in fuku_rows]
        ),
        "",
        "## Recommended Source",
        "",
        "- Use `NL_HR.TanUmaban/TanPay` for win payout.",
        "- Do not treat `NL_H1.Hyo` as place payout; it is vote count.",
        "- Treat `NL_HR.FukuUmaban/FukuPay` as an incomplete place payout source until parser/schema confirmation.",
        "- `NL_HA.PayKumi1-3/PayAmount1-3` would be a natural source if populated, but it appears empty in this DB.",
        "- Check whether the data loader lost repeated JRA-VAN HR payout slots for place payouts, or whether another table/file contains the full repeated place payouts.",
        "",
        "## Edge Cases",
        "",
        "- Small fields: races with low starter counts can have fewer place payout targets; target/evaluation rules must follow JRA place-payout rules for the race size.",
        "- Scratches/exclusions: use `IJyoCD`, refund flags, and `HenkanUma*` fields so canceled runners do not receive normal losing outcomes.",
        "- Dead heats: multiple horses can share the same placing; payout rows may exceed the usual 2 or 3 targets and must be decoded from official payout slots.",
        "- Coupled or special refund cases: `TokubaraiFlag*`, `FuseirituFlag*`, and refund fields should remain evaluation-only metadata.",
        "",
        "## Implementation Notes",
        "",
        "Keep payout extraction as a separate decoder that outputs `race_id, Umaban, fuku_pay, source_table, source_columns`. Do not merge it into feature extraction. Add assertions on sampled races: every normal top-3 horse should have a place payout unless race-size rules say otherwise.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(exist_ok=True)

    print(f"[db] {DB_PATH}", flush=True)
    with connect_readonly() as conn:
        print("[pragma] NL_HR/NL_H1 payout-related columns", flush=True)
        payout_cols = payout_related_columns(conn)
        h1_cols = fetch_columns(conn, "NL_H1")

        print("[sample] NL_H1 LIMIT 50", flush=True)
        h1_first50 = h1_limit50(conn)

        print("[sample] race keys from NL_HR LIMIT 5", flush=True)
        races = select_races(conn, 5)

        fuku_rows = []
        race_reports = []
        h1_race_sample = []
        for race in races:
            rid = race_id(race)
            print(f"[race] {rid}", flush=True)
            top3 = se_top3(conn, race)
            hr = hr_payout(conn, race)
            h1 = h1_rows(conn, race, 80)
            if not h1_race_sample:
                h1_race_sample = h1_bettype_sample(conn, race)
            fuku_rows.extend(fuku_rows_from_hr(race, top3, hr))

            top_by_rank = {row["KakuteiJyuni"]: row["Umaban"] for row in top3}
            h1_fuku = [f"{row['Kumi']}:{row['Hyo']}票(ninki={row['Ninki']})" for row in h1 if row["BetType"] == "Fukusyo"][:20]
            hr_fuku_umaban = "" if not hr else hr.get("FukuUmaban", "")
            hr_fuku_pay = "" if not hr else hr.get("FukuPay", "")
            represented = {int(hr_fuku_umaban)} if str(hr_fuku_umaban).isdigit() else set()
            top3_set = {int(row["Umaban"]) for row in top3}
            if top3_set and represented == top3_set:
                conclusion = "NL_HR alone covers all sampled top-3 place horses."
            elif represented & top3_set:
                conclusion = "NL_HR covers only part of the top-3 place horses in this sample."
            else:
                conclusion = "NL_HR Fuku fields do not match top-3 horses in this sample."

            race_reports.append({
                "race_id": rid,
                "first_umaban": top_by_rank.get(1, ""),
                "second_umaban": top_by_rank.get(2, ""),
                "third_umaban": top_by_rank.get(3, ""),
                "hr_fuku_umaban": hr_fuku_umaban,
                "hr_fuku_pay": hr_fuku_pay,
                "h1_fukusyo_candidates": ", ".join(h1_fuku),
                "conclusion": conclusion,
            })

        write_csv(
            PAYOUT_COLUMNS_CSV,
            ["table_name", "cid", "column_name", "data_type", "notnull", "default_value", "primary_key", "inferred_role"],
            payout_cols,
        )
        write_csv(
            FUKU_SAMPLE_CSV,
            [
                "race_id", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
                "Umaban", "KakuteiJyuni", "source_table", "source_column_umaban",
                "source_column_pay", "source_umaban_value", "source_pay_value",
                "fuku_pay", "matched", "notes",
            ],
            fuku_rows,
        )

        PAYOUT_COLUMNS_DOC.write_text(
            build_payout_columns_doc(payout_cols, h1_cols, h1_first50, h1_race_sample),
            encoding="utf-8-sig",
        )
        FUKU_DESIGN_DOC.write_text(
            build_fuku_design_doc(race_reports, fuku_rows),
            encoding="utf-8-sig",
        )

    print(f"[done] wrote {PAYOUT_COLUMNS_CSV}", flush=True)
    print(f"[done] wrote {FUKU_SAMPLE_CSV}", flush=True)
    print(f"[done] wrote {PAYOUT_COLUMNS_DOC}", flush=True)
    print(f"[done] wrote {FUKU_DESIGN_DOC}", flush=True)


if __name__ == "__main__":
    main()
