import csv
import sqlite3
import time
from pathlib import Path


DB_PATH = Path(r"D:\keiba\new_jra_2016-2026\keiba.db")
OUT_DIR = Path("outputs")
DOC_DIR = Path("docs")

SUMMARY_CSV = OUT_DIR / "new_db_fuku_payout_full_summary.csv"
MISMATCH_CSV = OUT_DIR / "new_db_fuku_payout_mismatches.csv"
SAMPLE_CSV = OUT_DIR / "new_db_fuku_payout_sample.csv"
VERTICAL_CSV = OUT_DIR / "new_db_fuku_payouts_vertical.csv"
DOC_PATH = DOC_DIR / "new_db_fuku_payout_validation.md"


RACE_KEY = ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum"]
FUKU_SLOTS = [
    ("FukuUmaban", "FukuPay", "FukuNinki"),
    ("FukuUmaban2", "FukuPay2", "FukuNinki2"),
    ("FukuUmaban3", "FukuPay3", "FukuNinki3"),
    ("FukuUmaban4", "FukuPay4", "FukuNinki4"),
    ("FukuUmaban5", "FukuPay5", "FukuNinki5"),
]


def connect_readonly() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def race_id(row) -> str:
    return (
        f"{int(row['Year']):04d}"
        f"{int(row['MonthDay']):04d}"
        f"{row['JyoCD']}"
        f"{int(row['Kaiji']):02d}"
        f"{int(row['Nichiji']):02d}"
        f"{int(row['RaceNum']):02d}"
    )


def payout_slots(row) -> dict[int, dict]:
    slots = {}
    for idx, (u_col, p_col, n_col) in enumerate(FUKU_SLOTS, start=1):
        value = row[u_col]
        if value is None or str(value).strip() == "":
            continue
        try:
            umaban = int(value)
        except ValueError:
            continue
        slots[umaban] = {
            "slot": idx,
            "pay": row[p_col],
            "ninki": row[n_col],
            "raw_umaban": value,
        }
    return slots


def expected_place_limit(syusso_tosu) -> int:
    # JRA place-payout rule used for validation: 8+ starters pay top 3, 7 or fewer pay top 2.
    return 3 if int(syusso_tosu or 0) >= 8 else 2


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(rank_column: str, normal_only: bool) -> tuple[dict, list[dict], list[dict]]:
    started = time.time()
    label = f"{rank_column}_{'normal_only' if normal_only else 'all'}"
    summary = {
        "validation": label,
        "db_path": str(DB_PATH),
        "rank_column": rank_column,
        "normal_only": normal_only,
        "races_seen": 0,
        "races_with_hr": 0,
        "eligible_place_rows": 0,
        "matched_place_rows": 0,
        "missing_place_rows": 0,
        "races_with_missing_place": 0,
        "slot1_used": 0,
        "slot2_used": 0,
        "slot3_used": 0,
        "slot4_used": 0,
        "slot5_used": 0,
        "max_slots_used": 0,
        "elapsed_sec": "",
    }
    mismatches = []
    sample_rows = []
    missing_races = set()

    normal_filter = "AND se.IJyoCD = '0'" if normal_only else ""
    sql = f"""
        SELECT
            hr.Year, hr.MonthDay, hr.JyoCD, hr.Kaiji, hr.Nichiji, hr.RaceNum,
            hr.SyussoTosu AS hr_syusso_tosu,
            hr.FukuUmaban, hr.FukuPay, hr.FukuNinki,
            hr.FukuUmaban2, hr.FukuPay2, hr.FukuNinki2,
            hr.FukuUmaban3, hr.FukuPay3, hr.FukuNinki3,
            hr.FukuUmaban4, hr.FukuPay4, hr.FukuNinki4,
            hr.FukuUmaban5, hr.FukuPay5, hr.FukuNinki5,
            se.Umaban, se.Bamei, se.NyusenJyuni, se.KakuteiJyuni, se.IJyoCD
        FROM NL_HR hr
        JOIN NL_SE se
          ON se.Year = hr.Year
         AND se.MonthDay = hr.MonthDay
         AND se.JyoCD = hr.JyoCD
         AND se.Kaiji = hr.Kaiji
         AND se.Nichiji = hr.Nichiji
         AND se.RaceNum = hr.RaceNum
        WHERE se.{rank_column} > 0
          AND se.{rank_column} <= CASE WHEN hr.SyussoTosu >= 8 THEN 3 ELSE 2 END
          {normal_filter}
        ORDER BY hr.Year, hr.MonthDay, hr.JyoCD, hr.Kaiji, hr.Nichiji, hr.RaceNum, se.{rank_column}
    """

    print(f"[query] validation={label}", flush=True)
    with connect_readonly() as con:
        current_race = None
        race_count = 0
        processed = 0
        for row in con.execute(sql):
            rid = race_id(row)
            if rid != current_race:
                current_race = rid
                race_count += 1
                summary["races_seen"] = race_count
                summary["races_with_hr"] = race_count
                race_slots = payout_slots(row)
                summary["max_slots_used"] = max(summary["max_slots_used"], len(race_slots))
                for slot in race_slots.values():
                    key = f"slot{slot['slot']}_used"
                    summary[key] += 1
                if race_count % 5000 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[progress] {label} races={race_count:,} eligible_rows={processed:,} "
                        f"missing={summary['missing_place_rows']:,} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            processed += 1
            summary["eligible_place_rows"] += 1
            slots = payout_slots(row)

            umaban = int(row["Umaban"])
            matched = umaban in slots
            if matched:
                summary["matched_place_rows"] += 1
                slot = slots[umaban]
                if len(sample_rows) < 50:
                    sample_rows.append({
                        "validation": label,
                        "race_id": rid,
                        "Umaban": umaban,
                        "Bamei": row["Bamei"],
                        "NyusenJyuni": row["NyusenJyuni"],
                        "KakuteiJyuni": row["KakuteiJyuni"],
                        "IJyoCD": row["IJyoCD"],
                        "fuku_slot": slot["slot"],
                        "fuku_pay": slot["pay"],
                        "fuku_ninki": slot["ninki"],
                        "status": "matched",
                    })
            else:
                summary["missing_place_rows"] += 1
                missing_races.add(rid)
                if len(mismatches) < 10000:
                    mismatches.append({
                        "validation": label,
                        "race_id": rid,
                        "Year": row["Year"],
                        "MonthDay": row["MonthDay"],
                        "JyoCD": row["JyoCD"],
                        "Kaiji": row["Kaiji"],
                        "Nichiji": row["Nichiji"],
                        "RaceNum": row["RaceNum"],
                        "Umaban": umaban,
                        "Bamei": row["Bamei"],
                        "NyusenJyuni": row["NyusenJyuni"],
                        "KakuteiJyuni": row["KakuteiJyuni"],
                        "IJyoCD": row["IJyoCD"],
                        "SyussoTosu": row["hr_syusso_tosu"],
                        "fuku_umaban_values": "|".join(str(row[c]) for c, _, _ in FUKU_SLOTS if row[c] not in (None, "")),
                        "notes": f"{rank_column} eligible horse not found in FukuUmaban1-5",
                    })

    summary["races_with_missing_place"] = len(missing_races)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    return summary, mismatches, sample_rows


def export_vertical_payouts() -> tuple[int, int]:
    rows_written = 0
    races_written = set()
    fieldnames = [
        "race_id", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "slot", "Umaban", "fuku_pay", "fuku_ninki", "Bamei", "NyusenJyuni",
        "KakuteiJyuni", "IJyoCD",
    ]
    slot_queries = []
    for slot, (u_col, p_col, n_col) in enumerate(FUKU_SLOTS, start=1):
        slot_queries.append(f"""
            SELECT
                hr.Year, hr.MonthDay, hr.JyoCD, hr.Kaiji, hr.Nichiji, hr.RaceNum,
                {slot} AS slot,
                CAST(hr.{u_col} AS INTEGER) AS Umaban,
                hr.{p_col} AS fuku_pay,
                hr.{n_col} AS fuku_ninki,
                se.Bamei, se.NyusenJyuni, se.KakuteiJyuni, se.IJyoCD
            FROM NL_HR hr
            LEFT JOIN NL_SE se
              ON se.Year = hr.Year
             AND se.MonthDay = hr.MonthDay
             AND se.JyoCD = hr.JyoCD
             AND se.Kaiji = hr.Kaiji
             AND se.Nichiji = hr.Nichiji
             AND se.RaceNum = hr.RaceNum
             AND se.Umaban = CAST(hr.{u_col} AS INTEGER)
            WHERE hr.{u_col} IS NOT NULL
              AND TRIM(hr.{u_col}) <> ''
              AND hr.{p_col} IS NOT NULL
        """)
    sql = (
        "SELECT * FROM (\n"
        + "\nUNION ALL\n".join(slot_queries)
        + "\n) ORDER BY Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, slot"
    )

    print("[export] vertical fuku payouts from NL_HR slots 1-5", flush=True)
    with connect_readonly() as con, VERTICAL_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in con.execute(sql):
            item = dict(row)
            item["race_id"] = race_id(row)
            writer.writerow({k: item.get(k, "") for k in fieldnames})
            rows_written += 1
            races_written.add(item["race_id"])
            if rows_written % 50000 == 0:
                print(f"[progress] vertical_rows={rows_written:,}", flush=True)
    return rows_written, len(races_written)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(exist_ok=True)

    print(f"[db] {DB_PATH}", flush=True)
    started = time.time()

    summaries = []
    all_mismatches = []
    all_samples = []
    for rank_column, normal_only in [("KakuteiJyuni", True), ("NyusenJyuni", False)]:
        summary, mismatches, sample_rows = run_validation(rank_column, normal_only)
        summaries.append(summary)
        all_mismatches.extend(mismatches)
        all_samples.extend(sample_rows)
    vertical_rows, vertical_races = export_vertical_payouts()

    print("[write] outputs", flush=True)
    write_csv(SUMMARY_CSV, summaries, list(summaries[0].keys()))
    write_csv(MISMATCH_CSV, all_mismatches, [
        "validation", "race_id", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum",
        "Umaban", "Bamei", "NyusenJyuni", "KakuteiJyuni", "IJyoCD", "SyussoTosu",
        "fuku_umaban_values", "notes",
    ])
    write_csv(SAMPLE_CSV, all_samples, [
        "validation", "race_id", "Umaban", "Bamei", "NyusenJyuni", "KakuteiJyuni", "IJyoCD",
        "fuku_slot", "fuku_pay", "fuku_ninki", "status",
    ])

    doc_lines = [
        "# New DB Fuku Payout Validation",
        "",
        "Validated `D:\\keiba\\new_jra_2016-2026\\keiba.db` in read-only mode.",
        "",
        "## Result",
        "",
    ]
    for summary in summaries:
        match_rate = (
            summary["matched_place_rows"] / summary["eligible_place_rows"]
            if summary["eligible_place_rows"]
            else 0
        )
        doc_lines.extend([
            f"### {summary['validation']}",
            "",
            f"- Eligible place rows checked: `{summary['eligible_place_rows']}`",
            f"- Matched place payout rows: `{summary['matched_place_rows']}`",
            f"- Missing place payout rows: `{summary['missing_place_rows']}`",
            f"- Match rate: `{match_rate:.6%}`",
            f"- Races with missing place payout: `{summary['races_with_missing_place']}`",
            f"- Max fuku slots used in a race: `{summary['max_slots_used']}`",
            f"- Elapsed seconds: `{summary['elapsed_sec']}`",
            "",
        ])
    doc_lines.extend([
        "## Interpretation",
        "",
        "The new DB schema includes `FukuUmaban2-5`, `FukuPay2-5`, and `FukuNinki2-5` in `NL_HR`/`RT_HR`.",
        "",
        f"The vertical payout export contains `{vertical_rows}` fuku payout rows across `{vertical_races}` races.",
        "",
        "`KakuteiJyuni_normal_only` validates normal runners by final placing. It has one apparent mismatch caused by an abnormal case where a horse has `NyusenJyuni=1`, `KakuteiJyuni=0`, and `IJyoCD=5`, while payout slots follow the arrival-order payout result.",
        "",
        "`NyusenJyuni_all` validates payout slots against arrival order and is the better check for whether payout records were fully preserved.",
        "",
        "The expected place cutoff used here is JRA's practical rule: 8+ starters pay top 3, 7 or fewer starters pay top 2.",
        "",
        "## Outputs",
        "",
        f"- `{SUMMARY_CSV}`",
        f"- `{MISMATCH_CSV}`",
        f"- `{SAMPLE_CSV}`",
        f"- `{VERTICAL_CSV}`",
        "",
    ])
    DOC_PATH.write_text(
        "\n".join(doc_lines),
        encoding="utf-8-sig",
    )

    for summary in summaries:
        print(f"[done] {summary['validation']} eligible={summary['eligible_place_rows']:,} matched={summary['matched_place_rows']:,} missing={summary['missing_place_rows']:,}", flush=True)
    print(f"[done] wrote {SUMMARY_CSV}", flush=True)
    print(f"[done] wrote {MISMATCH_CSV}", flush=True)
    print(f"[done] wrote {SAMPLE_CSV}", flush=True)
    print(f"[done] wrote {VERTICAL_CSV} rows={vertical_rows:,}", flush=True)
    print(f"[done] wrote {DOC_PATH}", flush=True)


if __name__ == "__main__":
    main()
