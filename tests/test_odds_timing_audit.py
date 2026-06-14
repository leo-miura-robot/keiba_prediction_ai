from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit.odds_schema import connect_readonly
from src.audit.odds_timing_audit import make_date_timing, timing_hypothesis_rows


def test_make_date_timing_buckets(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE NL_SE(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER);
        CREATE TABLE NL_O1(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, MakeDate TEXT, TanOdds REAL, FukuOddsLow REAL, FukuOddsHigh REAL);
        INSERT INTO NL_SE VALUES (2016,101,'01',1,1,1,1),(2016,101,'01',1,1,1,2),(2016,101,'01',1,1,1,3);
        INSERT INTO NL_O1 VALUES (2016,101,'01',1,1,1,1,'20151231',2.0,1.1,1.2),(2016,101,'01',1,1,1,2,'20160101',NULL,NULL,NULL),(2016,101,'01',1,1,1,3,'20160102',3.0,1.2,1.3);
        """
    )
    con.commit()
    con.close()
    with connect_readonly(db) as ro:
        rows = make_date_timing(ro)
    buckets = {r["timing_bucket"] for r in rows}
    assert {"before_race", "same_day", "day_after"} <= buckets
    checks = timing_hypothesis_rows(rows)
    assert any(r["hypothesis"].startswith("H1") for r in checks)
