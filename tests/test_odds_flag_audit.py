from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit.odds_flag_audit import flag_datakubun_cross, flag_value_summary
from src.audit.odds_schema import connect_readonly


def make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE NL_SE(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER);
        CREATE TABLE NL_O1(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, DataKubun TEXT, TanFlag TEXT, FukuFlag TEXT, WakurenFlag TEXT, TanOdds REAL, FukuOddsLow REAL, FukuOddsHigh REAL);
        INSERT INTO NL_SE VALUES (2016,101,'01',1,1,1,1),(2016,101,'01',1,1,1,2),(2016,101,'01',1,1,1,3);
        INSERT INTO NL_O1 VALUES (2016,101,'01',1,1,1,1,'5','1','1','1',2.0,1.1,1.2),(2016,101,'01',1,1,1,2,'5','0','0','0',NULL,NULL,NULL);
        """
    )
    con.commit()
    con.close()


def test_flag_datakubun_cross(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    with connect_readonly(db) as con:
        rows = flag_datakubun_cross(con)
    statuses = {(r["DataKubun"], r["TanFlag"], r["tan_odds_status"]) for r in rows}
    assert ("5", "1", "valid") in statuses
    assert ("5", "0", "null") in statuses
    assert ("missing_row", "missing_row", "missing_row") in statuses


def test_flag_value_summary(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    with connect_readonly(db) as con:
        rows = flag_value_summary(con)
    assert any(r["column_name"] == "TanFlag" and r["value"] == "1" and r["valid_tan_rows"] == 1 for r in rows)
