from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit.odds_missingness import join_coverage_by_year, value_encoding_summary
from src.audit.odds_schema import connect_readonly, duplicate_key_count


def make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE NL_SE(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, Odds REAL, Ninki INTEGER, IJyoCD TEXT, DataKubun TEXT, MakeDate TEXT);
        CREATE TABLE NL_O1(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, TanOdds, TanNinki, FukuOddsLow, FukuOddsHigh, FukuNinki, DataKubun TEXT, MakeDate TEXT);
        INSERT INTO NL_SE VALUES
          (2016,101,'01',1,1,1,1,2.0,1,'0','7','20160101'),
          (2016,101,'01',1,1,1,2,3.0,2,'0','7','20160101'),
          (2016,101,'01',1,1,1,3,4.0,3,'0','7','20160101'),
          (2016,101,'01',1,1,1,4,5.0,4,'1','7','20160101');
        INSERT INTO NL_O1 VALUES
          (2016,101,'01',1,1,1,1,2.0,1,1.1,1.2,1,'5','20160101'),
          (2016,101,'01',1,1,1,2,NULL,2,NULL,NULL,2,'5','20160101'),
          (2016,101,'01',1,1,1,3,0,3,0,0,3,'5','20160101');
        """
    )
    con.commit()
    con.close()


def test_join_coverage_classifies_missing_null_zero_valid(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    with connect_readonly(db) as con:
        rows = join_coverage_by_year(con)
    row = rows[0]
    assert row["se_rows"] == 4
    assert row["o1_missing_rows"] == 1
    assert row["valid_tan_rows"] == 1
    assert row["null_tan_rows"] == 1
    assert row["invalid_tan_rows"] == 1
    assert row["valid_place_rows"] == 1


def test_readonly_connection_rejects_write(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    with connect_readonly(db) as con:
        try:
            con.execute("CREATE TABLE nope(x)")
        except sqlite3.OperationalError:
            pass
        else:
            raise AssertionError("read-only connection allowed write")


def test_duplicate_key_detection(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    con = sqlite3.connect(db)
    con.execute("INSERT INTO NL_O1 SELECT * FROM NL_O1 WHERE Umaban=1")
    con.commit()
    con.close()
    with connect_readonly(db) as con:
        assert duplicate_key_count(con, "NL_O1", ["Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban"]) == 1


def test_value_encoding_summary(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    make_db(db)
    with connect_readonly(db) as con:
        rows = value_encoding_summary(con)
    tan = [r for r in rows if r["table_name"] == "NL_O1" and r["column_name"] == "TanOdds"][0]
    assert tan["null_count"] == 1
    assert tan["zero_count"] == 1
