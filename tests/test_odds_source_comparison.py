from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit.odds_schema import connect_readonly
from src.audit.odds_source_comparison import se_o1_comparison


def test_se_o1_scale_and_rank_comparison(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE NL_SE(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, Odds REAL, Ninki INTEGER);
        CREATE TABLE NL_O1(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, TanOdds REAL, TanNinki INTEGER, DataKubun TEXT, MakeDate TEXT);
        INSERT INTO NL_SE VALUES (2016,101,'01',1,1,1,1,2.0,1),(2016,101,'01',1,1,1,2,3.0,2);
        INSERT INTO NL_O1 VALUES (2016,101,'01',1,1,1,1,2.0,1,'5','20160101'),(2016,101,'01',1,1,1,2,3.5,2,'5','20160101');
        """
    )
    con.commit()
    con.close()
    with connect_readonly(db) as ro:
        summary, by_year, samples = se_o1_comparison(ro)
    direct = [r for r in summary if r["comparison"] == "se_vs_o1"][0]
    assert direct["compared_rows"] == 2
    assert direct["exact_match_count"] == 1
    assert by_year[0]["ranking_exact_match_rate"] == 1.0
    assert len(samples) == 1
