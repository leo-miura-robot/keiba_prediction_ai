from __future__ import annotations

import sqlite3
from pathlib import Path

from src.audit.odds_race_pattern_audit import race_level_patterns, race_pattern_summary, runner_count_consistency
from src.audit.odds_schema import connect_readonly


def test_race_patterns_and_runner_counts(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE NL_SE(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER);
        CREATE TABLE NL_RA(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, TorokuTosu INTEGER, SyussoTosu INTEGER);
        CREATE TABLE NL_O1(Year INTEGER, MonthDay INTEGER, JyoCD TEXT, Kaiji INTEGER, Nichiji INTEGER, RaceNum INTEGER, Umaban INTEGER, DataKubun TEXT, TorokuTosu INTEGER, SyussoTosu INTEGER, TanOdds REAL, FukuOddsLow REAL, FukuOddsHigh REAL);
        INSERT INTO NL_SE VALUES (2016,101,'01',1,1,1,1),(2016,101,'01',1,1,1,2),(2016,101,'01',1,1,2,1),(2016,101,'01',1,1,2,2);
        INSERT INTO NL_RA VALUES (2016,101,'01',1,1,1,2,2),(2016,101,'01',1,1,2,2,2);
        INSERT INTO NL_O1 VALUES (2016,101,'01',1,1,1,1,'5',2,2,2.0,1.1,1.2),(2016,101,'01',1,1,1,2,'5',2,2,3.0,1.2,1.3),(2016,101,'01',1,1,2,1,'5',2,2,NULL,NULL,NULL);
        """
    )
    con.commit()
    con.close()
    with connect_readonly(db) as ro:
        patterns = race_level_patterns(ro)
        consistency, anomalies = runner_count_consistency(ro)
    assert any(r["tan_race_pattern"] == "all_valid" for r in patterns)
    assert any(r["tan_race_pattern"] == "missing_o1_rows" for r in patterns)
    assert race_pattern_summary(patterns)
    assert anomalies
