from __future__ import annotations

from pathlib import Path

from src.audit.odds_external_import_audit import import_candidates


def test_external_import_candidates_exclude_noise(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "audit").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "loader.py").write_text("cur.executemany('INSERT INTO NL_O1 VALUES (?)', rows)", encoding="utf-8")
    (tmp_path / "tasks" / "task.md").write_text("INSERT OR REPLACE NL_O1", encoding="utf-8")
    (tmp_path / "tests" / "test.py").write_text("INSERT OR REPLACE NL_O1", encoding="utf-8")
    (tmp_path / "docs" / "x.md").write_text("TanOdds", encoding="utf-8")
    (tmp_path / "src" / "audit" / "audit.py").write_text("NL_O1", encoding="utf-8")
    rows = import_candidates(tmp_path)
    assert len(rows) == 1
    assert rows[0]["is_actual_import_code"] is True
    assert "loader.py" in rows[0]["file"]
