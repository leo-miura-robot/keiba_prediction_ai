from __future__ import annotations

from pathlib import Path

from src.audit.odds_import_audit import audit_import_code, overwrite_risk_rows


def test_import_audit_finds_replace_risk(tmp_path: Path) -> None:
    f = tmp_path / "loader.py"
    f.write_text("cur.executemany('INSERT OR REPLACE INTO NL_O1 VALUES (?)', rows)", encoding="utf-8")
    rows = audit_import_code(tmp_path)
    assert rows
    risk = overwrite_risk_rows(rows, has_history_tables=False)[0]
    assert risk["status"] == "possible"
