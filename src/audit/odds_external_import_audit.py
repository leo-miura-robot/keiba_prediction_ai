from __future__ import annotations

from pathlib import Path


TERMS = ["NL_O1", "TanOdds", "FukuOddsLow", "INSERT OR REPLACE", "REPLACE INTO", "ON CONFLICT", "executemany", "JVOpen", "JVGets", "JVRead", "JVLink", "DataLab"]
EXCLUDED_PARTS = {"tasks", "tests", "docs", "outputs", ".git", "__pycache__", ".pytest_cache"}


def import_candidates(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in {".py", ".sql", ".bat", ".ps1", ".cmd", ".yaml", ".yml", ".toml", ".ini"}:
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if len(rel.parts) >= 2 and rel.parts[0] == "src" and rel.parts[1] == "audit":
            continue
        if rel.parts[0] == "scripts" and path.name.startswith("audit_"):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, start=1):
            if any(term.lower() in line.lower() for term in TERMS):
                text = line.strip()
                is_actual = any(x in text for x in ["sqlite3", "connect", "INSERT", "REPLACE", "JVOpen", "JVGets", "JVRead", "executemany"])
                rows.append({
                    "file": str(path),
                    "line": lineno,
                    "matched_text": text[:500],
                    "is_actual_import_code": bool(is_actual),
                    "reason": "candidate ingestion code" if is_actual else "reference only",
                })
    return rows
