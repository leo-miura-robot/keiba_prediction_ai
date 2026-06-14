from __future__ import annotations

from pathlib import Path


SEARCH_TERMS = [
    "NL_O1", "TanOdds", "FukuOdds", "INSERT OR REPLACE", "REPLACE INTO",
    "ON CONFLICT", "executemany", "DataKubun", "MakeDate", "sqlite",
]


def audit_import_code(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in {".py", ".sql", ".md", ".txt", ".yaml", ".yml"}:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for term in SEARCH_TERMS:
            if term.lower() in text.lower():
                rel = path.relative_to(root)
                is_audit_or_test = (
                    "tests" in rel.parts
                    or ("src" in rel.parts and "audit" in rel.parts)
                    or path.name.startswith("audit_odds_missingness")
                )
                is_code = path.suffix.lower() in {".py", ".sql"} and not is_audit_or_test
                rows.append({
                    "file_path": str(path),
                    "term": term,
                    "is_code_file": is_code,
                    "contains_insert_or_replace": "INSERT OR REPLACE" in text.upper(),
                    "contains_replace_into": "REPLACE INTO" in text.upper(),
                    "contains_on_conflict": "ON CONFLICT" in text.upper(),
                    "contains_executemany": "executemany" in text,
                    "notes": "match in repository text; inspect manually for ingestion semantics",
                })
    return rows


def overwrite_risk_rows(import_rows: list[dict], has_history_tables: bool) -> list[dict]:
    risky = [
        r for r in import_rows
        if r.get("is_code_file") and (r["contains_insert_or_replace"] or r["contains_replace_into"] or r["contains_on_conflict"])
    ]
    return [{
        "risk_item": "empty_later_record_overwrites_valid_odds",
        "status": "possible" if risky else "not_confirmed_from_repo_code",
        "evidence": f"{len(risky)} repository matches with replace/upsert syntax",
        "history_available_in_db": has_history_tables,
        "conclusion": "現DBに履歴テーブルがなければ、有効値から欠損への遷移はDB単体では証明できない。",
    }]
