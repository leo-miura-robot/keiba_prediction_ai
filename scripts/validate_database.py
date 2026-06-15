from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db_validation_cache import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_DB_PATH,
    DatabaseValidationError,
    status,
    validate_or_require_full,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SQLite DB with cached full integrity_check manifest.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force-integrity-check", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    db = Path(args.db)
    cfg = Path(args.config)
    try:
        if args.status:
            print(json.dumps(status(db, cfg), ensure_ascii=False, indent=2, default=str))
            return 0
        result = validate_or_require_full(db, cfg, full=args.full, force_integrity_check=args.force_integrity_check)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except DatabaseValidationError as exc:
        print(f"database validation failed: {exc}", file=sys.stderr)
        if exc.reasons:
            print("reasons: " + "; ".join(exc.reasons), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
