# Database Validation Cache Design

## Purpose

The fixed JRA SQLite database is about 18.9 GB. Running `PRAGMA integrity_check` for every pipeline command is too expensive, so this repository now has a shared validator that performs the full check only when a validated cache manifest is missing or stale.

The validator never changes the database. SQLite connections use `mode=ro`; `immutable=1` is disabled by default and can only be enabled from config.

## Cache Hit Conditions

Full `integrity_check` is skipped only when all of these match the saved full-validation manifest:

- DB path hash
- file size
- `mtime_ns`
- SQLite header hash
- first / middle / last chunk hashes
- `page_size`
- `page_count`
- `schema_version`
- `user_version`
- light fingerprint
- required table presence
- previous `integrity_check == ok`
- `validation_completed == true`
- full DB SHA-256 is present
- WAL, SHM, and journal files are absent

If any item differs, normal mode stops and prints the full-validation command. It does not start the 20-minute full check automatically.

## Fingerprint

The light fingerprint reads only:

- SQLite header
- first chunk
- middle chunk
- last chunk
- file metadata
- SQLite PRAGMA metadata
- required table presence
- WAL/journal state

The default chunk size is 64 MiB. The full SHA-256 is computed only during `--full` or `--force-integrity-check`, using streaming reads with progress logs.

## Files

- Module: `src/database/db_validation_cache.py`
- CLI: `scripts/validate_database.py`
- Config: `config/database_validation.yaml`
- Cache root: `outputs/db_validation_cache/`

## Runner Integration

The following runners now use the shared validator:

- `scripts/build_full_runner_dataset_o1_fixed.py`
- `scripts/build_model_features_v2_1_2.py`
- `scripts/run_final_odds_two_models_v1.py`
- `scripts/run_roi_strategy_refinement_v1.py`

They support:

```bash
--force-integrity-check
--skip-db-validation
--db-validation-config config/database_validation.yaml
```

`--skip-db-validation` is intentionally explicit and prints a warning. It is not recommended for production.

Strict resume fingerprints include:

- DB validation manifest path
- DB light fingerprint
- DB full SHA-256
- integrity check timestamp
- validator manifest version

## Operations

Status:

```bash
python scripts/validate_database.py --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" --status
```

Initial full validation:

```bash
python scripts/validate_database.py --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" --full
```

Normal validation:

```bash
python scripts/validate_database.py --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db"
```

## Current Production DB Note

The current production DB has `keiba.db-wal` and `keiba.db-shm` present. The validator correctly rejects cache use and refuses full validation while these files exist. This is intentional: WAL/SHM files mean the main DB file alone is not a stable validation target.
