# Database Validation Cache Results

## Implementation

Implemented shared DB validation cache in `src/database/db_validation_cache.py` and CLI in `scripts/validate_database.py`.

The validator supports:

- full validation with `PRAGMA integrity_check`
- streaming full DB SHA-256
- light fingerprint validation
- atomic manifest writes
- cache hit/miss reporting
- read-only SQLite connections
- WAL/SHM/journal rejection
- runner fingerprint integration

## Test Results

Unit and integration tests use small SQLite databases, not the 18.9 GB production DB.

```text
python -m pytest -q tests/test_db_validation_cache.py
10 passed

python -m pytest -q
97 passed
```

Small SQLite CLI acceptance:

```text
--full: integrity_check ok, full SHA-256 saved
normal: cache HIT, full integrity_check skipped, elapsed 0.008s
--status: cache_valid true
```

## Production DB Check

Target:

```text
D:\keiba\new_jra_2016-2026_fixed\keiba.db
```

Observed:

```text
DB size: 18,918,297,600 bytes
WAL exists: true
WAL size: 0 bytes
SHM exists: true
SHM size: 32,768 bytes
journal exists: false
```

Status command produced:

```text
cache_found: false
cache_valid: false
invalid_reasons: cache_manifest_missing
light_fingerprint: 79f3447d0dc9e0fb093f2f660dacb1e6c471ec90f4c6c31806c8320416df373b
manifest_path: outputs\db_validation_cache\69d2651619b12d36\validation_manifest.json
```

Normal command stopped safely:

```text
database validation cache: MISS
reason: wal_exists; shm_exists
full integrity_check not started automatically
```

Full command also stopped safely:

```text
database validation failed: WAL/journal exists; refusing full validation
reasons: wal_or_journal_exists
```

## Acceptance Status

The implementation acceptance succeeded on small SQLite DBs. Production full validation is intentionally blocked until the WAL/SHM state is resolved outside this task. No DB file was updated, vacuumed, copied, or deleted.

## Next Operational Step

Close any process using the DB and resolve the WAL/SHM files through the DB owner workflow. After WAL/SHM are absent, run:

```bash
python scripts/validate_database.py --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db" --full
python scripts/validate_database.py --db "D:\keiba\new_jra_2016-2026_fixed\keiba.db"
```

The second command should show cache HIT and skip `integrity_check`.
