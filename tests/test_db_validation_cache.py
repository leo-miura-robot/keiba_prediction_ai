from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from src.database.db_validation_cache import (
    DatabaseValidationError,
    ValidationConfig,
    cache_paths,
    connect_readonly,
    create_full_manifest,
    db_validation_fingerprint,
    light_fingerprint,
    load_manifest,
    sha256_file,
    validate_cached,
    validate_or_require_full,
)


def make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE NL_RA(id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("CREATE TABLE NL_SE(id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("CREATE TABLE NL_O1(id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("CREATE TABLE NL_HR(id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO NL_RA(value) VALUES ('a')")
        con.commit()
    finally:
        con.close()


def cfg(tmp_path: Path) -> ValidationConfig:
    return ValidationConfig(
        cache_dir=tmp_path / "cache",
        required_tables=("NL_RA", "NL_SE", "NL_O1", "NL_HR"),
        light_hash_chunk_mib=1,
        full_hash_chunk_mib=1,
    )


def test_full_manifest_and_cache_hit(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    manifest = create_full_manifest(db, c)
    assert manifest["integrity_check"] == "ok"
    assert manifest["full_file_sha256"] == sha256_file(db, chunk_bytes=1024 * 1024)
    hit = validate_cached(db, c)
    assert hit["cache_hit"] is True
    assert hit["full_integrity_check_skipped"] is True
    assert db_validation_fingerprint(db, None if False else c)["db_full_sha256"] == manifest["full_file_sha256"]


def test_read_only_connection_rejects_write(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    with connect_readonly(db) as con:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE x(id INTEGER)")


def test_cache_miss_by_size_mtime_and_tail_hash(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    create_full_manifest(db, c)
    with db.open("ab") as f:
        f.write(b"x")
    with pytest.raises(DatabaseValidationError) as exc:
        validate_cached(db, c)
    assert any("file_size" in r or "tail_chunk" in r or "mtime" in r for r in exc.value.reasons)


def test_cache_miss_by_head_middle_tail_hash_with_same_size(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = ValidationConfig(
        cache_dir=tmp_path / "cache",
        required_tables=("NL_RA", "NL_SE", "NL_O1", "NL_HR"),
        light_hash_chunk_mib=1,
        full_hash_chunk_mib=1,
        require_mtime_match=False,
    )
    create_full_manifest(db, c)
    size = db.stat().st_size
    positions = [0, max(0, size // 2), max(0, size - 1)]
    for pos in positions:
        original = db.read_bytes()
        with db.open("r+b") as f:
            f.seek(pos)
            b = f.read(1) or b"\0"
            f.seek(pos)
            f.write(bytes([(b[0] + 1) % 256]))
        with pytest.raises(DatabaseValidationError) as exc:
            validate_cached(db, c)
        assert any("chunk" in r or "sqlite_header" in r or "light_fingerprint" in r or "sqlite_metadata_error" in r for r in exc.value.reasons)
        db.write_bytes(original)


def test_cache_miss_by_page_count_and_schema_version(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    create_full_manifest(db, c)
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE added(id INTEGER)")
        con.commit()
    finally:
        con.close()
    with pytest.raises(DatabaseValidationError) as exc:
        validate_cached(db, c)
    assert any("schema_version" in r or "page_count" in r for r in exc.value.reasons)


def test_wal_and_journal_detection(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    create_full_manifest(db, c)
    Path(str(db) + "-wal").write_bytes(b"wal")
    with pytest.raises(DatabaseValidationError) as exc:
        validate_cached(db, c)
    assert "wal_exists" in exc.value.reasons
    db2 = tmp_path / "keiba_journal.db"
    make_db(db2)
    create_full_manifest(db2, c)
    Path(str(db2) + "-journal").write_bytes(b"journal")
    with pytest.raises(DatabaseValidationError) as exc2:
        validate_cached(db2, c)
    assert "journal_exists" in exc2.value.reasons


def test_corrupt_manifest_and_failed_integrity_manifest(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    create_full_manifest(db, c)
    paths = cache_paths(db, c)
    paths["manifest"].write_text("{bad", encoding="utf-8")
    with pytest.raises(DatabaseValidationError):
        load_manifest(db, c)
    create_full_manifest(db, c)
    manifest = load_manifest(db, c)
    assert manifest is not None
    manifest["integrity_check"] = "failed"
    paths["manifest"].write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(DatabaseValidationError) as exc:
        validate_cached(db, c)
    assert "integrity_check_not_ok" in exc.value.reasons


def test_required_table_missing_and_auto_full_default_false(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = ValidationConfig(cache_dir=tmp_path / "cache", required_tables=("NL_RA", "MISSING"))
    current = light_fingerprint(db, c)
    assert current["table_presence"]["MISSING"] is False
    with pytest.raises(DatabaseValidationError):
        validate_or_require_full(db, None)


def test_atomic_manifest_has_no_tmp_after_full(tmp_path: Path) -> None:
    db = tmp_path / "keiba.db"
    make_db(db)
    c = cfg(tmp_path)
    create_full_manifest(db, c)
    paths = cache_paths(db, c)
    assert paths["manifest"].exists()
    assert not paths["manifest"].with_name(paths["manifest"].name + ".tmp").exists()


def test_runner_fingerprints_include_db_validation(monkeypatch) -> None:
    import scripts.run_final_odds_two_models_v1 as final_mod
    from scripts.run_roi_strategy_refinement_v1 import manifest_fingerprint

    dbfp = {
        "db_validation_manifest_path": "cache/validation_manifest.json",
        "db_light_fingerprint": "light",
        "db_full_sha256": "full",
        "integrity_checked_at": "2026-01-01T00:00:00+00:00",
        "validator_manifest_version": 1,
    }
    monkeypatch.setattr(final_mod, "dataset_hash", lambda _cfg: "dataset")
    final_fp = final_mod.expected_fingerprint(
        {
            "version": "v",
            "feature_set_yaml": "config/database_validation.yaml",
            "feature_set": "market_aware",
            "folds": [],
            "final_train_years": [],
            "test_year": 2025,
            "latest_holdout_year": 2026,
            "input_dataset_dir": "outputs/none",
        },
        {},
        "code",
        dbfp,
    )
    assert final_fp["db_validation"] == dbfp
    roi_fp = manifest_fingerprint({"source_output_root": "outputs/final_odds_two_models_v1"}, Path("config/database_validation.yaml"), dbfp)
    assert roi_fp["db_validation"] == dbfp
