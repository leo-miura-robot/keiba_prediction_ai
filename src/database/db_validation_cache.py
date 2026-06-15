from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml


MANIFEST_VERSION = 1
DEFAULT_DB_PATH = Path(r"D:\keiba\new_jra_2016-2026_fixed\keiba.db")
DEFAULT_CONFIG_PATH = Path("config/database_validation.yaml")


class DatabaseValidationError(RuntimeError):
    def __init__(self, message: str, reasons: list[str] | None = None):
        super().__init__(message)
        self.reasons = reasons or [message]


@dataclass(frozen=True)
class ValidationConfig:
    cache_dir: Path = Path("outputs/db_validation_cache")
    required_tables: tuple[str, ...] = ("NL_RA", "NL_SE", "NL_O1", "NL_HR")
    light_hash_chunk_mib: int = 64
    full_hash_chunk_mib: int = 32
    require_mtime_match: bool = True
    require_full_sha256_in_cache: bool = True
    reject_wal: bool = True
    reject_journal: bool = True
    auto_full_check_on_cache_miss: bool = False
    immutable_read: bool = False

    @property
    def light_hash_chunk_bytes(self) -> int:
        return int(self.light_hash_chunk_mib) * 1024 * 1024

    @property
    def full_hash_chunk_bytes(self) -> int:
        return int(self.full_hash_chunk_mib) * 1024 * 1024


def load_validation_config(path: Path | str | ValidationConfig | None = None, overrides: dict[str, Any] | None = None) -> ValidationConfig:
    if isinstance(path, ValidationConfig):
        if not overrides:
            return path
        return ValidationConfig(**{**path.__dict__, **{k: v for k, v in overrides.items() if v is not None}})
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data.update(raw.get("database_validation", raw))
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return ValidationConfig(
        cache_dir=Path(data.get("cache_dir", "outputs/db_validation_cache")),
        required_tables=tuple(data.get("required_tables", ["NL_RA", "NL_SE", "NL_O1", "NL_HR"])),
        light_hash_chunk_mib=int(data.get("light_hash_chunk_mib", 64)),
        full_hash_chunk_mib=int(data.get("full_hash_chunk_mib", 32)),
        require_mtime_match=bool(data.get("require_mtime_match", True)),
        require_full_sha256_in_cache=bool(data.get("require_full_sha256_in_cache", True)),
        reject_wal=bool(data.get("reject_wal", True)),
        reject_journal=bool(data.get("reject_journal", True)),
        auto_full_check_on_cache_miss=bool(data.get("auto_full_check_on_cache_miss", False)),
        immutable_read=bool(data.get("immutable_read", False)),
    )


def canonical_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def db_path_hash(path: Path | str) -> str:
    normalized = str(canonical_path(path)).replace("\\", "/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_paths(db_path: Path | str, cfg: ValidationConfig) -> dict[str, Path]:
    root = cfg.cache_dir / db_path_hash(db_path)[:16]
    return {
        "root": root,
        "manifest": root / "validation_manifest.json",
        "integrity": root / "integrity_check.txt",
        "metadata": root / "metadata.json",
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def sqlite_uri(db_path: Path | str, immutable: bool = False) -> str:
    path = canonical_path(db_path)
    uri = "file:" + quote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return uri


def connect_readonly(db_path: Path | str, immutable: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(sqlite_uri(db_path, immutable=immutable), uri=True)
    con.row_factory = sqlite3.Row
    return con


def validator_code_hash() -> str:
    return sha256_file(Path(__file__))


def sha256_file(path: Path | str, chunk_bytes: int = 32 * 1024 * 1024, progress: bool = False) -> str:
    p = Path(path)
    total = p.stat().st_size
    h = hashlib.sha256()
    read = 0
    next_pct = 0
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if progress and total:
                pct = int(read * 100 / total)
                if pct >= next_pct:
                    print(f"hashing database: {pct}% ({read}/{total} bytes)", flush=True)
                    next_pct += 10
    return h.hexdigest()


def read_range_hash(path: Path, start: int, length: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size <= 0:
        data = b""
        actual_start = 0
    else:
        actual_start = max(0, min(start, size - 1))
        with path.open("rb") as f:
            f.seek(actual_start)
            data = f.read(max(0, min(length, size - actual_start)))
    return {
        "start": actual_start,
        "length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def journal_paths(db_path: Path) -> dict[str, bool]:
    return {
        "wal_exists": Path(str(db_path) + "-wal").exists(),
        "journal_exists": Path(str(db_path) + "-journal").exists(),
        "shm_exists": Path(str(db_path) + "-shm").exists(),
    }


def table_presence(con: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, bool]:
    out = {}
    for table in tables:
        row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        out[table] = row is not None
    return out


def sqlite_metadata(db_path: Path, cfg: ValidationConfig) -> dict[str, Any]:
    with connect_readonly(db_path, immutable=cfg.immutable_read) as con:
        page_size = int(con.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(con.execute("PRAGMA page_count").fetchone()[0])
        schema_version = int(con.execute("PRAGMA schema_version").fetchone()[0])
        user_version = int(con.execute("PRAGMA user_version").fetchone()[0])
        freelist_count = int(con.execute("PRAGMA freelist_count").fetchone()[0])
        journal_mode = str(con.execute("PRAGMA journal_mode").fetchone()[0])
        presence = table_presence(con, cfg.required_tables)
        sqlite_version = sqlite3.sqlite_version
    return {
        "page_size": page_size,
        "page_count": page_count,
        "schema_version": schema_version,
        "user_version": user_version,
        "freelist_count": freelist_count,
        "journal_mode": journal_mode,
        "required_tables": list(cfg.required_tables),
        "table_presence": presence,
        "sqlite_version": sqlite_version,
    }


def light_fingerprint(db_path: Path | str, cfg: ValidationConfig) -> dict[str, Any]:
    path = canonical_path(db_path)
    stat = path.stat()
    header = read_range_hash(path, 0, 100)
    chunk = cfg.light_hash_chunk_bytes
    head = read_range_hash(path, 0, min(chunk, stat.st_size))
    middle_start = max(0, (stat.st_size // 2) - (chunk // 2))
    middle = read_range_hash(path, middle_start, min(chunk, stat.st_size))
    tail_start = max(0, stat.st_size - chunk)
    tail = read_range_hash(path, tail_start, min(chunk, stat.st_size))
    try:
        meta = sqlite_metadata(path, cfg)
    except sqlite3.DatabaseError as exc:
        raise DatabaseValidationError(f"sqlite metadata read failed: {exc}", ["sqlite_metadata_error"]) from exc
    journals = journal_paths(path)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "db_path": str(path).replace("\\", "/"),
        "db_path_hash": db_path_hash(path),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sqlite_header_sha256": header["sha256"],
        "head_chunk": head,
        "middle_chunk": middle,
        "tail_chunk": tail,
        **meta,
        **journals,
    }
    payload["light_fingerprint"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def load_manifest(db_path: Path | str, cfg: ValidationConfig) -> dict[str, Any] | None:
    path = cache_paths(db_path, cfg)["manifest"]
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise DatabaseValidationError(f"manifest corrupt: {path}: {exc}", ["manifest_corrupt"])


def invalid_reasons(current: dict[str, Any], manifest: dict[str, Any] | None, cfg: ValidationConfig) -> list[str]:
    if manifest is None:
        return ["cache_manifest_missing"]
    reasons: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        reasons.append("manifest_version_mismatch")
    if not manifest.get("validation_completed"):
        reasons.append("validation_not_completed")
    if manifest.get("integrity_check") != "ok":
        reasons.append("integrity_check_not_ok")
    if cfg.require_full_sha256_in_cache and not manifest.get("full_file_sha256"):
        reasons.append("full_sha256_missing")
    for key in ["db_path_hash", "file_size", "page_size", "page_count", "schema_version", "user_version", "light_fingerprint"]:
        if manifest.get(key) != current.get(key):
            reasons.append(f"{key}_mismatch")
    if cfg.require_mtime_match and manifest.get("mtime_ns") != current.get("mtime_ns"):
        reasons.append("mtime_ns_mismatch")
    if manifest.get("sqlite_header_sha256") != current.get("sqlite_header_sha256"):
        reasons.append("sqlite_header_sha256_mismatch")
    for prefix in ["head_chunk", "middle_chunk", "tail_chunk"]:
        old = manifest.get(prefix, {})
        new = current.get(prefix, {})
        if old.get("sha256") != new.get("sha256") or old.get("start") != new.get("start") or old.get("length") != new.get("length"):
            reasons.append(f"{prefix}_mismatch")
    if cfg.reject_wal and current.get("wal_exists"):
        reasons.append("wal_exists")
    if cfg.reject_journal and current.get("journal_exists"):
        reasons.append("journal_exists")
    if cfg.reject_journal and current.get("shm_exists"):
        reasons.append("shm_exists")
    missing_tables = [t for t, ok in current.get("table_presence", {}).items() if not ok]
    if missing_tables:
        reasons.append("required_table_missing:" + ",".join(missing_tables))
    old_missing = [t for t, ok in manifest.get("table_presence", {}).items() if not ok]
    if old_missing:
        reasons.append("manifest_required_table_missing:" + ",".join(old_missing))
    return reasons


def run_integrity_check(db_path: Path | str, cfg: ValidationConfig) -> str:
    with connect_readonly(db_path, immutable=cfg.immutable_read) as con:
        return str(con.execute("PRAGMA integrity_check").fetchone()[0])


def create_full_manifest(db_path: Path | str, cfg: ValidationConfig, force_integrity_check: bool = False) -> dict[str, Any]:
    del force_integrity_check
    path = canonical_path(db_path)
    paths = cache_paths(path, cfg)
    started = time.time()
    current = light_fingerprint(path, cfg)
    if (cfg.reject_wal and current.get("wal_exists")) or (cfg.reject_journal and (current.get("journal_exists") or current.get("shm_exists"))):
        raise DatabaseValidationError("WAL/journal exists; refusing full validation", ["wal_or_journal_exists"])
    print("database validation: FULL", flush=True)
    print("integrity_check start", flush=True)
    integrity_started = time.time()
    integrity = run_integrity_check(path, cfg)
    integrity_elapsed = time.time() - integrity_started
    print(f"integrity_check result: {integrity} elapsed={integrity_elapsed:.1f}s", flush=True)
    if integrity != "ok":
        atomic_write_text(paths["integrity"], integrity + "\n")
        raise DatabaseValidationError(f"integrity_check failed: {integrity}", ["integrity_check_failed"])
    print("full SHA-256 start", flush=True)
    hash_started = time.time()
    full_sha = sha256_file(path, chunk_bytes=cfg.full_hash_chunk_bytes, progress=True)
    hash_elapsed = time.time() - hash_started
    manifest = {
        **current,
        "full_file_sha256": full_sha,
        "integrity_check": integrity,
        "integrity_checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "integrity_elapsed_sec": round(integrity_elapsed, 3),
        "full_hash_elapsed_sec": round(hash_elapsed, 3),
        "validation_elapsed_sec": round(time.time() - started, 3),
        "validation_completed": True,
        "validator_code_hash": validator_code_hash(),
        "python_version": sys.version,
    }
    atomic_write_text(paths["integrity"], integrity + "\n")
    atomic_write_json(paths["metadata"], {k: v for k, v in manifest.items() if k not in {"full_file_sha256"}})
    atomic_write_json(paths["manifest"], manifest)
    print(f"cache updated: {paths['manifest']}", flush=True)
    return manifest


def validate_cached(db_path: Path | str, cfg: ValidationConfig) -> dict[str, Any]:
    started = time.time()
    journals = journal_paths(canonical_path(db_path))
    early_reasons = []
    if cfg.reject_wal and journals.get("wal_exists"):
        early_reasons.append("wal_exists")
    if cfg.reject_journal and journals.get("journal_exists"):
        early_reasons.append("journal_exists")
    if cfg.reject_journal and journals.get("shm_exists"):
        early_reasons.append("shm_exists")
    if early_reasons:
        raise DatabaseValidationError("database validation cache MISS: " + "; ".join(early_reasons), early_reasons)
    current = light_fingerprint(db_path, cfg)
    manifest = load_manifest(db_path, cfg)
    reasons = invalid_reasons(current, manifest, cfg)
    if reasons:
        raise DatabaseValidationError(
            "database validation cache MISS: " + "; ".join(reasons),
            reasons,
        )
    assert manifest is not None
    result = {
        "status": "hit",
        "cache_hit": True,
        "full_integrity_check_skipped": True,
        "reason": "unchanged database verified by cached validation manifest",
        "elapsed_sec": round(time.time() - started, 3),
        "manifest_path": str(cache_paths(db_path, cfg)["manifest"]),
        "db_path": current["db_path"],
        "db_path_hash": current["db_path_hash"],
        "file_size": current["file_size"],
        "mtime_ns": current["mtime_ns"],
        "light_fingerprint": current["light_fingerprint"],
        "full_file_sha256": manifest["full_file_sha256"],
        "integrity_checked_at": manifest["integrity_checked_at"],
        "manifest_version": manifest["manifest_version"],
    }
    return result


def validate_or_require_full(
    db_path: Path | str,
    config_path: Path | str | None = None,
    *,
    full: bool = False,
    force_integrity_check: bool = False,
    allow_auto_full: bool | None = None,
    skip: bool = False,
) -> dict[str, Any]:
    if skip:
        print("WARNING: DB validation skipped by explicit user option; not recommended for production.", flush=True)
        return {"status": "skipped", "cache_hit": False, "full_integrity_check_skipped": True, "db_validation_skipped": True}
    cfg = load_validation_config(config_path)
    if allow_auto_full is not None:
        cfg = ValidationConfig(**{**cfg.__dict__, "auto_full_check_on_cache_miss": allow_auto_full})
    if full or force_integrity_check:
        return create_full_manifest(db_path, cfg, force_integrity_check=True)
    try:
        result = validate_cached(db_path, cfg)
        print("database validation cache: HIT", flush=True)
        print(f"light fingerprint: matched {result['light_fingerprint']}", flush=True)
        print(f"last full integrity check: {result['integrity_checked_at']}", flush=True)
        print(f"full integrity_check: skipped elapsed={result['elapsed_sec']}s", flush=True)
        return result
    except DatabaseValidationError as exc:
        print("database validation cache: MISS", flush=True)
        print("reason: " + "; ".join(exc.reasons), flush=True)
        if cfg.auto_full_check_on_cache_miss:
            print("auto_full_check_on_cache_miss=true; starting full validation", flush=True)
            return create_full_manifest(db_path, cfg, force_integrity_check=True)
        print("full integrity_check not started automatically", flush=True)
        print(f"run: python scripts/validate_database.py --db \"{db_path}\" --full", flush=True)
        raise


def db_validation_fingerprint(db_path: Path | str, config_path: Path | str | None = None) -> dict[str, Any]:
    cfg = load_validation_config(config_path)
    manifest = validate_cached(db_path, cfg)
    return {
        "db_validation_manifest_path": manifest["manifest_path"],
        "db_light_fingerprint": manifest["light_fingerprint"],
        "db_full_sha256": manifest["full_file_sha256"],
        "integrity_checked_at": manifest["integrity_checked_at"],
        "validator_manifest_version": manifest["manifest_version"],
        "db_path_hash": manifest["db_path_hash"],
    }


def status(db_path: Path | str, config_path: Path | str | None = None) -> dict[str, Any]:
    cfg = load_validation_config(config_path)
    current = light_fingerprint(db_path, cfg)
    manifest = load_manifest(db_path, cfg)
    reasons = invalid_reasons(current, manifest, cfg)
    paths = cache_paths(db_path, cfg)
    return {
        "cache_found": manifest is not None,
        "cache_valid": not reasons,
        "invalid_reasons": reasons,
        "manifest_path": str(paths["manifest"]),
        "db_path": current["db_path"],
        "db_size": current["file_size"],
        "mtime_ns": current["mtime_ns"],
        "full_sha256": manifest.get("full_file_sha256") if manifest else None,
        "light_fingerprint": current["light_fingerprint"],
        "integrity_result": manifest.get("integrity_check") if manifest else None,
        "last_full_check": manifest.get("integrity_checked_at") if manifest else None,
    }
