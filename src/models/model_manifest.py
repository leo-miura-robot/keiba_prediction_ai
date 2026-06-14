from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).replace("\\", "/").encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def git_info(root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception:
        return {"git_commit_sha": "unknown", "git_is_dirty": None}


def package_versions() -> dict[str, Any]:
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ["catboost", "pandas", "numpy", "sklearn", "pyarrow"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"missing: {exc}"
    return versions


def dataset_fingerprints(base: Path = Path("outputs/model_feature_dataset_v2_1_1")) -> list[dict[str, Any]]:
    return [file_fingerprint(base / f"year={year}" / "data.parquet") for year in range(2016, 2027)]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old = pl.read_csv(path)
        new = pl.concat([old, new], how="diagonal_relaxed")
    new.write_csv(path)
