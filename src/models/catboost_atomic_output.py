from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import polars as pl


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]] | pl.DataFrame) -> str:
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    csv_text = df.write_csv()
    atomic_write_text(path, csv_text)
    return hashlib.sha256(csv_text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table_content_hash(path: Path) -> str:
    return file_sha256(path)
