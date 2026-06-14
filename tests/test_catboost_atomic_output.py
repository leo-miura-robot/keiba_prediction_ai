from __future__ import annotations

from pathlib import Path

import polars as pl

from src.models.catboost_atomic_output import atomic_write_csv, file_sha256
from scripts.analyze_catboost_baseline_v1_0_2 import calibration_bins


def test_atomic_write_replaces_and_drops_old_rows(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    atomic_write_csv(path, [{"id": 1, "value": "old"}, {"id": 2, "value": "stale"}])
    first = file_sha256(path)
    atomic_write_csv(path, [{"id": 1, "value": "new"}])
    second = file_sha256(path)
    df = pl.read_csv(path)
    assert first != second
    assert df.height == 1
    assert df["value"][0] == "new"


def test_atomic_write_idempotent_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    rows = [{"id": 1, "value": "a"}]
    h1 = atomic_write_csv(path, rows)
    h2 = atomic_write_csv(path, rows)
    assert h1 == h2
    assert file_sha256(path) == h1


def test_quantile_calibration_keeps_equal_probability_together() -> None:
    pred = pl.DataFrame({
        "data_split": ["validation"] * 12,
        "actual": [0, 1] * 6,
        "pred_probability": [0.1] * 6 + [0.8] * 6,
    })
    rows = calibration_bins(pred, "quantile", requested=10)
    assert sum(r["count"] for r in rows) == 12
    assert len(rows) <= 2
    assert {r["lower_bound"] for r in rows} == {0.1, 0.8}


def test_fixed_width_calibration_still_exists() -> None:
    pred = pl.DataFrame({
        "data_split": ["validation"] * 10,
        "actual": [0, 1] * 5,
        "pred_probability": [i / 10 for i in range(10)],
    })
    rows = calibration_bins(pred, "fixed_width", requested=10)
    assert sum(r["count"] for r in rows) == 10
