from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_latest_model_on_jrvltsql_db import apply_official_calibration
from src.calibration.official_calibrator_loader import (
    apply_official_platt_calibrator,
    load_official_platt_calibrator,
)


def write_artifact(path: Path, **overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "strategy": "ROLLING_10Y",
        "calibrator_type": "PLATT_SCALING",
        "input_space": "logit_probability_raw",
        "coef": 1.0162527329694642,
        "intercept": 0.016713944459665484,
        "clip_min": 0.000001,
        "clip_max": 0.999999,
        "created_from_existing_certified_parameters": True,
        "refit_performed": False,
    }
    artifact.update(overrides)
    payload_hash = hashlib.sha256(json.dumps(artifact, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    artifact["artifact_payload_sha256"] = payload_hash
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def test_official_platt_formula_matches_expected(tmp_path: Path) -> None:
    path = tmp_path / "platt.json"
    expected = write_artifact(path)
    artifact = load_official_platt_calibrator(path)
    p = np.array([0.2, 0.5, 0.8], dtype=float)

    actual = apply_official_platt_calibrator(artifact, p)

    clipped = np.clip(p, expected["clip_min"], expected["clip_max"])
    x = np.log(clipped / (1.0 - clipped))
    z = expected["coef"] * x + expected["intercept"]
    manual = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(actual, manual)


def test_official_platt_loader_fails_closed_on_strategy_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "platt.json"
    write_artifact(path, strategy="ROLLING_15Y")

    with pytest.raises(ValueError, match="strategy mismatch"):
        load_official_platt_calibrator(path)


def test_apply_official_calibration_does_not_call_phase6a_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "platt.json"
    write_artifact(path)
    pred = __import__("pandas").DataFrame(
        {
            "strategy": ["ROLLING_10Y", "ROLLING_15Y"],
            "entry_id": ["e1", "e2"],
            "race_id": ["r1", "r1"],
            "race_date": ["2026-06-13", "2026-06-13"],
            "market_logit": [0.0, 0.0],
            "probability_raw": [0.4, 0.4],
        }
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit/refit must not be called")

    import scripts.run_place_market_offset_safe_calibration_phase6a_v1 as phase6a

    monkeypatch.setattr(phase6a, "fit_calibrator", fail)
    out, audit = apply_official_calibration(pred, path, tmp_path / "missing.parquet", 1e-6)

    assert audit["refit_performed"] is False
    assert out.loc[out["strategy"].eq("ROLLING_10Y"), "probability_official_platt"].notna().all()
    assert out.loc[out["strategy"].eq("ROLLING_15Y"), "official_calibration_status"].item() == "BLOCKED_MISSING_ISOTONIC_THRESHOLDS"
