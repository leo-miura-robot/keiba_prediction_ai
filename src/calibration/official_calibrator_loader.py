from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_PLATT_KEYS = {
    "strategy",
    "calibrator_type",
    "input_space",
    "coef",
    "intercept",
    "clip_min",
    "clip_max",
    "created_from_existing_certified_parameters",
    "refit_performed",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def load_official_platt_calibrator(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_PLATT_KEYS - set(artifact))
    if missing:
        raise ValueError(f"Official Platt artifact missing keys: {missing}")
    if artifact["strategy"] != "ROLLING_10Y":
        raise ValueError("Official Platt artifact strategy mismatch")
    if artifact["calibrator_type"] != "PLATT_SCALING":
        raise ValueError("Official Platt artifact calibrator_type mismatch")
    if artifact["input_space"] != "logit_probability_raw":
        raise ValueError("Official Platt artifact input_space mismatch")
    if artifact["created_from_existing_certified_parameters"] is not True:
        raise ValueError("Official Platt artifact must be created from existing certified parameters")
    if artifact["refit_performed"] is not False:
        raise ValueError("Official Platt artifact indicates refit was performed")

    artifact["coef"] = _finite_float(artifact["coef"], "coef")
    artifact["intercept"] = _finite_float(artifact["intercept"], "intercept")
    artifact["clip_min"] = _finite_float(artifact["clip_min"], "clip_min")
    artifact["clip_max"] = _finite_float(artifact["clip_max"], "clip_max")
    if not 0.0 < artifact["clip_min"] < artifact["clip_max"] < 1.0:
        raise ValueError("Official Platt artifact clip bounds are invalid")

    expected_hash = artifact.get("artifact_payload_sha256")
    if expected_hash:
        payload = dict(artifact)
        payload.pop("artifact_payload_sha256", None)
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if actual != expected_hash:
            raise ValueError("Official Platt artifact payload hash mismatch")
    artifact["artifact_file_sha256"] = sha256_file(artifact_path)
    artifact["artifact_path"] = str(artifact_path.resolve())
    return artifact


def apply_official_platt_calibrator(artifact: dict[str, Any], probability_raw: np.ndarray) -> np.ndarray:
    p = np.asarray(probability_raw, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("probability_raw contains NaN or inf")
    p_clipped = np.clip(p, float(artifact["clip_min"]), float(artifact["clip_max"]))
    x = np.log(p_clipped / (1.0 - p_clipped))
    z = float(artifact["coef"]) * x + float(artifact["intercept"])
    out = 1.0 / (1.0 + np.exp(-z))
    out = np.clip(out, float(artifact["clip_min"]), float(artifact["clip_max"]))
    if not np.isfinite(out).all():
        raise ValueError("official Platt output contains NaN or inf")
    return out
