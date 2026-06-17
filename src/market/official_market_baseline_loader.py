from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_KEYS = {
    "strategy",
    "validation_year",
    "market_feature_order",
    "standard_scaler",
    "logistic_regression",
    "created_from_existing_parameters",
    "refit_performed",
    "parameter_generation_performed",
}


def load_official_market_baseline(
    artifact_root: str | Path,
    expected_strategy: str = "ROLLING_10Y",
    expected_validation_year: int = 2026,
) -> dict[str, Any]:
    root = Path(artifact_root)
    manifest_path = root / "official_market_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    artifact = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_KEYS - set(artifact))
    if missing:
        raise ValueError(f"Official market artifact is incomplete: {missing}")
    if artifact["strategy"] != expected_strategy:
        raise ValueError("Official market artifact strategy mismatch")
    if int(artifact["validation_year"]) != int(expected_validation_year):
        raise ValueError("Official market artifact validation_year mismatch")
    if artifact["refit_performed"] is not False:
        raise ValueError("Official market artifact indicates refit")
    if artifact["parameter_generation_performed"] is not False:
        raise ValueError("Official market artifact indicates parameter generation")
    return artifact


def apply_official_market_baseline(artifact: dict[str, Any], market_feature_frame: pd.DataFrame) -> np.ndarray:
    features = list(artifact["market_feature_order"])
    missing = sorted(set(features) - set(market_feature_frame.columns))
    if missing:
        raise ValueError(f"Missing market features: {missing}")
    scaler = artifact["standard_scaler"]
    logistic = artifact["logistic_regression"]
    for required in ["mean_", "scale_"]:
        if required not in scaler:
            raise ValueError(f"Missing scaler parameter: {required}")
    for required in ["coef_", "intercept_"]:
        if required not in logistic:
            raise ValueError(f"Missing logistic parameter: {required}")
    x = market_feature_frame[features].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("NaN/inf in market feature frame")
    mean = np.asarray(scaler["mean_"], dtype=float)
    scale = np.asarray(scaler["scale_"], dtype=float)
    coef = np.asarray(logistic["coef_"], dtype=float).reshape(-1)
    intercept = float(np.asarray(logistic["intercept_"], dtype=float).reshape(-1)[0])
    z = ((x - mean) / scale) @ coef + intercept
    if not np.isfinite(z).all():
        raise ValueError("NaN/inf in market_logit output")
    return z
