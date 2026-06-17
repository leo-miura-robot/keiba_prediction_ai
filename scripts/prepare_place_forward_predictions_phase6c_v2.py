from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402
from scripts.run_place_market_offset_forward_paper_phase6c_v2 import load_yaml, sha256_json, utc_now  # noqa: E402
from src.calibration.official_calibrator_loader import (  # noqa: E402
    apply_official_platt_calibrator,
    load_official_platt_calibrator,
    sha256_file,
)


REQUIRED_ID_COLUMNS = {"race_id", "entry_id", "race_date", "Umaban", "market_logit", "fuku_odds_low"}
DEFAULT_FORWARD_CONFIG = ROOT / "config/place_market_offset_forward_paper_phase6c_v2.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "config/place_market_offset_champion_challenger_phase5c_v1.yaml"


def sigmoid(z: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float))), eps, 1.0 - eps)


def validate_official_artifacts(forward_cfg: dict[str, Any]) -> tuple[Path, str, dict[str, Any], Path, str]:
    champion = forward_cfg["champion"]
    if champion["strategy"] != "ROLLING_10Y":
        raise ValueError("Champion strategy mismatch")
    calibration = champion["calibration"]
    artifact_path = Path(calibration["artifact_path"])
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    artifact_hash = sha256_file(artifact_path)
    if artifact_hash != calibration["required_artifact_sha256"]:
        raise ValueError("Official Platt artifact hash mismatch")
    artifact = load_official_platt_calibrator(artifact_path)
    if artifact["input_space"] != calibration["input_space"]:
        raise ValueError("Official Platt input_space mismatch")
    if artifact["refit_performed"] is not False:
        raise ValueError("Official Platt refit flag mismatch")

    model_path = Path(champion["model_artifact_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model_hash = sha256_file(model_path)
    if model_hash != forward_cfg["model_hash"]:
        raise ValueError("Official model artifact hash mismatch")
    return artifact_path, artifact_hash, artifact, model_path, model_hash


def validate_feature_input(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    missing = sorted((REQUIRED_ID_COLUMNS | set(numeric) | set(categorical)) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required pre-race feature columns: {missing[:20]}")
    for col in ["market_logit", "fuku_odds_low", *numeric]:
        vals = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(vals.to_numpy(float)).all():
            raise ValueError(f"NaN/inf in numeric input column: {col}")
    if pd.to_numeric(df["fuku_odds_low"], errors="raise").le(0).any():
        raise ValueError("odds missing or <= 0")
    if df.duplicated(["race_id", "entry_id"]).any():
        raise ValueError("duplicate prediction input key")


def make_prediction_run_id(race_date: str, df: pd.DataFrame, cfg_hash: str) -> str:
    keys = df[["race_id", "entry_id"]].astype(str).sort_values(["race_id", "entry_id"]).to_dict("list")
    payload = json.dumps({"race_date": race_date, "keys": keys, "cfg_hash": cfg_hash}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def prepare_predictions(
    race_date: str,
    pre_race_feature_csv: Path,
    output_csv: Path,
    forward_config: Path = DEFAULT_FORWARD_CONFIG,
    model_config: Path = DEFAULT_MODEL_CONFIG,
    odds_snapshot_type: str = "FINAL_ODDS",
    odds_observed_at: str | None = None,
    retrospective_only: bool = True,
    fixture: bool = False,
) -> dict[str, Any]:
    forward_cfg = load_yaml(forward_config)
    model_cfg = phase5b.load_yaml(model_config)
    numeric, categorical = phase5b.load_feature_allowlist(model_cfg)
    artifact_path, artifact_hash, artifact, model_path, model_hash = validate_official_artifacts(forward_cfg)
    if odds_snapshot_type == "FINAL_ODDS" and not retrospective_only:
        raise ValueError("FINAL_ODDS predictions must be retrospective_only=true")
    if odds_snapshot_type not in {"FINAL_ODDS", "PRE_RACE"}:
        raise ValueError(f"Unsupported odds_snapshot_type: {odds_snapshot_type}")

    df = pd.read_csv(pre_race_feature_csv)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    df = df[df["race_date"].eq(pd.to_datetime(race_date, errors="raise").strftime("%Y-%m-%d"))].copy()
    if df.empty:
        raise ValueError(f"No rows for race_date={race_date}")
    validate_feature_input(df, numeric, categorical)

    valid = df.copy()
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    predicted = phase5b.predict_raw(model, valid, numeric, categorical, float(model_cfg["epsilon"]))
    predicted["probability_market"] = sigmoid(predicted["market_logit"].to_numpy(float), float(model_cfg["epsilon"]))
    predicted["probability_calibrated"] = apply_official_platt_calibrator(artifact, predicted["probability_raw"].to_numpy(float))
    predicted["expected_value"] = predicted["probability_calibrated"] * pd.to_numeric(predicted["fuku_odds_low"], errors="raise")
    for col in ["probability_market", "probability_raw", "probability_calibrated", "expected_value"]:
        vals = predicted[col].to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"NaN/inf in output column: {col}")
    if not predicted["probability_raw"].between(0, 1).all() or not predicted["probability_calibrated"].between(0, 1).all():
        raise ValueError("probability out of range")

    now = utc_now()
    odds_at = odds_observed_at or now
    cfg_hash = sha256_json(forward_cfg)
    run_id = make_prediction_run_id(race_date, predicted, cfg_hash)
    out = pd.DataFrame(
        {
            "prediction_run_id": run_id,
            "race_id": predicted["race_id"].astype(str),
            "entry_id": predicted["entry_id"].astype(str),
            "race_date": predicted["race_date"],
            "Umaban": predicted["Umaban"].astype(str),
            "horse_no": predicted["Umaban"].astype(str),
            "strategy": "ROLLING_10Y",
            "calibration_method": "PLATT_SCALING",
            "model_artifact_path": str(model_path.resolve()),
            "model_artifact_sha256": model_hash,
            "calibrator_artifact_path": str(artifact_path.resolve()),
            "calibrator_artifact_sha256": artifact_hash,
            "calibrator_type": "PLATT_SCALING",
            "calibrator_input_space": artifact["input_space"],
            "calibrator_refit_performed": False,
            "probability_market": predicted["probability_market"],
            "probability_raw": predicted["probability_raw"],
            "probability_calibrated": predicted["probability_calibrated"],
            "market_logit": predicted["market_logit"],
            "residual_raw": predicted["catboost_residual_score"],
            "fuku_odds_low": predicted["fuku_odds_low"],
            "fuku_odds_low_at_prediction": predicted["fuku_odds_low"],
            "expected_value": predicted["expected_value"],
            "ev_at_prediction": predicted["expected_value"],
            "odds_snapshot_type": odds_snapshot_type,
            "odds_observed_at": odds_at,
            "prediction_created_at": now,
            "retrospective_only": bool(retrospective_only),
            "fixture": bool(fixture),
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    tier_counts = {}
    for tier in forward_cfg["threshold_tiers"]:
        tier_counts[tier["tier"]] = int(out["expected_value"].ge(float(tier["threshold"])).sum())
    manifest = {
        "prediction_run_id": run_id,
        "race_date": race_date,
        "created_at": now,
        "strategy": "ROLLING_10Y",
        "model_artifact_path": str(model_path.resolve()),
        "model_artifact_sha256": model_hash,
        "calibrator_artifact_path": str(artifact_path.resolve()),
        "calibrator_artifact_sha256": artifact_hash,
        "calibrator_type": "PLATT_SCALING",
        "calibrator_input_space": artifact["input_space"],
        "calibrator_refit_performed": False,
        "ev_definition": "probability_calibrated * fuku_odds_low",
        "tier_thresholds": forward_cfg["threshold_tiers"],
        "odds_snapshot_type": odds_snapshot_type,
        "retrospective_only": bool(retrospective_only),
        "fixture": bool(fixture),
        "row_count": int(len(out)),
        "core_count": tier_counts.get("CORE", 0),
        "margin_count": tier_counts.get("MARGIN", 0),
        "high_count": tier_counts.get("HIGH", 0),
        "very_high_count": tier_counts.get("VERY_HIGH", 0),
        "output_csv": str(output_csv.resolve()),
        "refit_performed": False,
        "raw_fallback_used": False,
        "shadow_15y_status": "BLOCKED_MISSING_ISOTONIC_THRESHOLDS",
    }
    manifest_dir = output_csv.parents[1] / "manifests" if output_csv.parent.name == "input" else output_csv.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"prediction_run_{pd.to_datetime(race_date).strftime('%Y%m%d')}_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-date", required=True)
    parser.add_argument("--pre-race-feature-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--forward-config", default=str(DEFAULT_FORWARD_CONFIG))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--odds-snapshot-type", default="FINAL_ODDS")
    parser.add_argument("--odds-observed-at", default=None)
    parser.add_argument("--retrospective-only", action="store_true", default=True)
    parser.add_argument("--fixture", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    manifest = prepare_predictions(
        race_date=args.race_date,
        pre_race_feature_csv=Path(args.pre_race_feature_csv),
        output_csv=Path(args.output_csv),
        forward_config=Path(args.forward_config),
        model_config=Path(args.model_config),
        odds_snapshot_type=args.odds_snapshot_type,
        odds_observed_at=args.odds_observed_at,
        retrospective_only=bool(args.retrospective_only),
        fixture=bool(args.fixture),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
