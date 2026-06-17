from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_place_forward_predictions_phase6c_v2 as prepared  # noqa: E402
from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402
from src.calibration.official_calibrator_loader import sha256_file  # noqa: E402


RAW_REQUIRED_COLUMNS = {
    "race_id",
    "entry_id",
    "race_date",
    "JyoCD",
    "RaceNum",
    "Umaban",
    "Wakuban",
    "KettoNum",
    "TrackCD",
    "Kyori",
    "SyussoTosu",
    "Barei",
    "SexCD",
    "Futan",
    "BaTaijyu",
    "ZogenSa",
    "KisyuCode",
    "ChokyosiCode",
    "tan_odds",
    "tan_ninki",
    "fuku_odds_low",
    "fuku_odds_high",
    "fuku_ninki",
    "odds_observed_at",
    "odds_snapshot_type",
    "retrospective_only",
}

FORBIDDEN_RESULT_COLUMNS = {
    "KakuteiJyuni",
    "target_place_paid",
    "fuku_pay",
    "払戻",
    "確定結果",
    "当日確定Time",
    "当日確定HaronTimeL3",
}

DEFAULT_OUTPUT_ROOT = Path("outputs/phase6c_raw_to_model_ready_bridge_v1")
DEFAULT_FORWARD_OUTPUT_ROOT = Path("outputs/place_market_offset_forward_paper_phase6c_v2")
DEFAULT_HISTORY_MANIFEST = Path("data/derived/history_extension_2006_phase5_v1/manifest.json")
DEFAULT_MODEL_CONFIG = Path("config/place_market_offset_champion_challenger_phase5c_v1.yaml")
DEFAULT_FORWARD_CONFIG = Path("config/place_market_offset_forward_paper_phase6c_v2.yaml")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def audit_prepare_contract() -> dict[str, Any]:
    required = sorted(prepared.REQUIRED_ID_COLUMNS)
    contract = "MODEL_READY_INPUT_ONLY"
    reason = "prepare_place_forward_predictions_phase6c_v2.py requires market_logit and the complete feature allowlist; it does not build history features from raw jrvltsql rows."
    return {
        "script": "scripts/prepare_place_forward_predictions_phase6c_v2.py",
        "contract": contract,
        "raw_input_supported": False,
        "required_id_columns": required,
        "requires_market_logit": "market_logit" in required,
        "reason": reason,
    }


def audit_raw_input(path: Path, race_date: str, output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    columns = list(df.columns)
    missing = sorted(RAW_REQUIRED_COLUMNS - set(columns))
    forbidden = sorted(FORBIDDEN_RESULT_COLUMNS.intersection(columns))
    duplicate_count = int(df.duplicated(["race_id", "entry_id"]).sum()) if {"race_id", "entry_id"}.issubset(df.columns) else -1
    date_values: list[str] = []
    if "race_date" in df.columns:
        date_values = sorted(pd.to_datetime(df["race_date"], errors="raise").dt.strftime("%Y-%m-%d").unique().tolist())
    race_date_canon = pd.to_datetime(race_date, errors="raise").strftime("%Y-%m-%d")
    date_match = date_values == [race_date_canon]
    odds_ok = False
    if {"fuku_odds_low", "fuku_odds_high"}.issubset(df.columns):
        odds_ok = bool(pd.to_numeric(df["fuku_odds_low"], errors="coerce").gt(0).all() and pd.to_numeric(df["fuku_odds_high"], errors="coerce").gt(0).all())
    audit = {
        "raw_input_path": str(path.resolve()),
        "raw_input_sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": columns,
        "missing_required_columns": missing,
        "forbidden_result_columns_present": forbidden,
        "duplicate_entry_count": duplicate_count,
        "race_date_values": date_values,
        "race_date_match": date_match,
        "odds_positive": odds_ok,
        "passed": not missing and not forbidden and duplicate_count == 0 and date_match and odds_ok,
    }
    pd.DataFrame([audit | {"columns": ",".join(columns), "missing_required_columns": ",".join(missing), "forbidden_result_columns_present": ",".join(forbidden), "race_date_values": ",".join(date_values)}]).to_csv(
        output_root / "raw_input_audit.csv",
        index=False,
    )
    if not audit["passed"]:
        raise ValueError(f"Raw input audit failed: {audit}")
    return df, audit


def audit_history_sources(race_date: str, output_root: Path, manifest_path: Path = DEFAULT_HISTORY_MANIFEST) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cutoff = (pd.to_datetime(race_date, errors="raise") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    source_paths = manifest.get("source_db_paths", [])
    source_exists = {p: Path(p).exists() for p in source_paths}
    audit = {
        "history_manifest_path": str(manifest_path.resolve()),
        "history_manifest_sha256": sha256_file(manifest_path),
        "history_source_paths": source_paths,
        "history_source_exists": source_exists,
        "history_source_sha256": manifest.get("source_db_sha256", []),
        "history_start_date": manifest.get("source_date_min"),
        "history_available_end_date": manifest.get("source_date_max"),
        "history_cutoff_date": cutoff,
        "history_cutoff_rule": "D-1 only",
        "current_race_excluded": manifest.get("current_race_excluded"),
        "same_day_future_excluded": manifest.get("same_day_future_excluded"),
        "passed": bool(source_paths) and all(source_exists.values()) and manifest.get("current_race_excluded") is True and manifest.get("same_day_future_excluded") is True,
    }
    write_json(output_root / "history_source_audit.json", audit)
    if not audit["passed"]:
        raise ValueError(f"History source audit failed: {audit}")
    return audit


def audit_feature_contract(output_root: Path, model_config_path: Path = DEFAULT_MODEL_CONFIG) -> dict[str, Any]:
    cfg = phase5b.load_yaml(model_config_path)
    numeric, categorical = phase5b.load_feature_allowlist(cfg)
    feature_columns = numeric + categorical
    audit = {
        "feature_allowlist_path": str(Path(cfg["feature_allowlist_path"]).resolve()),
        "feature_allowlist_sha256": sha256_file(Path(cfg["feature_allowlist_path"])),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "feature_count": len(feature_columns),
        "expected_feature_count": 79,
        "duplicate_features": sorted({c for c in feature_columns if feature_columns.count(c) > 1}),
        "forbidden_features_present": sorted(set(feature_columns).intersection({"Year", "p_market", "market_logit", "tan_odds", "fuku_odds_low", "fuku_odds_high", "KisyuCode", "ChokyosiCode"})),
        "passed_contract_only": len(feature_columns) == 79 and len(set(feature_columns)) == len(feature_columns),
        "feature_columns": feature_columns,
    }
    pd.DataFrame({"feature": feature_columns, "position": range(len(feature_columns))}).to_csv(output_root / "feature_schema_parity.csv", index=False)
    return audit


def find_market_artifact(output_root: Path) -> dict[str, Any]:
    # Phase 5B/5C fits the market LogisticRegression in-process per fold and only
    # stores provenance CSVs. No sklearn artifact is available for read-only forward use.
    candidate_patterns = ["*market*.joblib", "*market*.pkl", "*market*.pickle", "*market*model*.json"]
    candidates: list[str] = []
    for base in [Path("models"), Path("outputs")]:
        if base.exists():
            for pattern in candidate_patterns:
                candidates.extend(str(p) for p in base.rglob(pattern))
    audit = {
        "market_artifact_required": True,
        "market_artifact_path": "NOT_FOUND",
        "market_artifact_sha256": "NOT_FOUND",
        "candidate_files": sorted(candidates),
        "passed": False,
        "blocker": "BLOCKED_MARKET_ARTIFACT",
        "reason": "No persisted training-time market baseline LogisticRegression/scaler artifact was found. Forward market_logit generation would require refit or reimplementation, both forbidden.",
    }
    pd.DataFrame([audit | {"candidate_files": ",".join(audit["candidate_files"])}]).to_csv(output_root / "market_input_audit.csv", index=False)
    return audit


def run_pipeline(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "model_ready").mkdir(exist_ok=True)
    (output_root / "predictions").mkdir(exist_ok=True)

    input_contract = audit_prepare_contract()
    write_json(output_root / "input_contract_audit.json", input_contract)

    status = "PHASE6C_RAW_TO_OFFICIAL_CHAMPION_BRIDGE_PASSED"
    error: str | None = None
    raw_audit: dict[str, Any] | None = None
    history_audit: dict[str, Any] | None = None
    feature_audit: dict[str, Any] | None = None
    market_audit: dict[str, Any] | None = None
    phase6c_registration_performed = False

    try:
        _raw, raw_audit = audit_raw_input(Path(args.raw_pre_race_csv), args.race_date, output_root)
        history_audit = audit_history_sources(args.race_date, output_root)
        feature_audit = audit_feature_contract(output_root)
        market_audit = find_market_artifact(output_root)
        if not market_audit["passed"]:
            status = "BLOCKED_MARKET_ARTIFACT"
            raise RuntimeError(market_audit["reason"])
    except Exception as exc:
        error = str(exc)
        if status == "PHASE6C_RAW_TO_OFFICIAL_CHAMPION_BRIDGE_PASSED":
            if raw_audit is None:
                status = "BLOCKED_RAW_SCHEMA"
            elif history_audit is None:
                status = "BLOCKED_HISTORY_SOURCE"
            elif feature_audit is None or not feature_audit.get("passed_contract_only", False):
                status = "BLOCKED_FEATURE_SCHEMA"
            else:
                status = "MULTIPLE_BLOCKERS"

    artifact_audit = {
        "model_artifact_path": "models/place_market_offset_champion_challenger_phase5c_v1/ROLLING_10Y/validation_2026/model.cbm",
        "model_artifact_sha256": sha256_file(Path("models/place_market_offset_champion_challenger_phase5c_v1/ROLLING_10Y/validation_2026/model.cbm")),
        "platt_artifact_path": "outputs/place_market_offset_official_calibrators_phase6a_v1/rolling_10y_platt_phase6a_v1.json",
        "platt_artifact_sha256": sha256_file(Path("outputs/place_market_offset_official_calibrators_phase6a_v1/rolling_10y_platt_phase6a_v1.json")),
        "market_artifact_path": market_audit["market_artifact_path"] if market_audit else "NOT_EVALUATED",
        "market_artifact_sha256": market_audit["market_artifact_sha256"] if market_audit else "NOT_EVALUATED",
        "refit_performed": False,
        "raw_fallback_used": False,
        "status": status,
    }
    write_json(output_root / "artifact_audit.json", artifact_audit)

    completeness = pd.DataFrame(
        [
            {"feature_group": "horse", "zero_rate": "NOT_COMPUTED", "reason": status},
            {"feature_group": "jockey", "zero_rate": "NOT_COMPUTED", "reason": status},
            {"feature_group": "trainer", "zero_rate": "NOT_COMPUTED", "reason": status},
        ]
    )
    completeness.to_csv(output_root / "history_feature_completeness.csv", index=False)

    manifest = {
        "prediction_run_id": "NOT_CREATED",
        "race_date": args.race_date,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "raw_input_path": str(Path(args.raw_pre_race_csv).resolve()),
        "raw_input_sha256": sha256_file(Path(args.raw_pre_race_csv)) if Path(args.raw_pre_race_csv).exists() else "NOT_FOUND",
        "history_source_paths": history_audit.get("history_source_paths") if history_audit else [],
        "history_cutoff_date": history_audit.get("history_cutoff_date") if history_audit else "NOT_COMPUTED",
        "history_rows": "NOT_COMPUTED",
        "history_races": "NOT_COMPUTED",
        "feature_allowlist_path": feature_audit.get("feature_allowlist_path") if feature_audit else "NOT_EVALUATED",
        "feature_allowlist_sha256": feature_audit.get("feature_allowlist_sha256") if feature_audit else "NOT_EVALUATED",
        "feature_count": feature_audit.get("feature_count") if feature_audit else "NOT_EVALUATED",
        "model_artifact_path": artifact_audit["model_artifact_path"],
        "model_artifact_sha256": artifact_audit["model_artifact_sha256"],
        "market_artifact_paths": [],
        "market_artifact_sha256": "NOT_FOUND",
        "calibrator_artifact_path": artifact_audit["platt_artifact_path"],
        "calibrator_artifact_sha256": artifact_audit["platt_artifact_sha256"],
        "odds_snapshot_type": raw_audit and "FINAL_ODDS_OR_INPUT",
        "retrospective_only": raw_audit and "FROM_INPUT",
        "row_count": raw_audit.get("rows") if raw_audit else 0,
        "horse_history_zero_rate": "NOT_COMPUTED",
        "jockey_history_zero_rate": "NOT_COMPUTED",
        "trainer_history_zero_rate": "NOT_COMPUTED",
        "probability_raw_min": "NOT_COMPUTED",
        "probability_raw_max": "NOT_COMPUTED",
        "probability_calibrated_min": "NOT_COMPUTED",
        "probability_calibrated_max": "NOT_COMPUTED",
        "core_count": "NOT_COMPUTED",
        "margin_count": "NOT_COMPUTED",
        "high_count": "NOT_COMPUTED",
        "very_high_count": "NOT_COMPUTED",
        "phase6c_registration_performed": phase6c_registration_performed,
        "fixture": bool(args.fixture),
        "raw_pre_race_supported": False,
        "history_generation_performed": False,
        "feature_79_parity_passed": bool(feature_audit and feature_audit.get("passed_contract_only")) and status != "BLOCKED_MARKET_ARTIFACT",
        "official_model_loaded": False,
        "official_platt_loaded": False,
        "refit_performed": False,
        "raw_fallback_used": False,
        "ready_for_real_forward_prediction": False,
        "final_status": status,
        "error": error,
    }
    manifest_path = output_root / f"run_manifest_{pd.to_datetime(args.race_date).strftime('%Y%m%d')}_{sha256_json(manifest)[:12]}.json"
    write_json(manifest_path, manifest)
    report = [
        "# Phase 6C Raw To Official Champion Bridge v1",
        "",
        f"- final_status: `{status}`",
        f"- prepare_input_contract: `{input_contract['contract']}`",
        f"- raw_input_supported: `{input_contract['raw_input_supported']}`",
        f"- history_cutoff: `{manifest['history_cutoff_date']}`",
        f"- feature_count: `{manifest['feature_count']}`",
        "- market_artifact: `NOT_FOUND`",
        f"- phase6c_registration_performed: `{phase6c_registration_performed}`",
        f"- error: `{error}`",
    ]
    (output_root / "pipeline_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return 0 if status == "PHASE6C_RAW_TO_OFFICIAL_CHAMPION_BRIDGE_PASSED" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-date", required=True)
    parser.add_argument("--raw-pre-race-csv", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase6c-output-root", default=str(DEFAULT_FORWARD_OUTPUT_ROOT))
    parser.add_argument("--fixture", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
