from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.official_calibrator_loader import sha256_file  # noqa: E402


OUT = Path("outputs/place_market_offset_official_market_phase5c_v1")
FOLD_MANIFEST = Path("outputs/place_market_offset_champion_challenger_phase5c_v1/fold_manifests/ROLLING_10Y_validation_2026.json")
PREDICTION_REF = Path("outputs/place_market_offset_champion_challenger_phase5c_v1/predictions/ROLLING_10Y/validation_2026.parquet")
MODEL_PATH = Path("models/place_market_offset_champion_challenger_phase5c_v1/ROLLING_10Y/validation_2026/model.cbm")
BASE_CONFIG = Path("config/place_market_offset_catboost_c1r0_v1.yaml")
MARKET_CODE = Path("scripts/run_place_market_offset_catboost_v1.py")
ROOTS_SEARCHED = [Path("models"), Path("outputs"), Path("config"), Path("scripts"), Path("src"), Path("docs")]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_manifest() -> dict[str, Any]:
    return json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))


def candidate_files() -> list[str]:
    patterns = ["*market*.joblib", "*market*.pkl", "*market*.pickle", "*market*scaler*.json", "*market*logistic*.json", "*market*.npz", "*market*.npy"]
    found: list[str] = []
    for base in ROOTS_SEARCHED:
        if not base.exists():
            continue
        for pat in patterns:
            found.extend(str(p) for p in base.rglob(pat))
    return sorted(set(found))


def search_report(manifest: dict[str, Any]) -> dict[str, Any]:
    mp = manifest["market_provenance"]
    files = candidate_files()
    serialized = [p for p in files if p.lower().endswith((".joblib", ".pkl", ".pickle", ".npz", ".npy"))]
    return {
        "search_status": "PARTIAL_PARAMETERS_FOUND",
        "serialized_artifact_found": bool(serialized),
        "serialized_artifact_candidates": serialized,
        "parameter_candidates": files,
        "roots_searched": [str(p) for p in ROOTS_SEARCHED],
        "search_terms": [
            "market_logit",
            "p_market",
            "market_model",
            "StandardScaler",
            "LogisticRegression",
            "mean_",
            "scale_",
            "var_",
            "coef_",
            "intercept_",
            "feature_names",
            "feature_order",
            "validation_2026",
            "ROLLING_10Y",
        ],
        "strategy": mp["strategy"],
        "validation_year": mp["validation_year"],
        "training_period_start": mp["market_train_start"],
        "training_period_end": mp["market_train_end"],
        "market_feature_count": len(mp["market_input_columns"]),
        "market_feature_names_ordered": mp["market_input_columns"],
        "source_paths": [str(FOLD_MANIFEST), str(BASE_CONFIG), str(MARKET_CODE)],
        "source_sha256": {
            str(FOLD_MANIFEST): sha256_file(FOLD_MANIFEST),
            str(BASE_CONFIG): sha256_file(BASE_CONFIG),
            str(MARKET_CODE): sha256_file(MARKET_CODE),
        },
    }


def parameter_inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    mp = manifest["market_provenance"]
    return {
        "classification": "PARTIAL_PARAMETERS_FOUND",
        "market_feature_names_found": True,
        "market_feature_order_found": True,
        "market_feature_count": len(mp["market_input_columns"]),
        "transform_rules_found": True,
        "transform_rules": {
            "market_x": "coerce each market feature to numeric, replace +/-inf with NaN, fill NaN with current frame column median",
            "scaler": "sklearn StandardScaler() in Pipeline",
            "logistic": mp["market_model_config"],
            "clip": "p_market clipped with epsilon before logit; epsilon inherited from phase config",
        },
        "standard_scaler": {
            "mean_": "NOT_FOUND",
            "scale_": "NOT_FOUND",
            "var_": "NOT_FOUND",
            "n_features_in_": "NOT_FOUND",
            "feature_names_in_": "NOT_FOUND",
            "with_mean": True,
            "with_std": True,
        },
        "logistic_regression": {
            "coef_": "NOT_FOUND",
            "intercept_": "NOT_FOUND",
            "classes_": "NOT_FOUND",
            "n_features_in_": "NOT_FOUND",
            "feature_names_in_": "NOT_FOUND",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": mp["market_model_config"].get("C"),
            "max_iter": mp["market_model_config"].get("max_iter"),
            "class_weight": None,
            "fit_intercept": True,
        },
        "refit_performed": False,
        "parameter_generation_performed": False,
    }


def reference_audit() -> dict[str, Any]:
    if not PREDICTION_REF.exists():
        return {"status": "BLOCKED_NO_MARKET_LOGIT_REFERENCE", "reference_path": str(PREDICTION_REF), "exists": False}
    cols = pd.read_parquet(PREDICTION_REF, columns=None).columns.tolist()
    has_market_logit = "market_logit" in cols
    market_input_cols = read_manifest()["market_provenance"]["market_input_columns"]
    missing_market_inputs = sorted(set(market_input_cols) - set(cols))
    return {
        "status": "REFERENCE_FOUND_BUT_NO_REPRODUCTION_WITHOUT_PARAMETERS",
        "reference_path": str(PREDICTION_REF),
        "reference_sha256": sha256_file(PREDICTION_REF),
        "exists": True,
        "has_market_logit": has_market_logit,
        "market_input_columns_present": not missing_market_inputs,
        "missing_market_input_columns": missing_market_inputs,
        "rows_compared": 0,
        "mean_absolute_error": "NOT_COMPUTED",
        "max_absolute_error": "NOT_COMPUTED",
        "p99_absolute_error": "NOT_COMPUTED",
        "allclose_at_1e-12": False,
        "allclose_at_1e-9": False,
        "parity_status": "BLOCKED_MISSING_MARKET_PARAMETERS",
    }


def recovery_attempts() -> list[dict[str, Any]]:
    return [
        {
            "method": "serialized_pipeline_or_parameter_artifact_search",
            "status": "FAILED_SAFE",
            "reason": "No read-only StandardScaler/LogisticRegression artifact for ROLLING_10Y validation_2026 was found under models/ or outputs/.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "phase5c_manifest_and_csv_inventory",
            "status": "PARTIAL_ONLY",
            "reason": "Fold manifest and window CSV preserve feature order, training period, row counts, and model config, but not fitted scaler/logistic parameters.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "catboost_model_metadata_inventory",
            "status": "FAILED_SAFE",
            "reason": "CatBoost metadata contains residual model params and feature names only; market baseline pipeline parameters are not embedded.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "saved_prediction_reference_inventory",
            "status": "PARTIAL_ONLY",
            "reason": "Prediction parquet preserves market input columns and market_logit, but not the fitted StandardScaler and LogisticRegression state.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "infer_linear_parameters_from_saved_market_logit",
            "status": "REJECTED_PROHIBITED",
            "reason": "Solving coefficients from saved inputs/logits would regenerate parameters from prediction data and violates the task prohibition.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "refit_market_model_on_2016_2025_training_rows",
            "status": "REJECTED_PROHIBITED",
            "reason": "This would refit StandardScaler and LogisticRegression and requires explicit user approval.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
        {
            "method": "fallback_to_saved_market_logit_lookup",
            "status": "REJECTED_UNSAFE",
            "reason": "A lookup cannot score future rows and would hide the missing official market artifact.",
            "refit_performed": False,
            "parameter_generation_performed": False,
        },
    ]


def migration_plan() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "action": "Add market artifact persistence to the Phase 5B/5C training runner for future folds.",
            "details": "Persist StandardScaler mean_/scale_/var_, LogisticRegression coef_/intercept_/classes_, feature order, preprocessing rules, sklearn version, training key hash, and source hashes.",
            "requires_user_approval": False,
        },
        {
            "step": 2,
            "action": "On the next approved training/refit run, materialize the ROLLING_10Y validation_2026 market artifact and compare validation market_logit.",
            "details": "Certification target: max absolute market_logit error <= 1e-12 against the saved 2026 validation prediction parquet.",
            "requires_user_approval": True,
        },
        {
            "step": 3,
            "action": "Only after certification passes, connect the raw Phase 6C bridge to the official market loader and run fixture smoke.",
            "details": "Until then, raw-to-official Champion forward prediction remains fail-closed.",
            "requires_user_approval": False,
        },
    ]


def run(output_root: Path = OUT) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    search = search_report(manifest)
    inventory = parameter_inventory(manifest)
    ref = reference_audit()
    feature_contract = {
        "strategy": "ROLLING_10Y",
        "validation_year": 2026,
        "training_period_start": manifest["market_provenance"]["market_train_start"],
        "training_period_end": manifest["market_provenance"]["market_train_end"],
        "market_feature_names": manifest["market_provenance"]["market_input_columns"],
        "market_feature_order": manifest["market_provenance"]["market_input_columns"],
        "market_feature_count": len(manifest["market_provenance"]["market_input_columns"]),
        "preprocessing": inventory["transform_rules"],
    }
    final = {
        "final_status": "BLOCKED_MISSING_MARKET_PARAMETERS",
        "search_status": search["search_status"],
        "serialized_artifact_found": search["serialized_artifact_found"],
        "full_parameters_found": False,
        "official_market_artifact_created": False,
        "refit_performed": False,
        "parameter_generation_performed": False,
        "market_logit_reproduction_passed": False,
        "raw_bridge_connected": False,
        "fixture_smoke_passed": False,
        "ready_for_real_forward_prediction": False,
        "reason": "Feature contract and model config are saved, but StandardScaler and LogisticRegression fitted parameters are not saved.",
        "attempted_recovery_methods": recovery_attempts(),
        "migration_plan": migration_plan(),
    }
    write_json(output_root / "market_artifact_search_report.json", search)
    write_json(output_root / "market_parameter_inventory.json", inventory)
    write_json(output_root / "market_feature_contract.json", feature_contract)
    pd.DataFrame([ref]).to_csv(output_root / "market_reproduction_comparison.csv", index=False)
    write_json(output_root / "official_market_manifest.json", final)
    write_json(output_root / "artifact_audit.json", final)
    write_json(output_root / "blocked_market_artifact_recovery.json", {**final, "reference_audit": ref})
    report = [
        "# Official Market Baseline Recovery Phase5C v1",
        "",
        f"- final_status: `{final['final_status']}`",
        f"- search_status: `{search['search_status']}`",
        f"- serialized_artifact_found: `{search['serialized_artifact_found']}`",
        "- scaler_parameters: `NOT_FOUND`",
        "- logistic_parameters: `NOT_FOUND`",
        f"- market_feature_count: `{feature_contract['market_feature_count']}`",
        f"- training_period: `{feature_contract['training_period_start']}-{feature_contract['training_period_end']}`",
        f"- reference_status: `{ref['status']}`",
        "",
        "No refit or parameter generation was performed.",
        "",
        "## Attempted safe recovery methods",
        "",
        *[
            f"- `{item['method']}`: `{item['status']}` - {item['reason']}"
            for item in final["attempted_recovery_methods"]
        ],
        "",
        "## Migration plan",
        "",
        *[
            f"- {item['step']}. {item['action']} {item['details']}"
            for item in final["migration_plan"]
        ],
    ]
    (output_root / "certification_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(OUT))
    args = parser.parse_args()
    return run(Path(args.output_root))


if __name__ == "__main__":
    raise SystemExit(main())
