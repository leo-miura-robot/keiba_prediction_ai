from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_FILES = [
    "strategy_definition.csv",
    "walk_forward_folds.csv",
    "legacy_parity_check.csv",
    "market_model_window_by_strategy.csv",
    "residual_model_window_by_strategy.csv",
    "sample_weight_summary.csv",
    "metrics_by_strategy_fold.csv",
    "metrics_by_strategy_2020_2024.csv",
    "metrics_by_strategy_2016_2019_aux.csv",
    "yearly_win_loss_matrix.csv",
    "worst_year_summary.csv",
    "residual_stability_by_strategy.csv",
    "paired_bootstrap_summary.csv",
    "roi_diagnostic_raw.csv",
    "roi_row_removed_raw.csv",
    "roi_payout_zeroed_stress_raw.csv",
    "selected_year_strategy.json",
    "phase5b_2025_2026_diagnostic.csv",
    "manifest.json",
    "audit_report.md",
    "phase5b_predictions.parquet",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def check_required_files(out: Path) -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED_FILES:
        path = out / name
        rows.append({"check": f"file:{name}", "passed": path.exists(), "detail": str(path)})
    return rows


def check_manifest(out: Path) -> list[dict[str, Any]]:
    path = out / "manifest.json"
    if not path.exists():
        return [{"check": "manifest", "passed": False, "detail": "missing"}]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "check": "manifest_probability_raw_only",
            "passed": manifest.get("probability_column") == "probability_raw",
            "detail": manifest.get("probability_column"),
        },
        {
            "check": "manifest_no_calibration_fit",
            "passed": manifest.get("calibration_fit") is False and manifest.get("probability_calibrated_generated") is False,
            "detail": str({k: manifest.get(k) for k in ["calibration_fit", "probability_calibrated_generated"]}),
        },
        {
            "check": "manifest_catboost_safety",
            "passed": manifest.get("catboost_safety", {}) == {
                "iterations": 300,
                "use_best_model": False,
                "early_stopping_enabled": False,
                "outer_validation_eval_set_used": False,
            },
            "detail": str(manifest.get("catboost_safety", {})),
        },
    ]


def check_parity(out: Path) -> list[dict[str, Any]]:
    parity = load_csv(out / "legacy_parity_check.csv")
    if parity.empty:
        return [{"check": "legacy_parity_gate", "passed": False, "detail": "empty or not run"}]
    passed = bool(parity["passed"].all()) if "passed" in parity.columns else False
    return [{"check": "legacy_parity_gate", "passed": passed, "detail": parity.to_json(orient="records", force_ascii=False)}]


def check_roi(out: Path) -> list[dict[str, Any]]:
    normal = load_csv(out / "roi_diagnostic_raw.csv")
    pz = load_csv(out / "roi_payout_zeroed_stress_raw.csv")
    rr = load_csv(out / "roi_row_removed_raw.csv")
    rows = []
    if pz.empty:
        rows.append({"check": "payout_zeroed_stress_le_normal", "passed": True, "detail": "no stress rows"})
    else:
        ok = pz["roi"].fillna(-math.inf).le(pz["normal_roi"].fillna(math.inf) + 1e-12).all()
        rows.append({"check": "payout_zeroed_stress_le_normal", "passed": bool(ok), "detail": f"rows={len(pz)}"})
    if not normal.empty:
        rows.append({
            "check": "prediction_without_bets_is_na",
            "passed": bool(normal.loc[normal["bet_count"].eq(0), "roi"].isna().all()),
            "detail": f"zero_bet_rows={int(normal['bet_count'].eq(0).sum())}",
        })
    if not rr.empty and not pz.empty:
        rows.append({
            "check": "stress_population_columns_present",
            "passed": {"normal_roi", "removed_count", "stake", "bet_count"}.issubset(rr.columns) and {"normal_roi", "stake", "bet_count"}.issubset(pz.columns),
            "detail": f"row_removed={len(rr)} payout_zeroed={len(pz)}",
        })
    return rows


def check_predictions(out: Path) -> list[dict[str, Any]]:
    path = out / "phase5b_predictions.parquet"
    if not path.exists():
        return [{"check": "prediction_columns", "passed": False, "detail": "missing predictions"}]
    cols = pd.read_parquet(path, columns=None).columns
    forbidden = {"probability_calibrated", "calibrated_ev"}
    return [
        {"check": "prediction_has_probability_raw", "passed": "probability_raw" in cols, "detail": ",".join(cols)},
        {"check": "prediction_no_calibrated_columns", "passed": not bool(forbidden.intersection(cols)), "detail": ",".join(sorted(forbidden.intersection(cols)))},
    ]


def run(output_root: Path, report: Path | None) -> int:
    rows: list[dict[str, Any]] = []
    rows.extend(check_required_files(output_root))
    rows.extend(check_manifest(output_root))
    rows.extend(check_parity(output_root))
    rows.extend(check_roi(output_root))
    rows.extend(check_predictions(output_root))
    audit = pd.DataFrame(rows)
    out_path = output_root / "phase5b_audit_checks.csv"
    audit.to_csv(out_path, index=False)
    text = "\n".join(
        [
            "# Phase 5B Artifact Audit",
            "",
            f"- output_root: `{output_root}`",
            f"- checks: {len(audit)}",
            f"- failed: {int((~audit['passed']).sum())}",
            "",
            audit.to_markdown(index=False),
            "",
        ]
    )
    target = report or (output_root / "phase5b_audit_report.md")
    target.write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(audit["passed"].all()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/place_market_offset_year_strategy_phase5b_v2")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.output_root), Path(args.report) if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
