from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FILES = [
    "threshold_grid_by_year.csv",
    "threshold_grid_combined_2020_2024.csv",
    "threshold_nested_walk_forward.csv",
    "threshold_eligibility.csv",
    "threshold_roi_bootstrap.csv",
    "threshold_row_removed_stress.csv",
    "threshold_payout_zeroed_stress.csv",
    "selected_threshold.json",
    "diagnostic_2025_2026.csv",
    "diagnostic_2025_2026_stress.csv",
    "diagnostic_2025_2026_bootstrap.csv",
    "shadow_threshold_comparison.csv",
    "bet_overlap.csv",
    "segment_diagnostic.csv",
    "manifest.json",
    "audit_report.md",
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def checks(out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        rows.append({"check": f"file:{name}", "passed": (out / name).exists(), "detail": str(out / name)})
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {"check": "no_new_training", "passed": manifest.get("new_catboost_training") is False, "detail": str(manifest.get("new_catboost_training"))},
                {"check": "no_db", "passed": manifest.get("db_connection") is False, "detail": str(manifest.get("db_connection"))},
                {"check": "no_calibration_refit", "passed": manifest.get("calibration_refit") is False, "detail": str(manifest.get("calibration_refit"))},
                {"check": "selection_years_only", "passed": manifest.get("selection_years") == [2020, 2021, 2022, 2023, 2024], "detail": str(manifest.get("selection_years"))},
                {"check": "operationally_false", "passed": manifest.get("operationally_activated") is False and manifest.get("champion_changed") is False, "detail": str(manifest)},
            ]
        )
    grid = load_csv(out / "threshold_grid_combined_2020_2024.csv")
    if not grid.empty:
        th = sorted(grid["threshold"].unique())
        rows.append({"check": "threshold_grid_1_00_1_30_step_0_01", "passed": len(th) == 31 and abs(th[0] - 1.0) < 1e-9 and abs(th[-1] - 1.3) < 1e-9, "detail": f"{th[:2]}...{th[-2:]} len={len(th)}"})
        rows.append({"check": "combined_roi_total_stake", "passed": bool((grid["roi"].fillna(-1).between(-1, 10000)).all()), "detail": f"rows={len(grid)}"})
    pz = load_csv(out / "threshold_payout_zeroed_stress.csv")
    if not pz.empty:
        rows.append({"check": "payout_zeroed_le_normal", "passed": bool(pz["payout_zeroed_roi"].fillna(float("-inf")).le(pz["normal_roi"].fillna(float("inf")) + 1e-12).all()), "detail": f"rows={len(pz)}"})
    boot = load_csv(out / "threshold_roi_bootstrap.csv")
    if not boot.empty:
        rows.append({"check": "race_bootstrap_5000", "passed": bool(boot["n_bootstrap"].eq(5000).all()), "detail": f"rows={len(boot)}"})
        rows.append({"check": "bootstrap_prob_bounds", "passed": bool(boot["probability_roi_ge_90"].between(0, 1).all()), "detail": "probability_roi_ge_90"})
    selected = out / "selected_threshold.json"
    if selected.exists():
        s = json.loads(selected.read_text(encoding="utf-8"))
        rows.append({"check": "selected_not_activated", "passed": s.get("operationally_activated") is False, "detail": str(s)})
        rows.append({"check": "status_valid", "passed": s.get("threshold_status") in {"THRESHOLD_CANDIDATE_CERTIFIED", "DIAGNOSTIC_ONLY", "NO_THRESHOLD_CERTIFIED"}, "detail": str(s.get("threshold_status"))})
    diag = load_csv(out / "diagnostic_2025_2026.csv")
    if not diag.empty:
        rows.append({"check": "diagnostic_years_present", "passed": {"2025", "2026", "2025_2026"}.issubset(set(diag["Year"].astype(str))), "detail": str(sorted(diag["Year"].astype(str).unique()))})
    return pd.DataFrame(rows)


def run(output_root: Path, report: Path | None = None) -> int:
    audit = checks(output_root)
    audit.to_csv(output_root / "phase6b_audit_checks.csv", index=False)
    text = "\n".join(
        [
            "# Phase 6B Artifact Audit",
            "",
            f"- output_root: `{output_root}`",
            f"- checks: {len(audit)}",
            f"- failed: {int((audit['passed'] != True).sum())}",
            "",
            audit.to_markdown(index=False),
            "",
        ]
    )
    (report or output_root / "artifact_audit_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(audit["passed"].all()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/place_market_offset_ev_threshold_phase6b_v1")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.output_root), Path(args.report) if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
