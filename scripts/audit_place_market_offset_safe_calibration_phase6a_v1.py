from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FILES = [
    "phase6a_combined_raw_predictions.parquet",
    "phase6a_calibrated_predictions.parquet",
    "calibrator_fit_provenance.csv",
    "calibration_metrics_by_year.csv",
    "calibrator_comparison_2020_2024.csv",
    "calibrator_selection.csv",
    "diagnostic_2025_2026.csv",
    "raw_vs_calibrated_bootstrap.csv",
    "reliability_table.csv",
    "roi_ev_ge_1_calibrated.csv",
    "roi_row_removed_calibrated.csv",
    "roi_payout_zeroed_stress_calibrated.csv",
    "target_integrity_audit.json",
    "manifest.json",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def checks(out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        path = out / name
        rows.append({"check": f"file:{name}", "passed": path.exists(), "detail": str(path)})

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {
                    "check": "operationally_not_activated",
                    "passed": manifest.get("operationally_activated") is False and manifest.get("champion_changed") is False,
                    "detail": str({k: manifest.get(k) for k in ["operationally_activated", "champion_changed"]}),
                },
                {
                    "check": "no_2025_2026_selection",
                    "passed": manifest.get("calibrator_selection_uses_2025_2026") is False,
                    "detail": str(manifest.get("diagnostic_years")),
                },
                {
                    "check": "roi_not_selection",
                    "passed": manifest.get("roi_used_for_selection") is False and manifest.get("ev_threshold") == 1.0,
                    "detail": str({k: manifest.get(k) for k in ["roi_used_for_selection", "ev_threshold"]}),
                },
                {
                    "check": "target_no_rank_transform",
                    "passed": manifest.get("target_integrity_audit", {}).get("rank_transform_used") is False
                    and manifest.get("target_integrity_audit", {}).get("le_3_transform_used") is False,
                    "detail": str(manifest.get("target_integrity_audit", {})),
                },
            ]
        )

    prov = load_csv(out / "calibrator_fit_provenance.csv")
    if not prov.empty:
        rows.append(
            {
                "check": "fit_uses_only_prior_years",
                "passed": bool(prov["uses_only_prior_years"].eq(True).all()),
                "detail": f"rows={len(prov)}",
            }
        )
        expected = {
            2020: (2016, 2019),
            2021: (2016, 2020),
            2022: (2016, 2021),
            2023: (2016, 2022),
            2024: (2016, 2023),
            2025: (2016, 2024),
            2026: (2016, 2025),
        }
        ok = True
        details = []
        for year, (start, end) in expected.items():
            p = prov[prov["evaluation_year"].eq(year)]
            got = (int(p["fit_start_year"].min()), int(p["fit_end_year"].max())) if len(p) else None
            ok = ok and got == (start, end)
            details.append(f"{year}:{got}")
        rows.append({"check": "walk_forward_fit_windows", "passed": bool(ok), "detail": "; ".join(details)})

    pred_path = out / "phase6a_calibrated_predictions.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path, columns=["Year", "strategy", "calibration_method", "probability_calibrated"])
        rows.extend(
            [
                {
                    "check": "expected_years",
                    "passed": set(pred["Year"].unique()) == set(range(2020, 2027)),
                    "detail": str(sorted(pred["Year"].unique())),
                },
                {
                    "check": "expected_methods",
                    "passed": set(pred["calibration_method"].unique())
                    == {"RAW_IDENTITY", "TEMPERATURE_SCALING", "PLATT_SCALING", "ISOTONIC"},
                    "detail": str(sorted(pred["calibration_method"].unique())),
                },
                {
                    "check": "probability_bounds",
                    "passed": bool(pred["probability_calibrated"].between(0, 1).all()),
                    "detail": f"rows={len(pred)}",
                },
            ]
        )

    pz = load_csv(out / "roi_payout_zeroed_stress_calibrated.csv")
    if not pz.empty:
        rows.append(
            {
                "check": "payout_zeroed_stress_le_normal",
                "passed": bool(pz["roi"].fillna(float("-inf")).le(pz["normal_roi"].fillna(float("inf")) + 1e-12).all()),
                "detail": f"rows={len(pz)}",
            }
        )

    return pd.DataFrame(rows)


def run(output_root: Path, report: Path | None) -> int:
    audit = checks(output_root)
    audit.to_csv(output_root / "phase6a_audit_checks.csv", index=False)
    text = "\n".join(
        [
            "# Phase 6A Artifact Audit",
            "",
            f"- output_root: `{output_root}`",
            f"- checks: {len(audit)}",
            f"- failed: {int((~audit['passed']).sum())}",
            "",
            audit.to_markdown(index=False),
            "",
        ]
    )
    target = report or output_root / "audit_report.md"
    target.write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(audit["passed"].all()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/place_market_offset_safe_calibration_phase6a_v1")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.output_root), Path(args.report) if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
