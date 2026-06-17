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


REQUIRED_FILES = [
    "phase5c_predictions.parquet",
    "metrics_2025_2026_by_strategy.csv",
    "metrics_2025_2026_combined.csv",
    "direct_pairwise_bootstrap.csv",
    "probability_agreement.csv",
    "ranking_agreement.csv",
    "error_win_loss.csv",
    "roi_by_strategy_year.csv",
    "roi_combined.csv",
    "bet_overlap.csv",
    "roi_row_removed.csv",
    "roi_payout_zeroed_stress.csv",
    "segment_comparison.csv",
    "champion_challenger_policy.json",
    "forward_prediction_schema.json",
    "manifest.json",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def checks(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        path = output_root / name
        rows.append({"check": f"file:{name}", "passed": path.exists(), "detail": str(path)})

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {
                    "check": "champion_not_changed",
                    "passed": manifest.get("champion_strategy") == "ROLLING_10Y" and manifest.get("champion_changed") is False,
                    "detail": str({k: manifest.get(k) for k in ["champion_strategy", "champion_changed"]}),
                },
                {
                    "check": "probability_raw_only",
                    "passed": manifest.get("probability_column") == "probability_raw" and manifest.get("calibration_fit") is False,
                    "detail": str({k: manifest.get(k) for k in ["probability_column", "calibration_fit"]}),
                },
                {
                    "check": "catboost_safety",
                    "passed": manifest.get("catboost_safety", {}) == {
                        "iterations": 300,
                        "use_best_model": False,
                        "early_stopping_enabled": False,
                        "outer_validation_eval_set_used": False,
                    },
                    "detail": str(manifest.get("catboost_safety", {})),
                },
            ]
        )

    pred_path = output_root / "phase5c_predictions.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        rows.extend(
            [
                {
                    "check": "only_2025_2026",
                    "passed": set(pred["Year"].unique()) == {2025, 2026},
                    "detail": str(sorted(pred["Year"].unique())),
                },
                {
                    "check": "only_champion_challenger",
                    "passed": set(pred["strategy"].unique()) == {"ROLLING_10Y", "ROLLING_15Y"},
                    "detail": str(sorted(pred["strategy"].unique())),
                },
                {
                    "check": "no_calibrated_probability",
                    "passed": "probability_calibrated" not in pred.columns,
                    "detail": "columns checked",
                },
                {
                    "check": "tree_count_300",
                    "passed": bool(pred["tree_count"].eq(300).all()),
                    "detail": str(pred["tree_count"].value_counts().to_dict()),
                },
            ]
        )

    folds = load_csv(output_root / "walk_forward_folds.csv")
    if not folds.empty:
        expected = {
            ("ROLLING_10Y", 2025): "2015,2016,2017,2018,2019,2020,2021,2022,2023,2024",
            ("ROLLING_10Y", 2026): "2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
            ("ROLLING_15Y", 2025): "2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024",
            ("ROLLING_15Y", 2026): "2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
        }
        ok = True
        details = []
        for key, years in expected.items():
            row = folds[(folds["strategy"].eq(key[0])) & (folds["validation_year"].eq(key[1]))]
            got = row["train_years"].iloc[0] if len(row) else None
            ok = ok and got == years
            details.append(f"{key}:{got}")
        rows.append({"check": "expected_train_windows", "passed": bool(ok), "detail": "; ".join(details)})

    pz = load_csv(output_root / "roi_payout_zeroed_stress.csv")
    if not pz.empty:
        ok = pz["roi"].fillna(float("-inf")).le(pz["normal_roi"].fillna(float("inf")) + 1e-12).all()
        rows.append({"check": "payout_zeroed_stress_le_normal", "passed": bool(ok), "detail": f"rows={len(pz)}"})

    return pd.DataFrame(rows)


def run(output_root: Path, report: Path | None = None) -> int:
    audit = checks(output_root)
    audit.to_csv(output_root / "phase5c_audit_checks.csv", index=False)
    text = "\n".join(
        [
            "# Phase 5C Artifact Audit",
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
    parser.add_argument("--output-root", default="outputs/place_market_offset_champion_challenger_phase5c_v1")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.output_root), Path(args.report) if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
