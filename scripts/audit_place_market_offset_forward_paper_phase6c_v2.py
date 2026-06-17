from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FILES = [
    "forward_paper.sqlite",
    "prediction_runs_export.parquet",
    "predictions_export.parquet",
    "prediction_tiers_export.parquet",
    "settlements_export.parquet",
    "daily_summary.csv",
    "monthly_summary.csv",
    "cumulative_summary.csv",
    "threshold_comparison.csv",
    "high_payout_stress.csv",
    "champion_challenger_overlap.csv",
    "manifest.json",
]


def db_path(root: Path) -> Path:
    return root / "forward_paper.sqlite"


def checks(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        rows.append({"check": f"file:{name}", "passed": (output_root / name).exists(), "detail": str(output_root / name)})
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {"check": "paper_trade_enabled", "passed": manifest.get("paper_trade_enabled") is True, "detail": str(manifest)},
                {"check": "real_money_disabled", "passed": manifest.get("real_money_betting") is False, "detail": str(manifest)},
                {"check": "operationally_false", "passed": manifest.get("operationally_activated") is False, "detail": str(manifest)},
            ]
        )
    if db_path(output_root).exists():
        con = sqlite3.connect(db_path(output_root))
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        for table in ["prediction_runs", "predictions", "prediction_tiers", "settlements"]:
            rows.append({"check": f"table:{table}", "passed": table in tables, "detail": ",".join(tables)})
        tiers = pd.read_sql_query("SELECT * FROM prediction_tiers", con)
        if not tiers.empty:
            rows.append({"check": "four_tiers", "passed": set(tiers["threshold_tier"]) == {"CORE", "MARGIN", "HIGH", "VERY_HIGH"}, "detail": str(sorted(tiers["threshold_tier"].unique()))})
            ok = True
            detail = []
            for (run_id, strategy), g in tiers.groupby(["prediction_run_id", "strategy"]):
                sets = {tier: set(g[g["threshold_tier"].eq(tier) & g["paper_bet_flag"].eq(1)]["entry_id"]) for tier in ["CORE", "MARGIN", "HIGH", "VERY_HIGH"]}
                local = sets["VERY_HIGH"].issubset(sets["HIGH"]) and sets["HIGH"].issubset(sets["MARGIN"]) and sets["MARGIN"].issubset(sets["CORE"])
                ok = ok and local
                detail.append(f"{run_id}:{strategy}:{local}")
            rows.append({"check": "tier_inclusion", "passed": ok, "detail": ";".join(detail)})
        preds = pd.read_sql_query("SELECT * FROM predictions", con)
        if not preds.empty:
            rows.append({"check": "prediction_columns_have_odds_snapshot", "passed": "fuku_odds_low_at_prediction" in preds.columns, "detail": ",".join(preds.columns)})
        settles = pd.read_sql_query("SELECT s.*, r.prediction_generated_at FROM settlements s JOIN prediction_runs r USING(prediction_run_id)", con)
        if not settles.empty:
            rows.append({"check": "settled_after_prediction", "passed": bool((pd.to_datetime(settles["settled_at"]) > pd.to_datetime(settles["prediction_generated_at"])).all()), "detail": f"rows={len(settles)}"})
    smoke = output_root / "fixture_smoke_audit.json"
    if smoke.exists():
        audit = json.loads(smoke.read_text(encoding="utf-8"))
        for key, value in audit.items():
            if isinstance(value, bool):
                rows.append({"check": f"fixture:{key}", "passed": value, "detail": str(value)})
        rows.append({"check": "fixture:settlement_append_only", "passed": audit.get("settlement_append_only_rows", 0) > 0, "detail": str(audit.get("settlement_append_only_rows"))})
    return pd.DataFrame(rows)


def run(output_root: Path, report: Path | None) -> int:
    audit = checks(output_root)
    audit.to_csv(output_root / "phase6c_audit_checks.csv", index=False)
    text = "\n".join(
        [
            "# Phase 6C v2 Artifact Audit",
            "",
            f"- output_root: `{output_root}`",
            f"- checks: {len(audit)}",
            f"- failed: {int((audit['passed'] != True).sum())}",
            "",
            audit.to_markdown(index=False),
            "",
        ]
    )
    (report or output_root / "audit_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(audit["passed"].all()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/place_market_offset_forward_paper_phase6c_v2")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.output_root), Path(args.report) if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
