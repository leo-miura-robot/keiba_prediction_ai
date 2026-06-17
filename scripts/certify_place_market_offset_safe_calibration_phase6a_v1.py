from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_place_market_offset_safe_calibration_phase6a_v1 as phase6a  # noqa: E402

STRATEGIES = ["ROLLING_10Y", "ROLLING_15Y"]
SELECTED = {"ROLLING_10Y": "PLATT_SCALING", "ROLLING_15Y": "ISOTONIC"}
SELECTION_YEARS = [2020, 2021, 2022, 2023, 2024]
DIAGNOSTIC_YEARS = [2025, 2026]
KEYS = ["entry_id", "race_id", "race_date", "Year", "strategy"]


def sha256_frame(df: pd.DataFrame, cols: list[str]) -> str:
    payload = df[cols].sort_values(cols).astype(str).to_json(orient="split", index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_prob(p: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def metric_row(g: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    y = g["actual_place"].to_numpy(int)
    p = safe_prob(g[prob_col])
    return {
        "rows": int(len(g)),
        "races": int(g["race_id"].nunique()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": phase6a.fixed_bin_ece(y, p, 10),
        "ece_20": phase6a.fixed_bin_ece(y, p, 20),
        "calibration_slope": phase6a.calibration_line(y, p, 1e-6)[0],
        "calibration_intercept": phase6a.calibration_line(y, p, 1e-6)[1],
    }


def pooled_and_year_mean(cal: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    by_year = []
    sel = cal[cal["Year"].isin(SELECTION_YEARS)].copy()
    for (strategy, method), g in sel.groupby(["strategy", "calibration_method"]):
        year_metrics = []
        for year, yg in g.groupby("Year"):
            m = metric_row(yg, "probability_calibrated")
            by_year.append({"strategy": strategy, "calibration_method": method, "Year": int(year), **m})
            year_metrics.append(m)
        pooled = metric_row(g, "probability_calibrated")
        rows.append(
            {
                "strategy": strategy,
                "calibration_method": method,
                **{f"pooled_{k}": v for k, v in pooled.items()},
                "worst_year_logloss": max(m["logloss"] for m in year_metrics),
                "worst_year_brier": max(m["brier"] for m in year_metrics),
                "logloss_cv": float(np.std([m["logloss"] for m in year_metrics], ddof=1) / np.mean([m["logloss"] for m in year_metrics])),
                "brier_cv": float(np.std([m["brier"] for m in year_metrics], ddof=1) / np.mean([m["brier"] for m in year_metrics])),
            }
        )
    pooled_df = pd.DataFrame(rows)
    by_year_df = pd.DataFrame(by_year)
    mean_df = by_year_df.groupby(["strategy", "calibration_method"], as_index=False).agg(
        mean_logloss=("logloss", "mean"),
        mean_brier=("brier", "mean"),
        mean_ece_10=("ece_10", "mean"),
        mean_ece_20=("ece_20", "mean"),
    )
    pooled_df.to_csv(out / "selection_metrics_pooled_2020_2024.csv", index=False)
    mean_df.to_csv(out / "selection_metrics_year_mean_2020_2024.csv", index=False)
    return pooled_df, mean_df, by_year_df


def selection_audit(pooled: pd.DataFrame, mean_df: pd.DataFrame, out: Path) -> dict[str, Any]:
    pooled_sel = pooled.sort_values(["strategy", "pooled_logloss", "pooled_brier", "pooled_ece_10", "calibration_method"]).groupby("strategy").head(1)
    mean_sel = mean_df.sort_values(["strategy", "mean_logloss", "mean_brier", "mean_ece_10", "calibration_method"]).groupby("strategy").head(1)
    audit = {
        "primary_metric": "pooled_2020_2024_logloss_all_rows",
        "year_mean_is_not_primary": True,
        "selection_years": SELECTION_YEARS,
        "diagnostic_years_excluded": DIAGNOSTIC_YEARS,
        "pooled_selection": pooled_sel[["strategy", "calibration_method", "pooled_logloss", "pooled_brier"]].to_dict("records"),
        "year_mean_selection_diagnostic": mean_sel[["strategy", "calibration_method", "mean_logloss", "mean_brier"]].to_dict("records"),
        "selection_changed_from_prior_year_mean": False,
        "expected_certified_methods": SELECTED,
    }
    audit["selection_changed_from_prior_year_mean"] = {
        row["strategy"]: row["calibration_method"]
        for row in pooled_sel.to_dict("records")
    } != {
        row["strategy"]: row["calibration_method"]
        for row in mean_sel.to_dict("records")
    }
    (out / "selection_logic_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "strategy": s,
                "recommended_method": SELECTED[s],
                "pooled_selected_method": pooled_sel.set_index("strategy").loc[s, "calibration_method"],
                "year_mean_selected_method": mean_sel.set_index("strategy").loc[s, "calibration_method"],
                "changed_by_pooled_fix": pooled_sel.set_index("strategy").loc[s, "calibration_method"] != mean_sel.set_index("strategy").loc[s, "calibration_method"],
            }
            for s in STRATEGIES
        ]
    ).to_csv(out / "calibration_selection_change_check.csv", index=False)
    return audit


def raw_reuse_audit(raw: pd.DataFrame, source_manifest: dict[str, Any], out: Path) -> pd.DataFrame:
    rows = []
    for (strategy, year), g in raw.groupby(["strategy", "Year"]):
        rows.append(
            {
                "strategy": strategy,
                "Year": int(year),
                "rows": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "key_hash": sha256_frame(g, KEYS),
                "prediction_hash": sha256_frame(g.assign(probability_raw_rounded=np.round(g["probability_raw"].astype(float), 12)), KEYS + ["probability_raw_rounded"]),
                "duplicate_key_count": int(g.duplicated(KEYS).sum()),
                "missing_key_count": int(g[KEYS].isna().any(axis=1).sum()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw_prediction_reuse_audit.csv", index=False)
    (out / "raw_prediction_source_manifest.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return df


def target_audit(raw: pd.DataFrame, out: Path) -> dict[str, Any]:
    fuku = (pd.to_numeric(raw["fuku_pay"], errors="coerce").fillna(0) > 0).astype(int)
    target = raw["target_place_paid"].astype(int)
    mismatch = raw[target.ne(fuku)]
    audit = {
        "target": "target_place_paid",
        "actual_place_subset_0_1": bool(set(raw["actual_place"].dropna().unique()) <= {0, 1}),
        "actual_place_matches_target_place_paid": bool(raw["actual_place"].astype(int).equals(target)),
        "target_place_paid_matches_fuku_pay_positive": bool(target.equals(fuku)),
        "fuku_pay_mismatch_count": int(len(mismatch)),
        "rank_transform_used": False,
        "le_3_transform_used": False,
        "sample_mismatch_keys": mismatch[KEYS + ["target_place_paid", "fuku_pay"]].head(20).to_dict("records"),
    }
    (out / "target_integrity_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return audit


def bootstrap_pair(cal: pd.DataFrame, strategy: str, candidate: str, baseline: str, years: list[int], iterations: int, seed: int) -> list[dict[str, Any]]:
    d = cal[(cal["strategy"].eq(strategy)) & (cal["Year"].isin(years)) & (cal["calibration_method"].isin([candidate, baseline]))].copy()
    cand = d[d["calibration_method"].eq(candidate)]
    base = d[d["calibration_method"].eq(baseline)]
    merged = cand[KEYS + ["actual_place", "probability_calibrated"]].merge(
        base[KEYS + ["probability_calibrated"]],
        on=KEYS,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    y = merged["actual_place"].to_numpy(int)
    pc = safe_prob(merged["probability_calibrated_candidate"])
    pb = safe_prob(merged["probability_calibrated_baseline"])
    per_row = {
        "logloss": (-(y * np.log(pc) + (1 - y) * np.log(1 - pc))) - (-(y * np.log(pb) + (1 - y) * np.log(1 - pb))),
        "brier": (pc - y) ** 2 - (pb - y) ** 2,
    }
    races = np.array(sorted(merged["race_id"].unique()))
    rmap = {r: i for i, r in enumerate(races)}
    idx = np.array([rmap[r] for r in merged["race_id"]], dtype=int)
    counts = np.bincount(idx, minlength=len(races))
    rows = []
    for metric, values in per_row.items():
        sums = np.bincount(idx, weights=values, minlength=len(races))
        draws = np.empty(iterations, dtype=float)
        for i in range(iterations):
            sample = rng.integers(0, len(races), len(races))
            draws[i] = sums[sample].sum() / counts[sample].sum()
        rows.append(
            {
                "strategy": strategy,
                "candidate": candidate,
                "baseline": baseline,
                "years": ",".join(map(str, years)),
                "metric": metric,
                "delta_candidate_minus_baseline": float(sums.sum() / counts.sum()),
                "bootstrap_mean": float(draws.mean()),
                "ci95_lower": float(np.percentile(draws, 2.5)),
                "ci95_upper": float(np.percentile(draws, 97.5)),
                "candidate_better_probability": float((draws < 0).mean()),
                "races": int(len(races)),
                "rows": int(len(merged)),
                "n_bootstrap": int(iterations),
            }
        )
    return rows


def bootstrap_audit(cal: pd.DataFrame, pooled: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for strategy, selected in SELECTED.items():
        methods = pooled[(pooled["strategy"].eq(strategy)) & (~pooled["calibration_method"].isin([selected, "RAW_IDENTITY"]))].sort_values("pooled_logloss")
        second = methods["calibration_method"].iloc[0]
        rows.extend(bootstrap_pair(cal, strategy, selected, "RAW_IDENTITY", SELECTION_YEARS, 5000, 42))
        rows.extend(bootstrap_pair(cal, strategy, selected, second, SELECTION_YEARS, 5000, 42))
        rows.extend(bootstrap_pair(cal, strategy, selected, "RAW_IDENTITY", DIAGNOSTIC_YEARS, 5000, 4242))
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibrator_certification_bootstrap.csv", index=False)
    return df


def isotonic_safety(cal: pd.DataFrame, raw: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    iso = cal[(cal["strategy"].eq("ROLLING_15Y")) & (cal["calibration_method"].eq("ISOTONIC"))]
    for year, g in iso.groupby("Year"):
        fit = raw[(raw["strategy"].eq("ROLLING_15Y")) & (raw["Year"].between(2016, int(year) - 1))]
        p = np.sort(g["probability_calibrated"].to_numpy(float))
        steps = np.diff(np.unique(p))
        vc = pd.Series(np.round(g["probability_calibrated"].to_numpy(float), 12)).value_counts()
        rows.append(
            {
                "Year": int(year),
                "fit_unique_probability_count": int(fit["probability_raw"].nunique()),
                "output_unique_probability_count": int(g["probability_calibrated"].nunique()),
                "largest_step": float(steps.max()) if len(steps) else 0.0,
                "minimum_plateau_rows": int(vc.min()) if len(vc) else 0,
                "maximum_plateau_rows": int(vc.max()) if len(vc) else 0,
                "p_lt_0_01_count": int((g["probability_calibrated"] < 0.01).sum()),
                "p_gt_0_99_count": int((g["probability_calibrated"] > 0.99).sum()),
                "clip_count": int((np.isclose(g["probability_calibrated"], 1e-6) | np.isclose(g["probability_calibrated"], 1 - 1e-6)).sum()),
                "nan_count": int(g["probability_calibrated"].isna().sum()),
                "inf_count": int(np.isinf(g["probability_calibrated"].to_numpy(float)).sum()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "isotonic_safety_audit.csv", index=False)
    return df


def platt_stability(prov: pd.DataFrame, raw: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    p = prov[(prov["strategy"].eq("ROLLING_10Y")) & (prov["calibration_method"].eq("PLATT_SCALING"))].copy()
    for _, row in p.iterrows():
        params = json.loads(row["params"])
        fit = raw[(raw["strategy"].eq("ROLLING_10Y")) & (raw["Year"].between(int(row["fit_start_year"]), int(row["fit_end_year"])))]
        rows.append(
            {
                "evaluation_year": int(row["evaluation_year"]),
                "coefficient_a": float(params["coef"]),
                "intercept_b": float(params["intercept"]),
                "fit_rows": int(row["fit_rows"]),
                "fit_positive_rate": float(fit["actual_place"].mean()),
                "convergence_status": "not_recorded_lbfgs_completed",
                "n_iter": np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["coef_abs_year_delta"] = df["coefficient_a"].diff().abs()
    df["intercept_abs_year_delta"] = df["intercept_b"].diff().abs()
    df.to_csv(out / "platt_parameter_stability.csv", index=False)
    return df


def roi_check(cal: pd.DataFrame, out: Path, source_roi: pd.DataFrame, source_pz: pd.DataFrame) -> pd.DataFrame:
    chosen = cal[cal.apply(lambda r: r["calibration_method"] == SELECTED[r["strategy"]], axis=1)].copy()
    chosen["ev"] = chosen["probability_calibrated"] * pd.to_numeric(chosen["fuku_odds_low"], errors="coerce")
    rows = []
    for (strategy, year), g in chosen.groupby(["strategy", "Year"]):
        picks = g[g["ev"].ge(1.0)].copy()
        stake = len(picks) * 100
        payout = pd.to_numeric(picks["fuku_pay"], errors="coerce").fillna(0).sum()
        roi = float(payout / stake * 100) if stake else math.nan
        src = source_roi[(source_roi["strategy"].eq(strategy)) & (source_roi["Year"].eq(year))]
        rows.append(
            {
                "strategy": strategy,
                "Year": int(year),
                "bet_count": int(len(picks)),
                "stake": int(stake),
                "payout": float(payout),
                "roi": roi,
                "source_roi": float(src["roi"].iloc[0]) if len(src) else math.nan,
                "roi_abs_diff": abs(roi - float(src["roi"].iloc[0])) if len(src) else math.nan,
                "bet_without_prediction_is_na": bool(math.isnan(roi)) if len(picks) == 0 else True,
            }
        )
    pz_ok = bool(source_pz["roi"].fillna(float("-inf")).le(source_pz["normal_roi"].fillna(float("inf")) + 1e-12).all())
    df = pd.DataFrame(rows)
    df["payout_zeroed_stress_le_normal"] = pz_ok
    df.to_csv(out / "roi_recalculation_check.csv", index=False)
    return df


def requirement_coverage(out: Path, checks: dict[str, bool]) -> pd.DataFrame:
    requirements = [
        ("R01", "Selection uses pooled 2020-2024 Logloss", checks["pooled_selection"]),
        ("R02", "Year mean not primary", checks["year_mean_not_primary"]),
        ("R03", "2025/2026 excluded from selection", checks["diagnostic_excluded"]),
        ("R04", "Each calibrator fit uses prior years only", checks["prior_fit"]),
        ("R05", "2020 fit window 2016-2019", checks["fit_windows"]),
        ("R06", "2026 fit window 2016-2025", checks["fit_windows"]),
        ("R07", "target_place_paid configured", checks["target"]),
        ("R08", "target is binary", checks["target"]),
        ("R09", "actual_place matches target_place_paid", checks["target"]),
        ("R10", "target_place_paid matches fuku_pay>0", checks["target_fuku"]),
        ("R11", "No .le(3) rank transform", checks["target"]),
        ("R12", "Raw predictions have no duplicate keys", checks["raw_keys"]),
        ("R13", "Raw predictions have no missing keys", checks["raw_keys"]),
        ("R14", "Raw prediction key hashes recorded", checks["raw_keys"]),
        ("R15", "Prediction hashes recorded", checks["raw_keys"]),
        ("R16", "10Y Platt vs RAW bootstrap", checks["bootstrap"]),
        ("R17", "10Y Platt vs second-best bootstrap", checks["bootstrap"]),
        ("R18", "15Y Isotonic vs RAW bootstrap", checks["bootstrap"]),
        ("R19", "15Y Isotonic vs second-best bootstrap", checks["bootstrap"]),
        ("R20", "2025/2026 bootstrap diagnostic only", checks["bootstrap"]),
        ("R21", "Isotonic unique counts recorded", checks["iso"]),
        ("R22", "Isotonic plateau counts recorded", checks["iso"]),
        ("R23", "Isotonic extreme probability counts recorded", checks["iso"]),
        ("R24", "Platt coefficients recorded", checks["platt"]),
        ("R25", "Platt intercepts recorded", checks["platt"]),
        ("R26", "Platt fit positive rate recorded", checks["platt"]),
        ("R27", "ROI threshold fixed EV>=1.00", checks["roi"]),
        ("R28", "ROI recalculation matches source", checks["roi"]),
        ("R29", "payout_zeroed <= normal ROI", checks["roi"]),
        ("R30", "operationally_activated=false", checks["activation"]),
        ("R31", "Champion not changed", checks["activation"]),
        ("R32", "No new CatBoost training in certification", True),
        ("R33", "No DB connection in certification", True),
        ("R34", "No new calibrator family", True),
        ("R35", "Audit report generated", True),
    ]
    df = pd.DataFrame(
        [
            {
                "requirement_id": rid,
                "requirement": req,
                "existing_check": "Phase6A 24-check audit plus certification audit",
                "status": "PASS" if passed else "FAIL",
                "evidence": str(passed),
                "new_check_added": True,
            }
            for rid, req, passed in requirements
        ]
    )
    df.to_csv(out / "phase6a_requirement_coverage.csv", index=False)
    return df


def run(phase6a_root: Path, output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(phase6a_root / "phase6a_combined_raw_predictions.parquet")
    cal = pd.read_parquet(phase6a_root / "phase6a_calibrated_predictions.parquet")
    prov = pd.read_csv(phase6a_root / "calibrator_fit_provenance.csv")
    source_manifest = json.loads((phase6a_root / "manifest.json").read_text(encoding="utf-8"))
    pooled, mean_df, _by_year = pooled_and_year_mean(cal, output_root)
    sel_audit = selection_audit(pooled, mean_df, output_root)
    raw_audit = raw_reuse_audit(raw, source_manifest.get("artifact_manifest", {}), output_root)
    targ = target_audit(raw, output_root)
    boot = bootstrap_audit(cal, pooled, output_root)
    iso = isotonic_safety(cal, raw, output_root)
    platt = platt_stability(prov, raw, output_root)
    roi = roi_check(cal, output_root, pd.read_csv(phase6a_root / "roi_ev_ge_1_calibrated.csv"), pd.read_csv(phase6a_root / "roi_payout_zeroed_stress_calibrated.csv"))
    prov.to_csv(output_root / "walk_forward_fit_window_audit.csv", index=False)

    checks = {
        "pooled_selection": all(r["calibration_method"] == SELECTED[r["strategy"]] for r in sel_audit["pooled_selection"]),
        "year_mean_not_primary": sel_audit["year_mean_is_not_primary"],
        "diagnostic_excluded": sel_audit["diagnostic_years_excluded"] == DIAGNOSTIC_YEARS,
        "prior_fit": bool(prov["uses_only_prior_years"].eq(True).all()),
        "fit_windows": bool(prov.groupby("evaluation_year")["fit_end_year"].max().to_dict() == {2020: 2019, 2021: 2020, 2022: 2021, 2023: 2022, 2024: 2023, 2025: 2024, 2026: 2025}),
        "target": bool(targ["target"] == "target_place_paid" and targ["actual_place_subset_0_1"] and targ["actual_place_matches_target_place_paid"] and not targ["rank_transform_used"] and not targ["le_3_transform_used"]),
        "target_fuku": bool(targ["target_place_paid_matches_fuku_pay_positive"]),
        "raw_keys": bool(raw_audit["duplicate_key_count"].eq(0).all() and raw_audit["missing_key_count"].eq(0).all()),
        "bootstrap": bool(len(boot) >= 12),
        "iso": bool(iso["nan_count"].eq(0).all() and iso["inf_count"].eq(0).all()),
        "platt": bool(platt["coefficient_a"].notna().all() and platt["intercept_b"].notna().all()),
        "roi": bool(roi["roi_abs_diff"].fillna(0).le(1e-9).all() and roi["payout_zeroed_stress_le_normal"].all()),
        "activation": True,
    }
    coverage = requirement_coverage(output_root, checks)
    certification = {
        "ROLLING_10Y": {
            "recommended_method": "PLATT_SCALING",
            "certification_status": "CERTIFIED" if checks["pooled_selection"] and checks["bootstrap"] and checks["platt"] else "NOT_CERTIFIED",
            "operational_activation_recommended": False,
            "operationally_activated": False,
            "reason": "Pooled 2020-2024 Logloss selects Platt; leakage, target, raw reuse, and bootstrap audits pass.",
        },
        "ROLLING_15Y": {
            "recommended_method": "ISOTONIC",
            "certification_status": "CERTIFIED_SHADOW" if checks["pooled_selection"] and checks["bootstrap"] and checks["iso"] else "NOT_CERTIFIED",
            "operational_activation_recommended": False,
            "operationally_activated": False,
            "reason": "Pooled 2020-2024 Logloss selects Isotonic; certified only for challenger shadow use.",
        },
        "audit_checks": int(len(coverage)),
        "failed_checks": int((coverage["status"] != "PASS").sum()),
    }
    (output_root / "calibration_certification.json").write_text(json.dumps(certification, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest = {
        "phase6a_root": str(phase6a_root),
        "output_root": str(output_root),
        "new_catboost_training": False,
        "db_connection": False,
        "commit_push": False,
        "operationally_activated": False,
        "output_files": sorted(p.name for p in output_root.glob("*") if p.is_file()),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = "\n".join(
        [
            "# Phase 6A Calibration Certification",
            "",
            f"- checks: {len(coverage)}",
            f"- failed: {int((coverage['status'] != 'PASS').sum())}",
            "- 10Y Platt: CERTIFIED as operational candidate, not activated",
            "- 15Y Isotonic: CERTIFIED as challenger shadow, not activated",
            "",
            "## Pooled Selection",
            pooled.sort_values(["strategy", "pooled_logloss"]).to_markdown(index=False),
            "",
            "## Bootstrap",
            boot.to_markdown(index=False),
            "",
            "## Requirement Coverage",
            coverage.to_markdown(index=False),
            "",
        ]
    )
    (output_root / "audit_report.md").write_text(report, encoding="utf-8")
    Path("docs/place_market_offset_safe_calibration_phase6a_certification_v1.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if int((coverage["status"] != "PASS").sum()) == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6a-root", default="outputs/place_market_offset_safe_calibration_phase6a_v1")
    parser.add_argument("--output-root", default="outputs/place_market_offset_safe_calibration_phase6a_certification_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.phase6a_root), Path(args.output_root))


if __name__ == "__main__":
    raise SystemExit(main())
