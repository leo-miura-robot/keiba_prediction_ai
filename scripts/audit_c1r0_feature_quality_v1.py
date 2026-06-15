from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_c1r0_v1 import feature_group
from scripts.run_place_market_offset_catboost_v1 import atomic_write_csv, atomic_write_json


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_allowlist(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    allow = json.loads((Path(cfg["base_fixed300_output_root"]).parent / "place_market_offset_catboost_c1r0_v1" / "feature_allowlist_c1r0.json").read_text(encoding="utf-8"))
    return list(allow["numeric"]), list(allow["categorical"])


def psi_numeric(base: pd.Series, comp: pd.Series, bins: int = 10) -> float:
    b = pd.to_numeric(base, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    c = pd.to_numeric(comp, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(b) < 10 or len(c) < 10:
        return np.nan
    edges = np.unique(np.nanquantile(b, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    bh = np.histogram(b, bins=edges)[0] / len(b)
    ch = np.histogram(c, bins=edges)[0] / len(c)
    eps = 1e-6
    return float(np.sum((ch - bh) * np.log((ch + eps) / (bh + eps))))


def feature_semantics(features: list[str], numeric: list[str], cat: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in {"KisyuCode", "ChokyosiCode"}:
            source = "base entry/race table; categorical identity code"
            leak = "no direct result leak, but high-cardinality person identity can memorize stable skill and sparse IDs"
        elif f.endswith("_past_starts"):
            source = "src/features/history_builder_v2_1.py cumulative pre-day history state"
            leak = "pre-day only; raw cumulative count can proxy era/coverage and experience"
        elif f.endswith(("_win_rate", "_top3_rate", "_ren_rate", "_place_paid_rate")):
            source = "src/features/history_builder_v2_1.py cumulative or recent pre-day rates"
            leak = "pre-day only; rates are unsmoothed ratios with null for zero starts"
        elif f in {"horse_last3_avg_time", "horse_last5_avg_time", "horse_last3_avg_haron_l3", "horse_last5_avg_haron_l3"}:
            source = "src/features/history_builder_v2_1.py simple prior-race raw time averages"
            leak = "pre-day only; not normalized by distance, venue, surface, going, or pace"
        elif f == "BaTaijyu":
            source = "base race-entry body weight field"
            leak = "not post-result, but operational availability timing must be confirmed for production"
        elif f in {"Kaiji", "Nichiji", "RaceNum", "MonthDay", "YoubiCD"}:
            source = "race schedule/key metadata"
            leak = "available before race; can encode meeting/order/season management effects"
        else:
            source = "existing feature dataset / race card field"
            leak = "no direct result/payoff column in C1R0 allowlist"
        rows.append({
            "feature": f,
            "dtype_group": "numeric" if f in numeric else "categorical",
            "feature_group": feature_group(f),
            "meaning_and_generation": source,
            "leakage_audit_note": leak,
            "smoothing_note": "not smoothed" if f.endswith(("_win_rate", "_top3_rate", "_ren_rate", "_place_paid_rate")) else "",
        })
    return pd.DataFrame(rows)


def distribution_audit(pred: pd.DataFrame, features: list[str], numeric: list[str], cat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for f in features:
        s = pred[f] if f in pred.columns else pd.Series(dtype=float)
        for y, g in pred.groupby("Year"):
            gy = g[f] if f in g.columns else pd.Series(dtype=float)
            row = {
                "feature": f,
                "Year": int(y),
                "missing_rate": float(gy.isna().mean()) if len(gy) else np.nan,
                "unique_count": int(gy.nunique(dropna=True)) if len(gy) else 0,
                "unique_ratio": float(gy.nunique(dropna=True) / len(gy)) if len(gy) else np.nan,
            }
            if f in numeric:
                vals = pd.to_numeric(gy, errors="coerce")
                row.update({
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p01": float(vals.quantile(0.01)),
                    "p50": float(vals.quantile(0.50)),
                    "p99": float(vals.quantile(0.99)),
                })
            rows.append(row)
    by_year = pd.DataFrame(rows)
    shift = []
    base = pred[pred["Year"].between(2020, 2024)]
    comp = pred[pred["Year"].isin([2025, 2026])]
    for f in features:
        if f in numeric:
            corr = spearmanr(pd.to_numeric(pred[f], errors="coerce"), pred["Year"], nan_policy="omit").statistic
            psi_2425 = psi_numeric(pred[pred["Year"].eq(2024)][f], pred[pred["Year"].eq(2025)][f]) if 2025 in set(pred["Year"]) else np.nan
        else:
            codes = pred[f].astype("object").where(pred[f].notna(), "__MISSING__").astype(str)
            corr = np.nan
            psi_2425 = np.nan
        shift.append({
            "feature": f,
            "dtype_group": "numeric" if f in numeric else "categorical",
            "feature_group": feature_group(f),
            "year_spearman": float(corr) if corr is not None and not math.isnan(corr) else np.nan,
            "abs_year_spearman": float(abs(corr)) if corr is not None and not math.isnan(corr) else np.nan,
            "psi_2024_to_2025": psi_2425,
            "missing_rate_2020_2024": float(base[f].isna().mean()),
            "missing_rate_2025_2026": float(comp[f].isna().mean()),
            "unique_count_all": int(pred[f].nunique(dropna=True)),
            "unique_ratio_all": float(pred[f].nunique(dropna=True) / len(pred)),
        })
    shift_df = pd.DataFrame(shift)
    unknown_rows = []
    for f in cat:
        known = set(base[f].astype("object").where(base[f].notna(), "__MISSING__").astype(str).unique())
        for y, g in pred[pred["Year"].isin([2025, 2026])].groupby("Year"):
            vals = g[f].astype("object").where(g[f].notna(), "__MISSING__").astype(str)
            unknown_rows.append({
                "feature": f,
                "Year": int(y),
                "rows": int(len(vals)),
                "unique_count": int(vals.nunique()),
                "unknown_category_rate_vs_2020_2024": float((~vals.isin(known)).mean()),
                "unknown_category_count": int((~vals.isin(known)).sum()),
            })
    return by_year, shift_df, pd.DataFrame(unknown_rows)


def decision_table(cfg: dict[str, Any], semantics: pd.DataFrame, shift: pd.DataFrame, unknown: pd.DataFrame, shap: pd.DataFrame, pvc: pd.DataFrame) -> pd.DataFrame:
    candidates = cfg["ablation_candidates"]
    shap_rank = {f: i + 1 for i, f in enumerate(shap.sort_values("mean_abs_shap", ascending=False)["feature"].tolist())}
    pvc_rank = {f: i + 1 for i, f in enumerate(pvc.sort_values("weighted_mean", ascending=False)["feature"].tolist())}
    rows = []
    for name, c in candidates.items():
        feats = c["drop_features"]
        reason = []
        required = False
        if name == "drop_person_codes":
            u = unknown[unknown["feature"].isin(feats)]
            if not u.empty:
                reason.append(f"unknown category max={u['unknown_category_rate_vs_2020_2024'].max():.4f}")
            reason.append("high-cardinality identity categorical; KisyuCode/ChokyosiCode are top SHAP/PVC")
            required = True
        elif name == "drop_global_cumulative_starts":
            s = shift[shift["feature"].isin(feats)]
            reason.append("raw cumulative starts with no window/smoothing; trainer_past_starts is top SHAP/PVC")
            if s["abs_year_spearman"].max(skipna=True) >= cfg["audit"]["year_abs_spearman_threshold"]:
                reason.append(f"year proxy signal max_abs_spearman={s['abs_year_spearman'].max():.3f}")
            required = True
        elif name == "drop_raw_body_weight":
            reason.append("raw current-race body weight; production availability timing is operationally sensitive")
            required = True
        elif name == "drop_unadjusted_raw_time":
            reason.append("horse_last*_avg_time are simple raw averages, not distance/venue/going adjusted")
            required = True
        elif name == "drop_meeting_admin":
            reason.append("Kaiji/Nichiji/RaceNum are schedule/admin fields and can encode meeting/order effects")
            required = True
        rows.append({
            "ablation_name": name,
            "drop_features": ",".join(feats),
            "ablation_required": bool(required),
            "decision_reason": "; ".join(reason),
            "best_shap_rank_in_group": min([shap_rank.get(f, 9999) for f in feats]),
            "best_pvc_rank_in_group": min([pvc_rank.get(f, 9999) for f in feats]),
        })
    return pd.DataFrame(rows)


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    numeric, cat = load_allowlist(cfg)
    features = numeric + cat
    needed_cols = sorted(set(features + ["Year", "tree_count_candidate"]))
    oof = pd.read_parquet(Path(cfg["base_fixed300_output_root"]) / "fixed_tree_predictions_2020_2024.parquet", columns=needed_cols)
    oof = oof[oof["tree_count_candidate"].eq(int(cfg["fixed_tree_count"]))].copy()
    diag_cols = sorted(set(features + ["Year"]))
    diag = pd.read_parquet(Path(cfg["base_fixed300_output_root"]) / "selected_fixed_tree_predictions_2025_2026.parquet", columns=diag_cols)
    pred = pd.concat([oof, diag], ignore_index=True, sort=False)
    semantics = feature_semantics(features, numeric, cat)
    by_year, shift, unknown = distribution_audit(pred, features, numeric, cat)
    shap = pd.read_csv(Path(cfg["base_fixed300_output_root"]) / "selected_tree_shap_summary.csv")
    pvc = pd.read_csv(Path(cfg["base_fixed300_output_root"]) / "selected_tree_catboost_pvc_summary.csv")
    decisions = decision_table(cfg, semantics, shift, unknown, shap, pvc)
    hashes = {
        "feature_semantics_audit.csv": atomic_write_csv(out / "feature_semantics_audit.csv", semantics),
        "feature_distribution_by_year.csv": atomic_write_csv(out / "feature_distribution_by_year.csv", by_year),
        "feature_shift_summary.csv": atomic_write_csv(out / "feature_shift_summary.csv", shift),
        "categorical_unknown_rates.csv": atomic_write_csv(out / "categorical_unknown_rates.csv", unknown),
        "ablation_decision_table.csv": atomic_write_csv(out / "ablation_decision_table.csv", decisions),
    }
    manifest = {
        "version": cfg["version"],
        "stage": "feature_quality_audit",
        "feature_count": len(features),
        "numeric_count": len(numeric),
        "categorical_count": len(cat),
        "db_usage": "not_read; existing parquet predictions and feature dataset artifacts only",
        "feature_dataset_rebuild": False,
        "fixed_reference": "C1R0_pure_market_offset_fixed300",
        "ablation_required": decisions[decisions["ablation_required"]]["ablation_name"].tolist(),
        "output_hashes": hashes,
    }
    atomic_write_json(out / "feature_audit_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml")
    run(path)
