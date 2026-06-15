from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_v1 import (
    add_eval_columns,
    apply_calibration,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    bootstrap_ci,
    cat_indices,
    clip_prob,
    comparison,
    dataset_hash,
    ece,
    feature_columns,
    gpu_smoke,
    load_config,
    load_dataset,
    logit,
    make_market_predictions,
    metric_row,
    prepare_x,
    resolve_params,
    roi_of,
    sha256_file,
    sha256_json,
    sigmoid,
    strategy_roi,
    target_frame,
    threshold_roi,
    top_removed_roi,
    train_residual,
)


MODEL_KEY = "C1R0_pure_market_offset"
FORBIDDEN_EXACT = {
    "Year",
    "fold",
    "fold_id",
    "split",
    "split_name",
    "dataset_index",
    "row_id",
    "p_market",
    "market_logit",
    "tan_odds",
    "tan_ninki",
    "fuku_odds_low",
    "fuku_odds_high",
    "fuku_ninki",
    "market_rank",
    "p_market_rank",
    "rank_gap",
    "race_id",
    "entry_id",
    "file_name",
    "KettoNum",
    "horse_id",
    "Bamei",
    "finish_position",
    "KakuteiJyuni",
    "payoff",
    "payout",
    "place_payout",
    "win_payout",
    "result",
    "target",
    "label",
    "target_place_paid",
    "fuku_pay",
    "tan_pay",
    "place_bet_available_by_rule",
}
FORBIDDEN_TOKENS = (
    "odds",
    "ninki",
    "market",
    "vote",
    "betting",
    "implied_probability",
    "inverse_odds",
    "odds_ratio",
    "race_odds_ratio",
    "odds_width",
    "odds_rank",
    "popularity_rank",
    "payout",
    "payoff",
)


def git_info() -> dict[str, Any]:
    try:
        return {
            "git_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
            "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).splitlines(),
        }
    except Exception as exc:
        return {"git_error": str(exc)}


def c1r0_exclusion_reason(feature: str) -> tuple[bool, str, str]:
    low = feature.lower()
    if feature == "market_logit":
        return False, "Pool baseline only; not passed to CatBoost residual features.", "baseline_only"
    if feature in {"p_market"}:
        return False, "Market probability retained for audit/evaluation only.", "market_forbidden"
    if feature in FORBIDDEN_EXACT:
        if feature in {"Year", "fold", "fold_id", "split", "split_name", "dataset_index", "row_id"}:
            return False, "Time/split management column.", "time_management_only"
        if feature in {"race_id", "entry_id", "file_name", "KettoNum", "horse_id", "Bamei"}:
            return False, "High-cardinality identity/management column.", "id_forbidden"
        if feature in {"target_place_paid", "finish_position", "KakuteiJyuni", "payoff", "payout", "place_payout", "win_payout", "result", "target", "label", "fuku_pay", "tan_pay"}:
            return False, "Outcome, payout, or post-race leakage column.", "result_leakage_forbidden"
        return False, "Market/raw odds/betting-derived column.", "market_forbidden"
    if any(token in low for token in FORBIDDEN_TOKENS):
        return False, "Name matches market/odds/popularity/vote/betting/payout token.", "market_forbidden"
    return True, "Allowed existing non-market racing feature.", "allowed_fundamental"


def build_c1r0_features(cfg: dict[str, Any], dataset_columns: set[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    numeric, cat = feature_columns(cfg)
    c1_features = list(dict.fromkeys(numeric + cat + ["p_market", "market_logit"]))
    all_seen = sorted(set(c1_features) | set(FORBIDDEN_EXACT) | set(cfg["market_baseline"]["features"]))
    allowed_num = []
    allowed_cat = []
    rows = []
    for feature in all_seen:
        included, reason, category = c1r0_exclusion_reason(feature)
        present = feature in dataset_columns or feature in {"p_market", "market_logit", "market_rank", "p_market_rank", "rank_gap"}
        in_c1 = feature in c1_features
        if included and feature in numeric:
            allowed_num.append(feature)
        elif included and feature in cat:
            allowed_cat.append(feature)
        else:
            included = False
            if category == "allowed_fundamental" and feature not in numeric and feature not in cat:
                category = "not_present" if not present else "not_in_market_free_set"
                reason = "Not part of the C1 market_free feature set."
        rows.append({
            "feature": feature,
            "present_in_dataset": bool(present),
            "present_in_c1": bool(in_c1),
            "included_in_c1r0": bool(included),
            "reason": reason,
            "category": category,
        })
    return list(dict.fromkeys(allowed_num)), list(dict.fromkeys(allowed_cat)), pd.DataFrame(rows)


def predict_with_parts(model: CatBoostClassifier, df: pd.DataFrame, numeric: list[str], cat: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = prepare_x(df, numeric, cat)
    cats = cat_indices(x, cat)
    pool_with_baseline = Pool(x, cat_features=cats, baseline=df["market_logit"].to_numpy(float))
    raw_with_baseline = np.asarray(model.predict(pool_with_baseline, prediction_type="RawFormulaVal"), dtype=float)
    residual_raw = np.asarray(model.predict(Pool(x, cat_features=cats), prediction_type="RawFormulaVal"), dtype=float)
    return raw_with_baseline, residual_raw, clip_prob(sigmoid(raw_with_baseline), 1e-6)


def train_c1r0(market_pred: pd.DataFrame, cfg: dict[str, Any], params: dict[str, Any], numeric: list[str], cat: list[str], model_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parts = []
    fold_rows = []
    consistency = {}
    eps = float(cfg["epsilon"])
    for fold in cfg["folds"]:
        scoped = market_pred[market_pred["baseline_scope"] == fold["name"]]
        train = scoped[scoped["Year"].isin(fold["train_years"])]
        valid = scoped[scoped["Year"] == fold["validation_year"]].copy()
        model, meta = train_residual(train, valid, numeric, cat, params, model_root / "C1R0" / "folds" / fold["name"] / "model.cbm")
        raw, residual_raw, p = predict_with_parts(model, valid, numeric, cat)
        valid["probability"] = p
        valid["final_probability_raw"] = raw
        valid["catboost_residual_score"] = residual_raw
        valid["final_probability"] = p
        valid["model_key"] = MODEL_KEY
        parts.append(valid)
        fold_rows.append({**meta, **metric_row(valid, "probability", {"model_key": MODEL_KEY, "fold": fold["name"], "validation_year": fold["validation_year"]}, eps)})
        consistency[fold["name"]] = float(np.max(np.abs(raw - (valid["market_logit"].to_numpy(float) + residual_raw))))
    scoped = market_pred[market_pred["baseline_scope"] == "final"]
    train = scoped[scoped["Year"].isin(cfg["final_train_years"])]
    eval_df = scoped[scoped["Year"].isin([cfg["test_year"], cfg["latest_holdout_year"]])].copy()
    valid_tail = scoped[scoped["Year"] == cfg["final_train_years"][-1]]
    model, meta = train_residual(train, valid_tail, numeric, cat, params, model_root / "C1R0" / "final" / "model.cbm")
    raw, residual_raw, p = predict_with_parts(model, eval_df, numeric, cat)
    eval_df["probability"] = p
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual_raw
    eval_df["final_probability"] = p
    eval_df["model_key"] = MODEL_KEY
    parts.append(eval_df)
    fold_rows.append({**meta, "model_key": MODEL_KEY, "fold": "final", "validation_year": cfg["test_year"]})
    consistency["final"] = float(np.max(np.abs(raw - (eval_df["market_logit"].to_numpy(float) + residual_raw))))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(fold_rows), {"baseline_raw_consistency_max_abs": consistency}


def load_existing_b_c1(cfg: dict[str, Any]) -> pd.DataFrame:
    c1_out = Path(cfg["current_c1_output_dir"])
    parts = [
        pd.read_parquet(c1_out / "market_baseline_oof.parquet"),
        pd.read_parquet(c1_out / "residual_oof_predictions.parquet"),
        pd.read_parquet(c1_out / "final_predictions_2025.parquet"),
        pd.read_parquet(c1_out / "final_predictions_2026.parquet"),
    ]
    d = pd.concat(parts, ignore_index=True, sort=False)
    keep_models = {"B_market_baseline", "C1_market_offset_fundamental"}
    d = d[d["model_key"].isin(keep_models)].copy()
    if "probability" not in d.columns:
        d["probability"] = d["final_probability"]
    needed = [
        "entry_id",
        "race_id",
        "race_date",
        "Year",
        "period",
        "actual_place",
        "fuku_odds_low",
        "fuku_pay",
        "model_key",
        "probability",
        "final_probability",
        "p_market",
        "market_logit",
        "fold",
        "final_probability_raw",
        "catboost_residual_score",
    ]
    return d[[c for c in needed if c in d.columns]].copy()


def residual_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = pred[pred["model_key"].isin(["C1_market_offset_fundamental", MODEL_KEY])].copy()
    for keys, g in d.groupby(["model_key", "period", "Year"], dropna=False):
        r = pd.to_numeric(g["catboost_residual_score"], errors="coerce").dropna().to_numpy(float)
        if len(r) == 0:
            continue
        ar = np.abs(r)
        rows.append({
            "model_key": keys[0],
            "period": keys[1],
            "Year": int(keys[2]),
            "rows": int(len(r)),
            "residual_raw_mean": float(np.mean(r)),
            "residual_raw_std": float(np.std(r)),
            "residual_raw_min": float(np.min(r)),
            "residual_raw_p01": float(np.percentile(r, 1)),
            "residual_raw_p05": float(np.percentile(r, 5)),
            "residual_raw_p10": float(np.percentile(r, 10)),
            "residual_raw_p25": float(np.percentile(r, 25)),
            "residual_raw_p50": float(np.percentile(r, 50)),
            "residual_raw_p75": float(np.percentile(r, 75)),
            "residual_raw_p90": float(np.percentile(r, 90)),
            "residual_raw_p95": float(np.percentile(r, 95)),
            "residual_raw_p99": float(np.percentile(r, 99)),
            "residual_raw_max": float(np.max(r)),
            "abs_residual_raw_p50": float(np.percentile(ar, 50)),
            "abs_residual_raw_p90": float(np.percentile(ar, 90)),
            "abs_residual_raw_p95": float(np.percentile(ar, 95)),
            "abs_residual_raw_p99": float(np.percentile(ar, 99)),
        })
    return pd.DataFrame(rows)


def ev_crossing(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = add_eval_columns(pred, "final_probability")
    for keys, g in d.groupby(["model_key", "period", "Year"], dropna=False):
        market_ev = pd.to_numeric(g["p_market"], errors="coerce") * pd.to_numeric(g["fuku_odds_low"], errors="coerce") if "p_market" in g else np.nan
        final_ev = g["adjusted_place_ev"]
        market_ge = market_ev >= 1.0
        final_ge = final_ev >= 1.0
        rows.append({
            "model_key": keys[0],
            "period": keys[1],
            "Year": int(keys[2]),
            "rows": int(len(g)),
            "market_only_ev_ge_1": int(np.asarray(market_ge).sum()),
            "final_ev_ge_1": int(np.asarray(final_ge).sum()),
            "market_lt1_to_final_ge1": int((~np.asarray(market_ge) & np.asarray(final_ge)).sum()),
            "market_ge1_to_final_lt1": int((np.asarray(market_ge) & ~np.asarray(final_ge)).sum()),
            "ev_ge_1_rate": float(np.asarray(final_ge).mean()) if len(g) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ev_ge_1_year_over_year_ratio"] = out.sort_values("Year").groupby("model_key")["final_ev_ge_1"].pct_change() + 1.0
    return out


def high_payout_dependency(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dep_rows = []
    boot_rows = []
    d = add_eval_columns(pred, "final_probability")
    for keys, g in d.groupby(["model_key", "period"], dropna=False):
        bets = g[g["adjusted_place_ev"] >= 1.0].copy()
        row = {"model_key": keys[0], "period": keys[1], "bets": int(len(bets)), "normal_roi": roi_of(bets)}
        for n in [1, 3, 5, 10]:
            row[f"top{n}_payout_removed_roi"] = top_removed_roi(bets, n)
        dep_rows.append(row)
        ci = bootstrap_ci(bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
        boot_rows.append({"model_key": keys[0], "period": keys[1], "bets": int(len(bets)), "roi_p025": ci[0], "roi_p500": ci[1], "roi_p975": ci[2]})
    return pd.DataFrame(dep_rows), pd.DataFrame(boot_rows)


def feature_group(feature: str) -> str:
    if feature.startswith("horse_last") or feature in {"horse_days_since_last", "horse_distance_diff_last", "horse_futan_diff_last", "horse_body_weight_diff_last", "horse_past_starts"}:
        return "horse_recent_form"
    if feature.startswith(("horse_jyo_", "horse_surface_", "horse_dist_band_", "horse_baba_")):
        return "horse_course_suitability"
    if feature.startswith(("jockey_jyo_", "jockey_dist_band_")):
        return "jockey_course_suitability"
    if feature.startswith("jockey_"):
        return "jockey_overall"
    if feature.startswith("trainer_"):
        return "trainer"
    if feature.startswith("horse_jockey_"):
        return "horse_jockey_pair"
    if feature == "JyoCD":
        return "venue_identity"
    if feature in {"TrackCD", "CourseKubunCD", "SibaBabaCD", "DirtBabaCD", "TenkoCD"}:
        return "course_context"
    if feature == "Kyori":
        return "distance"
    if feature in {"Wakuban", "Umaban", "Futan", "BaTaijyu", "ZogenSa", "ZogenFugo", "Barei", "SexCD"}:
        return "weight_and_gate"
    if feature in {"MonthDay", "Kaiji", "Nichiji", "RaceNum", "TorokuTosu", "SyussoTosu", "place_rank_limit", "YoubiCD", "GradeCD", "SyubetuCD"} or feature.startswith("Jyoken"):
        return "race_metadata"
    return "other"


def feature_importance_tables(cfg: dict[str, Any], numeric: list[str], cat: list[str], pred: pd.DataFrame, out: Path, model_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pvc_rows = []
    lfc_rows = []
    features = numeric + cat
    for fold in cfg["folds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_root / "C1R0" / "folds" / fold["name"] / "model.cbm"))
        pvc = model.get_feature_importance(type="PredictionValuesChange")
        for f, v in zip(features, pvc):
            pvc_rows.append({"fold": fold["name"], "Year": fold["validation_year"], "feature": f, "group": feature_group(f), "importance": float(v)})
        valid = pred[(pred["model_key"] == MODEL_KEY) & (pred["Year"] == fold["validation_year"])]
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        lfc = model.get_feature_importance(data=pool, type="LossFunctionChange")
        for f, v in zip(features, lfc):
            lfc_rows.append({"fold": fold["name"], "Year": fold["validation_year"], "feature": f, "group": feature_group(f), "importance": float(v)})
    pvc_df = pd.DataFrame(pvc_rows)
    lfc_df = pd.DataFrame(lfc_rows)
    for name, df in [("catboost_pvc_by_fold.csv", pvc_df), ("catboost_lfc_by_fold.csv", lfc_df)]:
        atomic_write_csv(out / name, df)
    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["feature", "group"], as_index=False).agg(
            weighted_mean=("importance", "mean"),
            unweighted_mean=("importance", "mean"),
            median=("importance", "median"),
            min=("importance", "min"),
            max=("importance", "max"),
            std=("importance", "std"),
            fold_count=("importance", "count"),
        ).sort_values("weighted_mean", ascending=False)
    return summarize(pvc_df), summarize(lfc_df)


def shap_tables(cfg: dict[str, Any], numeric: list[str], cat: list[str], pred: pd.DataFrame, out: Path, model_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(cfg["random_seed"]))
    features = numeric + cat
    rows = []
    add_rows = []
    for fold in cfg["folds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_root / "C1R0" / "folds" / fold["name"] / "model.cbm"))
        valid = pred[(pred["model_key"] == MODEL_KEY) & (pred["Year"] == fold["validation_year"])].copy()
        if len(valid) > int(cfg["shap_sample_per_year"]):
            valid = valid.iloc[rng.choice(len(valid), int(cfg["shap_sample_per_year"]), replace=False)].copy()
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        shap = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)
        vals = shap[:, :-1]
        expected = shap[:, -1]
        residual = pd.to_numeric(valid["catboost_residual_score"], errors="coerce").to_numpy(float)
        final_raw = pd.to_numeric(valid["final_probability_raw"], errors="coerce").to_numpy(float)
        add_rows.append({
            "fold": fold["name"],
            "Year": fold["validation_year"],
            "rows": int(len(valid)),
            "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1))))),
            "final_logit_additivity_max_abs": float(np.max(np.abs(final_raw - (valid["market_logit"].to_numpy(float) + expected + vals.sum(axis=1))))),
        })
        for idx, f in enumerate(features):
            v = vals[:, idx]
            rows.append({
                "Year": fold["validation_year"],
                "feature": f,
                "group": feature_group(f),
                "mean_abs_shap": float(np.mean(np.abs(v))),
                "mean_signed_shap": float(np.mean(v)),
                "median_abs_shap": float(np.median(np.abs(v))),
                "p90_abs_shap": float(np.percentile(np.abs(v), 90)),
                "p99_abs_shap": float(np.percentile(np.abs(v), 99)),
                "positive_share": float((v > 0).mean()),
                "sample_rows": int(len(valid)),
            })
    by_year = pd.DataFrame(rows)
    global_df = by_year.groupby(["feature", "group"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        median_abs_shap=("median_abs_shap", "mean"),
        p90_abs_shap=("p90_abs_shap", "mean"),
        p99_abs_shap=("p99_abs_shap", "mean"),
        positive_share=("positive_share", "mean"),
        sample_rows=("sample_rows", "sum"),
    ).sort_values("mean_abs_shap", ascending=False)
    add_df = pd.DataFrame(add_rows)
    atomic_write_csv(out / "shap_additivity_check.csv", add_df)
    return global_df, by_year, add_df


def shap_for_subset(cfg: dict[str, Any], numeric: list[str], cat: list[str], pred: pd.DataFrame, model_root: Path, subset: str) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["random_seed"]) + len(subset))
    features = numeric + cat
    rows = []
    for fold in cfg["folds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_root / "C1R0" / "folds" / fold["name"] / "model.cbm"))
        valid = pred[(pred["model_key"] == MODEL_KEY) & (pred["Year"] == fold["validation_year"])].copy()
        jyo = valid["JyoCD"].astype(str).str.zfill(2)
        track = pd.to_numeric(valid["TrackCD"], errors="coerce")
        if subset == "nakayama":
            valid = valid[jyo.eq("06")]
        elif subset == "nakayama_turf":
            valid = valid[jyo.eq("06") & track.between(10, 22, inclusive="both")]
        elif subset == "nakayama_dirt":
            valid = valid[jyo.eq("06") & track.between(23, 29, inclusive="both")]
        else:
            raise ValueError(subset)
        if valid.empty:
            continue
        sample_limit = int(cfg["shap_sample_per_year"])
        if len(valid) > sample_limit:
            valid = valid.iloc[rng.choice(len(valid), sample_limit, replace=False)].copy()
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        shap = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)[:, :-1]
        for idx, f in enumerate(features):
            v = shap[:, idx]
            rows.append({
                "subset": subset,
                "Year": fold["validation_year"],
                "feature": f,
                "group": feature_group(f),
                "mean_abs_shap": float(np.mean(np.abs(v))),
                "mean_signed_shap": float(np.mean(v)),
                "median_abs_shap": float(np.median(np.abs(v))),
                "p90_abs_shap": float(np.percentile(np.abs(v), 90)),
                "p99_abs_shap": float(np.percentile(np.abs(v), 99)),
                "positive_share": float((v > 0).mean()),
                "sample_rows": int(len(valid)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["subset", "feature", "group"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        median_abs_shap=("median_abs_shap", "mean"),
        p90_abs_shap=("p90_abs_shap", "mean"),
        p99_abs_shap=("p99_abs_shap", "mean"),
        positive_share=("positive_share", "mean"),
        sample_rows=("sample_rows", "sum"),
    ).sort_values("mean_abs_shap", ascending=False)


def load_dataset_columns(cfg: dict[str, Any]) -> set[str]:
    p = Path(cfg["input_dataset_dir"]) / "year=2020" / "data.parquet"
    return set(pd.read_parquet(p).columns)


def write_report(path: Path, manifest: dict[str, Any], comp_2020: pd.DataFrame, comp_diag: pd.DataFrame, pvc: pd.DataFrame, shap_global: pd.DataFrame, ev_cross: pd.DataFrame) -> None:
    c1r0_val = comp_2020[comp_2020["model_key"] == MODEL_KEY]
    lines = [
        "# C1R0 Pure Market Offset Results",
        "",
        f"- Model: `{MODEL_KEY}`",
        f"- DB read: `{manifest['db_usage']}`",
        f"- Feature dataset rebuild: `not_performed`",
        f"- Selection years: `2020-2024 only`",
        f"- 2025/2026: `fixed diagnostic only`",
        f"- C1R0 feature count: `{manifest['feature_counts']['C1R0_total']}`",
        f"- C1 feature count: `{manifest['feature_counts']['C1_total']}`",
        f"- Baseline raw consistency max abs: `{manifest['catboost_baseline_checks']['max_abs_all']}`",
        "",
        "## 2020-2024 Model Comparison",
        comp_2020.to_markdown(index=False) if not comp_2020.empty else "(none)",
        "",
        "## 2025/2026 Fixed Diagnostic",
        comp_diag.to_markdown(index=False) if not comp_diag.empty else "(none)",
        "",
        "## C1R0 Top PVC",
        pvc.head(15).to_markdown(index=False) if not pvc.empty else "(none)",
        "",
        "## C1R0 Top SHAP",
        shap_global.head(15).to_markdown(index=False) if not shap_global.empty else "(none)",
        "",
        "## EV Crossing",
        ev_cross[ev_cross["model_key"].isin(["C1_market_offset_fundamental", MODEL_KEY])].to_markdown(index=False) if not ev_cross.empty else "(none)",
        "",
        "## Adoption Note",
    ]
    if not c1r0_val.empty:
        lines.append("C1R0 adoption judgment is based only on 2020-2024 metrics. ROI is not used as the sole criterion.")
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(config_path: Path, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["version"] += "_smoke"
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        cfg["bootstrap_iterations"] = 50
        cfg["shap_sample_per_year"] = 100
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    if out.exists() and any(out.iterdir()) and not force:
        raise RuntimeError(f"output already exists; pass --force to update C1R0 outputs only: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    params = resolve_params(cfg, smoke)
    gpu = gpu_smoke(params)
    dataset_columns = load_dataset_columns(cfg)
    numeric, cat, exclusion = build_c1r0_features(cfg, dataset_columns)
    allowlist = {
        "model_key": MODEL_KEY,
        "numeric": numeric,
        "categorical": cat,
        "all_features": numeric + cat,
        "baseline_only": ["market_logit"],
        "audit_only_market_probability": ["p_market"],
    }
    atomic_write_json(out / "feature_allowlist_c1r0.json", allowlist)
    atomic_write_csv(out / "feature_exclusion_c1r0.csv", exclusion)
    df = load_dataset(cfg, numeric, cat, smoke)
    tdf = target_frame(df, cfg)
    if int(tdf["Year"].min()) < 2016:
        raise RuntimeError("2015 or earlier rows detected")
    market_pred, market_meta = make_market_predictions(tdf, cfg, out, model_root, smoke)
    c1r0_pred, fold_metrics, residual_meta = train_c1r0(market_pred, cfg, params, numeric, cat, model_root)
    c1r0_pred["raw_prediction_with_baseline"] = c1r0_pred["final_probability_raw"]
    existing = load_existing_b_c1(cfg)
    base_keep = ["entry_id", "race_id", "race_date", "Year", "period", "actual_place", "fuku_odds_low", "fuku_pay", "model_key", "probability", "final_probability", "p_market", "market_logit", "fold", "final_probability_raw", "catboost_residual_score"]
    c1r0_eval = c1r0_pred[[c for c in base_keep if c in c1r0_pred.columns]].copy()
    pred = pd.concat([existing, c1r0_eval], ignore_index=True, sort=False)
    pred, cal_metrics, selected_cal = apply_calibration(pred, cfg)
    metrics_by_year = []
    for keys, g in pred.groupby(["model_key", "period", "Year"], dropna=False):
        metrics_by_year.append(metric_row(g, "final_probability", {"model_key": keys[0], "period": keys[1], "Year": int(keys[2])}, float(cfg["epsilon"])))
    metrics_by_year_df = pd.DataFrame(metrics_by_year)
    metrics_summary, model_comp = comparison(pred, "final_probability", cfg)
    model_comp_2020 = model_comp[model_comp["period"] == "validation_2020_2024"].copy()
    model_comp_diag = model_comp[model_comp["period"].isin(["test_2025", "latest_holdout_2026"])].copy()
    ev_roi = threshold_roi(pred, "final_probability")
    roi_comp = strategy_roi(pred, "final_probability", cfg)
    residual_dist = residual_distribution(pred)
    ev_cross = ev_crossing(pred)
    high_dep, boot = high_payout_dependency(pred, cfg)
    pvc, lfc = feature_importance_tables(cfg, numeric, cat, c1r0_pred, out, model_root)
    shap_global, shap_by_year, shap_add = shap_tables(cfg, numeric, cat, c1r0_pred, out, model_root)
    prediction_hash = atomic_write_parquet(out / "prediction_output.parquet", pred)
    hashes = {
        "prediction_output.parquet": prediction_hash,
        "model_comparison_2020_2024.csv": atomic_write_csv(out / "model_comparison_2020_2024.csv", model_comp_2020),
        "model_comparison_2025_2026_diagnostic.csv": atomic_write_csv(out / "model_comparison_2025_2026_diagnostic.csv", model_comp_diag),
        "metrics_by_year.csv": atomic_write_csv(out / "metrics_by_year.csv", metrics_by_year_df),
        "metrics_summary.csv": atomic_write_csv(out / "metrics_summary.csv", metrics_summary),
        "residual_distribution_by_year.csv": atomic_write_csv(out / "residual_distribution_by_year.csv", residual_dist),
        "ev_threshold_crossing_by_year.csv": atomic_write_csv(out / "ev_threshold_crossing_by_year.csv", ev_cross),
        "ev_roi_by_year.csv": atomic_write_csv(out / "ev_roi_by_year.csv", ev_roi),
        "roi_comparison.csv": atomic_write_csv(out / "roi_comparison.csv", roi_comp),
        "high_payout_dependency.csv": atomic_write_csv(out / "high_payout_dependency.csv", high_dep),
        "bootstrap_summary.csv": atomic_write_csv(out / "bootstrap_summary.csv", boot),
        "catboost_pvc_summary.csv": atomic_write_csv(out / "catboost_pvc_summary.csv", pvc),
        "catboost_lfc_summary.csv": atomic_write_csv(out / "catboost_lfc_summary.csv", lfc),
        "shap_global_2020_2024.csv": atomic_write_csv(out / "shap_global_2020_2024.csv", shap_global),
        "shap_by_year.csv": atomic_write_csv(out / "shap_by_year.csv", shap_by_year),
        "calibration_metrics.csv": atomic_write_csv(out / "calibration_metrics.csv", cal_metrics),
        "fold_metrics.csv": atomic_write_csv(out / "fold_metrics.csv", fold_metrics),
    }
    for subset_name in ["nakayama", "nakayama_turf", "nakayama_dirt"]:
        atomic_write_csv(out / f"shap_{subset_name}.csv", shap_for_subset(cfg, numeric, cat, c1r0_pred, model_root, subset_name))
    c1_manifest = json.loads((Path(cfg["current_c1_output_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    c1_features = c1_manifest.get("C1_features", [[], []])
    baseline_checks = residual_meta["baseline_raw_consistency_max_abs"]
    manifest = {
        "version": cfg["version"],
        "model_key": MODEL_KEY,
        "read_files": [
            "tasks/place_market_offset_catboost_c1r0_v1_task.md",
            "tasks/audit_place_market_offset_feature_importance_v1_task.md",
            "docs/place_market_offset_feature_audit_v1_results.md",
            "docs/place_market_offset_catboost_v1_design.md",
            "docs/place_market_offset_catboost_v1_results.md",
            "config/place_market_offset_catboost_v1.yaml",
            str(config_path),
        ],
        "missing_optional_files": ["keiba_ai_handover_market_offset_v1.md"] if not Path("keiba_ai_handover_market_offset_v1.md").exists() else [],
        "db_usage": "not_read; existing parquet feature dataset only",
        "feature_dataset_rebuild": False,
        "random_split_used": False,
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "diagnostic_only_years": [2025, 2026],
        "feature_counts": {
            "C1_total": len(c1_features[0]) + len(c1_features[1]) if isinstance(c1_features, list) and len(c1_features) == 2 else None,
            "C1R0_numeric": len(numeric),
            "C1R0_categorical": len(cat),
            "C1R0_total": len(numeric) + len(cat),
        },
        "allowlist_hash": sha256_json(allowlist),
        "input_feature_hash": dataset_hash(cfg),
        "catboost_params": params,
        "market_baseline_features": cfg["market_baseline"]["features"],
        "catboost_baseline_checks": {
            "by_fold": baseline_checks,
            "max_abs_all": max(baseline_checks.values()) if baseline_checks else None,
            "formula": "raw_prediction_with_baseline == market_logit + catboost_residual_score",
        },
        "calibration_by_model": selected_cal,
        "existing_c1_retrained": False,
        "c1r0_trained": True,
        "test_2025_used_for_selection": False,
        "latest_2026_used_for_selection": False,
        "gpu": gpu,
        "catboost_version": __import__("catboost").__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "output_hashes": hashes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    atomic_write_text(out / "run_log.txt", json.dumps({"elapsed_seconds": manifest["elapsed_seconds"], "status": "completed"}, indent=2))
    write_report(Path("docs/place_market_offset_catboost_c1r0_v1_results.md"), manifest, model_comp_2020, model_comp_diag, pvc, shap_global, ev_cross)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_v1.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), smoke=args.smoke_test, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
