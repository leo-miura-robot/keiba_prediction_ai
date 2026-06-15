from __future__ import annotations

import argparse
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
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    bootstrap_ci,
    cat_indices,
    clip_prob,
    ece,
    load_config,
    load_dataset,
    make_market_predictions,
    metric_row,
    nested_calibration,
    prepare_x,
    roi_of,
    sha256_file,
    sha256_json,
    sigmoid,
    target_frame,
    top_removed_roi,
)
from scripts.run_place_market_offset_catboost_c1r0_v1 import feature_group


MODEL_KEY_PREFIX = "C1R0_fixed_tree"
CURRENT_MODEL_KEY = "C1R0_pure_market_offset"


def git_info() -> dict[str, Any]:
    try:
        return {
            "git_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
            "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).splitlines(),
        }
    except Exception as exc:
        return {"git_error": str(exc)}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fixed_params(base_params: dict[str, Any], tree_count: int, smoke: bool = False, gpu_ram_part: float | None = None) -> dict[str, Any]:
    params = dict(base_params)
    params["iterations"] = int(tree_count)
    params["verbose"] = False if smoke else base_params.get("verbose", 100)
    if gpu_ram_part is not None:
        params["gpu_ram_part"] = float(gpu_ram_part)
    params.pop("od_type", None)
    params.pop("od_wait", None)
    return params


def train_fixed_model(train: pd.DataFrame, numeric: list[str], cat: list[str], params: dict[str, Any], model_path: Path) -> CatBoostClassifier:
    x = prepare_x(train, numeric, cat)
    model = CatBoostClassifier(**params)
    model.fit(Pool(x, train["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=train["market_logit"].to_numpy(float)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return model


def predict_parts(model: CatBoostClassifier, df: pd.DataFrame, numeric: list[str], cat: list[str], ntree_end: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = prepare_x(df, numeric, cat)
    cats = cat_indices(x, cat)
    kwargs = {"prediction_type": "RawFormulaVal"}
    if ntree_end is not None:
        kwargs["ntree_end"] = int(ntree_end)
    raw = np.asarray(model.predict(Pool(x, cat_features=cats, baseline=df["market_logit"].to_numpy(float)), **kwargs), dtype=float)
    residual = np.asarray(model.predict(Pool(x, cat_features=cats), **kwargs), dtype=float)
    return raw, residual, clip_prob(sigmoid(raw), 1e-6)


def residual_stats(r: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
    r = np.asarray(r, dtype=float)
    ar = np.abs(r)
    return {
        **label,
        "residual_mean": float(np.mean(r)),
        "residual_std": float(np.std(r)),
        "abs_residual_p90": float(np.percentile(ar, 90)),
        "abs_residual_p95": float(np.percentile(ar, 95)),
        "abs_residual_p99": float(np.percentile(ar, 99)),
    }


def ev_spearman(df: pd.DataFrame) -> float:
    d = add_eval_columns(df, "final_probability")
    bins = [-np.inf, .85, .90, .95, 1.00, 1.02, 1.05, 1.10, np.inf]
    d["ev_band_order"] = pd.cut(d["adjusted_place_ev"], bins, labels=False, right=False)
    rows = []
    for k, g in d.groupby("ev_band_order", dropna=True):
        rows.append((int(k), roi_of(g)))
    if len(rows) < 2:
        return np.nan
    arr = np.asarray(rows, dtype=float)
    return float(spearmanr(arr[:, 0], arr[:, 1], nan_policy="omit").statistic)


def ev_row(df: pd.DataFrame, label: dict[str, Any]) -> dict[str, Any]:
    d = add_eval_columns(df, "final_probability")
    market_ev = pd.to_numeric(d["p_market"], errors="coerce") * pd.to_numeric(d["fuku_odds_low"], errors="coerce")
    final_ev = d["adjusted_place_ev"]
    market_ge = np.asarray(market_ev >= 1.0)
    final_ge = np.asarray(final_ev >= 1.0)
    return {
        **label,
        "rows": int(len(d)),
        "ev_ge_1_count": int(final_ge.sum()),
        "ev_ge_1_rate": float(final_ge.mean()) if len(d) else np.nan,
        "market_only_ev_ge_1_count": int(market_ge.sum()),
        "market_lt1_to_final_ge1": int((~market_ge & final_ge).sum()),
        "market_ge1_to_final_lt1": int((market_ge & ~final_ge).sum()),
        "ev_roi_spearman": ev_spearman(d),
    }


def roi_diag(df: pd.DataFrame, cfg: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    d = add_eval_columns(df, "final_probability")
    bets = d[d["adjusted_place_ev"] >= 1.0].copy()
    ci = bootstrap_ci(bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
    return {
        **label,
        "bets": int(len(bets)),
        "roi": roi_of(bets),
        "top1_payout_removed_roi": top_removed_roi(bets, 1),
        "top3_payout_removed_roi": top_removed_roi(bets, 3),
        "top5_payout_removed_roi": top_removed_roi(bets, 5),
        "top10_payout_removed_roi": top_removed_roi(bets, 10),
        "bootstrap_roi_p025": ci[0],
        "bootstrap_roi_p500": ci[1],
        "bootstrap_roi_p975": ci[2],
    }


def audit_saved_models(base_cfg: dict[str, Any], base_out: Path, base_model_root: Path, market_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    params = base_cfg["training_params"]
    fold_metrics = pd.read_csv(base_out / "fold_metrics.csv")
    for fold in base_cfg["folds"]:
        path = base_model_root / "C1R0" / "folds" / fold["name"] / "model.cbm"
        model = CatBoostClassifier()
        model.load_model(str(path))
        train_rows = len(market_pred[(market_pred["baseline_scope"] == fold["name"]) & (market_pred["Year"].isin(fold["train_years"]))])
        valid_rows = len(market_pred[(market_pred["baseline_scope"] == fold["name"]) & (market_pred["Year"] == fold["validation_year"])])
        fm = fold_metrics[fold_metrics["fold"].eq(fold["name"])].head(1)
        rows.append({
            "scope": fold["name"],
            "train_years": ",".join(map(str, fold["train_years"])),
            "eval_year": fold["validation_year"],
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "config_iterations": params["iterations"],
            "tree_count": int(model.tree_count_),
            "best_iteration": int(model.get_best_iteration() if model.get_best_iteration() is not None else (fm["best_iteration"].iloc[0] if not fm.empty else -1)),
            "eval_set_used": True,
            "use_best_model": True,
            "early_stopping_enabled": True,
            "od_type": params.get("od_type"),
            "od_wait": params.get("od_wait"),
            "learning_rate": params.get("learning_rate"),
            "depth": params.get("depth"),
            "random_seed": params.get("random_seed"),
            "model_sha256": sha256_file(path),
        })
    path = base_model_root / "C1R0" / "final" / "model.cbm"
    model = CatBoostClassifier()
    model.load_model(str(path))
    fm = fold_metrics[fold_metrics["fold"].eq("final")].head(1)
    final_train = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"].isin(base_cfg["final_train_years"]))]
    valid_tail = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"] == base_cfg["final_train_years"][-1])]
    for eval_year in [base_cfg["test_year"], base_cfg["latest_holdout_year"]]:
        rows.append({
            "scope": f"final_for_{eval_year}",
            "train_years": ",".join(map(str, base_cfg["final_train_years"])),
            "eval_year": eval_year,
            "train_rows": int(len(final_train)),
            "valid_rows": int(len(valid_tail)),
            "config_iterations": params["iterations"],
            "tree_count": int(model.tree_count_),
            "best_iteration": int(model.get_best_iteration() if model.get_best_iteration() is not None else (fm["best_iteration"].iloc[0] if not fm.empty else -1)),
            "eval_set_used": True,
            "eval_set_note": "2024 tail was also included in final training data in current C1R0 runner",
            "use_best_model": True,
            "early_stopping_enabled": True,
            "od_type": params.get("od_type"),
            "od_wait": params.get("od_wait"),
            "learning_rate": params.get("learning_rate"),
            "depth": params.get("depth"),
            "random_seed": params.get("random_seed"),
            "model_sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


def summarize_best_iterations(audit: pd.DataFrame) -> pd.DataFrame:
    vals = audit[audit["scope"].str.startswith("fold_")]["best_iteration"].to_numpy(float)
    median = float(np.median(vals))
    return pd.DataFrame([{
        "min": float(np.min(vals)),
        "p25": float(np.percentile(vals, 25)),
        "median": median,
        "mean": float(np.mean(vals)),
        "p75": float(np.percentile(vals, 75)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals, ddof=1)),
        "median_rounded_to_50": int(round(median / 50.0) * 50),
    }])


def run_fixed_candidates(base_cfg: dict[str, Any], cfg: dict[str, Any], market_pred: pd.DataFrame, numeric: list[str], cat: list[str], model_root: Path, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_parts = []
    fold_rows = []
    residual_rows = []
    ev_rows = []
    roi_rows = []
    gpu_ram_part = cfg.get("execution_constraints", {}).get("gpu_ram_part")
    for tree_count in cfg["candidate_tree_counts"]:
        params = fixed_params(base_cfg["training_params"], int(tree_count), smoke, gpu_ram_part=gpu_ram_part)
        for fold in base_cfg["folds"]:
            scoped = market_pred[market_pred["baseline_scope"] == fold["name"]]
            train = scoped[scoped["Year"].isin(fold["train_years"])]
            valid = scoped[scoped["Year"] == fold["validation_year"]].copy()
            model = train_fixed_model(train, numeric, cat, params, model_root / f"tree_{tree_count}" / "folds" / fold["name"] / "model.cbm")
            raw, residual, p = predict_parts(model, valid, numeric, cat)
            valid["model_key"] = f"{MODEL_KEY_PREFIX}_{tree_count}"
            valid["tree_count_candidate"] = int(tree_count)
            valid["final_probability_raw"] = raw
            valid["catboost_residual_score"] = residual
            valid["probability"] = p
            valid["final_probability"] = p
            label = {"tree_count": int(tree_count), "fold": fold["name"], "Year": int(fold["validation_year"]), "model_key": valid["model_key"].iloc[0], "tree_count_actual": int(model.tree_count_)}
            fold_rows.append({**label, **metric_row(valid, "final_probability", {}, float(base_cfg["epsilon"]))})
            residual_rows.append(residual_stats(residual, label))
            ev_rows.append(ev_row(valid, label))
            roi_rows.append(roi_diag(valid, cfg, label))
            pred_parts.append(valid)
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(residual_rows), pd.DataFrame(ev_rows), pd.DataFrame(roi_rows)


def calibrate_candidate_predictions(pred: pd.DataFrame, base_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    parts = []
    cal_rows = []
    selected = {}
    for model_key in sorted(pred["model_key"].unique()):
        sel, metrics, cal = nested_calibration(pred[pred["Year"].between(2020, 2024)], model_key, base_cfg)
        selected[model_key] = sel
        if not metrics.empty:
            cal_rows.append(metrics)
        d = pred[pred["model_key"].eq(model_key)].copy()
        d["final_probability"] = cal.transform(d["probability"].to_numpy(float))
        parts.append(d)
    return pd.concat(parts, ignore_index=True), pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame(), selected


def recompute_candidate_tables(pred: pd.DataFrame, residual: pd.DataFrame, cfg: dict[str, Any], base_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    ev_rows = []
    roi_rows = []
    for keys, g in pred.groupby(["tree_count_candidate", "fold", "Year", "model_key"], dropna=False):
        label = {"tree_count": int(keys[0]), "fold": keys[1], "Year": int(keys[2]), "model_key": keys[3], "tree_count_actual": int(keys[0])}
        fold_rows.append({**label, **metric_row(g, "final_probability", {}, float(base_cfg["epsilon"]))})
        ev_rows.append(ev_row(g, label))
        roi_rows.append(roi_diag(g, cfg, label))
    return pd.DataFrame(fold_rows), residual, pd.DataFrame(ev_rows), pd.DataFrame(roi_rows)


def summarize_candidates(by_fold: pd.DataFrame, residual: pd.DataFrame, ev: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tc, g in by_fold.groupby("tree_count"):
        r = residual[residual["tree_count"].eq(tc)]
        e = ev[ev["tree_count"].eq(tc)]
        ro = roi[roi["tree_count"].eq(tc)]
        rows.append({
            "tree_count": int(tc),
            "mean_logloss": float(g["logloss"].mean()),
            "mean_brier": float(g["brier"].mean()),
            "mean_ece": float(g["ece"].mean()),
            "mean_calibration_slope": float(g["calibration_slope"].mean()),
            "mean_calibration_intercept": float(g["calibration_intercept"].mean()),
            "residual_std_mean": float(r["residual_std"].mean()),
            "residual_std_cv": float(r["residual_std"].std(ddof=1) / r["residual_std"].mean()),
            "abs_residual_p95_mean": float(r["abs_residual_p95"].mean()),
            "abs_residual_p95_cv": float(r["abs_residual_p95"].std(ddof=1) / r["abs_residual_p95"].mean()),
            "ev_ge_1_count_sum": int(e["ev_ge_1_count"].sum()),
            "ev_ge_1_count_std": float(e["ev_ge_1_count"].std(ddof=1)),
            "ev_ge_1_count_cv": float(e["ev_ge_1_count"].std(ddof=1) / e["ev_ge_1_count"].mean()),
            "ev_roi_spearman_mean": float(e["ev_roi_spearman"].mean()),
            "ev_ge_1_roi": float(roi_of_from_rows(ro)),
            "mean_roi": float(ro["roi"].mean()),
        })
    out = pd.DataFrame(rows)
    out["selection_score"] = (
        out["mean_logloss"].rank(method="min")
        + out["mean_brier"].rank(method="min")
        + out["residual_std_cv"].rank(method="min")
        + out["abs_residual_p95_cv"].rank(method="min")
        + out["ev_ge_1_count_cv"].rank(method="min")
        + out["mean_ece"].rank(method="min")
        - out["ev_roi_spearman_mean"].rank(method="min") * 0.05
    )
    return out.sort_values(["selection_score", "mean_logloss"])


def roi_of_from_rows(ro: pd.DataFrame) -> float:
    # Candidate selection does not rely on this aggregate; keep mean ROI separately.
    return float(ro["roi"].mean()) if not ro.empty else np.nan


def select_tree_count(summary: pd.DataFrame) -> dict[str, Any]:
    s = summary.copy()
    best_logloss = float(s["mean_logloss"].min())
    best_brier = float(s["mean_brier"].min())
    # Primary probability metrics dominate. Treat very small differences as a tier,
    # then pick the candidate that improves stability without inflating EV volume.
    tier = s[(s["mean_logloss"] <= best_logloss + 0.00020) & (s["mean_brier"] <= best_brier + 0.00010)].copy()
    if tier.empty:
        tier = s.copy()
    tier["stability_score"] = (
        tier["residual_std_cv"].rank(method="min")
        + tier["abs_residual_p95_cv"].rank(method="min")
        + tier["ev_ge_1_count_cv"].rank(method="min")
        + tier["ev_ge_1_count_sum"].rank(method="min") * 0.5
        + tier["abs_residual_p95_mean"].rank(method="min") * 0.5
    )
    row = tier.sort_values(["stability_score", "mean_logloss", "mean_brier"]).iloc[0].to_dict()
    return {
        "selected_tree_count": int(row["tree_count"]),
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "selection_basis": "2020-2024 only: primary Logloss/Brier tier first, then residual p95/std stability, EV count stability and EV count inflation; ROI auxiliary only.",
        "selected_row": row,
    }


def validate_existing_artifacts(cfg: dict[str, Any], base_cfg: dict[str, Any], out: Path, model_root: Path) -> pd.DataFrame:
    rows = []
    pred_path = out / "fixed_tree_predictions_2020_2024.parquet"
    pred_rows = None
    pred_counts: dict[tuple[int, int], int] = {}
    if pred_path.exists():
        pred = pd.read_parquet(pred_path, columns=["tree_count_candidate", "Year"])
        pred_rows = int(len(pred))
        pred_counts = pred.groupby(["tree_count_candidate", "Year"]).size().to_dict()
    for tree_count in cfg["candidate_tree_counts"]:
        for fold in base_cfg["folds"]:
            path = model_root / f"tree_{tree_count}" / "folds" / fold["name"] / "model.cbm"
            status = "missing_model"
            tree_count_actual = np.nan
            model_hash = ""
            if path.exists():
                model = CatBoostClassifier()
                model.load_model(str(path))
                tree_count_actual = int(model.tree_count_)
                model_hash = sha256_file(path)
                status = "ok" if tree_count_actual == int(tree_count) else "tree_count_mismatch"
            expected_rows = int(pred_counts.get((int(tree_count), int(fold["validation_year"])), 0))
            final_status = status if expected_rows > 0 else f"missing_prediction_rows;{status}"
            rows.append({
                "tree_count": int(tree_count),
                "fold": fold["name"],
                "Year": int(fold["validation_year"]),
                "model_path": str(path),
                "model_exists": path.exists(),
                "tree_count_actual": tree_count_actual,
                "tree_count_matches": bool(tree_count_actual == int(tree_count)) if path.exists() else False,
                "model_sha256": model_hash,
                "prediction_parquet": str(pred_path),
                "prediction_parquet_exists": pred_path.exists(),
                "prediction_total_rows": pred_rows,
                "prediction_rows_for_fold": expected_rows,
                "prediction_rows_match": expected_rows > 0,
                "status": final_status,
            })
    return pd.DataFrame(rows)


def recompute_from_existing_predictions(pred: pd.DataFrame, cfg: dict[str, Any], base_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_rows = []
    for keys, g in pred.groupby(["tree_count_candidate", "fold", "Year", "model_key"], dropna=False):
        label = {"tree_count": int(keys[0]), "fold": keys[1], "Year": int(keys[2]), "model_key": keys[3], "tree_count_actual": int(keys[0])}
        residual_rows.append(residual_stats(g["catboost_residual_score"], label))
    residual = pd.DataFrame(residual_rows)
    by_fold, residual, ev, roi = recompute_candidate_tables(pred, residual, cfg, base_cfg)
    summary = summarize_candidates(by_fold, residual, ev, roi)
    return summary, by_fold, residual, ev


def predict_selected_final_existing(base_cfg: dict[str, Any], cfg: dict[str, Any], market_pred: pd.DataFrame, numeric: list[str], cat: list[str], model_root: Path, selected_tree_count: int, calibrator: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = model_root / f"selected_tree_{selected_tree_count}" / "final" / "model.cbm"
    if not path.exists():
        raise FileNotFoundError(f"selected final model missing: {path}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    if int(model.tree_count_) != int(selected_tree_count):
        raise RuntimeError(f"selected final model tree_count mismatch: expected {selected_tree_count}, got {model.tree_count_}")
    eval_df = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"].isin([base_cfg["test_year"], base_cfg["latest_holdout_year"]]))].copy()
    raw, residual, p = predict_parts(model, eval_df, numeric, cat)
    eval_df["model_key"] = f"{MODEL_KEY_PREFIX}_{selected_tree_count}_final"
    eval_df["tree_count"] = int(selected_tree_count)
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual
    eval_df["probability"] = p
    eval_df["final_probability"] = calibrator.transform(p)
    metric_rows = []
    residual_rows = []
    ev_rows = []
    for year, g in eval_df.groupby("Year"):
        label = {"model_key": eval_df["model_key"].iloc[0], "tree_count": int(selected_tree_count), "Year": int(year), "period": "test_2025" if int(year) == 2025 else "latest_holdout_2026", "tree_count_actual": int(model.tree_count_)}
        metric_rows.append({**label, **metric_row(g, "final_probability", {}, float(base_cfg["epsilon"]))})
        residual_rows.append(residual_stats(g["catboost_residual_score"], label))
        ev_rows.append({**ev_row(g, label), **roi_diag(g, cfg, label)})
    return eval_df, pd.DataFrame(metric_rows), pd.DataFrame(residual_rows), pd.DataFrame(ev_rows)


def selected_feature_importance_and_shap(base_cfg: dict[str, Any], cfg: dict[str, Any], pred: pd.DataFrame, numeric: list[str], cat: list[str], model_root: Path, selected_tree_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = numeric + cat
    pvc_rows = []
    lfc_rows = []
    shap_rows = []
    add_rows = []
    rng = np.random.default_rng(int(cfg["random_seed"]))
    sample_per_fold = int(cfg.get("shap_sample_per_fold", 800))
    for fold in base_cfg["folds"]:
        model_path = model_root / f"tree_{selected_tree_count}" / "folds" / fold["name"] / "model.cbm"
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        valid = pred[(pred["tree_count_candidate"].eq(selected_tree_count)) & (pred["fold"].eq(fold["name"]))].copy()
        pvc = model.get_feature_importance(type="PredictionValuesChange")
        for f, v in zip(features, pvc):
            pvc_rows.append({"tree_count": selected_tree_count, "fold": fold["name"], "Year": fold["validation_year"], "feature": f, "group": feature_group(f), "importance": float(v)})
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        lfc = model.get_feature_importance(data=pool, type="LossFunctionChange")
        for f, v in zip(features, lfc):
            lfc_rows.append({"tree_count": selected_tree_count, "fold": fold["name"], "Year": fold["validation_year"], "feature": f, "group": feature_group(f), "importance": float(v)})
        sample = valid
        if len(sample) > sample_per_fold:
            sample = sample.iloc[rng.choice(len(sample), sample_per_fold, replace=False)].copy()
        xs = prepare_x(sample, numeric, cat)
        spool = Pool(xs, sample["actual_place"].to_numpy(int), cat_features=cat_indices(xs, cat), baseline=sample["market_logit"].to_numpy(float))
        shap = np.asarray(model.get_feature_importance(data=spool, type="ShapValues"), dtype=float)
        vals = shap[:, :-1]
        expected = shap[:, -1]
        residual = sample["catboost_residual_score"].to_numpy(float)
        final_raw = sample["final_probability_raw"].to_numpy(float)
        add_rows.append({
            "tree_count": selected_tree_count,
            "fold": fold["name"],
            "Year": fold["validation_year"],
            "rows": int(len(sample)),
            "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1))))),
            "final_logit_additivity_max_abs": float(np.max(np.abs(final_raw - (sample["market_logit"].to_numpy(float) + expected + vals.sum(axis=1))))),
        })
        for idx, f in enumerate(features):
            v = vals[:, idx]
            shap_rows.append({
                "tree_count": selected_tree_count,
                "fold": fold["name"],
                "Year": fold["validation_year"],
                "feature": f,
                "group": feature_group(f),
                "mean_abs_shap": float(np.mean(np.abs(v))),
                "mean_signed_shap": float(np.mean(v)),
                "median_abs_shap": float(np.median(np.abs(v))),
                "p90_abs_shap": float(np.percentile(np.abs(v), 90)),
                "p99_abs_shap": float(np.percentile(np.abs(v), 99)),
                "positive_share": float((v > 0).mean()),
                "sample_rows": int(len(sample)),
            })
    def summarize_importance(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["tree_count", "feature", "group"], as_index=False).agg(
            weighted_mean=("importance", "mean"),
            median=("importance", "median"),
            min=("importance", "min"),
            max=("importance", "max"),
            std=("importance", "std"),
            fold_count=("importance", "count"),
        ).sort_values("weighted_mean", ascending=False)
    shap = pd.DataFrame(shap_rows)
    shap_summary = shap.groupby(["tree_count", "feature", "group"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        median_abs_shap=("median_abs_shap", "mean"),
        p90_abs_shap=("p90_abs_shap", "mean"),
        p99_abs_shap=("p99_abs_shap", "mean"),
        positive_share=("positive_share", "mean"),
        sample_rows=("sample_rows", "sum"),
    ).sort_values("mean_abs_shap", ascending=False)
    return summarize_importance(pd.DataFrame(pvc_rows)), summarize_importance(pd.DataFrame(lfc_rows)), shap_summary, pd.DataFrame(add_rows)


def run_reuse_existing(config_path: Path, selected_tree_count: int | None = None) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    allow = json.loads((Path(cfg["base_c1r0_output_root"]) / "feature_allowlist_c1r0.json").read_text(encoding="utf-8"))
    numeric = list(allow["numeric"])
    cat = list(allow["categorical"])
    artifact_check = validate_existing_artifacts(cfg, base_cfg, out, model_root)
    atomic_write_csv(out / "reuse_artifact_check.csv", artifact_check)
    bad = artifact_check[~artifact_check["status"].astype(str).str.startswith("ok")]
    if not bad.empty:
        raise RuntimeError(f"cannot reuse all candidate artifacts; bad rows written to reuse_artifact_check.csv: {len(bad)}")
    pred = pd.read_parquet(out / "fixed_tree_predictions_2020_2024.parquet")
    summary, by_fold, residual, ev = recompute_from_existing_predictions(pred, cfg, base_cfg)
    selected = select_tree_count(summary)
    if selected_tree_count is not None:
        row = summary[summary["tree_count"].eq(int(selected_tree_count))]
        if row.empty:
            raise RuntimeError(f"selected tree count not present in summary: {selected_tree_count}")
        selected = {
            "selected_tree_count": int(selected_tree_count),
            "selection_years": [2020, 2021, 2022, 2023, 2024],
            "selection_basis": "User-pinned after reviewing 2020-2024 reuse summary; 2025/2026 not used for selection.",
            "selected_row": row.iloc[0].to_dict(),
            "user_pinned": True,
        }
    selected_key = f"{MODEL_KEY_PREFIX}_{selected['selected_tree_count']}"
    selected_calibration_method, _, selected_calibrator = nested_calibration(pred[pred["Year"].between(2020, 2024)], selected_key, base_cfg)
    selected["selected_calibration_method"] = selected_calibration_method
    atomic_write_json(out / "selected_fixed_tree_count.json", selected)
    df = load_dataset(base_cfg, numeric, cat, smoke=False)
    tdf = target_frame(df, base_cfg)
    market_pred, _ = make_market_predictions(tdf, base_cfg, out, model_root, smoke=False)
    selected_pred, diag_metrics, diag_resid, diag_ev = predict_selected_final_existing(base_cfg, cfg, market_pred, numeric, cat, model_root, int(selected["selected_tree_count"]), selected_calibrator)
    pvc, lfc, shap, shap_add = selected_feature_importance_and_shap(base_cfg, cfg, pred, numeric, cat, model_root, int(selected["selected_tree_count"]))
    cur_metrics, cur_resid, cur_ev = current_c1r0_diagnostic(Path(cfg["base_c1r0_output_root"]))
    fixed_diag = pd.concat([cur_metrics, diag_metrics], ignore_index=True)
    fixed_resid = pd.concat([cur_resid, diag_resid], ignore_index=True)
    fixed_ev = pd.concat([cur_ev, diag_ev], ignore_index=True)
    hashes = {
        "reuse_artifact_check.csv": sha256_file(out / "reuse_artifact_check.csv"),
        "fixed_tree_comparison_2020_2024.csv": atomic_write_csv(out / "fixed_tree_comparison_2020_2024.csv", summary),
        "fixed_tree_comparison_by_fold.csv": atomic_write_csv(out / "fixed_tree_comparison_by_fold.csv", by_fold),
        "fixed_tree_residual_stability.csv": atomic_write_csv(out / "fixed_tree_residual_stability.csv", residual),
        "fixed_tree_ev_stability.csv": atomic_write_csv(out / "fixed_tree_ev_stability.csv", ev),
        "fixed_tree_2025_2026_diagnostic.csv": atomic_write_csv(out / "fixed_tree_2025_2026_diagnostic.csv", fixed_diag),
        "fixed_tree_residual_2025_2026.csv": atomic_write_csv(out / "fixed_tree_residual_2025_2026.csv", fixed_resid),
        "fixed_tree_ev_2025_2026.csv": atomic_write_csv(out / "fixed_tree_ev_2025_2026.csv", fixed_ev),
        "selected_fixed_tree_predictions_2025_2026.parquet": atomic_write_parquet(out / "selected_fixed_tree_predictions_2025_2026.parquet", selected_pred),
        "selected_tree_catboost_pvc_summary.csv": atomic_write_csv(out / "selected_tree_catboost_pvc_summary.csv", pvc),
        "selected_tree_catboost_lfc_summary.csv": atomic_write_csv(out / "selected_tree_catboost_lfc_summary.csv", lfc),
        "selected_tree_shap_summary.csv": atomic_write_csv(out / "selected_tree_shap_summary.csv", shap),
        "selected_tree_shap_additivity.csv": atomic_write_csv(out / "selected_tree_shap_additivity.csv", shap_add),
    }
    old_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8")) if (out / "manifest.json").exists() else {}
    manifest = {
        **old_manifest,
        "selected_fixed_tree_count": selected["selected_tree_count"],
        "selected_calibration_method": selected_calibration_method,
        "reuse_existing": True,
        "retrained_models": [],
        "reuse_note": "Candidate fold models and OOF predictions were reused after tree_count, row-count, hash, and prediction availability checks.",
        "execution_constraints": {
            "gpu_ram_part": cfg.get("execution_constraints", {}).get("gpu_ram_part"),
            "note": cfg.get("execution_constraints", {}).get("note", "Execution stability constraint only; not a model-selection hyperparameter."),
            "model_selection_hyperparameter": False,
        },
        "fixed_training_params_example": fixed_params(
            base_cfg["training_params"],
            int(selected["selected_tree_count"]),
            smoke=False,
            gpu_ram_part=cfg.get("execution_constraints", {}).get("gpu_ram_part"),
        ),
        "git": git_info(),
        "output_hashes": {**old_manifest.get("output_hashes", {}), **hashes},
        "elapsed_seconds_reuse_selection": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_report(Path("docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md"), selected, pd.read_csv(out / "model_tree_count_audit.csv"), summary, fixed_diag, fixed_resid, fixed_ev)
    return manifest


def ntree_end_diagnostic(base_cfg: dict[str, Any], cfg: dict[str, Any], market_pred: pd.DataFrame, numeric: list[str], cat: list[str], base_model_root: Path) -> pd.DataFrame:
    model = CatBoostClassifier()
    model.load_model(str(base_model_root / "C1R0" / "final" / "model.cbm"))
    rows = []
    full = int(model.tree_count_)
    for val in list(cfg["ntree_end_values"]) + [full]:
        ntree = min(int(val), full)
        eval_df = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"].isin([2025, 2026]))].copy()
        raw, residual, p = predict_parts(model, eval_df, numeric, cat, ntree_end=ntree)
        eval_df["final_probability_raw"] = raw
        eval_df["catboost_residual_score"] = residual
        eval_df["final_probability"] = p
        for year, g in eval_df.groupby("Year"):
            label = {"diagnostic": "saved_final_ntree_end", "ntree_end": int(ntree), "Year": int(year), "period": "test_2025" if int(year) == 2025 else "latest_holdout_2026"}
            rows.append({**label, **metric_row(g, "final_probability", {}, float(base_cfg["epsilon"])), **residual_stats(g["catboost_residual_score"], {}), **ev_row(g, {})})
    return pd.DataFrame(rows)


def train_selected_final(base_cfg: dict[str, Any], cfg: dict[str, Any], market_pred: pd.DataFrame, numeric: list[str], cat: list[str], model_root: Path, selected_tree_count: int, calibrator: Any, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gpu_ram_part = cfg.get("execution_constraints", {}).get("gpu_ram_part")
    params = fixed_params(base_cfg["training_params"], selected_tree_count, smoke, gpu_ram_part=gpu_ram_part)
    train = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"].isin(base_cfg["final_train_years"]))]
    eval_df = market_pred[(market_pred["baseline_scope"] == "final") & (market_pred["Year"].isin([base_cfg["test_year"], base_cfg["latest_holdout_year"]]))].copy()
    model = train_fixed_model(train, numeric, cat, params, model_root / f"selected_tree_{selected_tree_count}" / "final" / "model.cbm")
    raw, residual, p = predict_parts(model, eval_df, numeric, cat)
    eval_df["model_key"] = f"{MODEL_KEY_PREFIX}_{selected_tree_count}_final"
    eval_df["tree_count"] = int(selected_tree_count)
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual
    eval_df["probability"] = p
    eval_df["final_probability"] = calibrator.transform(p)
    metric_rows = []
    residual_rows = []
    ev_rows = []
    for year, g in eval_df.groupby("Year"):
        label = {"model_key": eval_df["model_key"].iloc[0], "tree_count": int(selected_tree_count), "Year": int(year), "period": "test_2025" if int(year) == 2025 else "latest_holdout_2026", "tree_count_actual": int(model.tree_count_)}
        metric_rows.append({**label, **metric_row(g, "final_probability", {}, float(base_cfg["epsilon"]))})
        residual_rows.append(residual_stats(g["catboost_residual_score"], label))
        ev_rows.append({**ev_row(g, label), **roi_diag(g, cfg, label)})
    return eval_df, pd.DataFrame(metric_rows), pd.DataFrame(residual_rows), pd.DataFrame(ev_rows)


def current_c1r0_diagnostic(base_out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_parquet(base_out / "prediction_output.parquet")
    cur = pred[pred["model_key"].eq(CURRENT_MODEL_KEY) & pred["Year"].isin([2025, 2026])].copy()
    metric_rows = []
    residual_rows = []
    ev_rows = []
    dummy_cfg = {"bootstrap_iterations": 1000, "random_seed": 42}
    for year, g in cur.groupby("Year"):
        label = {"model_key": CURRENT_MODEL_KEY, "tree_count": 3000, "Year": int(year), "period": "test_2025" if int(year) == 2025 else "latest_holdout_2026"}
        metric_rows.append({**label, **metric_row(g, "final_probability", {}, 1e-6)})
        residual_rows.append(residual_stats(g["catboost_residual_score"], label))
        ev_rows.append({**ev_row(g, label), **roi_diag(g, dummy_cfg, label)})
    return pd.DataFrame(metric_rows), pd.DataFrame(residual_rows), pd.DataFrame(ev_rows)


def write_report(path: Path, selected: dict[str, Any], audit: pd.DataFrame, summary: pd.DataFrame, diag: pd.DataFrame, residual_diag: pd.DataFrame, ev_diag: pd.DataFrame) -> None:
    lines = [
        "# C1R0 Tree Count Audit Results",
        "",
        f"- Selected fixed tree count: `{selected['selected_tree_count']}`",
        f"- Selection basis: {selected['selection_basis']}",
        "- Selection used only 2020-2024 walk-forward validation.",
        "- 2025/2026 were fixed diagnostics only.",
        "",
        "## Saved Model Tree Count Audit",
        audit.to_markdown(index=False),
        "",
        "## Fixed Tree Summary 2020-2024",
        summary.to_markdown(index=False),
        "",
        "## 2025/2026 Diagnostic Metrics",
        diag.to_markdown(index=False),
        "",
        "## 2025/2026 Residual Diagnostic",
        residual_diag.to_markdown(index=False),
        "",
        "## 2025/2026 EV Diagnostic",
        ev_diag.to_markdown(index=False),
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(config_path: Path, smoke: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        cfg["candidate_tree_counts"] = cfg["smoke_overrides"]["candidate_tree_counts"]
        cfg["ntree_end_values"] = cfg["smoke_overrides"]["ntree_end_values"]
        cfg["bootstrap_iterations"] = cfg["smoke_overrides"]["bootstrap_iterations"]
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    if out.exists() and any(out.iterdir()) and not force:
        raise RuntimeError(f"output already exists; pass --force to update tree-count outputs only: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    allow = json.loads((Path(cfg["base_c1r0_output_root"]) / "feature_allowlist_c1r0.json").read_text(encoding="utf-8"))
    numeric = list(allow["numeric"])
    cat = list(allow["categorical"])
    forbidden = {"Year", "p_market", "market_logit", "tan_odds", "fuku_odds_low", "fuku_ninki", "race_id", "entry_id", "KettoNum", "target_place_paid", "fuku_pay"}
    if forbidden.intersection(numeric + cat):
        raise RuntimeError(f"forbidden features in C1R0 allowlist: {sorted(forbidden.intersection(numeric + cat))}")
    load_cfg = dict(base_cfg)
    if smoke:
        load_cfg["smoke_overrides"] = dict(base_cfg.get("smoke_overrides", {}))
        load_cfg["smoke_overrides"]["train_rows_per_year"] = 800
        load_cfg["smoke_overrides"]["eval_rows_per_year"] = 400
    df = load_dataset(load_cfg, numeric, cat, smoke)
    tdf = target_frame(df, load_cfg)
    if int(tdf["Year"].min()) < 2016:
        raise RuntimeError("2015 or earlier rows detected")
    market_pred, _ = make_market_predictions(tdf, load_cfg, out, model_root, smoke)
    audit = audit_saved_models(base_cfg, Path(cfg["base_c1r0_output_root"]), Path(cfg["base_c1r0_model_root"]), market_pred)
    best_summary = summarize_best_iterations(audit)
    ntree = ntree_end_diagnostic(base_cfg, cfg, market_pred, numeric, cat, Path(cfg["base_c1r0_model_root"]))
    pred, by_fold_raw, residual, ev_raw, roi_raw = run_fixed_candidates(base_cfg, cfg, market_pred, numeric, cat, model_root, smoke)
    pred, cal_metrics, calibration_by_model = calibrate_candidate_predictions(pred, base_cfg)
    by_fold, residual, ev, roi = recompute_candidate_tables(pred, residual, cfg, base_cfg)
    summary = summarize_candidates(by_fold, residual, ev, roi)
    selected = select_tree_count(summary)
    selected_key = f"{MODEL_KEY_PREFIX}_{selected['selected_tree_count']}"
    selected_calibration_method, _, selected_calibrator = nested_calibration(pred[pred["Year"].between(2020, 2024)], selected_key, base_cfg)
    selected["selected_calibration_method"] = selected_calibration_method
    atomic_write_json(out / "selected_fixed_tree_count.json", selected)
    selected_pred, diag_metrics, diag_resid, diag_ev = train_selected_final(base_cfg, cfg, market_pred, numeric, cat, model_root, int(selected["selected_tree_count"]), selected_calibrator, smoke)
    cur_metrics, cur_resid, cur_ev = current_c1r0_diagnostic(Path(cfg["base_c1r0_output_root"]))
    fixed_diag = pd.concat([cur_metrics, diag_metrics], ignore_index=True)
    fixed_resid = pd.concat([cur_resid, diag_resid], ignore_index=True)
    fixed_ev = pd.concat([cur_ev, diag_ev], ignore_index=True)
    hashes = {
        "model_tree_count_audit.csv": atomic_write_csv(out / "model_tree_count_audit.csv", audit),
        "fold_best_iteration_summary.csv": atomic_write_csv(out / "fold_best_iteration_summary.csv", best_summary),
        "ntree_end_diagnostic_2025_2026.csv": atomic_write_csv(out / "ntree_end_diagnostic_2025_2026.csv", ntree),
        "fixed_tree_comparison_by_fold.csv": atomic_write_csv(out / "fixed_tree_comparison_by_fold.csv", by_fold),
        "fixed_tree_comparison_2020_2024.csv": atomic_write_csv(out / "fixed_tree_comparison_2020_2024.csv", summary),
        "fixed_tree_residual_stability.csv": atomic_write_csv(out / "fixed_tree_residual_stability.csv", residual),
        "fixed_tree_ev_stability.csv": atomic_write_csv(out / "fixed_tree_ev_stability.csv", ev),
        "fixed_tree_roi_diagnostic.csv": atomic_write_csv(out / "fixed_tree_roi_diagnostic.csv", roi),
        "fixed_tree_calibration_metrics.csv": atomic_write_csv(out / "fixed_tree_calibration_metrics.csv", cal_metrics),
        "fixed_tree_2025_2026_diagnostic.csv": atomic_write_csv(out / "fixed_tree_2025_2026_diagnostic.csv", fixed_diag),
        "fixed_tree_residual_2025_2026.csv": atomic_write_csv(out / "fixed_tree_residual_2025_2026.csv", fixed_resid),
        "fixed_tree_ev_2025_2026.csv": atomic_write_csv(out / "fixed_tree_ev_2025_2026.csv", fixed_ev),
        "fixed_tree_predictions_2020_2024.parquet": atomic_write_parquet(out / "fixed_tree_predictions_2020_2024.parquet", pred),
        "selected_fixed_tree_predictions_2025_2026.parquet": atomic_write_parquet(out / "selected_fixed_tree_predictions_2025_2026.parquet", selected_pred),
    }
    manifest = {
        "version": cfg["version"],
        "base_c1r0_config": cfg["base_c1r0_config"],
        "base_c1r0_output_root": cfg["base_c1r0_output_root"],
        "base_c1r0_model_root": cfg["base_c1r0_model_root"],
        "output_root": cfg["output_root"],
        "model_root": cfg["model_root"],
        "candidate_tree_counts": cfg["candidate_tree_counts"],
        "selected_fixed_tree_count": selected["selected_tree_count"],
        "calibration_by_candidate": calibration_by_model,
        "selected_calibration_method": selected_calibration_method,
        "selection_used_years": cfg["selection_years"],
        "diagnostic_only_years": cfg["diagnostic_years"],
        "db_usage": "not_read; existing parquet feature dataset only",
        "feature_dataset_rebuild": False,
        "random_split_used": False,
        "allowlist_hash": sha256_json(allow),
        "allowlist_unchanged_from_c1r0": True,
        "tree_count_only_change": True,
        "execution_constraints": {
            "gpu_ram_part": cfg.get("execution_constraints", {}).get("gpu_ram_part"),
            "note": cfg.get("execution_constraints", {}).get("note", "Execution stability constraint only; not a model-selection hyperparameter."),
            "model_selection_hyperparameter": False,
        },
        "base_training_params": base_cfg["training_params"],
        "fixed_training_params_example": fixed_params(
            base_cfg["training_params"],
            int(selected["selected_tree_count"]),
            smoke,
            gpu_ram_part=cfg.get("execution_constraints", {}).get("gpu_ram_part"),
        ),
        "catboost_version": __import__("catboost").__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "output_hashes": hashes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    atomic_write_text(out / "run_log.txt", json.dumps({"status": "completed", "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))
    write_report(Path("docs/place_market_offset_catboost_c1r0_tree_count_v1_results.md"), selected, audit, summary, fixed_diag, fixed_resid, fixed_ev)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_tree_count_v1.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--selected-tree-count", type=int)
    args = parser.parse_args()
    if args.resume or args.reuse_existing:
        run_reuse_existing(Path(args.config), selected_tree_count=args.selected_tree_count)
        return 0
    run(Path(args.config), smoke=args.smoke_test, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
