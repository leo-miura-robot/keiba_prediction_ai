from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import brier_score_loss, log_loss

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
    ece,
    fit_one_calibrator,
    load_config,
    load_dataset,
    make_market_predictions,
    metric_row,
    prepare_x,
    roi_of,
    sha256_file,
    sha256_json,
    target_frame,
    top_removed_roi,
)
from scripts.run_place_market_offset_catboost_c1r0_tree_count_v1 import (
    ev_row,
    fixed_params,
    predict_parts,
    residual_stats,
    train_fixed_model,
)


RAW_KEY = "C1R0_fixed300_ablation_drop_person_codes"
FINAL_KEY = "C1R0_300_feature_clean_phase3_v1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_info() -> dict[str, Any]:
    return {
        "git_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).splitlines(),
    }


def load_allowlist(fixed_out: Path) -> tuple[list[str], list[str]]:
    allow = json.loads((fixed_out.parent / "place_market_offset_catboost_c1r0_v1" / "feature_allowlist_c1r0.json").read_text(encoding="utf-8"))
    return list(allow["numeric"]), list(allow["categorical"])


def candidate_features(numeric: list[str], cat: list[str], drops: list[str]) -> tuple[list[str], list[str]]:
    ds = set(drops)
    return [f for f in numeric if f not in ds], [f for f in cat if f not in ds]


def phase3_key(name: str) -> str:
    return f"C1R0_300_cleanbase_{name}"


def summarize(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, resid_rows, ev_rows, roi_rows = [], [], [], []
    for keys, g in pred.groupby(["model_key", "period", "Year"], dropna=False):
        label = {"model_key": keys[0], "period": keys[1], "Year": int(keys[2])}
        metric_rows.append(metric_row(g, "final_probability", label, float(cfg["epsilon"])))
        if "catboost_residual_score" in g:
            resid_rows.append(residual_stats(g["catboost_residual_score"], label))
        ev_rows.append(ev_row(g, label))
        d = add_eval_columns(g, "final_probability")
        bets = d[d["adjusted_place_ev"] >= 1.0]
        ci = bootstrap_ci(bets, int(cfg.get("bootstrap_iterations_roi", 1000)), int(cfg.get("random_seed", 42)))
        roi_rows.append({
            **label,
            "bets": int(len(bets)),
            "roi": roi_of(bets),
            "top1_removed_roi": top_removed_roi(bets, 1),
            "top3_removed_roi": top_removed_roi(bets, 3),
            "top5_removed_roi": top_removed_roi(bets, 5),
            "top10_removed_roi": top_removed_roi(bets, 10),
            "bootstrap_roi_p025": ci[0],
            "bootstrap_roi_p500": ci[1],
            "bootstrap_roi_p975": ci[2],
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(resid_rows), pd.DataFrame(ev_rows), pd.DataFrame(roi_rows)


def full_summary(metrics: pd.DataFrame, residual: pd.DataFrame, ev: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in metrics[metrics["Year"].between(2020, 2024)].groupby("model_key"):
        r = residual[residual["model_key"].eq(key) & residual["Year"].between(2020, 2024)]
        e = ev[ev["model_key"].eq(key) & ev["Year"].between(2020, 2024)]
        ro = roi[roi["model_key"].eq(key) & roi["Year"].between(2020, 2024)]
        rows.append({
            "model_key": key,
            "combined_logloss": float(g["logloss"].mean()),
            "combined_brier": float(g["brier"].mean()),
            "combined_ece": float(g["ece"].mean()),
            "calibration_slope": float(g["calibration_slope"].mean()),
            "calibration_intercept": float(g["calibration_intercept"].mean()),
            "worst_year_logloss": float(g["logloss"].max()),
            "worst_year_brier": float(g["brier"].max()),
            "residual_std": float(r["residual_std"].mean()) if not r.empty else np.nan,
            "residual_std_cv": float(r["residual_std"].std(ddof=1) / r["residual_std"].mean()) if not r.empty else np.nan,
            "abs_residual_p90": float(r["abs_residual_p90"].mean()) if not r.empty else np.nan,
            "abs_residual_p95": float(r["abs_residual_p95"].mean()) if not r.empty else np.nan,
            "abs_residual_p99": float(r["abs_residual_p99"].mean()) if not r.empty else np.nan,
            "abs_residual_p95_cv": float(r["abs_residual_p95"].std(ddof=1) / r["abs_residual_p95"].mean()) if not r.empty else np.nan,
            "ev_ge_1_count_sum": int(e["ev_ge_1_count"].sum()),
            "ev_ge_1_count_cv": float(e["ev_ge_1_count"].std(ddof=1) / e["ev_ge_1_count"].mean()),
            "ev_roi_spearman": float(e["ev_roi_spearman"].mean()),
            "ev_ge_1_roi": float(ro["roi"].mean()),
            "top1_removed_roi": float(ro["top1_removed_roi"].mean()),
            "top3_removed_roi": float(ro["top3_removed_roi"].mean()),
            "top5_removed_roi": float(ro["top5_removed_roi"].mean()),
            "top10_removed_roi": float(ro["top10_removed_roi"].mean()),
            "bootstrap_roi_p025": float(ro["bootstrap_roi_p025"].mean()),
            "bootstrap_roi_p975": float(ro["bootstrap_roi_p975"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["combined_logloss", "combined_brier"])


def load_raw_and_transformed(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase1 = Path(cfg["phase1_output_root"])
    phase2 = Path(cfg["phase2_output_root"])
    raw = pd.read_parquet(phase1 / "ablation_oof_predictions.parquet")
    raw = raw[raw["model_key"].eq(RAW_KEY) & raw["Year"].between(2020, 2024)].copy()
    parts = []
    for year in [2020, 2021, 2022, 2023, 2024]:
        parts.append(pd.read_parquet(phase2 / "predictions" / "starts_clip_p99_log1p" / f"fold_{year}.parquet"))
    trans = pd.concat(parts, ignore_index=True)
    trans = trans[trans["Year"].between(2020, 2024)].copy()
    return raw, trans


def paired_artifact_check(raw: pd.DataFrame, trans: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    keys = ["entry_id", "race_id", "Year"]
    r = raw[keys + ["actual_place", "fuku_odds_low", "fuku_pay", "p_market", "market_logit", "final_probability"]].copy()
    t = trans[keys + ["actual_place", "fuku_odds_low", "fuku_pay", "p_market", "market_logit", "final_probability"]].copy()
    merged = r.merge(t, on=keys, suffixes=("_raw", "_trans"), how="outer", indicator=True)
    rows = [{
        "check": "entry_alignment",
        "raw_rows": len(raw),
        "transformed_rows": len(trans),
        "merged_rows": len(merged),
        "left_only": int((merged["_merge"] == "left_only").sum()),
        "right_only": int((merged["_merge"] == "right_only").sum()),
        "duplicate_raw_entry": int(raw.duplicated(keys).sum()),
        "duplicate_transformed_entry": int(trans.duplicated(keys).sum()),
        "target_mismatch": int((merged["actual_place_raw"] != merged["actual_place_trans"]).fillna(False).sum()),
        "baseline_mismatch_max_abs": float(np.nanmax(np.abs(merged["market_logit_raw"] - merged["market_logit_trans"]))),
        "odds_mismatch": int((merged["fuku_odds_low_raw"] != merged["fuku_odds_low_trans"]).fillna(False).sum()),
        "status": "ok",
    }]
    phase2_params = pd.read_csv(Path(cfg["phase2_output_root"]) / "cumulative_starts_transform_params_by_fold.csv")
    ok_params = phase2_params[phase2_params["model_key"].eq(cfg["transformed_starts"]["model_key"])]
    rows.append({
        "check": "transform_params",
        "raw_rows": len(ok_params),
        "transformed_rows": 0,
        "merged_rows": 0,
        "left_only": 0,
        "right_only": 0,
        "duplicate_raw_entry": 0,
        "duplicate_transformed_entry": 0,
        "target_mismatch": 0,
        "baseline_mismatch_max_abs": np.nan,
        "odds_mismatch": 0,
        "status": "ok" if not ok_params.empty else "missing",
    })
    return pd.DataFrame(rows)


def metric_values(y: np.ndarray, p: np.ndarray, bins: int) -> dict[str, float]:
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p, bins=bins),
    }


def paired_bootstrap(raw: pd.DataFrame, trans: pd.DataFrame, unit_col: str, n_boot: int, seed: int, bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = ["entry_id", "race_id", "race_date", "Year", "actual_place", "final_probability"]
    m = raw[keep].merge(trans[keep], on=["entry_id", "race_id", "race_date", "Year", "actual_place"], suffixes=("_raw", "_trans"))
    y = m["actual_place"].to_numpy(int)
    pr = m["final_probability_raw"].to_numpy(float)
    pt = m["final_probability_trans"].to_numpy(float)
    point_raw = metric_values(y, pr, bins)
    point_trans = metric_values(y, pt, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    m["_ll_raw"] = -(y * np.log(np.clip(pr, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - pr, 1e-6, 1)))
    m["_ll_trans"] = -(y * np.log(np.clip(pt, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - pt, 1e-6, 1)))
    m["_br_raw"] = (pr - y) ** 2
    m["_br_trans"] = (pt - y) ** 2
    raw_bin = np.minimum(np.searchsorted(edges, pr, side="right") - 1, bins - 1)
    trans_bin = np.minimum(np.searchsorted(edges, pt, side="right") - 1, bins - 1)
    m["_raw_bin"] = np.maximum(raw_bin, 0)
    m["_trans_bin"] = np.maximum(trans_bin, 0)
    g = m.groupby(unit_col, sort=False)
    units = np.asarray(list(g.groups.keys()))
    n_units = len(units)
    count = g.size().to_numpy(float)
    ll_raw = g["_ll_raw"].sum().to_numpy(float)
    ll_trans = g["_ll_trans"].sum().to_numpy(float)
    br_raw = g["_br_raw"].sum().to_numpy(float)
    br_trans = g["_br_trans"].sum().to_numpy(float)
    raw_counts = np.zeros((n_units, bins), dtype=float)
    raw_actual = np.zeros((n_units, bins), dtype=float)
    raw_pred = np.zeros((n_units, bins), dtype=float)
    trans_counts = np.zeros((n_units, bins), dtype=float)
    trans_actual = np.zeros((n_units, bins), dtype=float)
    trans_pred = np.zeros((n_units, bins), dtype=float)
    unit_index = {u: i for i, u in enumerate(units)}
    codes = m[unit_col].map(unit_index).to_numpy(int)
    for b in range(bins):
        mask = m["_raw_bin"].to_numpy(int) == b
        np.add.at(raw_counts[:, b], codes[mask], 1)
        np.add.at(raw_actual[:, b], codes[mask], y[mask])
        np.add.at(raw_pred[:, b], codes[mask], pr[mask])
        mask = m["_trans_bin"].to_numpy(int) == b
        np.add.at(trans_counts[:, b], codes[mask], 1)
        np.add.at(trans_actual[:, b], codes[mask], y[mask])
        np.add.at(trans_pred[:, b], codes[mask], pt[mask])

    def ece_from_agg(cnt: np.ndarray, act: np.ndarray, pred_sum: np.ndarray) -> float:
        total = float(cnt.sum())
        if total <= 0:
            return np.nan
        mask = cnt > 0
        return float(np.sum(np.abs(act[mask] / cnt[mask] - pred_sum[mask] / cnt[mask]) * (cnt[mask] / total)))

    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_boot):
        chosen = rng.integers(0, len(units), len(units))
        total = count[chosen].sum()
        raw_ece = ece_from_agg(raw_counts[chosen].sum(axis=0), raw_actual[chosen].sum(axis=0), raw_pred[chosen].sum(axis=0))
        trans_ece = ece_from_agg(trans_counts[chosen].sum(axis=0), trans_actual[chosen].sum(axis=0), trans_pred[chosen].sum(axis=0))
        samples.append({
            "bootstrap_id": i,
            "unit": unit_col,
            "delta_logloss": float((ll_trans[chosen].sum() - ll_raw[chosen].sum()) / total),
            "delta_brier": float((br_trans[chosen].sum() - br_raw[chosen].sum()) / total),
            "delta_ece": float(trans_ece - raw_ece),
        })
    s = pd.DataFrame(samples)
    rows = []
    for metric in ["logloss", "brier", "ece"]:
        col = f"delta_{metric}"
        vals = s[col].to_numpy(float)
        rows.append({
            "unit": unit_col,
            "metric": col,
            "point_estimate_delta": point_trans[metric] - point_raw[metric],
            "bootstrap_mean_delta": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
            "prob_delta_below_zero": float((vals < 0).mean()),
            "prob_delta_above_zero": float((vals > 0).mean()),
            "better_model": "transformed" if point_trans[metric] < point_raw[metric] else "raw",
            "decision": "transformed_clear" if np.percentile(vals, 97.5) < 0 else ("raw_clear" if np.percentile(vals, 2.5) > 0 else "ambiguous"),
        })
    return pd.DataFrame(rows), s


def bootstrap_by_year(raw: pd.DataFrame, trans: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for year in [2020, 2021, 2022, 2023, 2024]:
        summary, _ = paired_bootstrap(
            raw[raw["Year"].eq(year)],
            trans[trans["Year"].eq(year)],
            "race_id",
            int(cfg["bootstrap"]["n_bootstrap"]),
            int(cfg["bootstrap"]["seed"]) + year,
            int(cfg["bootstrap"]["ece_bins"]),
        )
        summary["Year"] = year
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def choose_starts(summary: pd.DataFrame, by_year: pd.DataFrame) -> dict[str, Any]:
    race = summary[summary["unit"].eq("race_id")]
    ll = race[race["metric"].eq("delta_logloss")].iloc[0]
    br = race[race["metric"].eq("delta_brier")].iloc[0]
    improved_years = int((by_year[by_year["metric"].eq("delta_logloss")]["point_estimate_delta"] < 0).sum())
    clear = ll["ci_upper"] < 0 and br["ci_upper"] < 0 and improved_years >= 3
    if clear:
        selected = "clip_p99_log1p"
        reason = "Race-level paired bootstrap CI for Logloss and Brier is below zero and most years improve."
    elif ll["ci_lower"] > 0 and br["ci_lower"] > 0:
        selected = "raw"
        reason = "Race-level paired bootstrap CI shows clip_p99_log1p worsens Logloss and Brier, so raw starts are retained."
    else:
        selected = "raw"
        reason = "Difference is statistically/practically ambiguous, so the simpler raw starts are retained."
    return {"selected_starts": selected, "selected_model_key": RAW_KEY if selected == "raw" else "C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p", "improved_years_logloss": improved_years, "reason": reason}


def valid_reuse(mp: Path, pp: Path, rows: int, fh: str) -> tuple[bool, str]:
    if not mp.exists() or not pp.exists():
        return False, "missing"
    try:
        model = CatBoostClassifier()
        model.load_model(str(mp))
        if int(model.tree_count_) != 300:
            return False, "tree_count_mismatch"
        p = pd.read_parquet(pp, columns=["feature_hash"])
        if len(p) != rows:
            return False, "row_count_mismatch"
        if str(p["feature_hash"].iloc[0]) != fh:
            return False, "feature_hash_mismatch"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def train_oof(name: str, drops: list[str], market_pred: pd.DataFrame, base_cfg: dict[str, Any], numeric: list[str], cat: list[str], params: dict[str, Any], out: Path, model_root: Path, resume: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    n2, c2 = candidate_features(numeric, cat, drops)
    fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": 300, "drops": drops})
    parts, logs = [], []
    for fold in base_cfg["folds"]:
        scoped = market_pred[market_pred["baseline_scope"].eq(fold["name"])]
        train = scoped[scoped["Year"].isin(fold["train_years"])].copy()
        valid = scoped[scoped["Year"].eq(fold["validation_year"])].copy()
        mp = model_root / name / "folds" / fold["name"] / "model.cbm"
        pp = out / "predictions" / name / f"{fold['name']}.parquet"
        ok, reason = valid_reuse(mp, pp, len(valid), fh) if resume else (False, "resume_disabled")
        if ok:
            print(f"[reuse] {name} {fold['name']}", flush=True)
            pred = pd.read_parquet(pp)
            parts.append(pred)
            logs.append({"model": name, "fold": fold["name"], "action": "reuse", "reason": reason})
            continue
        print(f"[train] {name} {fold['name']}: {reason}", flush=True)
        model = train_fixed_model(train, n2, c2, params, mp)
        raw, residual, prob = predict_parts(model, valid, n2, c2)
        valid["model_key"] = phase3_key(name)
        valid["variant_name"] = name
        valid["probability"] = prob
        valid["final_probability"] = prob
        valid["final_probability_raw"] = raw
        valid["catboost_residual_score"] = residual
        valid["tree_count"] = int(model.tree_count_)
        valid["feature_hash"] = fh
        atomic_write_parquet(pp, valid)
        parts.append(valid)
        logs.append({"model": name, "fold": fold["name"], "action": "train", "reason": reason, "model_sha256": sha256_file(mp)})
    return pd.concat(parts, ignore_index=True), logs


def calibrate_oof(pred: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for key, d in pred.groupby("model_key"):
        cal = fit_one_calibrator("isotonic", d[d["Year"].between(2020, 2024)], "probability")
        x = d.copy()
        x["final_probability"] = cal.transform(x["probability"].to_numpy(float))
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def select_additional(summary: pd.DataFrame, base_key: str) -> dict[str, Any]:
    base = summary[summary["model_key"].eq(base_key)].iloc[0]
    candidates = summary[~summary["model_key"].eq(base_key)].copy()
    candidates["effective"] = (
        (candidates["combined_logloss"] < float(base["combined_logloss"]) - 0.00002)
        & (candidates["combined_brier"] <= float(base["combined_brier"]))
        & (candidates["worst_year_logloss"] <= float(base["worst_year_logloss"]))
        & (candidates["abs_residual_p95"] <= float(base["abs_residual_p95"]) + 0.01)
    )
    eff = candidates[candidates["effective"]].copy()
    if eff.empty:
        return {"selected_model_key": base_key, "selected_additional": None, "reason": "No additional ablation had clear individual benefit under 2020-2024 priority metrics.", "selected_row": base.to_dict()}
    row = eff.sort_values(["combined_logloss", "combined_brier", "combined_ece", "abs_residual_p95", "worst_year_logloss"]).iloc[0]
    return {"selected_model_key": row["model_key"], "selected_additional": str(row["model_key"]).replace("C1R0_300_cleanbase_", ""), "reason": "Best individually effective additional ablation selected on 2020-2024 metrics.", "selected_row": row.to_dict()}


def train_final(name: str, key: str, drops: list[str], market_pred: pd.DataFrame, base_cfg: dict[str, Any], numeric: list[str], cat: list[str], params: dict[str, Any], oof: pd.DataFrame, out: Path, model_root: Path, resume: bool) -> pd.DataFrame:
    n2, c2 = candidate_features(numeric, cat, drops)
    fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": 300, "drops": drops, "final": True})
    train = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin(base_cfg["final_train_years"])].copy()
    eval_df = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin([2025, 2026])].copy()
    mp = model_root / name / "final" / "model.cbm"
    pp = out / "predictions" / name / "final_2025_2026.parquet"
    ok, reason = valid_reuse(mp, pp, len(eval_df), fh) if resume else (False, "resume_disabled")
    if ok:
        print(f"[reuse] {name} final", flush=True)
        return pd.read_parquet(pp)
    print(f"[train] {name} final: {reason}", flush=True)
    model = train_fixed_model(train, n2, c2, params, mp)
    raw, residual, prob = predict_parts(model, eval_df, n2, c2)
    cal = fit_one_calibrator("isotonic", oof[oof["model_key"].eq(key)], "probability")
    eval_df["model_key"] = key
    eval_df["variant_name"] = name
    eval_df["probability"] = prob
    eval_df["final_probability"] = cal.transform(prob)
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual
    eval_df["tree_count"] = 300
    eval_df["feature_hash"] = fh
    atomic_write_parquet(pp, eval_df)
    return eval_df


def fi_shap(name: str, key: str, drops: list[str], market_pred: pd.DataFrame, base_cfg: dict[str, Any], numeric: list[str], cat: list[str], out: Path, model_root: Path, sample: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n2, c2 = candidate_features(numeric, cat, drops)
    features = n2 + c2
    rng = np.random.default_rng(seed)
    pvc_rows, shap_rows, add_rows = [], [], []
    for fold in base_cfg["folds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_root / name / "folds" / fold["name"] / "model.cbm"))
        for f, v in zip(features, model.get_feature_importance(type="PredictionValuesChange")):
            pvc_rows.append({"feature": f, "Year": fold["validation_year"], "importance": float(v)})
        valid = market_pred[market_pred["baseline_scope"].eq(fold["name"]) & market_pred["Year"].eq(fold["validation_year"])].copy()
        if len(valid) > sample:
            valid = valid.iloc[rng.choice(len(valid), sample, replace=False)].copy()
        x = prepare_x(valid, n2, c2)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, c2), baseline=valid["market_logit"].to_numpy(float))
        sv = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)
        vals, expected = sv[:, :-1], sv[:, -1]
        residual = np.asarray(model.predict(Pool(x, cat_features=cat_indices(x, c2)), prediction_type="RawFormulaVal"), dtype=float)
        add_rows.append({"model_key": key, "Year": fold["validation_year"], "rows": len(valid), "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1)))))})
        for idx, f in enumerate(features):
            v = vals[:, idx]
            shap_rows.append({"feature": f, "Year": fold["validation_year"], "mean_abs_shap": float(np.mean(np.abs(v))), "mean_signed_shap": float(np.mean(v)), "p99_abs_shap": float(np.percentile(np.abs(v), 99))})
    pvc = pd.DataFrame(pvc_rows).groupby("feature", as_index=False).agg(weighted_mean=("importance", "mean")).sort_values("weighted_mean", ascending=False)
    shap = pd.DataFrame(shap_rows).groupby("feature", as_index=False).agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), p99_abs_shap=("p99_abs_shap", "mean")).sort_values("mean_abs_shap", ascending=False)
    return pvc, shap, pd.DataFrame(add_rows)


def write_report(path: Path, boot: pd.DataFrame, decision: dict[str, Any], add_summary: pd.DataFrame, selected: dict[str, Any], diag: pd.DataFrame, ev: pd.DataFrame, roi: pd.DataFrame, logs: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# C1R0 Feature Cleanup Phase3 Results",
        "",
        "## Paired Bootstrap Summary",
        boot.to_markdown(index=False),
        "",
        "## Cumulative Starts Decision",
        json.dumps(decision, ensure_ascii=False, indent=2),
        "",
        "## Additional Ablation Comparison",
        add_summary.to_markdown(index=False),
        "",
        "## Selected Feature Set",
        json.dumps(selected, ensure_ascii=False, indent=2),
        "",
        "## 2025/2026 Diagnostic",
        diag.to_markdown(index=False),
        "",
        "## 2025/2026 EV",
        ev.to_markdown(index=False),
        "",
        "## 2025/2026 ROI",
        roi.to_markdown(index=False),
        "",
        "## Reuse Training Log",
        logs.to_markdown(index=False),
        "",
        f"Elapsed seconds: `{elapsed:.1f}`",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(config_path: Path, resume: bool = True, smoke: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        cfg["bootstrap"]["n_bootstrap"] = cfg["smoke_overrides"]["n_bootstrap"]
        cfg["bootstrap_iterations_roi"] = cfg["smoke_overrides"]["bootstrap_iterations_roi"]
        cfg["shap_sample_per_year"] = cfg["smoke_overrides"]["shap_sample_per_year"]
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    if smoke:
        base_cfg = dict(base_cfg)
        base_cfg["smoke_overrides"] = {**base_cfg.get("smoke_overrides", {}), "train_rows_per_year": cfg["smoke_overrides"]["train_rows_per_year"], "eval_rows_per_year": cfg["smoke_overrides"]["eval_rows_per_year"]}
    out, model_root = Path(cfg["output_root"]), Path(cfg["model_root"])
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    numeric, cat = load_allowlist(Path(cfg["fixed300_output_root"]))
    params = fixed_params(base_cfg["training_params"], 300, smoke, gpu_ram_part=cfg["execution_constraints"]["gpu_ram_part"])

    raw, trans = load_raw_and_transformed(cfg)
    check = paired_artifact_check(raw, trans, cfg)
    race_summary, race_samples = paired_bootstrap(raw, trans, "race_id", int(cfg["bootstrap"]["n_bootstrap"]), int(cfg["bootstrap"]["seed"]), int(cfg["bootstrap"]["ece_bins"]))
    day_summary, _ = paired_bootstrap(raw, trans, "race_date", int(cfg["bootstrap"]["n_bootstrap"]), int(cfg["bootstrap"]["seed"]) + 17, int(cfg["bootstrap"]["ece_bins"]))
    by_year = bootstrap_by_year(raw, trans, cfg)
    boot_summary = pd.concat([race_summary, day_summary], ignore_index=True)
    starts_decision = choose_starts(boot_summary, by_year)
    clean_base_pred = raw.copy() if starts_decision["selected_starts"] == "raw" else trans.copy()
    clean_base_key = starts_decision["selected_model_key"]
    clean_base_pred["model_key"] = clean_base_key

    df = load_dataset(base_cfg, numeric, cat, smoke)
    tdf = target_frame(df, base_cfg)
    market_pred, _ = make_market_predictions(tdf, base_cfg, out, model_root, smoke)
    all_oof = [clean_base_pred]
    logs = []
    for name, spec in cfg["additional_ablation_candidates"].items():
        pred, lg = train_oof(name, spec["drop_features"], market_pred, base_cfg, numeric, cat, params, out, model_root, resume)
        all_oof.append(calibrate_oof(pred))
        logs.extend(lg)
    add_pred = pd.concat(all_oof, ignore_index=True, sort=False)
    m, r, e, ro = summarize(add_pred, {**base_cfg, "bootstrap_iterations_roi": cfg["bootstrap_iterations_roi"], "random_seed": cfg["random_seed"]})
    add_summary = full_summary(m, r, e, ro)
    selected = select_additional(add_summary, clean_base_key)

    selected_key = selected["selected_model_key"]
    selected_name = "clean_base" if selected["selected_additional"] is None else selected["selected_additional"]
    selected_drops = cfg["clean_base"]["drop_features"] if selected["selected_additional"] is None else cfg["additional_ablation_candidates"][selected["selected_additional"]]["drop_features"]
    selected_oof = add_pred[add_pred["model_key"].eq(selected_key)].copy()
    raw_final = pd.read_parquet(Path(cfg["phase1_output_root"]) / "predictions" / "drop_person_codes" / "final_2025_2026.parquet")
    if selected_key == RAW_KEY:
        final_pred = raw_final.copy()
        diag_pred = raw_final.copy()
    else:
        final_pred = train_final(selected_name, selected_key, selected_drops, market_pred, base_cfg, numeric, cat, params, selected_oof, out, model_root, resume)
        diag_pred = pd.concat([raw_final, final_pred], ignore_index=True, sort=False)
    dm, dr, de, dro = summarize(diag_pred, {**base_cfg, "bootstrap_iterations_roi": cfg["bootstrap_iterations_roi"], "random_seed": cfg["random_seed"]})
    if selected_name == "clean_base":
        phase1 = Path(cfg["phase1_output_root"])
        pvc = pd.read_csv(phase1 / "selected_model_pvc_summary.csv")
        shap = pd.read_csv(phase1 / "selected_model_shap_summary.csv")
        shap_add = pd.read_csv(phase1 / "selected_model_shap_additivity.csv")
        pvc["source"] = "reused_phase1_drop_person_codes"
        shap["source"] = "reused_phase1_drop_person_codes"
        shap_add["source"] = "reused_phase1_drop_person_codes"
    else:
        pvc, shap, shap_add = fi_shap(selected_name, selected_key, selected_drops, market_pred, base_cfg, numeric, cat, out, model_root, int(cfg["shap_sample_per_year"]), int(cfg["random_seed"]))

    hashes = {
        "paired_prediction_artifact_check.csv": atomic_write_csv(out / "paired_prediction_artifact_check.csv", check),
        "paired_bootstrap_summary.csv": atomic_write_csv(out / "paired_bootstrap_summary.csv", boot_summary),
        "paired_bootstrap_samples.parquet": atomic_write_parquet(out / "paired_bootstrap_samples.parquet", race_samples),
        "paired_bootstrap_by_year.csv": atomic_write_csv(out / "paired_bootstrap_by_year.csv", by_year),
        "paired_bootstrap_day_level_summary.csv": atomic_write_csv(out / "paired_bootstrap_day_level_summary.csv", day_summary),
        "cumulative_starts_transform_decision.json": atomic_write_json(out / "cumulative_starts_transform_decision.json", starts_decision),
        "additional_ablation_artifact_check.csv": atomic_write_csv(out / "additional_ablation_artifact_check.csv", pd.DataFrame(logs)),
        "additional_ablation_by_fold.csv": atomic_write_csv(out / "additional_ablation_by_fold.csv", m),
        "additional_ablation_2020_2024.csv": atomic_write_csv(out / "additional_ablation_2020_2024.csv", add_summary),
        "additional_ablation_residual_stability.csv": atomic_write_csv(out / "additional_ablation_residual_stability.csv", r),
        "additional_ablation_ev_stability.csv": atomic_write_csv(out / "additional_ablation_ev_stability.csv", e),
        "additional_ablation_roi_diagnostic.csv": atomic_write_csv(out / "additional_ablation_roi_diagnostic.csv", ro),
        "selected_feature_set_phase3.json": atomic_write_json(out / "selected_feature_set_phase3.json", selected),
        "phase3_2025_2026_diagnostic.csv": atomic_write_csv(out / "phase3_2025_2026_diagnostic.csv", dm),
        "phase3_2025_2026_residual.csv": atomic_write_csv(out / "phase3_2025_2026_residual.csv", dr),
        "phase3_2025_2026_ev.csv": atomic_write_csv(out / "phase3_2025_2026_ev.csv", de),
        "phase3_2025_2026_roi.csv": atomic_write_csv(out / "phase3_2025_2026_roi.csv", dro),
        "selected_model_feature_importance.csv": atomic_write_csv(out / "selected_model_feature_importance.csv", pvc),
        "selected_model_shap.csv": atomic_write_csv(out / "selected_model_shap.csv", shap),
        "selected_model_shap_additivity.csv": atomic_write_csv(out / "selected_model_shap_additivity.csv", shap_add),
    }
    elapsed = time.time() - started
    manifest = {
        "version": cfg["version"],
        "db_usage": "not_read; existing parquet feature dataset only",
        "feature_dataset_rebuild": False,
        "source_parquet_modified": False,
        "random_split_used": False,
        "tree_count": 300,
        "calibration_method": "isotonic",
        "selection_years": cfg["selection_years"],
        "diagnostic_years": cfg["diagnostic_years"],
        "starts_decision": starts_decision,
        "selected_feature_set": selected,
        "monthday_changed": False,
        "execution_constraints": {**cfg["execution_constraints"], "model_selection_hyperparameter": False},
        "catboost_params": params,
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "output_hashes": hashes,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_report(Path("docs/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1_results.md"), boot_summary, starts_decision, add_summary, selected, dm, de, dro, pd.DataFrame(logs), elapsed)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_feature_cleanup_phase3_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume or not args.no_resume, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
