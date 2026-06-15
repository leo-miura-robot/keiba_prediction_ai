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


BASE_KEY = "C1R0_pure_market_offset_fixed300_base"
WORKING_KEY = "C1R0_fixed300_ablation_drop_person_codes"


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


def model_key(name: str) -> str:
    return f"C1R0_300_feature_cleanup_phase2_{name}"


def phase1_model_key(name: str) -> str:
    if name == "base_fixed300":
        return BASE_KEY
    return f"C1R0_fixed300_ablation_{name}"


def summarize(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, resid_rows, ev_rows, roi_rows = [], [], [], []
    for keys, g in pred.groupby(["model_key", "period", "Year"], dropna=False):
        label = {"model_key": keys[0], "period": keys[1], "Year": int(keys[2])}
        metric_rows.append(metric_row(g, "final_probability", label, float(cfg["epsilon"])))
        resid_rows.append(residual_stats(g["catboost_residual_score"], label))
        ev_rows.append(ev_row(g, label))
        d = add_eval_columns(g, "final_probability")
        bets = d[d["adjusted_place_ev"] >= 1.0]
        ci = bootstrap_ci(bets, int(cfg.get("bootstrap_iterations", 1000)), int(cfg.get("random_seed", 42)))
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
            "residual_mean": float(r["residual_mean"].mean()),
            "residual_std": float(r["residual_std"].mean()),
            "residual_std_cv": float(r["residual_std"].std(ddof=1) / r["residual_std"].mean()),
            "abs_residual_p90": float(r["abs_residual_p90"].mean()),
            "abs_residual_p95": float(r["abs_residual_p95"].mean()),
            "abs_residual_p99": float(r["abs_residual_p99"].mean()),
            "abs_residual_p95_cv": float(r["abs_residual_p95"].std(ddof=1) / r["abs_residual_p95"].mean()),
            "abs_residual_p99_cv": float(r["abs_residual_p99"].std(ddof=1) / r["abs_residual_p99"].mean()),
            "ev_ge_1_count_sum": int(e["ev_ge_1_count"].sum()),
            "ev_ge_1_count_cv": float(e["ev_ge_1_count"].std(ddof=1) / e["ev_ge_1_count"].mean()),
            "market_lt1_to_final_ge1_sum": int(e["market_lt1_to_final_ge1"].sum()),
            "market_ge1_to_final_lt1_sum": int(e["market_ge1_to_final_lt1"].sum()),
            "ev_roi_spearman": float(e["ev_roi_spearman"].mean()),
            "ev_ge_1_roi": float(ro["roi"].mean()),
            "top1_removed_roi": float(ro["top1_removed_roi"].mean()),
            "top3_removed_roi": float(ro["top3_removed_roi"].mean()),
            "top5_removed_roi": float(ro["top5_removed_roi"].mean()),
            "top10_removed_roi": float(ro["top10_removed_roi"].mean()),
            "bootstrap_roi_p025": float(ro["bootstrap_roi_p025"].mean()),
            "bootstrap_roi_p500": float(ro["bootstrap_roi_p500"].mean()),
            "bootstrap_roi_p975": float(ro["bootstrap_roi_p975"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["combined_logloss", "combined_brier"])


def select_by_priority(summary: pd.DataFrame, preferred: str | None = None) -> dict[str, Any]:
    s = summary.copy()
    s = s.sort_values([
        "combined_logloss",
        "combined_brier",
        "combined_ece",
        "abs_residual_p95",
        "abs_residual_p99",
        "residual_std_cv",
        "ev_ge_1_count_sum",
        "ev_ge_1_count_cv",
        "ev_roi_spearman",
    ], ascending=[True, True, True, True, True, True, True, True, False])
    row = s.iloc[0].to_dict()
    if preferred and preferred in set(s["model_key"]):
        best = s.iloc[0]
        pref = s[s["model_key"].eq(preferred)].iloc[0]
        if float(pref["combined_logloss"]) <= float(best["combined_logloss"]) + 0.0005 and float(pref["combined_brier"]) <= float(best["combined_brier"]) + 0.0002:
            row = pref.to_dict()
    return {
        "selected_model_key": row["model_key"],
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "selection_rule": "2020-2024 only; Logloss, Brier, calibration, residual stability, EV count stability, EV-ROI Spearman; ROI auxiliary.",
        "selected_row": row,
    }


def select_starts_preprocessing(summary: pd.DataFrame) -> dict[str, Any]:
    s = summary.copy()
    best_logloss = float(s["combined_logloss"].min())
    best_brier = float(s["combined_brier"].min())
    tier = s[(s["combined_logloss"] <= best_logloss + 0.00001) & (s["combined_brier"] <= best_brier + 0.00001)].copy()
    if tier.empty:
        tier = s.copy()
    simplicity = {
        "C1R0_300_feature_cleanup_phase2_starts_raw": 0,
        "C1R0_300_feature_cleanup_phase2_starts_log1p": 1,
        "C1R0_300_feature_cleanup_phase2_starts_clip_p99": 2,
        "C1R0_300_feature_cleanup_phase2_starts_clip_p99_log1p": 3,
        "C1R0_300_feature_cleanup_phase2_starts_drop": 4,
    }
    tier["simplicity_rank"] = tier["model_key"].map(simplicity).fillna(9)
    row = tier.sort_values([
        "combined_ece",
        "ev_roi_spearman",
        "abs_residual_p95",
        "abs_residual_p99",
        "ev_ge_1_count_cv",
        "simplicity_rank",
        "combined_logloss",
    ], ascending=[True, False, True, True, True, True, True]).iloc[0].to_dict()
    return {
        "selected_model_key": row["model_key"],
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "selection_rule": "2020-2024 only; Logloss/Brier near-tie tier, then calibration, residual tails, EV stability, EV-ROI Spearman, and simplicity. ROI auxiliary.",
        "selected_row": row,
    }


def load_phase1_predictions(phase1: Path, fixed: Path) -> pd.DataFrame:
    base = pd.read_parquet(fixed / "fixed_tree_predictions_2020_2024.parquet")
    base = base[base["tree_count_candidate"].eq(300)].copy()
    base["model_key"] = BASE_KEY
    base["ablation_name"] = "base_fixed300"
    ab = pd.read_parquet(phase1 / "ablation_oof_predictions.parquet")
    return pd.concat([base, ab], ignore_index=True, sort=False)


def artifact_check(cfg: dict[str, Any]) -> pd.DataFrame:
    phase1 = Path(cfg["phase1_output_root"])
    model_root = Path(cfg["phase1_model_root"])
    pred = pd.read_parquet(phase1 / "ablation_oof_predictions.parquet", columns=["ablation_name", "fold", "Year", "tree_count", "feature_hash"])
    rows = []
    for name in ["drop_person_codes", "drop_global_cumulative_starts", "drop_raw_body_weight", "drop_unadjusted_raw_time", "drop_meeting_admin"]:
        for year in [2020, 2021, 2022, 2023, 2024]:
            fold = f"fold_{year}"
            mp = model_root / name / "folds" / fold / "model.cbm"
            g = pred[pred["ablation_name"].eq(name) & pred["fold"].eq(fold) & pred["Year"].eq(year)]
            ok = mp.exists() and not g.empty and int(g["tree_count"].dropna().iloc[0]) == 300
            rows.append({
                "ablation_name": name,
                "fold": fold,
                "Year": year,
                "model_exists": mp.exists(),
                "prediction_rows": int(len(g)),
                "tree_count": int(g["tree_count"].dropna().iloc[0]) if not g.empty else None,
                "feature_hash": str(g["feature_hash"].dropna().iloc[0]) if not g.empty else "",
                "model_sha256": sha256_file(mp) if mp.exists() else "",
                "status": "ok" if ok else "bad",
            })
    return pd.DataFrame(rows)


def transform_starts(df: pd.DataFrame, features: list[str], transform: str, params: dict[str, float] | None = None) -> pd.DataFrame:
    if transform in {"raw", "drop"}:
        return df
    out = df.copy()
    for f in features:
        x = pd.to_numeric(out[f], errors="coerce")
        if transform in {"clip_p99", "clip_p99_log1p"}:
            x = x.clip(upper=float(params[f]))
        if transform in {"log1p", "clip_p99_log1p"}:
            x = np.log1p(x.clip(lower=0))
        out[f] = x
    return out


def transform_params(train: pd.DataFrame, features: list[str], transform: str, label: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    params, rows = {}, []
    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        p99 = float(x.quantile(0.99))
        params[f] = p99
        rows.append({**label, "feature": f, "transform": transform, "train_p99": p99, "train_min": float(x.min()), "train_median": float(x.median()), "train_mean": float(x.mean()), "train_max": float(x.max())})
    return params, rows


def valid_reuse(mp: Path, pp: Path, tree_count: int, rows: int, feature_hash: str) -> tuple[bool, str]:
    if not mp.exists() or not pp.exists():
        return False, "missing"
    try:
        m = CatBoostClassifier()
        m.load_model(str(mp))
        if int(m.tree_count_) != int(tree_count):
            return False, "tree_count_mismatch"
        p = pd.read_parquet(pp, columns=["feature_hash"])
        if len(p) != rows:
            return False, "row_count_mismatch"
        if str(p["feature_hash"].iloc[0]) != feature_hash:
            return False, "feature_hash_mismatch"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def train_candidate_oof(
    name: str,
    market_pred: pd.DataFrame,
    base_cfg: dict[str, Any],
    numeric: list[str],
    cat: list[str],
    params: dict[str, Any],
    transform: str,
    transform_features: list[str],
    out: Path,
    model_root: Path,
    resume: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    parts, reuse_rows, param_rows = [], [], []
    fh = sha256_json({"numeric": numeric, "categorical": cat, "transform": transform, "features": transform_features, "tree_count": 300})
    for fold in base_cfg["folds"]:
        scoped = market_pred[market_pred["baseline_scope"].eq(fold["name"])]
        train = scoped[scoped["Year"].isin(fold["train_years"])].copy()
        valid = scoped[scoped["Year"].eq(fold["validation_year"])].copy()
        label = {"model_key": model_key(name), "fold_eval_year": int(fold["validation_year"]), "train_start_year": min(fold["train_years"]), "train_end_year": max(fold["train_years"])}
        tparams, rows = transform_params(train, transform_features, transform, label)
        param_rows.extend(rows)
        train = transform_starts(train, transform_features, transform, tparams)
        valid = transform_starts(valid, transform_features, transform, tparams)
        mp = model_root / name / "folds" / fold["name"] / "model.cbm"
        pp = out / "predictions" / name / f"{fold['name']}.parquet"
        ok, reason = valid_reuse(mp, pp, 300, len(valid), fh) if resume else (False, "resume_disabled")
        if ok:
            print(f"[reuse] {name} {fold['name']}", flush=True)
            parts.append(pd.read_parquet(pp))
            reuse_rows.append({"model": name, "fold": fold["name"], "action": "reuse", "reason": reason})
            continue
        print(f"[train] {name} {fold['name']}: {reason}", flush=True)
        model = train_fixed_model(train, numeric, cat, params, mp)
        raw, residual, prob = predict_parts(model, valid, numeric, cat)
        valid["model_key"] = model_key(name)
        valid["variant_name"] = name
        valid["probability"] = prob
        valid["final_probability"] = prob
        valid["final_probability_raw"] = raw
        valid["catboost_residual_score"] = residual
        valid["tree_count"] = int(model.tree_count_)
        valid["feature_hash"] = fh
        atomic_write_parquet(pp, valid)
        parts.append(valid)
        reuse_rows.append({"model": name, "fold": fold["name"], "action": "train", "reason": reason, "model_sha256": sha256_file(mp)})
    return pd.concat(parts, ignore_index=True), reuse_rows, param_rows


def calibrate(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    parts = []
    for key, d in pred.groupby("model_key"):
        cal = fit_one_calibrator("isotonic", d[d["Year"].between(2020, 2024)], "probability")
        x = d.copy()
        x["final_probability"] = cal.transform(x["probability"].to_numpy(float))
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def monthday_audit(pred: pd.DataFrame, out: Path) -> pd.DataFrame:
    d = pred[pred["model_key"].eq(WORKING_KEY)].copy()
    m = pd.to_numeric(d["MonthDay"], errors="coerce")
    month = (m // 100).astype("Int64")
    day = (m % 100).astype("Int64")
    rows = []
    for y, g in d.groupby("Year"):
        mg = pd.to_numeric(g["MonthDay"], errors="coerce")
        rows.append({"Year": int(y), "rows": len(g), "missing_rate": float(mg.isna().mean()), "unique_count": int(mg.nunique()), "min": float(mg.min()), "max": float(mg.max()), "spearman_with_year_all": float(spearmanr(m, d["Year"], nan_policy="omit").statistic)})
    audit = pd.DataFrame(rows)
    md = pd.DataFrame({
        "feature": ["MonthDay"],
        "encoding": ["MMDD integer"],
        "numeric_or_categorical": ["numeric"],
        "missing_rate": [float(m.isna().mean())],
        "unique_count": [int(m.nunique())],
        "spearman_with_year": [float(spearmanr(m, d["Year"], nan_policy="omit").statistic)],
        "spearman_with_month": [float(spearmanr(m, month, nan_policy="omit").statistic)],
        "spearman_with_kaiji": [float(spearmanr(m, pd.to_numeric(d["Kaiji"], errors="coerce"), nan_policy="omit").statistic)],
        "spearman_with_nichiji": [float(spearmanr(m, pd.to_numeric(d["Nichiji"], errors="coerce"), nan_policy="omit").statistic)],
        "december_january_discontinuity": [True],
        "interpretation": ["season/date-order signal, not direct result leakage; numeric MMDD has artificial year-end discontinuity"],
    })
    atomic_write_text(out / "monthday_audit.md", md.to_markdown(index=False) + "\n\n" + audit.to_markdown(index=False) + "\n")
    return md


def train_final(name: str, selected_key: str, market_pred: pd.DataFrame, base_cfg: dict[str, Any], numeric: list[str], cat: list[str], params: dict[str, Any], transform: str, transform_features: list[str], oof: pd.DataFrame, out: Path, model_root: Path, resume: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fh = sha256_json({"numeric": numeric, "categorical": cat, "transform": transform, "features": transform_features, "tree_count": 300, "final": True})
    train = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin(base_cfg["final_train_years"])].copy()
    eval_df = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin([2025, 2026])].copy()
    tparams, param_rows = transform_params(train, transform_features, transform, {"model_key": selected_key, "fold_eval_year": "final_2025_2026", "train_start_year": 2016, "train_end_year": 2024})
    train = transform_starts(train, transform_features, transform, tparams)
    eval_df = transform_starts(eval_df, transform_features, transform, tparams)
    mp = model_root / name / "final" / "model.cbm"
    pp = out / "predictions" / name / "final_2025_2026.parquet"
    ok, reason = valid_reuse(mp, pp, 300, len(eval_df), fh) if resume else (False, "resume_disabled")
    if ok:
        print(f"[reuse] {name} final", flush=True)
        return pd.read_parquet(pp), param_rows
    print(f"[train] {name} final: {reason}", flush=True)
    model = train_fixed_model(train, numeric, cat, params, mp)
    raw, residual, prob = predict_parts(model, eval_df, numeric, cat)
    cal = fit_one_calibrator("isotonic", oof[oof["model_key"].eq(selected_key)], "probability")
    eval_df["model_key"] = selected_key
    eval_df["variant_name"] = name
    eval_df["probability"] = prob
    eval_df["final_probability"] = cal.transform(prob)
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual
    eval_df["feature_hash"] = fh
    eval_df["tree_count"] = 300
    atomic_write_parquet(pp, eval_df)
    return eval_df, param_rows


def fi_shap_selected(name: str, selected_key: str, market_pred: pd.DataFrame, base_cfg: dict[str, Any], numeric: list[str], cat: list[str], out: Path, model_root: Path, sample: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = numeric + cat
    pvc_rows, shap_rows, add_rows = [], [], []
    for fold in base_cfg["folds"]:
        mp = model_root / name / "folds" / fold["name"] / "model.cbm"
        model = CatBoostClassifier()
        model.load_model(str(mp))
        for f, v in zip(features, model.get_feature_importance(type="PredictionValuesChange")):
            pvc_rows.append({"feature": f, "Year": fold["validation_year"], "importance": float(v)})
        valid = market_pred[market_pred["baseline_scope"].eq(fold["name"]) & market_pred["Year"].eq(fold["validation_year"])].copy()
        if len(valid) > sample:
            valid = valid.iloc[rng.choice(len(valid), sample, replace=False)].copy()
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        sv = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)
        vals, expected = sv[:, :-1], sv[:, -1]
        residual = np.asarray(model.predict(Pool(x, cat_features=cat_indices(x, cat)), prediction_type="RawFormulaVal"), dtype=float)
        add_rows.append({"model_key": selected_key, "Year": fold["validation_year"], "rows": len(valid), "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1)))))})
        for idx, f in enumerate(features):
            v = vals[:, idx]
            shap_rows.append({"feature": f, "Year": fold["validation_year"], "mean_abs_shap": float(np.mean(np.abs(v))), "mean_signed_shap": float(np.mean(v)), "p99_abs_shap": float(np.percentile(np.abs(v), 99))})
    pvc = pd.DataFrame(pvc_rows).groupby("feature", as_index=False).agg(weighted_mean=("importance", "mean")).sort_values("weighted_mean", ascending=False)
    shap = pd.DataFrame(shap_rows).groupby("feature", as_index=False).agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), p99_abs_shap=("p99_abs_shap", "mean")).sort_values("mean_abs_shap", ascending=False)
    return pvc, shap, pd.DataFrame(add_rows)


def run(config_path: Path, resume: bool = True, smoke: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        cfg["bootstrap_iterations"] = cfg["smoke_overrides"]["bootstrap_iterations"]
        cfg["shap_sample_per_year"] = cfg["smoke_overrides"]["shap_sample_per_year"]
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    if smoke:
        base_cfg = dict(base_cfg)
        base_cfg["smoke_overrides"] = {**base_cfg.get("smoke_overrides", {}), "train_rows_per_year": cfg["smoke_overrides"]["train_rows_per_year"], "eval_rows_per_year": cfg["smoke_overrides"]["eval_rows_per_year"]}
    out, model_root = Path(cfg["output_root"]), Path(cfg["model_root"])
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    fixed, phase1 = Path(cfg["fixed300_output_root"]), Path(cfg["phase1_output_root"])
    numeric, cat = load_allowlist(fixed)
    params = fixed_params(base_cfg["training_params"], 300, smoke, gpu_ram_part=cfg["execution_constraints"]["gpu_ram_part"])

    check = artifact_check(cfg)
    phase1_pred = load_phase1_predictions(phase1, fixed)
    m, r, e, ro = summarize(phase1_pred, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    existing_summary = full_summary(m, r, e, ro)
    working = select_by_priority(existing_summary, preferred=WORKING_KEY)
    month_audit = monthday_audit(phase1_pred, out)

    work_num, work_cat = candidate_features(numeric, cat, cfg["working_base_drop_features"])
    df = load_dataset(base_cfg, numeric, cat, smoke)
    tdf = target_frame(df, base_cfg)
    market_pred, _ = make_market_predictions(tdf, base_cfg, out, model_root, smoke)

    month_num, month_cat = candidate_features(numeric, cat, cfg["monthday_ablation"]["drop_features"])
    month_oof, month_reuse, month_params = train_candidate_oof("no_monthday", market_pred, base_cfg, month_num, month_cat, params, "raw", [], out, model_root, resume)
    month_oof = calibrate(month_oof, base_cfg)
    work_oof = phase1_pred[phase1_pred["model_key"].eq(WORKING_KEY) & phase1_pred["Year"].between(2020, 2024)].copy()
    month_comp_pred = pd.concat([work_oof, month_oof], ignore_index=True, sort=False)
    mm, mr, me, mro = summarize(month_comp_pred, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    month_summary = full_summary(mm, mr, me, mro)

    parts = [work_oof.copy()]
    parts[0]["model_key"] = model_key("starts_raw")
    reuse_rows = month_reuse
    param_rows = month_params
    starts = cfg["cumulative_starts"]["features"]
    for vname, vcfg in cfg["cumulative_starts"]["variants"].items():
        if vname == "raw":
            continue
        n2, c2 = candidate_features(numeric, cat, vcfg["drop_features"])
        tf = vcfg["transform"]
        tfeatures = [] if tf == "drop" else starts
        pred, rr, pr = train_candidate_oof(f"starts_{vname}", market_pred, base_cfg, n2, c2, params, tf, tfeatures, out, model_root, resume)
        parts.append(calibrate(pred, base_cfg))
        reuse_rows.extend(rr)
        param_rows.extend(pr)
    starts_pred = pd.concat(parts, ignore_index=True, sort=False)
    sm, sr, se, sro = summarize(starts_pred, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    starts_summary = full_summary(sm, sr, se, sro)
    starts_selected = select_starts_preprocessing(starts_summary)

    month_selected = select_by_priority(month_summary, preferred=WORKING_KEY)
    selected_key = starts_selected["selected_model_key"]
    selected_name = selected_key.replace("C1R0_300_feature_cleanup_phase2_", "")
    selected_variant = selected_name.replace("starts_", "")
    transform = cfg["cumulative_starts"]["variants"].get(selected_variant, {"transform": "raw"})["transform"]
    selected_drops = cfg["cumulative_starts"]["variants"].get(selected_variant, {"drop_features": cfg["working_base_drop_features"]})["drop_features"]
    selected_transform_features = [] if transform in {"raw", "drop"} else starts
    final_num, final_cat = candidate_features(numeric, cat, selected_drops)
    final_diag, final_param_rows = train_final(selected_name, selected_key, market_pred, base_cfg, final_num, final_cat, params, transform, selected_transform_features, starts_pred, out, model_root, resume)
    param_rows.extend(final_param_rows)
    base_diag = pd.read_parquet(phase1 / "predictions" / "drop_person_codes" / "final_2025_2026.parquet")
    diag_pred = pd.concat([base_diag, final_diag], ignore_index=True, sort=False)
    dm, dr, de, dro = summarize(diag_pred, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    if selected_name == "starts_raw":
        pvc = pd.read_csv(phase1 / "selected_model_pvc_summary.csv")
        shap = pd.read_csv(phase1 / "selected_model_shap_summary.csv")
        shap_add = pd.read_csv(phase1 / "selected_model_shap_additivity.csv")
        pvc["source"] = "reused_phase1_drop_person_codes"
        shap["source"] = "reused_phase1_drop_person_codes"
        shap_add["source"] = "reused_phase1_drop_person_codes"
    else:
        pvc, shap, shap_add = fi_shap_selected(selected_name, selected_key, market_pred, base_cfg, final_num, final_cat, out, model_root, int(cfg["shap_sample_per_year"]), int(cfg["random_seed"]))

    hashes = {
        "existing_ablation_artifact_check.csv": atomic_write_csv(out / "existing_ablation_artifact_check.csv", check),
        "existing_five_ablation_full_comparison.csv": atomic_write_csv(out / "existing_five_ablation_full_comparison.csv", existing_summary),
        "existing_five_ablation_by_year.csv": atomic_write_csv(out / "existing_five_ablation_by_year.csv", m),
        "working_base_model.json": atomic_write_json(out / "working_base_model.json", working),
        "monthday_feature_audit.csv": atomic_write_csv(out / "monthday_feature_audit.csv", month_audit),
        "monthday_ablation_comparison.csv": atomic_write_csv(out / "monthday_ablation_comparison.csv", month_summary),
        "cumulative_starts_transform_params_by_fold.csv": atomic_write_csv(out / "cumulative_starts_transform_params_by_fold.csv", pd.DataFrame(param_rows)),
        "cumulative_starts_comparison_by_fold.csv": atomic_write_csv(out / "cumulative_starts_comparison_by_fold.csv", sm),
        "cumulative_starts_comparison_2020_2024.csv": atomic_write_csv(out / "cumulative_starts_comparison_2020_2024.csv", starts_summary),
        "cumulative_starts_residual_stability.csv": atomic_write_csv(out / "cumulative_starts_residual_stability.csv", sr),
        "cumulative_starts_ev_stability.csv": atomic_write_csv(out / "cumulative_starts_ev_stability.csv", se),
        "selected_preprocessing.json": atomic_write_json(out / "selected_preprocessing.json", starts_selected),
        "phase2_model_comparison_2020_2024.csv": atomic_write_csv(out / "phase2_model_comparison_2020_2024.csv", starts_summary),
        "phase2_2025_2026_diagnostic.csv": atomic_write_csv(out / "phase2_2025_2026_diagnostic.csv", dm),
        "phase2_2025_2026_residual.csv": atomic_write_csv(out / "phase2_2025_2026_residual.csv", dr),
        "phase2_2025_2026_ev.csv": atomic_write_csv(out / "phase2_2025_2026_ev.csv", de),
        "phase2_2025_2026_roi.csv": atomic_write_csv(out / "phase2_2025_2026_roi.csv", dro),
        "selected_model_feature_importance.csv": atomic_write_csv(out / "selected_model_feature_importance.csv", pvc),
        "selected_model_shap.csv": atomic_write_csv(out / "selected_model_shap.csv", shap),
        "selected_model_shap_additivity.csv": atomic_write_csv(out / "selected_model_shap_additivity.csv", shap_add),
        "reuse_training_log.csv": atomic_write_csv(out / "reuse_training_log.csv", pd.DataFrame(reuse_rows)),
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
        "working_base": working,
        "monthday_selection": month_selected,
        "selected_preprocessing": starts_selected,
        "integrated_model_created": False,
        "integrated_model_reason": "MonthDay removal was evaluated separately; cumulative-starts preprocessing selection did not require combining independent changes unless both clearly won.",
        "execution_constraints": {**cfg["execution_constraints"], "model_selection_hyperparameter": False},
        "catboost_params": params,
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "output_hashes": hashes,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_report(Path("docs/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1_results.md"), existing_summary, working, month_audit, month_summary, starts_summary, starts_selected, dm, de, dro, pd.DataFrame(reuse_rows), elapsed)
    return manifest


def write_report(path: Path, existing: pd.DataFrame, working: dict[str, Any], month_audit: pd.DataFrame, month_summary: pd.DataFrame, starts_summary: pd.DataFrame, selected: dict[str, Any], diag: pd.DataFrame, ev: pd.DataFrame, roi: pd.DataFrame, reuse: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# C1R0 Feature Cleanup Phase2 Results",
        "",
        "## Existing Five Ablations",
        existing.to_markdown(index=False),
        "",
        "## Working Base",
        json.dumps(working, ensure_ascii=False, indent=2),
        "",
        "## MonthDay Audit",
        month_audit.to_markdown(index=False),
        "",
        "## MonthDay Ablation",
        month_summary.to_markdown(index=False),
        "",
        "## Cumulative Starts Comparison",
        starts_summary.to_markdown(index=False),
        "",
        "## Selected Preprocessing",
        json.dumps(selected, ensure_ascii=False, indent=2),
        "",
        "## 2025/2026 Diagnostic Metrics",
        diag.to_markdown(index=False),
        "",
        "## 2025/2026 EV",
        ev.to_markdown(index=False),
        "",
        "## 2025/2026 ROI",
        roi.to_markdown(index=False),
        "",
        "## Reuse Training Log",
        reuse.to_markdown(index=False),
        "",
        f"Elapsed seconds: `{elapsed:.1f}`",
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_feature_cleanup_phase2_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume or not args.no_resume, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
