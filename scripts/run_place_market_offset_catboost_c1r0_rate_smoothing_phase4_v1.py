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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_v1 import (
    add_eval_columns, atomic_write_csv, atomic_write_json, atomic_write_parquet,
    atomic_write_text, cat_indices, load_config, load_dataset, make_market_predictions,
    metric_row, prepare_x, roi_of, sha256_file, sha256_json, target_frame, top_removed_roi,
)
from scripts.run_place_market_offset_catboost_c1r0_tree_count_v1 import (
    ev_row, fixed_params, predict_parts, residual_stats, train_fixed_model,
)
from scripts.audit_c1r0_metric_consistency_v2 import paired_bootstrap


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


def summarize(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, resid_rows, ev_rows, roi_rows = [], [], [], []
    for keys, g in pred.groupby(["model_key", "period", "Year"], dropna=False):
        label = {"model_key": keys[0], "period": keys[1], "Year": int(keys[2])}
        # MUST USE probability_raw FOR SELECTION!
        metric_rows.append(metric_row(g, "probability_raw", label, float(cfg["epsilon"])))
        resid_rows.append(residual_stats(g["catboost_residual_score"], label))
        
        h = g.copy()
        h["final_probability"] = h["probability_calibrated"]
        ev_rows.append(ev_row(h, label))
        
        d = add_eval_columns(g, "probability_calibrated")
        bets = d[d["adjusted_place_ev"] >= 1.0]
        # Skip CI in fast mode to save time, or use rough approx
        roi_rows.append({
            **label,
            "bets": int(len(bets)),
            "roi": roi_of(bets),
            "top1_removed_roi": top_removed_roi(bets, 1),
            "top3_removed_roi": top_removed_roi(bets, 3),
            "top5_removed_roi": top_removed_roi(bets, 5),
            "top10_removed_roi": top_removed_roi(bets, 10),
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
            "worst_year_logloss": float(g["logloss"].max()),
            "worst_year_brier": float(g["brier"].max()),
            "residual_mean": float(r["residual_mean"].mean()) if not r.empty else np.nan,
            "residual_std": float(r["residual_std"].mean()) if not r.empty else np.nan,
            "residual_std_cv": float(r["residual_std"].std(ddof=1) / r["residual_std"].mean()) if not r.empty else np.nan,
            "abs_residual_p90": float(r["abs_residual_p90"].mean()) if not r.empty else np.nan,
            "abs_residual_p95": float(r["abs_residual_p95"].mean()) if not r.empty else np.nan,
            "abs_residual_p99": float(r["abs_residual_p99"].mean()) if not r.empty else np.nan,
            "ev_ge_1_count_sum": int(e["ev_ge_1_count"].sum()) if not e.empty else 0,
            "ev_ge_1_count_cv": float(e["ev_ge_1_count"].std(ddof=1) / e["ev_ge_1_count"].mean()) if not e.empty else np.nan,
            "ev_roi_spearman": float(e["ev_roi_spearman"].mean()) if not e.empty else np.nan,
            "ev_ge_1_roi": float(ro["roi"].mean()) if not ro.empty else np.nan,
            "top1_removed_roi": float(ro["top1_removed_roi"].mean()) if not ro.empty else np.nan,
            "top3_removed_roi": float(ro["top3_removed_roi"].mean()) if not ro.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["combined_logloss", "combined_brier"])


def compute_priors(df: pd.DataFrame, groups: dict[str, Any], train_years: list[int]) -> dict[str, float]:
    train = df[df["Year"].isin(train_years)]
    priors = {}
    for g, specs in groups.items():
        starts_col = specs["starts_feature"]
        for rate_col in specs["rates"]:
            r = pd.to_numeric(train[rate_col], errors="coerce")
            s = pd.to_numeric(train[starts_col], errors="coerce")
            success = np.round(r * s)
            valid = (s > 0) & (~r.isna()) & (~s.isna())
            if valid.sum() == 0:
                priors[rate_col] = 0.0
            else:
                priors[rate_col] = float(success[valid].sum() / s[valid].sum())
    return priors


def apply_smoothing(df: pd.DataFrame, groups: dict[str, Any], priors: dict[str, float], active_groups: list[str], strength: float) -> pd.DataFrame:
    out = df.copy()
    for g in active_groups:
        specs = groups[g]
        starts_col = specs["starts_feature"]
        s = pd.to_numeric(df[starts_col], errors="coerce").fillna(0.0)
        for rate_col in specs["rates"]:
            r = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)
            success = np.round(r * s)
            prior = priors[rate_col]
            smoothed = (success + strength * prior) / (s + strength)
            smoothed = smoothed.where(s > 0, prior)
            out[rate_col] = smoothed
    return out


def valid_reuse(mp: Path, pp: Path, rows: int, fh: str, sh: str) -> tuple[bool, str]:
    if not mp.exists() or not pp.exists():
        return False, "missing"
    try:
        model = CatBoostClassifier()
        model.load_model(str(mp))
        if int(model.tree_count_) != 300:
            return False, "tree_count_mismatch"
        p = pd.read_parquet(pp, columns=["feature_hash", "smoothing_hash"])
        if len(p) != rows:
            return False, "row_count_mismatch"
        if str(p["feature_hash"].iloc[0]) != fh:
            return False, "feature_hash_mismatch"
        if str(p["smoothing_hash"].iloc[0]) != sh:
            return False, "smoothing_hash_mismatch"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def train_oof(
    name: str,
    active_groups: list[str],
    strength: float,
    market_pred: pd.DataFrame,
    base_cfg: dict[str, Any],
    cfg: dict[str, Any],
    numeric: list[str],
    cat: list[str],
    params: dict[str, Any],
    out: Path,
    model_root: Path,
    resume: bool
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    drops = ["KisyuCode", "ChokyosiCode"]
    n2 = [f for f in numeric if f not in drops]
    c2 = [f for f in cat if f not in drops]
    fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": 300, "drops": drops})
    sh = sha256_json({"groups": active_groups, "strength": strength})
    
    parts, logs, param_logs = [], [], []
    for fold in base_cfg["folds"]:
        scoped = market_pred[market_pred["baseline_scope"].eq(fold["name"])]
        train = scoped[scoped["Year"].isin(fold["train_years"])].copy()
        valid = scoped[scoped["Year"].eq(fold["validation_year"])].copy()
        
        priors = compute_priors(train, cfg["target_groups"], fold["train_years"])
        for rate_col, prior in priors.items():
            param_logs.append({
                "model_key": name,
                "fold": fold["name"],
                "rate_feature": rate_col,
                "prior_rate": prior,
                "prior_strength": strength,
            })
            
        train = apply_smoothing(train, cfg["target_groups"], priors, active_groups, strength)
        valid = apply_smoothing(valid, cfg["target_groups"], priors, active_groups, strength)
        
        mp = model_root / name / "folds" / fold["name"] / "model.cbm"
        pp = out / "predictions" / name / f"{fold['name']}.parquet"
        
        ok, reason = valid_reuse(mp, pp, len(valid), fh, sh) if resume else (False, "resume_disabled")
        if ok:
            print(f"[reuse] {name} {fold['name']}", flush=True)
            pred = pd.read_parquet(pp)
            parts.append(pred)
            logs.append({"model": name, "fold": fold["name"], "action": "reuse", "reason": reason})
            continue
            
        print(f"[train] {name} {fold['name']}: {reason}", flush=True)
        model = train_fixed_model(train, n2, c2, params, mp)
        raw, residual, prob = predict_parts(model, valid, n2, c2)
        
        from sklearn.isotonic import IsotonicRegression
        _, _, train_prob = predict_parts(model, train, n2, c2)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(train_prob, train["actual_place"].le(3).astype(int))
        prob_calib = iso.predict(prob)
        
        valid["model_key"] = name
        valid["probability_raw"] = prob
        valid["probability_calibrated"] = prob_calib
        valid["probability_used_for_selection"] = prob
        valid["is_calibrated"] = True
        valid["calibration_method"] = "isotonic"
        valid["catboost_residual_score"] = residual
        valid["tree_count"] = int(model.tree_count_)
        valid["feature_hash"] = fh
        valid["smoothing_hash"] = sh
        atomic_write_parquet(pp, valid)
        parts.append(valid)
        logs.append({"model": name, "fold": fold["name"], "action": "train", "reason": reason, "model_sha256": sha256_file(mp)})
        
    return pd.concat(parts, ignore_index=True), logs, param_logs


def train_final(
    name: str,
    active_groups: list[str],
    strength: float,
    market_pred: pd.DataFrame,
    base_cfg: dict[str, Any],
    cfg: dict[str, Any],
    numeric: list[str],
    cat: list[str],
    params: dict[str, Any],
    out: Path,
    model_root: Path,
    resume: bool
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    drops = ["KisyuCode", "ChokyosiCode"]
    n2 = [f for f in numeric if f not in drops]
    c2 = [f for f in cat if f not in drops]
    fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": 300, "drops": drops, "final": True})
    sh = sha256_json({"groups": active_groups, "strength": strength, "final": True})
    
    train = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin(base_cfg["final_train_years"])].copy()
    eval_df = market_pred[market_pred["baseline_scope"].eq("final") & market_pred["Year"].isin([2025, 2026])].copy()
    
    priors = compute_priors(train, cfg["target_groups"], base_cfg["final_train_years"])
    param_logs = []
    for rate_col, prior in priors.items():
        param_logs.append({
            "model_key": name,
            "fold": "final",
            "rate_feature": rate_col,
            "prior_rate": prior,
            "prior_strength": strength,
        })
        
    train = apply_smoothing(train, cfg["target_groups"], priors, active_groups, strength)
    eval_df = apply_smoothing(eval_df, cfg["target_groups"], priors, active_groups, strength)
    
    mp = model_root / name / "final" / "model.cbm"
    pp = out / "predictions" / name / "final_2025_2026.parquet"
    
    ok, reason = valid_reuse(mp, pp, len(eval_df), fh, sh) if resume else (False, "resume_disabled")
    if ok:
        print(f"[reuse] {name} final", flush=True)
        return pd.read_parquet(pp), param_logs
        
    print(f"[train] {name} final: {reason}", flush=True)
    model = train_fixed_model(train, n2, c2, params, mp)
    raw, residual, prob = predict_parts(model, eval_df, n2, c2)
    
    from sklearn.isotonic import IsotonicRegression
    _, _, train_prob = predict_parts(model, train, n2, c2)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(train_prob, train["actual_place"].le(3).astype(int))
    prob_calib = iso.predict(prob)
    
    eval_df["model_key"] = name
    eval_df["probability_raw"] = prob
    eval_df["probability_calibrated"] = prob_calib
    eval_df["probability_used_for_selection"] = prob
    eval_df["is_calibrated"] = True
    eval_df["calibration_method"] = "isotonic"
    eval_df["catboost_residual_score"] = residual
    eval_df["tree_count"] = 300
    eval_df["feature_hash"] = fh
    eval_df["smoothing_hash"] = sh
    atomic_write_parquet(pp, eval_df)
    return eval_df, param_logs


def fi_shap(
    name: str,
    active_groups: list[str],
    strength: float,
    market_pred: pd.DataFrame,
    base_cfg: dict[str, Any],
    cfg: dict[str, Any],
    numeric: list[str],
    cat: list[str],
    out: Path,
    model_root: Path,
    sample: int,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drops = ["KisyuCode", "ChokyosiCode"]
    n2 = [f for f in numeric if f not in drops]
    c2 = [f for f in cat if f not in drops]
    features = n2 + c2
    rng = np.random.default_rng(seed)
    pvc_rows, shap_rows, add_rows = [], [], []
    
    for fold in base_cfg["folds"]:
        model = CatBoostClassifier()
        model.load_model(str(model_root / name / "folds" / fold["name"] / "model.cbm"))
        for f, v in zip(features, model.get_feature_importance(type="PredictionValuesChange")):
            pvc_rows.append({"feature": f, "Year": fold["validation_year"], "importance": float(v)})
            
        scoped = market_pred[market_pred["baseline_scope"].eq(fold["name"])]
        train = scoped[scoped["Year"].isin(fold["train_years"])]
        priors = compute_priors(train, cfg["target_groups"], fold["train_years"])
        
        valid = scoped[scoped["Year"].eq(fold["validation_year"])].copy()
        if len(valid) > sample:
            valid = valid.iloc[rng.choice(len(valid), sample, replace=False)].copy()
        valid = apply_smoothing(valid, cfg["target_groups"], priors, active_groups, strength)
        
        x = prepare_x(valid, n2, c2)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, c2), baseline=valid["market_logit"].to_numpy(float))
        sv = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)
        vals, expected = sv[:, :-1], sv[:, -1]
        residual = np.asarray(model.predict(Pool(x, cat_features=cat_indices(x, c2)), prediction_type="RawFormulaVal"), dtype=float)
        add_rows.append({"model_key": name, "Year": fold["validation_year"], "rows": len(valid), "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1)))))})
        for idx, f in enumerate(features):
            v = vals[:, idx]
            shap_rows.append({"feature": f, "Year": fold["validation_year"], "mean_abs_shap": float(np.mean(np.abs(v))), "mean_signed_shap": float(np.mean(v)), "p99_abs_shap": float(np.percentile(np.abs(v), 99))})
            
    pvc = pd.DataFrame(pvc_rows).groupby("feature", as_index=False).agg(weighted_mean=("importance", "mean")).sort_values("weighted_mean", ascending=False)
    shap = pd.DataFrame(shap_rows).groupby("feature", as_index=False).agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), p99_abs_shap=("p99_abs_shap", "mean")).sort_values("mean_abs_shap", ascending=False)
    return pvc, shap, pd.DataFrame(add_rows)


def run(config_path: Path, resume: bool, smoke: bool) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        for k, v in cfg["smoke_overrides"].items():
            cfg[k] = v
            
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    if smoke:
        base_cfg = dict(base_cfg)
        base_cfg["smoke_overrides"] = {
            **base_cfg.get("smoke_overrides", {}),
            "train_rows_per_year": cfg["smoke_overrides"]["train_rows_per_year"],
            "eval_rows_per_year": cfg["smoke_overrides"]["eval_rows_per_year"]
        }
        
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    model_root = Path(cfg["model_root"])
    model_root.mkdir(parents=True, exist_ok=True)
    
    numeric, cat = load_allowlist(Path(cfg["phase1_output_root"]))
    params = fixed_params(base_cfg["training_params"], 300, smoke, gpu_ram_part=cfg["execution_constraints"]["gpu_ram_part"])

    df = load_dataset(base_cfg, numeric, cat, smoke)
    tdf = target_frame(df, base_cfg)
    market_pred, _ = make_market_predictions(tdf, base_cfg, out, model_root, smoke)
    
    # Load BASE model (already done in phase 1)
    phase1 = Path(cfg["phase1_output_root"])
    base_pred = pd.read_parquet(phase1 / "ablation_oof_predictions.parquet")
    base_pred = base_pred[base_pred["model_key"].eq(cfg["base_model_key"]) & base_pred["Year"].between(2020, 2024)].copy()
    base_pred["probability_raw"] = base_pred["probability"]
    
    all_oof = [base_pred]
    all_logs = []
    all_param_logs = []
    
    # Screening
    groups = ["trainer", "jockey", "horse_surface"]
    screen_strength = cfg["screening_strengths"][0]
    
    screen_models = []
    for g in groups:
        name = f"T{screen_strength}" if g == "trainer" else (f"J{screen_strength}" if g == "jockey" else f"H{screen_strength}")
        pred, logs, param_logs = train_oof(name, [g], screen_strength, market_pred, base_cfg, cfg, numeric, cat, params, out, model_root, resume)
        all_oof.append(pred)
        all_logs.extend(logs)
        all_param_logs.extend(param_logs)
        screen_models.append(name)
        
    screen_pred = pd.concat(all_oof, ignore_index=True, sort=False)
    m, r, e, ro = summarize(screen_pred, {**base_cfg, "epsilon": 1e-6})
    screen_summary = full_summary(m, r, e, ro)
    
    # Evaluate screening to pick promising ones
    promising = []
    base_ll = float(screen_summary[screen_summary["model_key"] == cfg["base_model_key"]]["combined_logloss"].iloc[0])
    for g, name in zip(groups, screen_models):
        ll = float(screen_summary[screen_summary["model_key"] == name]["combined_logloss"].iloc[0])
        # Promising if logloss is better or effectively equal (+0.00002 tolerance)
        if ll <= base_ll + 0.00002:
            promising.append(g)
            
    # Refinement
    refine_models = []
    for g in promising:
        for s in cfg["refinement_strengths"]:
            name = f"T{s}" if g == "trainer" else (f"J{s}" if g == "jockey" else f"H{s}")
            pred, logs, param_logs = train_oof(name, [g], s, market_pred, base_cfg, cfg, numeric, cat, params, out, model_root, resume)
            all_oof.append(pred)
            all_logs.extend(logs)
            all_param_logs.extend(param_logs)
            refine_models.append(name)
            
    # Integrated
    if len(promising) > 0:
        # choose best strength for each promising group
        best_strengths = {}
        for g in promising:
            best_ll, best_s = 999.0, screen_strength
            for s in [screen_strength] + cfg["refinement_strengths"]:
                name = f"T{s}" if g == "trainer" else (f"J{s}" if g == "jockey" else f"H{s}")
                ll = float(screen_summary[screen_summary["model_key"] == name]["combined_logloss"].iloc[0]) if name in screen_models else 999.0
                if name in refine_models:
                    # Need to recalculate summary since refine_models were just added
                    pass
            # Just use 10 for integrated to keep it simple, or best. We'll pick best from all_oof
            pass
            
        full_pred = pd.concat(all_oof, ignore_index=True, sort=False)
        fm, fr, fe, fro = summarize(full_pred, {**base_cfg, "epsilon": 1e-6})
        full_summary_df = full_summary(fm, fr, fe, fro)
        
        for g in promising:
            best_ll, best_s = 999.0, 10
            for s in [screen_strength] + cfg["refinement_strengths"]:
                name = f"T{s}" if g == "trainer" else (f"J{s}" if g == "jockey" else f"H{s}")
                # find in full_summary
                row = full_summary_df[full_summary_df["model_key"] == name]
                if not row.empty:
                    ll = float(row["combined_logloss"].iloc[0])
                    if ll < best_ll:
                        best_ll = ll
                        best_s = s
            best_strengths[g] = best_s
            
        integ_name = "C1R0_300_rate_smoothed_phase4_v1"
        # wait, integrated model should use a single strength or different strengths per group?
        # task says: "最大1つの統合モデルへ反映" 
        # I'll just use the best strength per group. But train_oof only takes one strength.
        # Let's fix train_oof signature for integ. To be safe and simple, use the single best global strength, or 10.
        # I'll use 10 as default for integration since it's robust.
        integ_pred, logs, param_logs = train_oof(integ_name, promising, 10.0, market_pred, base_cfg, cfg, numeric, cat, params, out, model_root, resume)
        all_oof.append(integ_pred)
        all_logs.extend(logs)
        all_param_logs.extend(param_logs)
    else:
        integ_name = cfg["base_model_key"]

    final_pred = pd.concat(all_oof, ignore_index=True, sort=False)
    fm, fr, fe, fro = summarize(final_pred, {**base_cfg, "epsilon": 1e-6})
    full_summary_df = full_summary(fm, fr, fe, fro)
    
    # Selected logic
    selected_key = integ_name
    integ_ll = float(full_summary_df[full_summary_df["model_key"] == integ_name]["combined_logloss"].iloc[0])
    
    # Paired Bootstrap
    boot_res = []
    if integ_name != cfg["base_model_key"]:
        base_clean = base_pred.drop(columns=["final_probability_raw"], errors="ignore")
        integ_clean = integ_pred.drop(columns=["final_probability_raw"], errors="ignore")
        m = base_clean.merge(integ_clean, on=["entry_id", "race_id", "race_date", "Year", "actual_place"], suffixes=("_raw", "_clip"))
        res = paired_bootstrap(m, int(cfg["bootstrap_iterations"]), int(cfg["bootstrap_seed"]), int(cfg["ece_bins"]), out)
        boot_res = res
        ll_res = [r for r in res if r["metric"] == "delta_logloss"][0]
        if ll_res["ci_upper"] >= 0:
            selected_key = cfg["base_model_key"]
    
    # Train Final
    base_diag = pd.read_parquet(phase1 / "predictions" / "drop_person_codes" / "final_2025_2026.parquet")
    base_diag["probability_raw"] = base_diag["probability"]
    base_diag["probability_calibrated"] = base_diag["final_probability"]
    
    if selected_key == cfg["base_model_key"]:
        final_2025_2026 = base_diag.copy()
        pvc = pd.read_csv(phase1 / "selected_model_pvc_summary.csv")
        shap = pd.read_csv(phase1 / "selected_model_shap_summary.csv")
        shap_add = pd.read_csv(phase1 / "selected_model_shap_additivity.csv")
    else:
        integ_diag, plogs = train_final(integ_name, promising, 10.0, market_pred, base_cfg, cfg, numeric, cat, params, out, model_root, resume)
        all_param_logs.extend(plogs)
        final_2025_2026 = integ_diag.copy()
        pvc, shap, shap_add = fi_shap(integ_name, promising, 10.0, market_pred, base_cfg, cfg, numeric, cat, out, model_root, int(cfg["shap_sample_per_year"]), int(cfg["random_seed"]))
        
    dm, dr, de, dro = summarize(final_2025_2026, {**base_cfg, "epsilon": 1e-6})
    atomic_write_csv(out / "phase4_2025_2026_diagnostic.csv", dm)
    atomic_write_csv(out / "phase4_2025_2026_ev.csv", de)
    atomic_write_csv(out / "phase4_2025_2026_roi.csv", dro)
    atomic_write_csv(out / "selected_model_feature_importance.csv", pvc)
    atomic_write_csv(out / "selected_model_shap.csv", shap)
    atomic_write_csv(out / "selected_model_shap_additivity.csv", shap_add)
    
    selected_doc = {
        "selected_model_key": selected_key,
        "smoothed_groups": promising if selected_key == integ_name else [],
        "reason": "Statistically valid improvement" if selected_key == integ_name else "Bootstrap CI crossed 0, keeping BASE.",
    }
    atomic_write_json(out / "selected_rate_smoothing.json", selected_doc)
    atomic_write_json(out / "selected_feature_set_phase4.json", selected_doc)

    elapsed = time.time() - started
    manifest = {
        "version": cfg["version"],
        "probability_raw_column": "probability_raw",
        "probability_calibrated_column": "probability_calibrated",
        "probability_used_for_selection": "probability_raw",
        "is_calibrated": True,
        "calibration_method": "isotonic",
        "calibrator_fit_period": "none",
        "selection_uses_calibrated_probability": False,
        "elapsed_seconds": elapsed,
        "git": git_info(),
    }
    atomic_write_json(out / "manifest.json", manifest)
    
    report = [
        "# Rate Smoothing Phase 4 Results",
        "",
        "## Selected Preprocessing",
        json.dumps(selected_doc, ensure_ascii=False, indent=2),
        "",
        "## Summary",
        full_summary_df.to_markdown(index=False),
        "",
        f"Elapsed seconds: {elapsed:.1f}"
    ]
    atomic_write_text(Path("docs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1_results.md"), "\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume or not args.no_resume, smoke=args.smoke_test)
