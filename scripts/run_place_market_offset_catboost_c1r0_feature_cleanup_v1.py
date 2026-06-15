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
    add_eval_columns,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    bootstrap_ci,
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
from scripts.run_place_market_offset_catboost_v1 import cat_indices
from scripts.run_place_market_offset_catboost_c1r0_tree_count_v1 import (
    ev_row,
    fixed_params,
    predict_parts,
    residual_stats,
    train_fixed_model,
)


BASE_KEY = "C1R0_pure_market_offset_fixed300_base"


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


def load_allowlist(base_out: Path) -> tuple[list[str], list[str]]:
    allow = json.loads((base_out.parent / "place_market_offset_catboost_c1r0_v1" / "feature_allowlist_c1r0.json").read_text(encoding="utf-8"))
    return list(allow["numeric"]), list(allow["categorical"])


def candidate_features(numeric: list[str], cat: list[str], drops: list[str]) -> tuple[list[str], list[str]]:
    drop_set = set(drops)
    return [f for f in numeric if f not in drop_set], [f for f in cat if f not in drop_set]


def model_key(name: str) -> str:
    return f"C1R0_fixed300_ablation_{name}"


def prediction_path(out: Path, name: str, fold_name: str) -> Path:
    return out / "predictions" / name / f"{fold_name}.parquet"


def model_path(model_root: Path, name: str, fold_name: str) -> Path:
    return model_root / name / "folds" / fold_name / "model.cbm"


def final_model_path(model_root: Path, name: str) -> Path:
    return model_root / name / "final" / "model.cbm"


def valid_reuse(path_model: Path, path_pred: Path, expected_tree_count: int, expected_rows: int, expected_feature_hash: str) -> tuple[bool, str]:
    if not path_model.exists() or not path_pred.exists():
        return False, "missing_model_or_prediction"
    try:
        m = CatBoostClassifier()
        m.load_model(str(path_model))
        if int(m.tree_count_) != int(expected_tree_count):
            return False, f"tree_count_mismatch:{m.tree_count_}"
        pred = pd.read_parquet(path_pred, columns=["feature_hash"])
        if len(pred) != expected_rows:
            return False, f"row_count_mismatch:{len(pred)}!={expected_rows}"
        if pred["feature_hash"].nunique() != 1 or str(pred["feature_hash"].iloc[0]) != expected_feature_hash:
            return False, "feature_hash_mismatch"
    except Exception as exc:
        return False, f"reuse_check_error:{exc}"
    return True, "ok"


def train_or_reuse_fold(
    name: str,
    fold: dict[str, Any],
    market_pred: pd.DataFrame,
    numeric: list[str],
    cat: list[str],
    params: dict[str, Any],
    out: Path,
    model_root: Path,
    feature_hash: str,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scoped = market_pred[market_pred["baseline_scope"] == fold["name"]]
    train = scoped[scoped["Year"].isin(fold["train_years"])]
    valid = scoped[scoped["Year"] == fold["validation_year"]].copy()
    mp = model_path(model_root, name, fold["name"])
    pp = prediction_path(out, name, fold["name"])
    reuse, reason = valid_reuse(mp, pp, params["iterations"], len(valid), feature_hash) if resume else (False, "resume_disabled")
    if reuse:
        print(f"[reuse] {name} {fold['name']}", flush=True)
        return pd.read_parquet(pp), {"ablation_name": name, "fold": fold["name"], "action": "reuse", "reason": reason, "rows": int(len(valid)), "model_path": str(mp)}
    print(f"[train] {name} {fold['name']}: {reason}", flush=True)
    model = train_fixed_model(train, numeric, cat, params, mp)
    raw, residual, prob = predict_parts(model, valid, numeric, cat)
    valid["model_key"] = model_key(name)
    valid["ablation_name"] = name
    valid["probability"] = prob
    valid["final_probability"] = prob
    valid["final_probability_raw"] = raw
    valid["catboost_residual_score"] = residual
    valid["feature_hash"] = feature_hash
    valid["tree_count"] = int(model.tree_count_)
    pp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(pp, valid)
    return valid, {"ablation_name": name, "fold": fold["name"], "action": "train", "reason": reason, "rows": int(len(valid)), "model_path": str(mp), "model_sha256": sha256_file(mp)}


def summarize(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    resid_rows = []
    ev_rows = []
    roi_rows = []
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


def summary_2020_2024(metrics: pd.DataFrame, residual: pd.DataFrame, ev: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in metrics[metrics["Year"].between(2020, 2024)].groupby("model_key"):
        r = residual[residual["model_key"].eq(key) & residual["Year"].between(2020, 2024)]
        e = ev[ev["model_key"].eq(key) & ev["Year"].between(2020, 2024)]
        ro = roi[roi["model_key"].eq(key) & roi["Year"].between(2020, 2024)]
        rows.append({
            "model_key": key,
            "mean_logloss": float(g["logloss"].mean()),
            "mean_brier": float(g["brier"].mean()),
            "mean_ece": float(g["ece"].mean()),
            "mean_calibration_slope": float(g["calibration_slope"].mean()),
            "residual_std_mean": float(r["residual_std"].mean()),
            "residual_std_cv": float(r["residual_std"].std(ddof=1) / r["residual_std"].mean()),
            "abs_residual_p95_mean": float(r["abs_residual_p95"].mean()),
            "abs_residual_p99_mean": float(r["abs_residual_p99"].mean()),
            "ev_ge_1_count_sum": int(e["ev_ge_1_count"].sum()),
            "ev_ge_1_count_cv": float(e["ev_ge_1_count"].std(ddof=1) / e["ev_ge_1_count"].mean()),
            "ev_roi_spearman_mean": float(e["ev_roi_spearman"].mean()),
            "mean_roi": float(ro["roi"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["mean_logloss", "mean_brier"])


def select_model(summary: pd.DataFrame) -> dict[str, Any]:
    base = summary[summary["model_key"].eq(BASE_KEY)].iloc[0]
    s = summary.copy()
    s["eligible"] = (
        (s["mean_logloss"] <= float(base["mean_logloss"]) + 0.00005)
        & (s["mean_brier"] <= float(base["mean_brier"]) + 0.00003)
        & (s["abs_residual_p95_mean"] <= float(base["abs_residual_p95_mean"]) + 0.005)
        & (s["ev_ge_1_count_sum"] <= float(base["ev_ge_1_count_sum"]) * 1.20)
    )
    eligible = s[s["eligible"]].copy()
    if eligible.empty:
        row = base.to_dict()
        reason = "No ablation stayed within the fixed300 probability metric tolerance and residual/EV stability guardrails."
    else:
        # The task priority is lexicographic: probability metrics first, then
        # residual size/stability and EV stability. ROI never drives selection.
        row = eligible.sort_values([
            "mean_logloss",
            "mean_brier",
            "abs_residual_p95_mean",
            "abs_residual_p99_mean",
            "residual_std_cv",
            "ev_ge_1_count_cv",
            "mean_ece",
        ]).iloc[0].to_dict()
        reason = "Selected using 2020-2024 probability metrics first, then residual and EV count stability; ROI auxiliary only."
    return {"selected_model_key": row["model_key"], "selection_years": [2020, 2021, 2022, 2023, 2024], "reason": reason, "selected_row": row}


def calibrate_oof(oof: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = []
    methods = {}
    for key, d in oof.groupby("model_key"):
        cal = fit_one_calibrator("isotonic", d[d["Year"].between(2020, 2024)], "probability")
        x = d.copy()
        x["final_probability"] = cal.transform(x["probability"].to_numpy(float))
        parts.append(x)
        methods[key] = "isotonic"
    return pd.concat(parts, ignore_index=True), methods


def load_base_predictions(base_out: Path) -> pd.DataFrame:
    oof = pd.read_parquet(base_out / "fixed_tree_predictions_2020_2024.parquet")
    oof = oof[oof["tree_count_candidate"].eq(300)].copy()
    diag = pd.read_parquet(base_out / "selected_fixed_tree_predictions_2025_2026.parquet")
    base = pd.concat([oof, diag], ignore_index=True, sort=False)
    base["model_key"] = BASE_KEY
    base["ablation_name"] = "base_fixed300"
    return base


def train_final_selected(
    selected_key: str,
    candidate_name: str,
    market_pred: pd.DataFrame,
    numeric: list[str],
    cat: list[str],
    params: dict[str, Any],
    oof: pd.DataFrame,
    base_cfg: dict[str, Any],
    out: Path,
    model_root: Path,
    feature_hash: str,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selected_key == BASE_KEY:
        return pd.DataFrame(), {"action": "reuse_base_final", "reason": "base fixed300 selected"}
    train = market_pred[(market_pred["baseline_scope"].eq("final")) & (market_pred["Year"].isin(base_cfg["final_train_years"]))]
    eval_df = market_pred[(market_pred["baseline_scope"].eq("final")) & (market_pred["Year"].isin([base_cfg["test_year"], base_cfg["latest_holdout_year"]]))].copy()
    mp = final_model_path(model_root, candidate_name)
    pp = out / "predictions" / candidate_name / "final_2025_2026.parquet"
    reuse, reason = valid_reuse(mp, pp, params["iterations"], len(eval_df), feature_hash) if resume else (False, "resume_disabled")
    if reuse:
        print(f"[reuse] {candidate_name} final_2025_2026", flush=True)
        return pd.read_parquet(pp), {"action": "reuse", "reason": reason, "model_path": str(mp)}
    print(f"[train] {candidate_name} final_2025_2026: {reason}", flush=True)
    model = train_fixed_model(train, numeric, cat, params, mp)
    raw, residual, prob = predict_parts(model, eval_df, numeric, cat)
    cal = fit_one_calibrator("isotonic", oof[oof["model_key"].eq(selected_key) & oof["Year"].between(2020, 2024)], "probability")
    eval_df["model_key"] = selected_key
    eval_df["ablation_name"] = candidate_name
    eval_df["probability"] = prob
    eval_df["final_probability"] = cal.transform(prob)
    eval_df["final_probability_raw"] = raw
    eval_df["catboost_residual_score"] = residual
    eval_df["feature_hash"] = feature_hash
    eval_df["tree_count"] = int(model.tree_count_)
    atomic_write_parquet(pp, eval_df)
    return eval_df, {"action": "train", "reason": reason, "model_path": str(mp), "model_sha256": sha256_file(mp)}


def selected_importance_and_shap(
    selected_key: str,
    candidate_name: str,
    base_out: Path,
    market_pred: pd.DataFrame,
    numeric: list[str],
    cat: list[str],
    out: Path,
    model_root: Path,
    sample_per_year: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selected_key == BASE_KEY:
        pvc = pd.read_csv(base_out / "selected_tree_catboost_pvc_summary.csv")
        shap = pd.read_csv(base_out / "selected_tree_shap_summary.csv")
        add = pd.read_csv(base_out / "selected_tree_shap_additivity.csv")
        pvc["source"] = "reused_fixed300_base"
        shap["source"] = "reused_fixed300_base"
        add["source"] = "reused_fixed300_base"
        return pvc, shap, add
    rng = np.random.default_rng(seed)
    features = numeric + cat
    pvc_rows = []
    shap_rows = []
    add_rows = []
    for fold_name in sorted((model_root / candidate_name / "folds").iterdir()):
        if not fold_name.is_dir():
            continue
        year = int(fold_name.name.replace("fold_", ""))
        model = CatBoostClassifier()
        model.load_model(str(fold_name / "model.cbm"))
        pvc = model.get_feature_importance(type="PredictionValuesChange")
        for f, v in zip(features, pvc):
            pvc_rows.append({"feature": f, "importance": float(v), "Year": year})
        valid = market_pred[(market_pred["baseline_scope"].eq(fold_name.name)) & (market_pred["Year"].eq(year))].copy()
        if len(valid) > sample_per_year:
            valid = valid.iloc[rng.choice(len(valid), sample_per_year, replace=False)].copy()
        x = prepare_x(valid, numeric, cat)
        pool = Pool(x, valid["actual_place"].to_numpy(int), cat_features=cat_indices(x, cat), baseline=valid["market_logit"].to_numpy(float))
        sv = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"), dtype=float)
        vals = sv[:, :-1]
        expected = sv[:, -1]
        residual = np.asarray(model.predict(Pool(x, cat_features=cat_indices(x, cat)), prediction_type="RawFormulaVal"), dtype=float)
        add_rows.append({"Year": year, "rows": int(len(valid)), "residual_additivity_max_abs": float(np.max(np.abs(residual - (expected + vals.sum(axis=1)))))})
        for idx, f in enumerate(features):
            v = vals[:, idx]
            shap_rows.append({
                "feature": f,
                "Year": year,
                "mean_abs_shap": float(np.mean(np.abs(v))),
                "mean_signed_shap": float(np.mean(v)),
                "p90_abs_shap": float(np.percentile(np.abs(v), 90)),
                "p99_abs_shap": float(np.percentile(np.abs(v), 99)),
            })
    pvc_df = pd.DataFrame(pvc_rows).groupby("feature", as_index=False).agg(weighted_mean=("importance", "mean"), fold_count=("importance", "count")).sort_values("weighted_mean", ascending=False)
    shap_df = pd.DataFrame(shap_rows).groupby("feature", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        p90_abs_shap=("p90_abs_shap", "mean"),
        p99_abs_shap=("p99_abs_shap", "mean"),
    ).sort_values("mean_abs_shap", ascending=False)
    add_df = pd.DataFrame(add_rows)
    atomic_write_csv(out / "selected_model_pvc_summary.csv", pvc_df)
    atomic_write_csv(out / "selected_model_shap_summary.csv", shap_df)
    atomic_write_csv(out / "selected_model_shap_additivity.csv", add_df)
    return pvc_df, shap_df, add_df


def write_report(path: Path, decisions: pd.DataFrame, summary: pd.DataFrame, selected: dict[str, Any], diag: pd.DataFrame, reuse_log: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# C1R0 Feature Cleanup Results",
        "",
        "- Fixed reference model: `C1R0_pure_market_offset_fixed300`",
        "- Tree count: `300` fixed",
        "- Selection years: `2020-2024 only`",
        "- 2025/2026: `fixed diagnostic only`",
        "- DB read: `not performed`",
        "",
        "## Feature Cleanup Decisions",
        decisions.to_markdown(index=False),
        "",
        "## 2020-2024 Comparison",
        summary.to_markdown(index=False),
        "",
        "## Selected Model",
        f"- `{selected['selected_model_key']}`",
        f"- Reason: {selected['reason']}",
        "",
        "## 2025/2026 Diagnostic",
        diag.to_markdown(index=False) if not diag.empty else "Base fixed300 diagnostics reused; no new final model was required.",
        "",
        "## Reuse And Training Log",
        reuse_log.to_markdown(index=False),
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
        cfg["bootstrap_iterations"] = cfg["smoke_overrides"]["bootstrap_iterations"]
    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    if smoke:
        base_cfg = dict(base_cfg)
        base_cfg["smoke_overrides"] = dict(base_cfg.get("smoke_overrides", {}))
        base_cfg["smoke_overrides"]["train_rows_per_year"] = cfg["smoke_overrides"]["train_rows_per_year"]
        base_cfg["smoke_overrides"]["eval_rows_per_year"] = cfg["smoke_overrides"]["eval_rows_per_year"]
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    base_out = Path(cfg["base_fixed300_output_root"])
    numeric, cat = load_allowlist(base_out)
    decisions = pd.read_csv(out / "ablation_decision_table.csv")
    selected_candidates = decisions[decisions["ablation_required"].astype(bool)]["ablation_name"].tolist()
    params = fixed_params(base_cfg["training_params"], int(cfg["fixed_tree_count"]), smoke, gpu_ram_part=cfg.get("execution_constraints", {}).get("gpu_ram_part"))
    df = load_dataset(base_cfg, numeric, cat, smoke)
    tdf = target_frame(df, base_cfg)
    market_pred, _ = make_market_predictions(tdf, base_cfg, out, model_root, smoke)
    parts = []
    reuse_rows = []
    for name in selected_candidates:
        drops = cfg["ablation_candidates"][name]["drop_features"]
        n2, c2 = candidate_features(numeric, cat, drops)
        fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": cfg["fixed_tree_count"], "calibration": cfg["fixed_calibration_method"]})
        for fold in base_cfg["folds"]:
            pred, row = train_or_reuse_fold(name, fold, market_pred, n2, c2, params, out, model_root, fh, resume)
            parts.append(pred)
            reuse_rows.append(row)
    ablation_oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ablation_oof, cal_methods = calibrate_oof(ablation_oof, base_cfg) if not ablation_oof.empty else (ablation_oof, {})
    base = load_base_predictions(base_out)
    comp_pred = pd.concat([base[base["Year"].between(2020, 2024)], ablation_oof], ignore_index=True, sort=False)
    metrics, residual, ev, roi = summarize(comp_pred, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    summary = summary_2020_2024(metrics, residual, ev, roi)
    selected = select_model(summary)
    final_diag = base[base["Year"].isin([2025, 2026])].copy()
    final_extra = pd.DataFrame()
    if selected["selected_model_key"] != BASE_KEY:
        cname = selected["selected_model_key"].replace("C1R0_fixed300_ablation_", "")
        drops = cfg["ablation_candidates"][cname]["drop_features"]
        n2, c2 = candidate_features(numeric, cat, drops)
        fh = sha256_json({"numeric": n2, "categorical": c2, "tree_count": cfg["fixed_tree_count"], "calibration": cfg["fixed_calibration_method"]})
        final_extra, row = train_final_selected(selected["selected_model_key"], cname, market_pred, n2, c2, params, ablation_oof, base_cfg, out, model_root, fh, resume)
        reuse_rows.append({"ablation_name": cname, "fold": "final_2025_2026", **row})
        final_diag = pd.concat([final_diag, final_extra], ignore_index=True, sort=False)
    diag_metrics, diag_resid, diag_ev, diag_roi = summarize(final_diag, {**base_cfg, "bootstrap_iterations": cfg["bootstrap_iterations"], "random_seed": cfg["random_seed"]})
    selected_candidate_name = selected["selected_model_key"].replace("C1R0_fixed300_ablation_", "") if selected["selected_model_key"] != BASE_KEY else "base_fixed300"
    selected_drops = cfg["ablation_candidates"].get(selected_candidate_name, {}).get("drop_features", [])
    selected_numeric, selected_cat = candidate_features(numeric, cat, selected_drops)
    pvc, shap, shap_add = selected_importance_and_shap(
        selected["selected_model_key"],
        selected_candidate_name,
        base_out,
        market_pred,
        selected_numeric,
        selected_cat,
        out,
        model_root,
        int(cfg.get("smoke_overrides", {}).get("shap_sample_per_year", 100) if smoke else 1200),
        int(cfg["random_seed"]),
    )
    elapsed = time.time() - started
    hashes = {
        "ablation_oof_predictions.parquet": atomic_write_parquet(out / "ablation_oof_predictions.parquet", ablation_oof) if not ablation_oof.empty else "",
        "ablation_metrics_by_year.csv": atomic_write_csv(out / "ablation_metrics_by_year.csv", metrics),
        "ablation_residual_by_year.csv": atomic_write_csv(out / "ablation_residual_by_year.csv", residual),
        "ablation_ev_by_year.csv": atomic_write_csv(out / "ablation_ev_by_year.csv", ev),
        "ablation_roi_by_year.csv": atomic_write_csv(out / "ablation_roi_by_year.csv", roi),
        "ablation_comparison_2020_2024.csv": atomic_write_csv(out / "ablation_comparison_2020_2024.csv", summary),
        "selected_model.json": atomic_write_json(out / "selected_model.json", selected),
        "diagnostic_2025_2026_metrics.csv": atomic_write_csv(out / "diagnostic_2025_2026_metrics.csv", diag_metrics),
        "diagnostic_2025_2026_residual.csv": atomic_write_csv(out / "diagnostic_2025_2026_residual.csv", diag_resid),
        "diagnostic_2025_2026_ev.csv": atomic_write_csv(out / "diagnostic_2025_2026_ev.csv", diag_ev),
        "diagnostic_2025_2026_roi.csv": atomic_write_csv(out / "diagnostic_2025_2026_roi.csv", diag_roi),
        "reuse_training_log.csv": atomic_write_csv(out / "reuse_training_log.csv", pd.DataFrame(reuse_rows)),
        "selected_model_pvc_summary.csv": sha256_file(out / "selected_model_pvc_summary.csv") if (out / "selected_model_pvc_summary.csv").exists() else atomic_write_csv(out / "selected_model_pvc_summary.csv", pvc),
        "selected_model_shap_summary.csv": sha256_file(out / "selected_model_shap_summary.csv") if (out / "selected_model_shap_summary.csv").exists() else atomic_write_csv(out / "selected_model_shap_summary.csv", shap),
        "selected_model_shap_additivity.csv": sha256_file(out / "selected_model_shap_additivity.csv") if (out / "selected_model_shap_additivity.csv").exists() else atomic_write_csv(out / "selected_model_shap_additivity.csv", shap_add),
    }
    manifest = {
        "version": cfg["version"],
        "fixed_reference": BASE_KEY,
        "fixed_tree_count": int(cfg["fixed_tree_count"]),
        "fixed_calibration_method": cfg["fixed_calibration_method"],
        "selection_years": cfg["selection_years"],
        "diagnostic_years": cfg["diagnostic_years"],
        "db_usage": "not_read; existing parquet feature dataset only",
        "feature_dataset_rebuild": False,
        "random_split_used": False,
        "tree_count_changed": False,
        "ablation_candidates_run": selected_candidates,
        "calibration_by_ablation": cal_methods,
        "selected": selected,
        "execution_constraints": {
            "gpu_ram_part": cfg.get("execution_constraints", {}).get("gpu_ram_part"),
            "note": cfg.get("execution_constraints", {}).get("note"),
            "model_selection_hyperparameter": False,
        },
        "catboost_params": params,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "output_hashes": hashes,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_report(Path("docs/place_market_offset_catboost_c1r0_feature_cleanup_v1_results.md"), decisions, summary, selected, diag_metrics, pd.DataFrame(reuse_rows), elapsed)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_c1r0_feature_cleanup_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume or not args.no_resume, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
