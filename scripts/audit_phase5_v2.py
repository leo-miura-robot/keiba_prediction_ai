import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path("outputs/place_market_offset_catboost_c1r0_history_extension_phase5_v1/audit_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("audit_v2")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for h in [logging.FileHandler(OUT_DIR / "audit_v2.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(formatter)
    logger.addHandler(h)

def load_predictions() -> pd.DataFrame:
    phase5_dir = Path("outputs/place_market_offset_catboost_c1r0_history_extension_phase5_v1")
    phase5_pred = pd.read_parquet(phase5_dir / "phase5_oof_predictions.parquet")
    if "actual_place" not in phase5_pred.columns and "target_place" in phase5_pred.columns:
        phase5_pred["actual_place"] = phase5_pred["target_place"]
    elif "actual_place" not in phase5_pred.columns:
        phase5_pred["actual_place"] = phase5_pred["target_place_paid"]
        
    phase5_pred = phase5_pred[phase5_pred["model_key"].isin(["WARMUP_2006_TRAIN_2016", "FULL_2006"])]
    phase5_pred["source_file"] = "phase5_oof_predictions.parquet"
    
    # Load BASE_2016
    phase1_dir = Path("outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1")
    base_oof = pd.read_parquet(phase1_dir / "ablation_oof_predictions.parquet")
    base_oof = base_oof[base_oof["model_key"] == "C1R0_fixed300_ablation_drop_person_codes"].copy()
    base_oof = base_oof[base_oof["Year"].between(2020, 2024)]
    base_oof["source_file"] = "ablation_oof_predictions.parquet"
    
    base_2526 = pd.read_parquet(phase1_dir / "predictions/drop_person_codes/final_2025_2026.parquet")
    base_2526["source_file"] = "predictions/drop_person_codes/final_2025_2026.parquet"
    
    base = pd.concat([base_oof, base_2526], ignore_index=True)
    base["model_key"] = "BASE_2016"
    base["probability_raw"] = base["probability"]
    if "actual_place" not in base.columns:
        base["actual_place"] = base["target_place_paid"]
        
    cols = ["model_key", "Year", "race_id", "probability_raw", "fuku_odds_low", "fuku_pay", "actual_place", "source_file"]
    
    return pd.concat([phase5_pred[cols], base[cols]], ignore_index=True)

def calc_logloss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def calc_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def evaluate_stress_roi(df: pd.DataFrame, limit: int) -> dict:
    picks = df[(df["probability_raw"] * df["fuku_odds_low"]) >= 1.0].copy()
    hits = picks[picks["actual_place"] == 1].copy()
    hits = hits.sort_values("fuku_pay", ascending=False)
    
    removed_idx = hits.head(limit).index
    
    # Base
    cost = len(picks) * 100
    ret = hits["fuku_pay"].sum()
    base_roi = ret / cost * 100 if cost else 0
    
    # Row Removed
    rr_picks = picks.drop(index=removed_idx)
    rr_cost = len(rr_picks) * 100
    rr_ret = rr_picks.loc[rr_picks["actual_place"] == 1, "fuku_pay"].sum()
    rr_roi = rr_ret / rr_cost * 100 if rr_cost else 0
    
    # Payout Zeroed
    pz_picks = picks.copy()
    pz_picks.loc[removed_idx, "fuku_pay"] = 0
    pz_cost = len(pz_picks) * 100
    pz_ret = pz_picks.loc[pz_picks["actual_place"] == 1, "fuku_pay"].sum()
    pz_roi = pz_ret / pz_cost * 100 if pz_cost else 0
    
    rem_count = len(removed_idx)
    rem_payout = hits.loc[removed_idx, "fuku_pay"].sum()
    
    return {
        "limit": limit,
        "bet_count": len(picks),
        "stake": cost,
        "payout": ret,
        "roi": base_roi,
        "removed_count": rem_count,
        "removed_payout": rem_payout,
        "rr_stake": rr_cost,
        "rr_payout": rr_ret,
        "rr_roi": rr_roi,
        "pz_stake": pz_cost,
        "pz_payout": pz_ret,
        "pz_roi": pz_roi,
    }

def audit_1_and_2(df: pd.DataFrame):
    logger.info("=== Audit 1 & 2: ROI detailed stress check ===")
    records = []
    
    for model in ["BASE_2016", "WARMUP_2006_TRAIN_2016", "FULL_2006"]:
        mdf = df[(df["model_key"] == model) & (df["Year"].between(2020, 2024))]
        for limit in [1, 3, 5, 10]:
            res = evaluate_stress_roi(mdf, limit)
            res["model"] = model
            records.append(res)
            
    res_df = pd.DataFrame(records)
    res_df.to_csv(OUT_DIR / "roi_stress_audit.csv", index=False)
    logger.info(f"Saved roi_stress_audit.csv")

def audit_3(df: pd.DataFrame):
    logger.info("=== Audit 3: 2025/2026 Evaluation ===")
    records = []
    for model in ["BASE_2016", "WARMUP_2006_TRAIN_2016", "FULL_2006"]:
        mdf = df[df["model_key"] == model]
        for y in [2025, 2026, "2025+2026"]:
            ydf = mdf[mdf["Year"].isin([2025, 2026])] if y == "2025+2026" else mdf[mdf["Year"] == y]
            if ydf.empty:
                continue
            picks = ydf[(ydf["probability_raw"] * ydf["fuku_odds_low"]) >= 1.0]
            cost = len(picks) * 100
            ret = picks.loc[picks["actual_place"] == 1, "fuku_pay"].sum()
            roi = ret / cost * 100 if cost else 0
            records.append({"model": model, "period": y, "roi": roi, "picks": len(picks)})
    res_df = pd.DataFrame(records)
    res_df.to_csv(OUT_DIR / "eval_2025_2026.csv", index=False)
    logger.info(f"Saved eval_2025_2026.csv")

def audit_4(df: pd.DataFrame):
    logger.info("=== Audit 4: Race-level Paired Bootstrap (BASE vs FULL) ===")
    np.random.seed(42)
    bdf = df[(df["model_key"] == "BASE_2016") & (df["Year"].between(2020, 2024))]
    fdf = df[(df["model_key"] == "FULL_2006") & (df["Year"].between(2020, 2024))]
    
    bdf = bdf.sort_values(["race_id", "probability_raw"])
    fdf = fdf.sort_values(["race_id", "probability_raw"])
    
    if len(bdf) != len(fdf):
        logger.warning(f"Row count mismatch! BASE: {len(bdf)}, FULL: {len(fdf)}")
    
    common_races = np.intersect1d(bdf["race_id"].unique(), fdf["race_id"].unique())
    logger.info(f"Common races: {len(common_races)}")
    
    bdf = bdf[bdf["race_id"].isin(common_races)].set_index("race_id")
    fdf = fdf[fdf["race_id"].isin(common_races)].set_index("race_id")
    
    def calc_race_metrics(df_grp):
        def _agg(x):
            return pd.Series({
                "ll_sum": -np.sum(x["actual_place"] * np.log(np.clip(x["probability_raw"], 1e-15, 1-1e-15)) + (1-x["actual_place"]) * np.log(1 - np.clip(x["probability_raw"], 1e-15, 1-1e-15))),
                "brier_sum": np.sum((x["probability_raw"] - x["actual_place"])**2),
                "count": len(x)
            })
        return df_grp.groupby(level=0).apply(_agg)
        
    b_metrics = calc_race_metrics(bdf)
    f_metrics = calc_race_metrics(fdf)
    
    diff_ll = f_metrics["ll_sum"] - b_metrics["ll_sum"]
    diff_brier = f_metrics["brier_sum"] - b_metrics["brier_sum"]
    counts = b_metrics["count"]
    
    n_boot = 1000
    ll_diffs = []
    brier_diffs = []
    races = diff_ll.index.to_numpy()
    
    for _ in range(n_boot):
        idx = np.random.choice(races, size=len(races), replace=True)
        tot_count = counts.loc[idx].sum()
        ll_diffs.append(diff_ll.loc[idx].sum() / tot_count)
        brier_diffs.append(diff_brier.loc[idx].sum() / tot_count)
        
    ll_diffs = np.array(ll_diffs)
    brier_diffs = np.array(brier_diffs)
    
    res = {
        "n_bootstrap": n_boot,
        "seed": 42,
        "races": len(races),
        "rows": counts.sum(),
        "prob_col": "probability_raw",
        "ll_diff_mean": np.mean(ll_diffs),
        "ll_diff_ci95": [np.percentile(ll_diffs, 2.5), np.percentile(ll_diffs, 97.5)],
        "brier_diff_mean": np.mean(brier_diffs),
        "brier_diff_ci95": [np.percentile(brier_diffs, 2.5), np.percentile(brier_diffs, 97.5)],
    }
    with open(OUT_DIR / "paired_bootstrap.json", "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Saved paired_bootstrap.json: {res}")

def audit_5():
    logger.info("=== Audit 5: market_logit extrapolation distribution ===")
    data_path = Path("data/derived/history_extension_2006_phase5_v1/history_features_2006_2026.parquet")
    df = pd.read_parquet(data_path, columns=["Year", "target_place", "fuku_odds_low", "tan_odds", "fuku_odds_high", "SyussoTosu", "place_rank_limit", "fuku_ninki", "tan_ninki", "race_id"])
    df["actual_place"] = df["target_place"]
    import scripts.run_place_market_offset_catboost_v1 as base_v1
    df = base_v1.add_market_features(df)
    
    import yaml
    with open("config/place_market_offset_catboost_c1r0_v1.yaml") as f:
        cfg = yaml.safe_load(f)
    eps = float(cfg["epsilon"])
    
    # Same logic as Phase 5 market_logit script
    d_recent = df[df["Year"] >= 2016].copy()
    d_recent = base_v1.expanding_market_predictions_for_train(d_recent, cfg, eps)
    
    d_old = df[df["Year"] < 2016].copy()
    train_extrapolate = df[(df["Year"] >= 2016) & (df["Year"] <= 2019)]
    model = base_v1.fit_market_model(train_extrapolate, cfg)
    p = base_v1.clip_prob(model.predict_proba(base_v1.market_x(d_old, cfg["market_baseline"]["features"]))[:, 1], eps)
    d_old["baseline_source"] = "extrapolated_from_2016_2019"
    d_old["p_market"] = p
    d_old["market_logit"] = base_v1.logit(p, eps)
    
    combined = pd.concat([d_old, d_recent], ignore_index=True)
    
    records = []
    for period, mask in [("2006-2015", combined["Year"] < 2016), ("2016-2019", combined["Year"].between(2016, 2019))]:
        mdf = combined[mask]
        ll = calc_logloss(mdf["actual_place"].fillna(0).to_numpy(), mdf["p_market"].to_numpy())
        records.append({
            "period": period,
            "mean": mdf["market_logit"].mean(),
            "std": mdf["market_logit"].std(),
            "min": mdf["market_logit"].min(),
            "max": mdf["market_logit"].max(),
            "p1": mdf["market_logit"].quantile(0.01),
            "p50": mdf["market_logit"].quantile(0.50),
            "p99": mdf["market_logit"].quantile(0.99),
            "nan_rate": mdf["market_logit"].isna().mean(),
            "inf_rate": np.isinf(mdf["market_logit"]).mean(),
            "clip_rate": (mdf["p_market"] <= eps).mean() + (mdf["p_market"] >= 1-eps).mean(),
            "market_logloss": ll
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(OUT_DIR / "market_logit_dist.csv", index=False)
    logger.info(f"Saved market_logit_dist.csv")

if __name__ == "__main__":
    preds = load_predictions()
    audit_1_and_2(preds)
    audit_3(preds)
    audit_4(preds)
    audit_5()
    logger.info("All audits completed.")
