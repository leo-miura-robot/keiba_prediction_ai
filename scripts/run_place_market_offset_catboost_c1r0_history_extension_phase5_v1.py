import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_place_market_offset_catboost_v1 as base_v1

CONFIG_PATH = Path("config/place_market_offset_catboost_c1r0_history_extension_phase5_v1.yaml")
OUT_DIR = Path("outputs/place_market_offset_catboost_c1r0_history_extension_phase5_v1")

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def get_market_logit(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eps = float(cfg["epsilon"])
    d = df.copy()
    
    # 2016-2026 using expanding
    d_recent = d[d["Year"] >= 2016].copy()
    d_recent = base_v1.expanding_market_predictions_for_train(d_recent, cfg, eps)
    
    # Train market model on 2016-2019 for extrapolating to 2006-2015
    d_old = d[d["Year"] < 2016].copy()
    if not d_old.empty:
        train_extrapolate = d[(d["Year"] >= 2016) & (d["Year"] <= 2019)]
        model = base_v1.fit_market_model(train_extrapolate, cfg)
        p = base_v1.clip_prob(model.predict_proba(base_v1.market_x(d_old, cfg["market_baseline"]["features"]))[:, 1], eps)
        d_old["baseline_source"] = "extrapolated_from_2016_2019"
        d_old["p_market"] = p
        d_old["market_logit"] = base_v1.logit(p, eps)
        
    out = pd.concat([d_old, d_recent], ignore_index=True) if not d_old.empty else d_recent
    return out

def row_removed_roi(df: pd.DataFrame, limit: int = 10000) -> dict:
    top_n = df.nlargest(limit, "probability_raw")
    mask = (top_n["target_place"] == 1) & (top_n["fuku_pay"] > 5000)
    filtered = top_n[~mask]
    cost = len(filtered) * 100
    ret = filtered.loc[filtered["target_place"] == 1, "fuku_pay"].sum()
    return {"cost": cost, "ret": ret, "roi": (ret / cost * 100) if cost else 0}

def payout_zeroed_stress_roi(df: pd.DataFrame, limit: int = 10000) -> dict:
    top_n = df.nlargest(limit, "probability_raw").copy()
    cost = len(top_n) * 100
    top_n.loc[top_n["fuku_pay"] > 5000, "fuku_pay"] = 0
    ret = top_n.loc[top_n["target_place"] == 1, "fuku_pay"].sum()
    return {"cost": cost, "ret": ret, "roi": (ret / cost * 100) if cost else 0}

def evaluate_roi(df: pd.DataFrame, ev_limit: float = 1.0) -> dict:
    evs = df["probability_raw"] * df["fuku_odds_low"]
    picks = df[evs >= ev_limit]
    cost = len(picks) * 100
    ret = picks.loc[picks["target_place"] == 1, "fuku_pay"].sum()
    sp, _ = stats.spearmanr(evs, df["target_place"] * df["fuku_pay"])
    return {
        "ev_gte_1_count": len(picks),
        "cost": cost,
        "ret": ret,
        "roi": (ret / cost * 100) if cost else 0,
        "spearman": sp
    }

def train_and_predict(train: pd.DataFrame, val: pd.DataFrame, num_features: list, cat_features: list, cfg: dict) -> pd.DataFrame:
    x_train = base_v1.prepare_x(train, num_features, cat_features)
    y_train = train["actual_place"].to_numpy(int)
    w_train = np.ones(len(train))
    baseline_train = train["market_logit"].to_numpy()
    
    x_val = base_v1.prepare_x(val, num_features, cat_features)
    y_val = val["actual_place"].to_numpy(int)
    baseline_val = val["market_logit"].to_numpy()
    
    c_idx = base_v1.cat_indices(x_train, cat_features)
    ptrain = Pool(x_train, label=y_train, weight=w_train, baseline=baseline_train, cat_features=c_idx)
    pval = Pool(x_val, label=y_val, baseline=baseline_val, cat_features=c_idx)
    
    model = CatBoostClassifier(**cfg["training_params"])
    model.fit(ptrain, eval_set=pval, verbose=100)
    
    val_out = val.copy()
    val_out["final_logit"] = model.predict(pval, prediction_type="RawFormulaVal")
    val_out["probability_raw"] = base_v1.sigmoid(val_out["final_logit"].to_numpy())
    return val_out

def main():
    with open("config/place_market_offset_catboost_c1r0_v1.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUT_DIR / "phase5.log")
    
    logger.info("Loading BASE_2016 predictions")
    phase1_dir = Path("outputs/place_market_offset_catboost_c1r0_feature_cleanup_v1")
    base_pred = pd.read_parquet(phase1_dir / "ablation_oof_predictions.parquet")
    base_pred = base_pred[base_pred["model_key"] == "C1R0_fixed300_ablation_drop_person_codes"].copy()
    base_pred["probability_raw"] = base_pred["probability"]
    base_pred["model_key"] = "BASE_2016"
    
    logger.info("Loading Phase 5 history features")
    data_path = Path("data/derived/history_extension_2006_phase5_v1/history_features_2006_2026.parquet")
    df = pd.read_parquet(data_path)
    df["actual_place"] = df["target_place"]
    df = base_v1.add_market_features(df)
    df = df[df["eligible_for_place_training"] == True].copy()
    
    logger.info("Generating market logit")
    df = get_market_logit(df, cfg)
    
    # Drop Person Codes
    with open(cfg["feature_set_yaml"], "r", encoding="utf-8") as f:
        feature_set = yaml.safe_load(f)["market_aware"]
    
    num_features = [c for c in feature_set.get("numeric", []) if c not in ["Year", "trainer_past_starts", "jockey_past_starts"]]
    num_features.extend(["trainer_past_starts", "jockey_past_starts"]) # keep raw
    cat_features = [c for c in feature_set.get("categorical", []) if c not in ["KisyuCode", "ChokyosiCode"]]
    
    cfg["training_params"]["iterations"] = 300
    
    all_oof = [base_pred[base_pred["Year"].between(2020, 2026)]]
    
    for candidate in ["WARMUP_2006_TRAIN_2016", "FULL_2006"]:
        logger.info(f"Training {candidate}")
        val_parts = []
        for year in range(2020, 2027):
            val = df[df["Year"] == year].copy()
            if candidate == "WARMUP_2006_TRAIN_2016":
                train = df[(df["Year"] >= 2016) & (df["Year"] < year)].copy()
            else:
                train = df[(df["Year"] >= 2006) & (df["Year"] < year)].copy()
                
            out = train_and_predict(train, val, num_features, cat_features, cfg)
            val_parts.append(out)
            
        cand_pred = pd.concat(val_parts, ignore_index=True)
        cand_pred["model_key"] = candidate
        all_oof.append(cand_pred)
        
    combined = pd.concat(all_oof, ignore_index=True)
    combined["race_date"] = combined["race_date"].astype(str)
    combined.to_parquet(OUT_DIR / "phase5_oof_predictions.parquet")
    
    logger.info("Evaluating ROI")
    summary = []
    for key in ["BASE_2016", "WARMUP_2006_TRAIN_2016", "FULL_2006"]:
        pdf = combined[combined["model_key"] == key]
        for year in range(2020, 2027):
            ydf = pdf[pdf["Year"] == year]
            if ydf.empty: continue
            
            brier = ((ydf["probability_raw"] - ydf["target_place"]) ** 2).mean()
            ll = -np.log(ydf.loc[ydf["target_place"]==1, "probability_raw"]).sum() - np.log(1 - ydf.loc[ydf["target_place"]==0, "probability_raw"]).sum()
            ll /= len(ydf)
            
            roi = evaluate_roi(ydf)
            rr = row_removed_roi(ydf)
            pz = payout_zeroed_stress_roi(ydf)
            
            summary.append({
                "model_key": key,
                "Year": year,
                "brier": brier,
                "logloss": ll,
                "ev_gte_1_count": roi["ev_gte_1_count"],
                "cost": roi["cost"],
                "ret": roi["ret"],
                "roi": roi["roi"],
                "row_removed_roi": rr["roi"],
                "payout_zeroed_stress_roi": pz["roi"],
                "ev_roi_spearman": roi["spearman"]
            })
            
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "phase5_evaluation_summary.csv", index=False)
    logger.info("Phase 5 Evaluation complete.")

if __name__ == "__main__":
    main()
