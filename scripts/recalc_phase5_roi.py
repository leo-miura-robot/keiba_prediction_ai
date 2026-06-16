import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

OUT_DIR = Path("outputs/place_market_offset_catboost_c1r0_history_extension_phase5_v1")

def row_removed_roi(df: pd.DataFrame, limit: int = 10000) -> dict:
    top_n = df.nlargest(limit, "probability_raw")
    mask = (top_n["actual_place"] == 1) & (top_n["fuku_pay"] > 5000)
    filtered = top_n[~mask]
    cost = len(filtered) * 100
    ret = filtered.loc[filtered["actual_place"] == 1, "fuku_pay"].sum()
    return {"cost": cost, "ret": ret, "roi": (ret / cost * 100) if cost else 0}

def payout_zeroed_stress_roi(df: pd.DataFrame, limit: int = 10000) -> dict:
    top_n = df.nlargest(limit, "probability_raw").copy()
    cost = len(top_n) * 100
    top_n.loc[top_n["fuku_pay"] > 5000, "fuku_pay"] = 0
    ret = top_n.loc[top_n["actual_place"] == 1, "fuku_pay"].sum()
    return {"cost": cost, "ret": ret, "roi": (ret / cost * 100) if cost else 0}

def evaluate_roi(df: pd.DataFrame, ev_limit: float = 1.0) -> dict:
    evs = df["probability_raw"] * df["fuku_odds_low"]
    picks = df[evs >= ev_limit]
    cost = len(picks) * 100
    ret = picks.loc[picks["actual_place"] == 1, "fuku_pay"].sum()
    sp, _ = stats.spearmanr(evs, df["actual_place"] * df["fuku_pay"])
    return {
        "ev_gte_1_count": len(picks),
        "cost": cost,
        "ret": ret,
        "roi": (ret / cost * 100) if cost else 0,
        "spearman": sp
    }

def main():
    combined = pd.read_parquet(OUT_DIR / "phase5_oof_predictions.parquet")
    if "target_place" in combined.columns and "actual_place" not in combined.columns:
        combined["actual_place"] = combined["target_place"]
    elif "actual_place" not in combined.columns:
        combined["actual_place"] = combined["target_place_paid"]
        
    summary = []
    for key in ["BASE_2016", "WARMUP_2006_TRAIN_2016", "FULL_2006"]:
        pdf = combined[combined["model_key"] == key]
        for year in range(2020, 2027):
            ydf = pdf[pdf["Year"] == year]
            if ydf.empty: continue
            
            brier = ((ydf["probability_raw"] - ydf["actual_place"]) ** 2).mean()
            ll = -np.log(ydf.loc[ydf["actual_place"]==1, "probability_raw"]).sum() - np.log(1 - ydf.loc[ydf["actual_place"]==0, "probability_raw"]).sum()
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
    print("Recalculation complete.")

if __name__ == "__main__":
    main()
