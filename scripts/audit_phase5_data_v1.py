import polars as pl
from pathlib import Path
import csv

def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    out_dir = Path("data/derived/history_extension_2006_phase5_v1")
    data_path = out_dir / "history_features_2006_2026.parquet"
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    print(f"Loading {data_path}...")
    df = pl.read_parquet(data_path)
    
    # 1. target_definition_consistency.csv
    print("Auditing target definitions...")
    target_rows = []
    # group by period (2006-2015 vs 2016-2026) and place_rank_limit
    df_pd = df.to_pandas()
    df_pd["period"] = df_pd["Year"].apply(lambda y: "2006-2015" if y <= 2015 else "2016-2026")
    
    for period in ["2006-2015", "2016-2026"]:
        pdf = df_pd[df_pd["period"] == period]
        if pdf.empty: continue
        for prl in pdf["place_rank_limit"].unique():
            pdf_prl = pdf[pdf["place_rank_limit"] == prl]
            for tosu in [7, 8]:
                # check 7 or less, vs 8 or more
                if tosu == 7:
                    sub = pdf_prl[pdf_prl["SyussoTosu"] <= 7]
                else:
                    sub = pdf_prl[pdf_prl["SyussoTosu"] >= 8]
                    
                if sub.empty: continue
                # target_place_paid is 1 if is_place_paid == 1
                pos_rate = sub["target_place_paid"].mean()
                target_rows.append({
                    "period": period,
                    "place_rank_limit": prl,
                    "SyussoTosu_group": "<=7" if tosu == 7 else ">=8",
                    "count": len(sub),
                    "positive_rate": pos_rate,
                })
    write_csv_rows(out_dir / "target_definition_consistency.csv", target_rows)

    # 2. db_concept_drift_summary.csv
    print("Auditing concept drift...")
    drift_rows = []
    for year in sorted(df_pd["Year"].unique()):
        ydf = df_pd[df_pd["Year"] == year]
        drift_rows.append({
            "year": year,
            "mean_SyussoTosu": ydf["SyussoTosu"].mean(),
            "mean_fuku_odds_low": ydf["fuku_odds_low"].mean() if "fuku_odds_low" in ydf.columns else None,
            "mean_fuku_pay": ydf["fuku_pay"].mean() if "fuku_pay" in ydf.columns else None,
        })
    write_csv_rows(out_dir / "db_concept_drift_summary.csv", drift_rows)
    
    # 3. history_saturation_by_year.csv
    print("Auditing history saturation...")
    sat_rows = []
    for year in sorted(df_pd["Year"].unique()):
        ydf = df_pd[df_pd["Year"] == year]
        
        for entity, prefix in [("jockey", "jockey_"), ("trainer", "trainer_"), ("horse", "horse_")]:
            starts_col = f"{prefix}past_starts"
            win_col = f"{prefix}win_rate"
            if starts_col in ydf.columns:
                starts = ydf[starts_col].dropna()
                win = ydf[win_col].dropna() if win_col in ydf.columns else None
                if len(starts) > 0:
                    sat_rows.append({
                        "year": year,
                        "entity": entity,
                        "median_past_starts": starts.median(),
                        "p90_past_starts": starts.quantile(0.9),
                        "p99_past_starts": starts.quantile(0.99),
                        "median_win_rate": win.median() if win is not None and len(win) > 0 else None,
                        "std_win_rate": win.std() if win is not None and len(win) > 0 else None,
                    })
    write_csv_rows(out_dir / "history_saturation_by_year.csv", sat_rows)
    
    # 4. market_logit_2006_2015_extrapolation_audit.csv
    print("Auditing market logit extrapolation...")
    if "market_logit" in df_pd.columns:
        logit_rows = []
        for year in sorted(df_pd["Year"].unique()):
            ydf = df_pd[df_pd["Year"] == year]
            ml = ydf["market_logit"].dropna()
            if len(ml) > 0:
                logit_rows.append({
                    "year": year,
                    "extrapolation": True if year <= 2015 else False,
                    "mean": ml.mean(),
                    "std": ml.std(),
                    "p01": ml.quantile(0.01),
                    "p99": ml.quantile(0.99)
                })
        write_csv_rows(out_dir / "market_logit_2006_2015_extrapolation_audit.csv", logit_rows)

    print("Audits complete.")

if __name__ == "__main__":
    main()
