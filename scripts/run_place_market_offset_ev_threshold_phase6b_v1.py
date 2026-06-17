from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def threshold_grid(cfg: dict[str, Any]) -> list[float]:
    start = int(round(float(cfg["threshold_min"]) * 100))
    end = int(round(float(cfg["threshold_max"]) * 100))
    step = int(round(float(cfg["threshold_step"]) * 100))
    return [x / 100.0 for x in range(start, end + 1, step)]


def load_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["phase6a_output_root"]) / "phase6a_calibrated_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    keep = (
        (df["strategy"].eq(cfg["champion_strategy"]) & df["calibration_method"].eq(cfg["champion_calibration_method"]))
        | (df["strategy"].eq(cfg["challenger_strategy"]) & df["calibration_method"].eq(cfg["challenger_calibration_method"]))
    )
    df = df[keep & df["Year"].isin(cfg["selection_years"] + cfg["diagnostic_years"])].copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    df["ev"] = pd.to_numeric(df[cfg["probability_column"]], errors="raise") * pd.to_numeric(df[cfg["odds_column"]], errors="coerce")
    df["payout"] = pd.to_numeric(df[cfg["payout_column"]], errors="coerce").fillna(0.0)
    if df.duplicated(["entry_id", "race_id", "race_date", "Year", "strategy"]).any():
        raise ValueError("Duplicate prediction keys")
    return df


def picks_for(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["ev"].ge(threshold)].copy()


def roi_summary(picks: pd.DataFrame, cfg: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    stake_yen = int(cfg["stake_yen"])
    stake = len(picks) * stake_yen
    payout = float(picks["payout"].sum()) if len(picks) else 0.0
    return {
        **label,
        "bet_count": int(len(picks)),
        "race_count_with_bet": int(picks["race_id"].nunique()) if len(picks) else 0,
        "stake": int(stake),
        "payout": payout,
        "roi": float(payout / stake * 100.0) if stake else math.nan,
        "hit_count": int((picks["payout"] > 0).sum()) if len(picks) else 0,
        "hit_rate": float((picks["payout"] > 0).mean()) if len(picks) else math.nan,
        "average_odds": float(picks[cfg["odds_column"]].mean()) if len(picks) else math.nan,
        "median_odds": float(picks[cfg["odds_column"]].median()) if len(picks) else math.nan,
        "average_probability": float(picks[cfg["probability_column"]].mean()) if len(picks) else math.nan,
        "average_ev": float(picks["ev"].mean()) if len(picks) else math.nan,
        "median_ev": float(picks["ev"].median()) if len(picks) else math.nan,
    }


def grid_tables(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly, combined = [], []
    for strategy, sd in df.groupby("strategy"):
        method = sd["calibration_method"].iloc[0]
        for th in threshold_grid(cfg):
            for year, g in sd[sd["Year"].isin(cfg["selection_years"])].groupby("Year"):
                yearly.append(roi_summary(picks_for(g, th), cfg, {"strategy": strategy, "calibration_method": method, "threshold": th, "Year": int(year)}))
            all_sel = sd[sd["Year"].isin(cfg["selection_years"])]
            combined.append(roi_summary(picks_for(all_sel, th), cfg, {"strategy": strategy, "calibration_method": method, "threshold": th, "Year": "2020_2024"}))
    return pd.DataFrame(yearly), pd.DataFrame(combined)


def stress_tables(df: pd.DataFrame, cfg: dict[str, Any], years: list[int], thresholds: list[tuple[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rr_rows, pz_rows = [], []
    for label_name, th in thresholds:
        for strategy, sd in df[df["Year"].isin(years)].groupby("strategy"):
            picks = picks_for(sd, th)
            normal = roi_summary(picks, cfg, {"strategy": strategy, "threshold_label": label_name, "threshold": th, "period": f"{min(years)}_{max(years)}"})
            hits = picks[picks["payout"].gt(0)].sort_values("payout", ascending=False)
            total_hit_payout = float(hits["payout"].sum())
            for limit in [1, 3, 5, 10]:
                removed_idx = hits.head(limit).index
                rr = picks.drop(index=removed_idx)
                pz = picks.copy()
                pz.loc[removed_idx, "payout"] = 0.0
                for kind, frame, rows in [("row_removed", rr, rr_rows), ("payout_zeroed", pz, pz_rows)]:
                    s = roi_summary(frame, cfg, {k: normal[k] for k in ["strategy", "threshold_label", "threshold", "period"]})
                    s.update(
                        {
                            "limit": limit,
                            "normal_roi": normal["roi"],
                            f"{kind}_roi": s["roi"],
                            "roi_drop_point": normal["roi"] - s["roi"] if not math.isnan(normal["roi"]) and not math.isnan(s["roi"]) else math.nan,
                            "remaining_bet_count": s["bet_count"],
                            "removed_or_zeroed_payout_share": float(hits.head(limit)["payout"].sum() / total_hit_payout) if total_hit_payout else math.nan,
                        }
                    )
                    if kind == "payout_zeroed" and not math.isnan(s["roi"]) and not math.isnan(normal["roi"]) and s["roi"] > normal["roi"] + 1e-12:
                        raise RuntimeError("payout_zeroed_stress_roi exceeded normal_roi")
                    rows.append(s)
    return pd.DataFrame(rr_rows), pd.DataFrame(pz_rows)


def race_bootstrap(df: pd.DataFrame, cfg: dict[str, Any], years: list[int], thresholds: list[float], strategy: str) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["random_seed"]))
    rows = []
    base = df[df["strategy"].eq(strategy) & df["Year"].isin(years)].copy()
    races = pd.DataFrame({"race_id": sorted(base["race_id"].unique())})
    for th in thresholds:
        picks = picks_for(base, th)
        race = races.merge(picks.groupby("race_id").agg(stake=("entry_id", lambda x: len(x) * int(cfg["stake_yen"])), payout=("payout", "sum")), on="race_id", how="left").fillna(0.0)
        vals = race[["stake", "payout"]].to_numpy(float)
        point = float(vals[:, 1].sum() / vals[:, 0].sum() * 100.0) if vals[:, 0].sum() else math.nan
        draws = np.empty(int(cfg["bootstrap_iterations"]), dtype=float)
        for i in range(len(draws)):
            idx = rng.integers(0, len(vals), len(vals))
            st = vals[idx, 0].sum()
            draws[i] = vals[idx, 1].sum() / st * 100.0 if st else math.nan
        good = draws[~np.isnan(draws)]
        rows.append(
            {
                "strategy": strategy,
                "threshold": th,
                "years": f"{min(years)}_{max(years)}",
                "point_roi": point,
                "bootstrap_mean_roi": float(np.mean(good)) if len(good) else math.nan,
                "roi_ci_lower": float(np.percentile(good, 2.5)) if len(good) else math.nan,
                "roi_ci_upper": float(np.percentile(good, 97.5)) if len(good) else math.nan,
                "probability_roi_ge_90": float(np.mean(good >= 90.0)) if len(good) else 0.0,
                "probability_roi_ge_100": float(np.mean(good >= 100.0)) if len(good) else 0.0,
                "races": int(len(vals)),
                "n_bootstrap": int(cfg["bootstrap_iterations"]),
            }
        )
    return pd.DataFrame(rows)


def eligibility(combined: pd.DataFrame, yearly: pd.DataFrame, bootstrap: pd.DataFrame, pz: pd.DataFrame, cfg: dict[str, Any], strategy: str) -> pd.DataFrame:
    rows = []
    elig = cfg["eligibility"]
    c = combined[combined["strategy"].eq(strategy)].copy()
    y = yearly[yearly["strategy"].eq(strategy)].copy()
    b = bootstrap[bootstrap["strategy"].eq(strategy)].copy()
    for _, row in c.iterrows():
        th = float(row["threshold"])
        yy = y[y["threshold"].eq(th)]
        bb = b[b["threshold"].eq(th)].iloc[0]
        pzz = pz[(pz["strategy"].eq(strategy)) & (pz["threshold"].eq(th)) & (pz["period"].eq("2020_2024"))]
        top3 = pzz[pzz["limit"].eq(3)]["payout_zeroed_roi"].iloc[0] if len(pzz[pzz["limit"].eq(3)]) else math.nan
        top5 = pzz[pzz["limit"].eq(5)]["payout_zeroed_roi"].iloc[0] if len(pzz[pzz["limit"].eq(5)]) else math.nan
        passed = (
            row["bet_count"] >= elig["combined_bet_count_min"]
            and yy["Year"].nunique() >= elig["bet_years_min"]
            and yy["bet_count"].min() >= elig["min_yearly_bet_count"]
            and int(yy["roi"].ge(90.0).sum()) >= elig["roi_ge_90_years_min"]
            and row["roi"] >= elig["combined_roi_min"]
            and bb["probability_roi_ge_90"] >= elig["probability_roi_ge_90_min"]
            and top3 >= elig["top3_payout_zeroed_roi_min"]
            and top5 >= elig["top5_payout_zeroed_roi_min"]
        )
        rows.append(
            {
                "strategy": strategy,
                "threshold": th,
                "eligible": bool(passed),
                "combined_roi": row["roi"],
                "combined_bet_count": int(row["bet_count"]),
                "bet_years": int(yy["Year"].nunique()),
                "minimum_yearly_bet_count": int(yy["bet_count"].min()) if len(yy) else 0,
                "roi_ge_90_years": int(yy["roi"].ge(90.0).sum()),
                "minimum_yearly_roi": float(yy["roi"].min()) if len(yy) else math.nan,
                "median_yearly_roi": float(yy["roi"].median()) if len(yy) else math.nan,
                "roi_std": float(yy["roi"].std(ddof=1)) if len(yy) > 1 else math.nan,
                "worst_year_drawdown_from_100": float(100.0 - yy["roi"].min()) if len(yy) else math.nan,
                "probability_roi_ge_90": float(bb["probability_roi_ge_90"]),
                "roi_ci_lower": float(bb["roi_ci_lower"]),
                "top3_payout_zeroed_roi": float(top3),
                "top5_payout_zeroed_roi": float(top5),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(elig_df: pd.DataFrame, nested: pd.DataFrame, cfg: dict[str, Any], strategy: str) -> dict[str, Any]:
    e = elig_df[elig_df["strategy"].eq(strategy) & elig_df["eligible"]].copy()
    if e.empty:
        return {"threshold_status": "NO_THRESHOLD_CERTIFIED", "recommended_threshold": None, "activation_recommended": False, "operationally_activated": False, "reason": "No threshold passed eligibility."}
    e = e.sort_values(["probability_roi_ge_90", "roi_ci_lower", "top5_payout_zeroed_roi", "minimum_yearly_roi", "combined_bet_count", "threshold"], ascending=[False, False, False, False, False, True])
    th = float(e.iloc[0]["threshold"])
    nw = nested[nested["validation_year"].isin([2021, 2022, 2023, 2024])]
    nested_roi = float(nw["validation_payout"].sum() / nw["validation_stake"].sum() * 100.0) if nw["validation_stake"].sum() else math.nan
    status = "THRESHOLD_CANDIDATE_CERTIFIED" if nested_roi >= cfg["eligibility"]["nested_validation_combined_roi_min"] else "DIAGNOSTIC_ONLY"
    return {
        "threshold_status": status,
        "recommended_threshold": th,
        "activation_recommended": status == "THRESHOLD_CANDIDATE_CERTIFIED",
        "operationally_activated": False,
        "nested_validation_combined_roi": nested_roi,
        "reason": "Threshold chosen by conservative ranking; operational activation remains false.",
    }


def nested_walk_forward(df: pd.DataFrame, cfg: dict[str, Any], strategy: str) -> pd.DataFrame:
    rows = []
    for validation_year in [2021, 2022, 2023, 2024]:
        sel_years = list(range(2020, validation_year))
        yearly, combined = grid_tables(df[df["strategy"].eq(strategy) & df["Year"].isin(sel_years + [validation_year])], cfg)
        boot = race_bootstrap(df, cfg, sel_years, threshold_grid(cfg), strategy)
        rr, pz = stress_tables(df[df["strategy"].eq(strategy)], cfg, sel_years, [("grid", th) for th in threshold_grid(cfg)])
        elig_df = eligibility(combined, yearly, boot, pz, cfg, strategy)
        ok = elig_df[elig_df["eligible"]].sort_values(["probability_roi_ge_90", "roi_ci_lower", "top5_payout_zeroed_roi", "minimum_yearly_roi", "combined_bet_count", "threshold"], ascending=[False, False, False, False, False, True])
        th = float(ok.iloc[0]["threshold"]) if len(ok) else 1.00
        val = df[df["strategy"].eq(strategy) & df["Year"].eq(validation_year)]
        ps = picks_for(val, th)
        s = roi_summary(ps, cfg, {"selection_end_year": validation_year - 1, "selected_threshold": th, "selection_bet_count": int(combined[combined["threshold"].eq(th)]["bet_count"].iloc[0]) if len(combined[combined["threshold"].eq(th)]) else 0, "selection_combined_roi": float(combined[combined["threshold"].eq(th)]["roi"].iloc[0]) if len(combined[combined["threshold"].eq(th)]) else math.nan, "validation_year": validation_year})
        for col in ["bet_count", "race_count_with_bet", "stake", "payout", "roi", "hit_count", "hit_rate"]:
            s[f"validation_{col}"] = s.pop(col)
        rows.append({"selection_years": ",".join(map(str, sel_years)), "eligibility_candidates": int(len(ok)), **s})
    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame, cfg: dict[str, Any], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for strategy, sd in df[df["Year"].isin(cfg["diagnostic_years"])].groupby("strategy"):
        for year, g in sd.groupby("Year"):
            rows.append(roi_summary(picks_for(g, threshold), cfg, {"strategy": strategy, "threshold": threshold, "Year": int(year)}))
        rows.append(roi_summary(picks_for(sd, threshold), cfg, {"strategy": strategy, "threshold": threshold, "Year": "2025_2026"}))
    boot = pd.concat([race_bootstrap(df, cfg, cfg["diagnostic_years"], [threshold], s) for s in df["strategy"].unique()], ignore_index=True)
    rr, pz = stress_tables(df, cfg, cfg["diagnostic_years"], [("recommended", threshold)])
    return pd.DataFrame(rows), pd.concat([rr.assign(stress_type="row_removed"), pz.assign(stress_type="payout_zeroed")], ignore_index=True), boot


def bet_overlap(df: pd.DataFrame, cfg: dict[str, Any], threshold: float) -> pd.DataFrame:
    rows = []
    for years, label in [(cfg["selection_years"], "2020_2024"), (cfg["diagnostic_years"], "2025_2026")]:
        p = {s: picks_for(g[g["Year"].isin(years)], threshold) for s, g in df.groupby("strategy")}
        keys = ["entry_id", "race_id", "race_date", "Year"]
        a, b = p[cfg["champion_strategy"]], p[cfg["challenger_strategy"]]
        aset = set(map(tuple, a[keys].to_numpy()))
        bset = set(map(tuple, b[keys].to_numpy()))
        common, only_a, only_b = aset & bset, aset - bset, bset - aset
        rows.append({"period": label, "threshold": threshold, "common_bets": len(common), "champion_only_bets": len(only_a), "challenger_only_bets": len(only_b), "jaccard": len(common) / len(aset | bset) if aset | bset else math.nan})
    return pd.DataFrame(rows)


def segment_diagnostic(df: pd.DataFrame, cfg: dict[str, Any], threshold: float) -> pd.DataFrame:
    d = picks_for(df[df["strategy"].eq(cfg["champion_strategy"])].copy(), threshold)
    d["period"] = np.where(d["Year"].isin(cfg["selection_years"]), "2020_2024", "2025_2026")
    track = pd.to_numeric(d["TrackCD"], errors="coerce") if "TrackCD" in d.columns else pd.Series(np.nan, index=d.index)
    kyori = pd.to_numeric(d["Kyori"], errors="coerce") if "Kyori" in d.columns else pd.Series(np.nan, index=d.index)
    field = pd.to_numeric(d["SyussoTosu"], errors="coerce") if "SyussoTosu" in d.columns else pd.Series(np.nan, index=d.index)
    d["surface"] = np.where(track.fillna(0).between(10, 22), "turf", "dirt_or_other")
    d["distance_bucket"] = pd.cut(kyori, [0, 1399, 1799, 2199, 10000], labels=["short", "mile", "middle", "long"])
    d["field_size_bucket"] = pd.cut(field, [0, 7, 11, 15, 99], labels=["small", "medium", "large", "max"])
    d["odds_bucket"] = pd.cut(pd.to_numeric(d[cfg["odds_column"]], errors="coerce"), [0, 2, 5, 10, 9999], labels=["low", "mid", "high", "long"])
    d["EV_bucket"] = pd.cut(d["ev"], [0, threshold, 1.1, 1.2, 9999], labels=["at_threshold", "ev_1_10", "ev_1_20", "ev_high"])
    rows = []
    for dim in ["Year", "JyoCD", "surface", "distance_bucket", "field_size_bucket", "odds_bucket", "EV_bucket"]:
        for (period, value), g in d.groupby(["period", dim], dropna=False, observed=False):
            s = roi_summary(g, cfg, {"period": period, "segment": dim, "value": str(value)})
            s["small_sample"] = s["bet_count"] < 20
            total_payout = d[d["period"].eq(period)]["payout"].sum()
            s["payout_share"] = float(g["payout"].sum() / total_payout) if total_payout else math.nan
            rows.append(s)
    return pd.DataFrame(rows)


def write_report(out: Path) -> None:
    selected = json.loads((out / "selected_threshold.json").read_text(encoding="utf-8"))
    elig = pd.read_csv(out / "threshold_eligibility.csv")
    boot = pd.read_csv(out / "threshold_roi_bootstrap.csv")
    diag = pd.read_csv(out / "diagnostic_2025_2026.csv")
    shadow = pd.read_csv(out / "shadow_threshold_comparison.csv")
    text = "\n".join(
        [
            "# Phase 6B Conservative EV Threshold Results",
            "",
            json.dumps(selected, ensure_ascii=False, indent=2),
            "",
            "## Eligibility",
            elig.to_markdown(index=False),
            "",
            "## Bootstrap",
            boot.to_markdown(index=False),
            "",
            "## 2025/2026 Diagnostic",
            diag.to_markdown(index=False),
            "",
            "## Shadow",
            shadow.to_markdown(index=False),
            "",
        ]
    )
    Path("docs/place_market_offset_ev_threshold_phase6b_v1_results.md").write_text(text, encoding="utf-8")
    (out / "audit_report.md").write_text(text, encoding="utf-8")


def run(config: Path) -> int:
    cfg = load_yaml(config)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    df = load_predictions(cfg)
    yearly, combined = grid_tables(df, cfg)
    yearly.to_csv(out / "threshold_grid_by_year.csv", index=False)
    combined.to_csv(out / "threshold_grid_combined_2020_2024.csv", index=False)

    champion = cfg["champion_strategy"]
    all_thresholds = threshold_grid(cfg)
    rr, pz = stress_tables(df[df["strategy"].eq(champion)], cfg, cfg["selection_years"], [("grid", th) for th in all_thresholds])
    rr.to_csv(out / "threshold_row_removed_stress.csv", index=False)
    pz.to_csv(out / "threshold_payout_zeroed_stress.csv", index=False)
    boot = race_bootstrap(df, cfg, cfg["selection_years"], all_thresholds, champion)
    boot.to_csv(out / "threshold_roi_bootstrap.csv", index=False)
    elig = eligibility(combined, yearly, boot, pz, cfg, champion)
    nested = nested_walk_forward(df, cfg, champion)
    selected = choose_threshold(elig, nested, cfg, champion)
    elig.to_csv(out / "threshold_eligibility.csv", index=False)
    nested.to_csv(out / "threshold_nested_walk_forward.csv", index=False)
    (out / "selected_threshold.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    th = float(selected["recommended_threshold"] or 1.0)
    diag, diag_stress, diag_boot = diagnostics(df, cfg, th)
    diag.to_csv(out / "diagnostic_2025_2026.csv", index=False)
    diag_stress.to_csv(out / "diagnostic_2025_2026_stress.csv", index=False)
    diag_boot.to_csv(out / "diagnostic_2025_2026_bootstrap.csv", index=False)
    shadow = combined[combined["threshold"].eq(th)].copy()
    shadow["comparison"] = "same_champion_threshold"
    shadow_best = combined[combined["strategy"].eq(cfg["challenger_strategy"])].sort_values(["roi", "bet_count"], ascending=[False, False]).head(1).copy()
    shadow_best["comparison"] = "shadow_diagnostic_best_roi_not_for_selection"
    pd.concat([shadow, shadow_best], ignore_index=True).to_csv(out / "shadow_threshold_comparison.csv", index=False)
    bet_overlap(df, cfg, th).to_csv(out / "bet_overlap.csv", index=False)
    segment_diagnostic(df, cfg, th).to_csv(out / "segment_diagnostic.csv", index=False)
    manifest = {
        "version": cfg["version"],
        "new_catboost_training": False,
        "db_connection": False,
        "calibration_refit": False,
        "selection_years": cfg["selection_years"],
        "diagnostic_years": cfg["diagnostic_years"],
        "operationally_activated": False,
        "champion_changed": False,
        "output_files": sorted(p.name for p in out.glob("*") if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_ev_threshold_phase6b_v1.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
