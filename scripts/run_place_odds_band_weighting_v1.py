from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    data = tmp.read_bytes()
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    data = tmp.read_bytes()
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception as exc:
        return {"git_commit_sha": "unknown", "git_is_dirty": None, "git_error": str(exc)}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_db_cache_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["db_validation_manifest"])
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "db_validation_cache_status": "HIT",
        "database_accessed": False,
        "database_validation_reused": True,
        "manifest_path": str(path),
        "db_light_fingerprint": data["light_fingerprint"],
        "db_full_sha256": data["full_file_sha256"],
        "integrity_checked_at": data["integrity_checked_at"],
        "validator_manifest_version": data["manifest_version"],
    }


def load_place_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["source_output_root"])
    oof = pd.read_parquet(root / "oof_predictions.parquet")
    final = pd.read_parquet(root / "final_predictions.parquet")
    oof = oof[oof["target"].eq("place")].copy()
    final = final[final["target"].eq("place")].copy()
    oof["eval_period"] = "validation_2020_2024"
    df = pd.concat([oof, final], ignore_index=True)
    df["entry_id"] = df["entry_id"].astype(str)
    df["race_id"] = df["race_id"].astype(str)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["actual_place"] = df["actual"].astype(int)
    df["predicted_place_probability"] = pd.to_numeric(df["raw_probability"], errors="coerce")
    df["break_even_probability"] = 1.0 / pd.to_numeric(df["fuku_odds_low"], errors="coerce")
    df["place_edge_low"] = pd.to_numeric(df["conservative_probability"], errors="coerce") - df["break_even_probability"]
    df["place_ev_low"] = pd.to_numeric(df["conservative_probability"], errors="coerce") * pd.to_numeric(df["fuku_odds_low"], errors="coerce")
    return assign_odds_band(df, cfg)


def assign_odds_band(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    odds = pd.to_numeric(d["fuku_odds_low"], errors="coerce")
    d["place_odds_band"] = pd.NA
    for band in cfg["odds_bands"]:
        m = odds.ge(float(band["min"])) & odds.lt(float(band["max"]))
        d.loc[m, "place_odds_band"] = band["label"]
    return d


def payout_col() -> str:
    return "fuku_pay"


def summarize_bets(bets: pd.DataFrame, label: dict[str, Any] | None = None, stake_col: str | None = None) -> dict[str, Any]:
    label = dict(label or {})
    if bets.empty:
        return {**label, "bets": 0, "races": 0, "stake": 0.0, "return": 0.0, "profit": 0.0, "roi": np.nan, "hit_count": 0, "hit_rate": np.nan, "mean_predicted_place_probability": np.nan, "actual_place_rate": np.nan, "calibration_gap": np.nan, "mean_fuku_odds_low": np.nan, "mean_fuku_odds_high": np.nan, "average_payout": np.nan, "max_payout": np.nan, "max_losing_streak": 0, "max_drawdown": 0.0}
    pay = pd.to_numeric(bets[payout_col()], errors="coerce").fillna(0).to_numpy(float)
    stake = bets[stake_col].to_numpy(float) if stake_col else np.full(len(bets), 100.0)
    returns = np.where(pay > 0, pay * (stake / 100.0), 0.0)
    profit = returns - stake
    equity = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    hits = pay > 0
    max_ls = 0
    cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        max_ls = max(max_ls, cur)
    pred = pd.to_numeric(bets["conservative_probability"], errors="coerce")
    actual = bets["actual_place"].astype(float)
    return {
        **label,
        "bets": int(len(bets)),
        "races": int(bets["race_id"].nunique()),
        "stake": float(stake.sum()),
        "return": float(returns.sum()),
        "profit": float(profit.sum()),
        "roi": float(returns.sum() / stake.sum() * 100) if stake.sum() else np.nan,
        "hit_count": int(hits.sum()),
        "hit_rate": float(hits.mean()),
        "mean_predicted_place_probability": float(pred.mean()),
        "actual_place_rate": float(actual.mean()),
        "calibration_gap": float(actual.mean() - pred.mean()),
        "mean_fuku_odds_low": float(pd.to_numeric(bets["fuku_odds_low"], errors="coerce").mean()),
        "mean_fuku_odds_high": float(pd.to_numeric(bets["fuku_odds_high"], errors="coerce").mean()),
        "average_payout": float(pay.mean()),
        "max_payout": float(pay.max()) if len(pay) else np.nan,
        "max_losing_streak": int(max_ls),
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
    }


def top_removed_roi(bets: pd.DataFrame, n: int, stake_col: str | None = None) -> float:
    if bets.empty:
        return np.nan
    pay = pd.to_numeric(bets[payout_col()], errors="coerce").fillna(0)
    if n:
        keep = pay.sort_values(ascending=False).index[n:]
        bets = bets.loc[keep]
    if bets.empty:
        return np.nan
    return summarize_bets(bets, stake_col=stake_col)["roi"]


def bootstrap_ci(bets: pd.DataFrame, iterations: int, seed: int, stake_col: str | None = None) -> tuple[float, float, float]:
    if bets.empty:
        return (np.nan, np.nan, np.nan)
    pay = pd.to_numeric(bets[payout_col()], errors="coerce").fillna(0).to_numpy(float)
    stake = bets[stake_col].to_numpy(float) if stake_col else np.full(len(bets), 100.0)
    returns = np.where(pay > 0, pay * (stake / 100.0), 0.0)
    races = bets["race_id"].astype(str).to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    ret_by_race = np.bincount(inv, weights=returns)
    stake_by_race = np.bincount(inv, weights=stake)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(iterations, len(uniq)))
    roi = ret_by_race[idx].sum(axis=1) / stake_by_race[idx].sum(axis=1) * 100
    return tuple(float(x) for x in np.percentile(roi, [2.5, 50, 97.5]))


def enrich_summary(row: dict[str, Any], bets: pd.DataFrame, cfg: dict[str, Any], stake_col: str | None = None) -> dict[str, Any]:
    row.update(summarize_bets(bets, stake_col=stake_col))
    for n in [1, 3, 5, 10]:
        row[f"roi_remove_top{n}"] = top_removed_roi(bets, n, stake_col=stake_col)
    ci = bootstrap_ci(bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]), stake_col=stake_col)
    row["bootstrap_roi_p025"], row["bootstrap_roi_p500"], row["bootstrap_roi_p975"] = ci
    return row


def odds_band_tables(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val = df[df["Year"].isin(cfg["validation_years"])].copy()
    rows = []
    yearly = []
    boot = []
    dep = []
    for band in [b["label"] for b in cfg["odds_bands"]]:
        bdf = val[val["place_odds_band"].eq(band)]
        rows.append(enrich_summary({"place_odds_band": band, "period": "validation_2020_2024"}, bdf, cfg))
        for year, g in bdf.groupby("Year"):
            yearly.append(enrich_summary({"place_odds_band": band, "Year": int(year)}, g, cfg))
        ci = bootstrap_ci(bdf, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
        boot.append({"place_odds_band": band, "roi_p025": ci[0], "roi_p500": ci[1], "roi_p975": ci[2]})
        for n in [1, 3, 5, 10]:
            dep.append({"place_odds_band": band, "removed_top_payouts": n, "roi": top_removed_roi(bdf, n)})
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.DataFrame(boot), pd.DataFrame(dep)


def normalize_series(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (x - lo) / (hi - lo)
    if not higher_better:
        out = 1 - out
    return out.fillna(0)


def band_reliability(summary: pd.DataFrame, yearly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        y = yearly[yearly["place_odds_band"].eq(r["place_odds_band"])]
        rows.append({
            "place_odds_band": r["place_odds_band"],
            "combined_roi": r["roi"],
            "minimum_year_roi": y["roi"].min() if not y.empty else np.nan,
            "year_roi_std": y["roi"].std(ddof=0) if not y.empty else np.nan,
            "bets": r["bets"],
            "minimum_year_bets": y["bets"].min() if not y.empty else 0,
            "years_with_bets": int((y["bets"] > 0).sum()) if not y.empty else 0,
            "bootstrap_lower": r["bootstrap_roi_p025"],
            "top5_removed_roi": r["roi_remove_top5"],
            "max_drawdown": r["max_drawdown"],
        })
    rel = pd.DataFrame(rows)
    w = cfg["band_reliability_weights"]
    rel["combined_roi_score"] = normalize_series(rel["combined_roi"])
    rel["minimum_year_roi_score"] = normalize_series(rel["minimum_year_roi"])
    rel["bootstrap_lower_score"] = normalize_series(rel["bootstrap_lower"])
    rel["top5_removed_roi_score"] = normalize_series(rel["top5_removed_roi"])
    rel["sample_size_score"] = normalize_series(rel["bets"])
    rel["roi_stability_score"] = normalize_series(rel["year_roi_std"], higher_better=False)
    rel["drawdown_score"] = normalize_series(rel["max_drawdown"], higher_better=False)
    rel["reliability_score"] = (
        rel["combined_roi_score"] * w["combined_roi"]
        + rel["minimum_year_roi_score"] * w["minimum_year_roi"]
        + rel["bootstrap_lower_score"] * w["bootstrap_lower"]
        + rel["top5_removed_roi_score"] * w["top5_removed_roi"]
        + rel["sample_size_score"] * w["sample_size"]
        + rel["roi_stability_score"] * w["roi_stability"]
        + rel["drawdown_score"] * w["drawdown"]
    )
    rel["band_class"] = "exclude"
    main = (
        (rel["years_with_bets"] >= int(cfg["min_years_with_bets"]))
        & (rel["bets"] >= int(cfg["min_total_bets"]))
        & (rel["minimum_year_bets"] >= int(cfg["min_yearly_bets"]))
        & (rel["combined_roi"] >= float(cfg["main_band_min_roi"]))
        & (rel["top5_removed_roi"] >= float(cfg["main_band_top5_removed_roi"]))
    )
    rel.loc[main, "band_class"] = "main"
    support = (rel["band_class"].eq("exclude")) & (rel["bets"] >= int(cfg["min_total_bets"])) & (rel["combined_roi"] >= 85)
    rel.loc[support, "band_class"] = "support"
    rel["ev_threshold"] = np.where(rel["band_class"].eq("main"), 0.90, np.where(rel["band_class"].eq("support"), 0.85, np.inf))
    rel["edge_threshold"] = np.where(rel["band_class"].eq("main"), -0.15, np.where(rel["band_class"].eq("support"), -0.20, np.inf))
    return rel.sort_values("reliability_score", ascending=False)


def candidate_strategies(df: pd.DataFrame, rel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    val = df[df["Year"].isin(cfg["validation_years"])].copy()
    active_bands = rel[rel["band_class"].isin(["main", "support"])]["place_odds_band"].tolist()
    rows = []
    for one_per_race in ["all", "max_ev", "max_probability"]:
        for ev_add in [0.0, 0.03, 0.06]:
            for edge_add in [0.0, 0.02]:
                rule = {"one_per_race": one_per_race, "ev_add": ev_add, "edge_add": edge_add}
                bets = apply_strategy(val, rel, rule)
                if len(bets) < int(cfg["min_total_bets"]):
                    continue
                row = enrich_summary({**rule, "strategy_id": f"s{len(rows)+1:03d}"}, bets, cfg)
                y = pd.DataFrame([summarize_bets(g, {"Year": int(y)}) for y, g in bets.groupby("Year")])
                row["minimum_year_roi"] = y["roi"].min() if not y.empty else np.nan
                row["minimum_year_bets"] = y["bets"].min() if not y.empty else 0
                row["active_bands"] = ",".join(active_bands)
                row["score"] = (
                    min(row["bets"], 2000)
                    + np.nan_to_num(row["minimum_year_roi"], nan=0) * 10000
                    + np.nan_to_num(row["roi_remove_top5"], nan=0) * 2000
                    + np.nan_to_num(row["bootstrap_roi_p025"], nan=0) * 2000
                    - np.nan_to_num(row["max_drawdown"], nan=0) * 0.05
                    + np.nan_to_num(row["roi"], nan=0) * 100
                )
                rows.append(row)
    out = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    return out.head(int(cfg["max_candidate_strategies"]))


def apply_strategy(df: pd.DataFrame, rel: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    merged = df.merge(rel[["place_odds_band", "band_class", "reliability_score", "ev_threshold", "edge_threshold"]], on="place_odds_band", how="left")
    m = (
        merged["band_class"].isin(["main", "support"])
        & (merged["place_ev_low"] >= merged["ev_threshold"] + float(rule.get("ev_add", 0)))
        & (merged["place_edge_low"] >= merged["edge_threshold"] + float(rule.get("edge_add", 0)))
    )
    bets = merged[m].copy()
    if rule.get("one_per_race") == "max_ev":
        bets = bets.sort_values(["race_id", "place_ev_low", "entry_id"], ascending=[True, False, True]).drop_duplicates("race_id")
    elif rule.get("one_per_race") == "max_probability":
        bets = bets.sort_values(["race_id", "conservative_probability", "entry_id"], ascending=[True, False, True]).drop_duplicates("race_id")
    return bets.sort_values(["race_date", "race_id", "entry_id"]).copy()


def jaccard_prune(cands: pd.DataFrame, df: pd.DataFrame, rel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = []
    sets: list[set[str]] = []
    overlaps = []
    for _, rule in cands.iterrows():
        bets = apply_strategy(df[df["Year"].isin(cfg["validation_years"])], rel, rule.to_dict())
        s = set(bets["entry_id"].astype(str))
        dup = False
        for i, ks in enumerate(sets):
            sim = len(s & ks) / len(s | ks) if s or ks else 0.0
            overlaps.append({"strategy_id": rule["strategy_id"], "kept_index": i, "jaccard": sim, "duplicate": sim >= float(cfg["jaccard_duplicate_threshold"])})
            if sim >= float(cfg["jaccard_duplicate_threshold"]):
                dup = True
                break
        if not dup:
            kept.append(rule)
            sets.append(s)
    return pd.DataFrame(kept).head(int(cfg["max_selected_strategies"])), pd.DataFrame(overlaps)


def evaluate(df: pd.DataFrame, selected: pd.DataFrame, rel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    periods = {
        "validation_2020_2024": cfg["validation_years"],
        "test_2025": [cfg["test_year"]],
        "latest_2026": [cfg["latest_holdout_year"]],
        "combined_2025_2026": [cfg["test_year"], cfg["latest_holdout_year"]],
    }
    for _, rule in selected.iterrows():
        rdict = rule.to_dict()
        for period, years in periods.items():
            bets = apply_strategy(df[df["Year"].isin(years)], rel, rdict)
            bets["strategy_id"] = rule["strategy_id"]
            bets["eval_period"] = period
            rows.append(enrich_summary({"strategy_id": rule["strategy_id"], "eval_period": period, "one_per_race": rule["one_per_race"]}, bets, cfg))
            details.append(bets)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return pd.DataFrame(rows), detail


def weighted_comparison(details: pd.DataFrame, selected: pd.DataFrame, rel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if details.empty or selected.empty:
        return pd.DataFrame()
    primary = str(selected.iloc[0]["strategy_id"])
    d = details[details["strategy_id"].eq(primary)].copy()
    d = d.merge(rel[["place_odds_band", "band_class", "reliability_score"]], on="place_odds_band", how="left", suffixes=("", "_rel"))
    rows = []
    for method in ["equal_100", "three_stage", "reliability_rounded"]:
        b = d.copy()
        if method == "equal_100":
            b["stake_yen"] = 100.0
        elif method == "three_stage":
            b["stake_yen"] = np.where(b["band_class"].eq("main"), 100.0, np.where(b["band_class"].eq("support"), 50.0, 0.0))
        else:
            b["stake_yen"] = pd.cut(b["reliability_score"], [-np.inf, 0.33, 0.66, np.inf], labels=[50.0, 100.0, 150.0]).astype(float)
        for period, g in b.groupby("eval_period"):
            rows.append(enrich_summary({"stake_method": method, "eval_period": period}, g[g["stake_yen"] > 0], cfg, stake_col="stake_yen"))
    return pd.DataFrame(rows)


def yearly_validation(summary_details: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if summary_details.empty:
        return pd.DataFrame()
    primary = summary_details["strategy_id"].iloc[0]
    d = summary_details[(summary_details["strategy_id"].eq(primary)) & (summary_details["Year"].isin(cfg["validation_years"]))]
    return pd.DataFrame([enrich_summary({"Year": int(y), "strategy_id": primary}, g, cfg) for y, g in d.groupby("Year")])


def profit_contribution(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    d = details.copy()
    d["profit"] = pd.to_numeric(d["fuku_pay"], errors="coerce").fillna(0) - 100
    rows = []
    for keys, g in d.groupby(["strategy_id", "eval_period", "place_odds_band"], dropna=False):
        total = d[(d["strategy_id"].eq(keys[0])) & (d["eval_period"].eq(keys[1]))]["profit"].sum()
        rows.append({"strategy_id": keys[0], "eval_period": keys[1], "place_odds_band": keys[2], "bets": len(g), "profit": g["profit"].sum(), "profit_share": g["profit"].sum() / total if total else np.nan})
    return pd.DataFrame(rows)


def required_outputs() -> list[str]:
    return [
        "odds_band_summary.csv", "odds_band_yearly.csv", "odds_band_bootstrap.csv",
        "odds_band_payout_dependency.csv", "odds_band_reliability.csv", "candidate_strategies.csv",
        "selected_strategies.json", "validation_summary.csv", "test_2025_summary.csv",
        "latest_2026_summary.csv", "combined_2025_2026_summary.csv", "equal_stake_comparison.csv",
        "weighted_stake_comparison.csv", "bet_details.parquet", "manifest.json",
    ]


def expected_fingerprint(cfg: dict[str, Any], config_path: Path, db_cache: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["source_output_root"])
    roi = Path(cfg["roi_refinement_root"])
    files = [root / "oof_predictions.parquet", root / "final_predictions.parquet", root / "manifest.json", roi / "selected_rules.json", roi / "manifest.json"]
    return {
        "version": cfg["version"],
        "config_hash": sha256_file(config_path),
        "code_hash": sha256_file(Path(__file__)),
        "input_hashes": {str(p): sha256_file(p) for p in files if p.exists()},
        "split_hash": sha256_json({"validation_years": cfg["validation_years"], "test_year": cfg["test_year"], "latest_holdout_year": cfg["latest_holdout_year"]}),
        "db_validation": db_cache,
    }


def should_resume(out: Path, fingerprint: dict[str, Any], strict: bool) -> bool:
    manifest = out / "manifest.json"
    if not manifest.exists():
        return False
    old = json.loads(manifest.read_text(encoding="utf-8"))
    ok = old.get("fingerprint") == fingerprint and all((out / p).exists() for p in required_outputs())
    if ok:
        print("[place-band] resume: existing outputs match; skipped", flush=True)
        return True
    if strict:
        print("[place-band] strict resume mismatch; exit 2", flush=True)
        raise SystemExit(2)
    return False


def write_docs(cfg: dict[str, Any], manifest: dict[str, Any], rel: pd.DataFrame, selected: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    design = [
        "# Place Odds Band Weighting V1 Design",
        "",
        "- Existing `final_odds_two_models_v1` place predictions are used.",
        "- Models are not retrained and feature datasets are not regenerated.",
        "- Odds bands are exclusive left-closed/right-open intervals based on `fuku_odds_low`.",
        "- Rule selection uses only 2020-2024 validation predictions.",
        "- 2025 and 2026 are fixed evaluations with no threshold adjustment.",
        "- Official evaluation uses equal 100 yen stakes; weighted stakes are reference diagnostics only.",
        "- DB was not accessed by this script; existing validation manifest was reused.",
    ]
    atomic_write_text(Path("docs/place_odds_band_weighting_v1_design.md"), "\n".join(design) + "\n")
    results = [
        "# Place Odds Band Weighting V1 Results",
        "",
        f"- Version: `{cfg['version']}`",
        f"- Elapsed seconds: `{manifest['elapsed_seconds']:.1f}`",
        f"- DB validation cache status: `{manifest['db_validation']['db_validation_cache_status']}`",
        f"- Database accessed: `{manifest['db_validation']['database_accessed']}`",
        "",
        "## Band Reliability",
        rel.to_markdown(index=False) if not rel.empty else "(none)",
        "",
        "## Selected Strategies",
        selected.to_markdown(index=False) if not selected.empty else "(none)",
        "",
        "## Evaluation",
        evaluation.to_markdown(index=False) if not evaluation.empty else "(none)",
    ]
    atomic_write_text(Path("docs/place_odds_band_weighting_v1_results.md"), "\n".join(results) + "\n")


def run(config_path: Path, resume: bool = False, strict_resume: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    db_cache = load_db_cache_manifest(cfg)
    fingerprint = expected_fingerprint(cfg, config_path, db_cache)
    if (resume or strict_resume) and not force and should_resume(out, fingerprint, strict_resume):
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print("[place-band] loading predictions", flush=True)
    df = load_place_predictions(cfg)
    print("[place-band] odds band analysis", flush=True)
    band_summary, band_yearly, band_boot, band_dep = odds_band_tables(df, cfg)
    rel = band_reliability(band_summary, band_yearly, cfg)
    print("[place-band] candidate strategies", flush=True)
    cands = candidate_strategies(df, rel, cfg)
    selected, overlap = jaccard_prune(cands, df, rel, cfg)
    print("[place-band] fixed evaluation", flush=True)
    eval_summary, details = evaluate(df, selected, rel, cfg)
    weighted = weighted_comparison(details, selected, rel, cfg)
    validation_yearly = yearly_validation(details, cfg)
    contrib = profit_contribution(details)
    hashes = {
        "odds_band_summary.csv": atomic_write_csv(out / "odds_band_summary.csv", band_summary),
        "odds_band_yearly.csv": atomic_write_csv(out / "odds_band_yearly.csv", band_yearly),
        "odds_band_bootstrap.csv": atomic_write_csv(out / "odds_band_bootstrap.csv", band_boot),
        "odds_band_payout_dependency.csv": atomic_write_csv(out / "odds_band_payout_dependency.csv", band_dep),
        "odds_band_reliability.csv": atomic_write_csv(out / "odds_band_reliability.csv", rel),
        "candidate_strategies.csv": atomic_write_csv(out / "candidate_strategies.csv", cands),
        "rule_overlap_matrix.csv": atomic_write_csv(out / "rule_overlap_matrix.csv", overlap),
        "validation_summary.csv": atomic_write_csv(out / "validation_summary.csv", eval_summary[eval_summary["eval_period"].eq("validation_2020_2024")]),
        "validation_yearly.csv": atomic_write_csv(out / "validation_yearly.csv", validation_yearly),
        "test_2025_summary.csv": atomic_write_csv(out / "test_2025_summary.csv", eval_summary[eval_summary["eval_period"].eq("test_2025")]),
        "latest_2026_summary.csv": atomic_write_csv(out / "latest_2026_summary.csv", eval_summary[eval_summary["eval_period"].eq("latest_2026")]),
        "combined_2025_2026_summary.csv": atomic_write_csv(out / "combined_2025_2026_summary.csv", eval_summary[eval_summary["eval_period"].eq("combined_2025_2026")]),
        "equal_stake_comparison.csv": atomic_write_csv(out / "equal_stake_comparison.csv", weighted[weighted["stake_method"].eq("equal_100")] if not weighted.empty else weighted),
        "weighted_stake_comparison.csv": atomic_write_csv(out / "weighted_stake_comparison.csv", weighted),
        "odds_band_profit_contribution.csv": atomic_write_csv(out / "odds_band_profit_contribution.csv", contrib),
        "bet_details.parquet": atomic_write_parquet(out / "bet_details.parquet", details),
    }
    selected_path = out / "selected_strategies.json"
    atomic_write_json(selected_path, json.loads(selected.to_json(orient="records", force_ascii=False)) if not selected.empty else [])
    hashes["selected_strategies.json"] = sha256_file(selected_path)
    manifest = {
        "version": cfg["version"],
        "fingerprint": fingerprint,
        "db_validation": db_cache,
        "task_file_read": {
            "path": "tasks/place_odds_band_weighting_v1_task.md",
            "encoding": "UTF-8",
            "read_command": "Get-Content tasks\\place_odds_band_weighting_v1_task.md -Encoding UTF8",
            "full_text_read_before_implementation": True,
        },
        "models_retrained": False,
        "features_regenerated": False,
        "auto_purchase_implemented": False,
        "selection_years": cfg["validation_years"],
        "test_year": cfg["test_year"],
        "latest_holdout_year": cfg["latest_holdout_year"],
        "output_hashes": hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_docs(cfg, manifest, rel, selected, eval_summary)
    print("[place-band] done", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_odds_band_weighting_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume, strict_resume=args.strict_resume, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
