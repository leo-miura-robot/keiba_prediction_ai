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


def atomic_write_json(path: Path, data: Any) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    atomic_write_text(path, text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_info() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception as exc:
        return {"git_commit_sha": "unknown", "git_is_dirty": None, "git_error": str(exc)}


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
    required = [
        "race_id", "entry_id", "race_date", "Year", "actual", "raw_probability",
        "calibrated_probability", "conservative_probability", "model_rank", "market_rank",
        "rank_gap", "fuku_odds_low", "fuku_odds_high", "fuku_pay",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df["entry_id"] = df["entry_id"].astype(str)
    df["race_id"] = df["race_id"].astype(str)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["actual_place"] = pd.to_numeric(df["actual"], errors="coerce").fillna(0).astype(int)
    df["predicted_place_probability"] = pd.to_numeric(df["raw_probability"], errors="coerce")
    df["fuku_odds_low"] = pd.to_numeric(df["fuku_odds_low"], errors="coerce")
    df["fuku_odds_high"] = pd.to_numeric(df["fuku_odds_high"], errors="coerce")
    df["fuku_pay"] = pd.to_numeric(df["fuku_pay"], errors="coerce").fillna(0)
    df["conservative_probability"] = pd.to_numeric(df["conservative_probability"], errors="coerce")
    df["break_even_probability"] = 1.0 / df["fuku_odds_low"]
    df["place_edge_low"] = df["conservative_probability"] - df["break_even_probability"]
    df["place_ev_low"] = df["conservative_probability"] * df["fuku_odds_low"]
    df = df[df["race_date"].dt.year.ge(2016)].copy()
    return df


def odds_ranges(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for low in cfg["odds_lower_candidates"]:
        for up in cfg["odds_upper_candidates"]:
            if up is not None and float(up) <= float(low):
                continue
            label = f"{float(low):.1f}+" if up is None else f"{float(low):.1f}-{float(up):.1f}"
            ranges.append({"odds_range": label, "odds_low_min": float(low), "odds_low_max": None if up is None else float(up)})
    return ranges


def mask_odds(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    m = df["fuku_odds_low"].ge(spec["odds_low_min"])
    if spec["odds_low_max"] is not None:
        m &= df["fuku_odds_low"].lt(spec["odds_low_max"])
    return m.fillna(False)


def apply_one_per_race(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty or method in ("none", None):
        return df.copy()
    sort_cols = ["race_id"]
    if method == "max_ev":
        d = df.sort_values(sort_cols + ["place_ev_low", "conservative_probability"], ascending=[True, False, False])
    else:
        d = df.sort_values(sort_cols + ["conservative_probability", "place_ev_low"], ascending=[True, False, False])
    return d.drop_duplicates("race_id", keep="first").copy()


def summarize_bets(bets: pd.DataFrame, label: dict[str, Any] | None = None) -> dict[str, Any]:
    label = dict(label or {})
    if bets.empty:
        return {**label, "bets": 0, "races": 0, "stake": 0.0, "return": 0.0, "profit": 0.0, "roi": np.nan, "hit_count": 0, "hit_rate": np.nan, "average_fuku_odds_low": np.nan, "average_fuku_pay": np.nan, "median_fuku_odds_low": np.nan, "median_fuku_pay": np.nan, "max_payout": np.nan, "max_losing_streak": 0, "max_drawdown": 0.0, "mean_predicted_place_probability": np.nan, "actual_place_rate": np.nan, "calibration_gap": np.nan}
    pay = bets["fuku_pay"].fillna(0).to_numpy(float)
    stake = np.full(len(bets), 100.0)
    returns = np.where(pay > 0, pay, 0.0)
    profit = returns - stake
    equity = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = peak - equity
    hits = pay > 0
    max_ls = 0
    cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        max_ls = max(max_ls, cur)
    pred = bets["conservative_probability"].astype(float)
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
        "average_fuku_odds_low": float(bets["fuku_odds_low"].mean()),
        "average_fuku_pay": float(pay.mean()),
        "median_fuku_odds_low": float(bets["fuku_odds_low"].median()),
        "median_fuku_pay": float(np.median(pay)),
        "max_payout": float(pay.max()),
        "max_losing_streak": int(max_ls),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "mean_predicted_place_probability": float(pred.mean()),
        "actual_place_rate": float(actual.mean()),
        "calibration_gap": float(actual.mean() - pred.mean()),
    }


def top_removed_roi(bets: pd.DataFrame, n: int) -> float:
    if bets.empty:
        return np.nan
    keep = bets["fuku_pay"].sort_values(ascending=False).index[n:] if n else bets.index
    sub = bets.loc[keep]
    return summarize_bets(sub)["roi"] if not sub.empty else np.nan


def bootstrap_ci(bets: pd.DataFrame, iterations: int, seed: int) -> tuple[float, float, float]:
    if bets.empty:
        return (np.nan, np.nan, np.nan)
    pay = bets["fuku_pay"].fillna(0).to_numpy(float)
    returns = np.where(pay > 0, pay, 0.0)
    stake = np.full(len(bets), 100.0)
    races = bets["race_id"].astype(str).to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    ret_by_race = np.bincount(inv, weights=returns)
    stake_by_race = np.bincount(inv, weights=stake)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(iterations, len(uniq)))
    roi = ret_by_race[idx].sum(axis=1) / stake_by_race[idx].sum(axis=1) * 100
    return tuple(float(x) for x in np.percentile(roi, [2.5, 50, 97.5]))


def enrich(row: dict[str, Any], bets: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    row = summarize_bets(bets, row)
    for n in [1, 3, 5, 10]:
        row[f"roi_remove_top{n}"] = top_removed_roi(bets, n)
    ci = bootstrap_ci(bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
    row["bootstrap_roi_p025"], row["bootstrap_roi_p500"], row["bootstrap_roi_p975"] = ci
    row["ev_ge_1_count"] = int(bets["place_ev_low"].ge(1.0).sum()) if "place_ev_low" in bets else 0
    row["ev_ge_1_rate"] = float(bets["place_ev_low"].ge(1.0).mean()) if len(bets) else np.nan
    return row


def yearly_counts(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    years = cfg["validation_years"] + [cfg["test_year"], cfg["latest_holdout_year"]]
    rows = []
    for year in years:
        y = df[df["Year"].eq(year)]
        days = max(1, int(y["race_date"].dt.date.nunique()))
        months = max(1, int(y["race_date"].dt.to_period("M").nunique()))
        for thr in [1.0, 1.05, 1.10]:
            t = y[y["place_ev_low"].ge(thr)]
            rows.append({"Year": year, "ev_threshold": thr, "horses": int(len(t)), "races": int(t["race_id"].nunique()), "avg_horses_per_year": float(len(t)), "avg_horses_per_month": float(len(t) / months), "avg_horses_per_race_day": float(len(t) / days)})
    return pd.DataFrame(rows)


def range_and_grid_tables(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val = df[df["Year"].isin(cfg["validation_years"])].copy()
    ranges = odds_ranges(cfg)
    range_rows, range_year_rows, grid_rows, grid_year_rows = [], [], [], []
    thresholds = [None] + list(cfg["ev_thresholds"])
    for spec in ranges:
        rbase = val[mask_odds(val, spec)]
        range_rows.append(enrich({**spec, "period": "validation_2020_2024"}, rbase, cfg))
        for year, y in rbase.groupby("Year"):
            range_year_rows.append(enrich({**spec, "Year": int(year)}, y, cfg))
        for thr in thresholds:
            g = rbase if thr is None else rbase[rbase["place_ev_low"].ge(float(thr))]
            label = "none" if thr is None else f"{thr:.2f}"
            grid_rows.append(enrich({**spec, "ev_threshold": label, "period": "validation_2020_2024"}, g, cfg))
            for year, y in g.groupby("Year"):
                grid_year_rows.append(enrich({**spec, "ev_threshold": label, "Year": int(year)}, y, cfg))
    return pd.DataFrame(range_rows), pd.DataFrame(range_year_rows), pd.DataFrame(grid_rows), pd.DataFrame(grid_year_rows)


def threshold_tables(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = df[df["Year"].isin(cfg["validation_years"])].copy()
    rows, yearly = [], []
    for thr in [None] + list(cfg["ev_thresholds"]):
        sub = val if thr is None else val[val["place_ev_low"].ge(float(thr))]
        label = "none" if thr is None else f"{thr:.2f}"
        rows.append(enrich({"ev_threshold": label, "period": "validation_2020_2024"}, sub, cfg))
        for year, y in sub.groupby("Year"):
            yearly.append(enrich({"ev_threshold": label, "Year": int(year)}, y, cfg))
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def ev_band_tables(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bins = [-np.inf, 0.85, 0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.15, 1.20, np.inf]
    labels = ["<0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00", "1.00-1.02", "1.02-1.05", "1.05-1.10", "1.10-1.15", "1.15-1.20", "1.20+"]
    d = df[df["Year"].isin(cfg["validation_years"])].copy()
    d["ev_band"] = pd.cut(d["place_ev_low"], bins=bins, labels=labels, right=False)
    rows, yearly = [], []
    for i, band in enumerate(labels):
        sub = d[d["ev_band"].astype(str).eq(band)]
        r = summarize_bets(sub, {"ev_band": band, "ev_band_order": i, "period": "validation_2020_2024"})
        rows.append(r)
        for year, y in sub.groupby("Year"):
            yearly.append(summarize_bets(y, {"ev_band": band, "ev_band_order": i, "Year": int(year)}))
    evs = pd.DataFrame(rows)
    valid = evs.dropna(subset=["roi"])
    corr = valid["ev_band_order"].corr(valid["roi"], method="spearman") if len(valid) > 1 else np.nan
    monotonic = bool(valid.sort_values("ev_band_order")["roi"].is_monotonic_increasing) if len(valid) > 1 else False
    return evs, pd.DataFrame(yearly), {"spearman_ev_band_roi": float(corr) if pd.notna(corr) else np.nan, "is_monotonic_increasing": monotonic}


def candidate_strategies(grid: pd.DataFrame, yearly: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in grid[grid["ev_threshold"].ne("none")].itertuples(index=False):
        key = {
            "odds_range": r.odds_range,
            "odds_low_min": r.odds_low_min,
            "odds_low_max": r.odds_low_max,
            "ev_threshold": r.ev_threshold,
        }
        yrs = yearly[(yearly["odds_range"].eq(r.odds_range)) & (yearly["ev_threshold"].eq(r.ev_threshold))]
        years_with_bets = int((yrs["bets"] > 0).sum()) if not yrs.empty else 0
        min_year_bets = int(yrs["bets"].min()) if not yrs.empty else 0
        min_year_roi = float(yrs["roi"].min()) if not yrs.empty else np.nan
        sample_ok = bool(r.bets >= cfg["min_total_bets"] and years_with_bets >= cfg["min_years_with_bets"] and min_year_bets >= cfg["min_yearly_bets"])
        if not sample_ok:
            continue
        score = (
            min_year_roi * 10000
            + r.roi_remove_top5 * 1000
            + r.bootstrap_roi_p025 * 100
            + r.roi * 10
            + min(float(r.bets), 5000) / 10
            - float(r.max_drawdown) / 1000
        )
        rows.append({**key, "strategy_id": f"evs{len(rows)+1:03d}", "bets": r.bets, "races": r.races, "roi": r.roi, "minimum_year_roi": min_year_roi, "minimum_year_bets": min_year_bets, "years_with_bets": years_with_bets, "roi_remove_top5": r.roi_remove_top5, "bootstrap_roi_p025": r.bootstrap_roi_p025, "max_drawdown": r.max_drawdown, "ev_ge_1_count": r.ev_ge_1_count, "score": score})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def strategy_mask(df: pd.DataFrame, row: pd.Series | dict[str, Any]) -> pd.Series:
    low = float(row["odds_low_min"])
    high = row.get("odds_low_max")
    m = df["fuku_odds_low"].ge(low)
    if high is not None and not pd.isna(high):
        m &= df["fuku_odds_low"].lt(float(high))
    m &= df["place_ev_low"].ge(float(row["ev_threshold"]))
    return m.fillna(False)


def jaccard_select(cands: pd.DataFrame, val: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = []
    overlap = []
    sets: list[set[str]] = []
    for _, row in cands.iterrows():
        entries = set(apply_one_per_race(val[strategy_mask(val, row)], cfg["one_per_race"])["entry_id"].astype(str))
        dup = False
        for i, prev in enumerate(sets):
            union = entries | prev
            jac = len(entries & prev) / len(union) if union else 1.0
            overlap.append({"strategy_id": row["strategy_id"], "kept_index": i, "jaccard": jac, "duplicate": jac >= cfg["jaccard_duplicate_threshold"]})
            if jac >= cfg["jaccard_duplicate_threshold"]:
                dup = True
                break
        if not dup:
            selected.append(row)
            sets.append(entries)
        if len(selected) >= cfg["max_selected_strategies"]:
            break
    return pd.DataFrame(selected), pd.DataFrame(overlap)


def eval_strategy_set(df: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, yearly, details, dep = [], [], [], []
    periods = [
        ("validation_2020_2024", cfg["validation_years"]),
        ("test_2025", [cfg["test_year"]]),
        ("latest_2026", [cfg["latest_holdout_year"]]),
        ("combined_2025_2026", [cfg["test_year"], cfg["latest_holdout_year"]]),
    ]
    for _, s in selected.iterrows():
        for period, years in periods:
            base = df[df["Year"].isin(years)]
            bets = apply_one_per_race(base[strategy_mask(base, s)], cfg["one_per_race"])
            label = {"strategy_id": s["strategy_id"], "eval_period": period, "odds_range": s["odds_range"], "ev_threshold": s["ev_threshold"], "one_per_race": cfg["one_per_race"]}
            rows.append(enrich(label, bets, cfg))
            if not bets.empty:
                d = bets.copy()
                for k, v in label.items():
                    d[k] = v
                details.append(d)
                for _, g in bets.groupby("Year"):
                    yearly.append(enrich({**label, "Year": int(g["Year"].iloc[0])}, g, cfg))
                for n in [1, 3, 5, 10]:
                    dep.append({**label, "removed_top_payouts": n, "roi": top_removed_roi(bets, n)})
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.concat(details, ignore_index=True) if details else pd.DataFrame(), pd.DataFrame(dep)


def comparison_strategies(df: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    specs = []
    current_path = Path(cfg["place_band_root"]) / "selected_strategies.json"
    if current_path.exists():
        current = json.loads(current_path.read_text(encoding="utf-8"))[0]
        specs.append({"comparison": "current_place_band_strategy", "odds_low_min": 1.0, "odds_low_max": 1.2, "ev_threshold": 0.90, "odds_range": "1.0-1.2"})
    for thr in cfg["comparison_ev_thresholds"]:
        specs.append({"comparison": f"all_odds_ev_ge_{thr:.2f}", "odds_low_min": 1.0, "odds_low_max": None, "ev_threshold": float(thr), "odds_range": "1.0+"})
    if not selected.empty:
        s = selected.iloc[0].to_dict()
        specs.append({"comparison": "best_odds_range_selected_ev", **{k: s[k] for k in ["odds_low_min", "odds_low_max", "ev_threshold", "odds_range"]}})
        s1 = dict(s)
        s1["ev_threshold"] = "1.00"
        specs.append({"comparison": "best_odds_range_ev_ge_1.00", **{k: s1[k] for k in ["odds_low_min", "odds_low_max", "ev_threshold", "odds_range"]}})
    rows = []
    for spec in specs:
        base = df[df["Year"].isin(cfg["validation_years"])]
        bets = apply_one_per_race(base[strategy_mask(base, spec)], cfg["one_per_race"])
        rows.append(enrich(spec, bets, cfg))
    return pd.DataFrame(rows)


def fingerprint(cfg: dict[str, Any], db_cache: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["source_output_root"])
    files = [root / "oof_predictions.parquet", root / "final_predictions.parquet", root / "manifest.json"]
    for p in [Path(cfg["roi_refinement_root"]) / "manifest.json", Path(cfg["place_band_root"]) / "manifest.json"]:
        if p.exists():
            files.append(p)
    return {
        "version": cfg["version"],
        "config_hash": sha256_json(cfg),
        "code_hash": sha256_file(Path(__file__)),
        "input_hashes": {str(p): sha256_file(p) for p in files if p.exists()},
        "db_validation": db_cache,
    }


def check_resume(cfg: dict[str, Any], fp: dict[str, Any], strict: bool) -> bool:
    manifest = Path(cfg["output_root"]) / "manifest.json"
    if not manifest.exists():
        return False
    old = json.loads(manifest.read_text(encoding="utf-8"))
    ok = old.get("fingerprint") == fp
    if strict and not ok:
        raise SystemExit("strict resume failed: fingerprint mismatch")
    return ok


def write_docs(cfg: dict[str, Any], manifest: dict[str, Any], selected: pd.DataFrame, eval_summary: pd.DataFrame, ev_diag: dict[str, Any]) -> None:
    docs = Path("docs")
    atomic_write_text(docs / "place_odds_ev_surface_v1_design.md", "\n".join([
        "# Place Odds EV Surface V1 Design",
        "",
        "- Existing `final_odds_two_models_v1` place predictions are used.",
        "- Models are not retrained and feature datasets are not regenerated.",
        "- Selection uses only 2020-2024; 2025/2026 are fixed holdout evaluations.",
        "- ROI uses actual `fuku_pay`, not estimated odds payout.",
        "- DB was not accessed; existing DB validation manifest was reused.",
        "- Odds ranges use lower-inclusive / upper-exclusive boundaries.",
    ]))
    lines = [
        "# Place Odds EV Surface V1 Results",
        "",
        f"- Version: `{cfg['version']}`",
        f"- Elapsed seconds: `{manifest['elapsed_seconds']:.1f}`",
        f"- DB cache status: `{manifest['db_validation']['db_validation_cache_status']}`",
        f"- EV band Spearman vs ROI: `{ev_diag['spearman_ev_band_roi']}`",
        f"- EV band monotonic increasing: `{ev_diag['is_monotonic_increasing']}`",
        "",
        "## Selected Strategies",
        selected.to_markdown(index=False) if not selected.empty else "No selected strategies.",
        "",
        "## Evaluation",
        eval_summary.to_markdown(index=False) if not eval_summary.empty else "No evaluation rows.",
    ]
    atomic_write_text(docs / "place_odds_ev_surface_v1_results.md", "\n".join(lines))


def run(config_path: Path, resume: bool = False, strict_resume: bool = False, force: bool = False) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    out = Path(cfg["output_root"])
    db_cache = load_db_cache_manifest(cfg)
    fp = fingerprint(cfg, db_cache)
    if resume and not force and check_resume(cfg, fp, strict_resume):
        print("[ev-surface] resume: existing outputs match; skipped", flush=True)
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print("[ev-surface] loading predictions", flush=True)
    df = load_place_predictions(cfg)
    print("[ev-surface] odds ranges and EV grid", flush=True)
    range_summary, range_yearly, grid, grid_yearly = range_and_grid_tables(df, cfg)
    ev_threshold_summary, ev_threshold_yearly = threshold_tables(df, cfg)
    ev_band_summary, ev_band_yearly, ev_diag = ev_band_tables(df, cfg)
    ev_counts = yearly_counts(df, cfg)
    print("[ev-surface] candidate selection", flush=True)
    cands = candidate_strategies(grid, grid_yearly, cfg)
    val = df[df["Year"].isin(cfg["validation_years"])]
    selected, overlap = jaccard_select(cands, val, cfg) if not cands.empty else (pd.DataFrame(), pd.DataFrame())
    print("[ev-surface] fixed evaluation", flush=True)
    eval_summary, eval_yearly, details, payout_dep = eval_strategy_set(df, selected, cfg)
    comparison = comparison_strategies(df, selected, cfg)
    bootstrap = eval_summary[["strategy_id", "eval_period", "bootstrap_roi_p025", "bootstrap_roi_p500", "bootstrap_roi_p975"]].copy() if not eval_summary.empty else pd.DataFrame()
    hashes = {
        "odds_range_summary.csv": atomic_write_csv(out / "odds_range_summary.csv", range_summary),
        "odds_range_yearly.csv": atomic_write_csv(out / "odds_range_yearly.csv", range_yearly),
        "ev_threshold_summary.csv": atomic_write_csv(out / "ev_threshold_summary.csv", ev_threshold_summary),
        "ev_threshold_yearly.csv": atomic_write_csv(out / "ev_threshold_yearly.csv", ev_threshold_yearly),
        "ev_band_summary.csv": atomic_write_csv(out / "ev_band_summary.csv", ev_band_summary),
        "ev_band_yearly.csv": atomic_write_csv(out / "ev_band_yearly.csv", ev_band_yearly),
        "odds_ev_grid.csv": atomic_write_csv(out / "odds_ev_grid.csv", grid),
        "odds_ev_yearly.csv": atomic_write_csv(out / "odds_ev_yearly.csv", grid_yearly),
        "candidate_strategies.csv": atomic_write_csv(out / "candidate_strategies.csv", cands),
        "validation_summary.csv": atomic_write_csv(out / "validation_summary.csv", eval_summary[eval_summary["eval_period"].eq("validation_2020_2024")] if not eval_summary.empty else eval_summary),
        "test_2025_summary.csv": atomic_write_csv(out / "test_2025_summary.csv", eval_summary[eval_summary["eval_period"].eq("test_2025")] if not eval_summary.empty else eval_summary),
        "latest_2026_summary.csv": atomic_write_csv(out / "latest_2026_summary.csv", eval_summary[eval_summary["eval_period"].eq("latest_2026")] if not eval_summary.empty else eval_summary),
        "combined_2025_2026_summary.csv": atomic_write_csv(out / "combined_2025_2026_summary.csv", eval_summary[eval_summary["eval_period"].eq("combined_2025_2026")] if not eval_summary.empty else eval_summary),
        "payout_dependency.csv": atomic_write_csv(out / "payout_dependency.csv", payout_dep),
        "bootstrap_ci.csv": atomic_write_csv(out / "bootstrap_ci.csv", bootstrap),
        "ev_yearly_counts.csv": atomic_write_csv(out / "ev_yearly_counts.csv", ev_counts),
        "comparison_summary.csv": atomic_write_csv(out / "comparison_summary.csv", comparison),
        "validation_yearly.csv": atomic_write_csv(out / "validation_yearly.csv", eval_yearly),
        "rule_overlap_matrix.csv": atomic_write_csv(out / "rule_overlap_matrix.csv", overlap),
        "bet_details.parquet": atomic_write_parquet(out / "bet_details.parquet", details),
    }
    atomic_write_json(out / "selected_strategies.json", json.loads(selected.to_json(orient="records", force_ascii=False)) if not selected.empty else [])
    hashes["selected_strategies.json"] = sha256_file(out / "selected_strategies.json")
    manifest = {
        "version": cfg["version"],
        "fingerprint": fp,
        "db_validation": db_cache,
        "task_file_read": {
            "path": "tasks/place_odds_ev_surface_v1_task.md",
            "encoding": "UTF-8",
            "read_command": "Get-Content tasks\\place_odds_ev_surface_v1_task.md -Encoding UTF8",
            "full_text_read_before_implementation": True,
        },
        "models_retrained": False,
        "features_regenerated": False,
        "auto_purchase_implemented": False,
        "single_win_strategy_added": False,
        "kelly_used": False,
        "selection_years": cfg["validation_years"],
        "test_year": cfg["test_year"],
        "latest_holdout_year": cfg["latest_holdout_year"],
        "ev_diagnostics": ev_diag,
        "candidate_strategy_count": int(len(cands)),
        "selected_strategy_count": int(len(selected)),
        "output_hashes": hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_docs(cfg, manifest, selected, eval_summary, ev_diag)
    print("[ev-surface] done", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_odds_ev_surface_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), resume=args.resume, strict_resume=args.strict_resume, force=args.force)


if __name__ == "__main__":
    main()
