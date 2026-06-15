from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_final_odds_two_models_v1 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    odds_col,
    payout_col,
    sha256_file,
    summarize_bets,
)
from src.database.db_validation_cache import DatabaseValidationError, DEFAULT_DB_PATH, db_validation_fingerprint, validate_or_require_full


TARGETS = ("win", "place")
EVAL_PERIODS = ("validation_2020_2024", "test_2025", "latest_holdout_2026", "test_latest_combined")


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


def sha256_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def load_source_predictions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg["source_output_root"])
    oof = pd.read_parquet(root / "oof_predictions.parquet")
    final = pd.read_parquet(root / "final_predictions.parquet")
    oof = oof.copy()
    oof["eval_period"] = "validation_2020_2024"
    final = final.copy()
    needed = {
        "entry_id", "race_id", "race_date", "Year", "month", "JyoCD", "TrackCD", "Kyori", "SyussoTosu",
        "Ninki", "TanNinki", "FukuNinki", "tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_pay",
        "fuku_pay", "distance_band", "field_size_band", "actual", "raw_probability", "target",
        "calibrated_probability", "normalized_market_probability", "model_rank", "market_rank", "rank_gap",
        "conservative_probability", "edge", "ev", "top1_probability", "top1_minus_top2_margin",
        "prediction_entropy", "top3_probability_sum", "eval_period",
    }
    missing = needed - set(oof.columns) | (needed - set(final.columns))
    if missing:
        raise RuntimeError(f"source predictions missing columns: {sorted(missing)}")
    all_pred = pd.concat([oof[list(needed)], final[list(needed)]], ignore_index=True)
    all_pred["race_date"] = pd.to_datetime(all_pred["race_date"])
    all_pred["entry_id"] = all_pred["entry_id"].astype(str)
    all_pred["race_id"] = all_pred["race_id"].astype(str)
    combined = final.copy()
    combined["entry_id"] = combined["entry_id"].astype(str)
    combined["race_id"] = combined["race_id"].astype(str)
    combined["eval_period"] = "test_latest_combined"
    combined["race_date"] = pd.to_datetime(combined["race_date"])
    all_pred = pd.concat([all_pred, combined[list(needed)]], ignore_index=True)
    baseline = pd.read_parquet(root / "bet_details.parquet")
    baseline["entry_id"] = baseline["entry_id"].astype(str)
    baseline["race_id"] = baseline["race_id"].astype(str)
    baseline["race_date"] = pd.to_datetime(baseline["race_date"])
    return all_pred, baseline


def load_feature_meta(cfg: dict[str, Any]) -> pd.DataFrame:
    cols = [
        "entry_id", "GradeCD", "SyubetuCD", "JyokenCD1", "JyokenCD2", "JyokenCD3", "JyokenCD4", "JyokenCD5",
        "JyokenName", "TorokuTosu", "SexCD", "Barei",
    ]
    frames = []
    base = Path(cfg["feature_dataset_dir"])
    for year in sorted(set(cfg["validation_years"] + [cfg["test_year"], cfg["latest_holdout_year"]])):
        p = base / f"year={year}" / "data.parquet"
        if not p.exists():
            continue
        available = pd.read_parquet(p, columns=None).columns
        use_cols = [c for c in cols if c in available]
        if "entry_id" not in use_cols:
            continue
        frames.append(pd.read_parquet(p, columns=use_cols))
    if not frames:
        return pd.DataFrame({"entry_id": []})
    meta = pd.concat(frames, ignore_index=True)
    meta["entry_id"] = meta["entry_id"].astype(str)
    return meta.drop_duplicates("entry_id")


def add_segments(df: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.copy()
    if meta is not None and not meta.empty:
        d = d.merge(meta, on="entry_id", how="left")
    track = pd.to_numeric(d["TrackCD"], errors="coerce")
    d["surface"] = np.select(
        [track.between(10, 22), track.between(23, 29), track.ge(50)],
        ["turf", "dirt", "jump"],
        default="other",
    )
    kyori = pd.to_numeric(d["Kyori"], errors="coerce")
    d["distance_group"] = pd.cut(
        kyori,
        [0, 1200, 1400, 1600, 1800, 2000, 2400, 10000],
        labels=["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"],
        include_lowest=True,
    ).astype(str)
    if "GradeCD" in d.columns:
        grade = d["GradeCD"].astype(str).str.strip()
        d["class_group"] = np.select(
            [grade.isin(["A", "B", "C"]), grade.eq("D"), grade.eq("E"), grade.eq("F"), grade.eq("G"), grade.eq("H")],
            ["grade", "open", "3wins", "2wins", "1win", "maiden"],
            default="other",
        )
    else:
        d["class_group"] = "unknown"
    jyoken = d["JyokenName"].astype(str) if "JyokenName" in d.columns else pd.Series("", index=d.index)
    d["handicap_flag"] = np.where(jyoken.str.contains("ハン|handicap|ハンデ", case=False, regex=True, na=False), "handicap", "non_handicap_or_unknown")
    d["month"] = pd.to_numeric(d["month"], errors="coerce").astype("Int64").astype(str)
    d["JyoCD"] = d["JyoCD"].astype(str)
    return d


def top_payout_removed_roi(bets: pd.DataFrame, target: str, n: int) -> float:
    if bets.empty:
        return np.nan
    pay = pd.to_numeric(bets[payout_col(target)], errors="coerce").fillna(0)
    if n > 0:
        keep = pay.sort_values(ascending=False).index[n:]
        bets = bets.loc[keep]
        pay = pay.loc[keep]
    if bets.empty:
        return np.nan
    return float(pay.sum() / (len(bets) * 100) * 100)


def profit_share(bets: pd.DataFrame, target: str, q: float) -> float:
    if bets.empty:
        return np.nan
    pay = pd.to_numeric(bets[payout_col(target)], errors="coerce").fillna(0).to_numpy(float)
    profit = pay - 100
    pos = np.sort(profit[profit > 0])[::-1]
    total = pos.sum()
    if total <= 0:
        return np.nan
    n = max(1, int(np.ceil(len(pos) * q)))
    return float(pos[:n].sum() / total)


def bootstrap_ci(bets: pd.DataFrame, target: str, iterations: int, seed: int) -> tuple[float, float, float]:
    if bets.empty:
        return (np.nan, np.nan, np.nan)
    pay = pd.to_numeric(bets[payout_col(target)], errors="coerce").fillna(0).to_numpy(float)
    races = bets["race_id"].astype(str).to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    returns = np.bincount(inv, weights=pay)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(iterations, len(uniq)))
    roi = returns[idx].sum(axis=1) / (counts[idx].sum(axis=1) * 100) * 100
    return tuple(float(x) for x in np.percentile(roi, [2.5, 50.0, 97.5]))


def enrich_summary(row: dict[str, Any], bets: pd.DataFrame, target: str, cfg: dict[str, Any]) -> dict[str, Any]:
    by_year = pd.DataFrame([summarize_bets(g, target, {"Year": int(y)}) for y, g in bets.groupby("Year")])
    row.update(summarize_bets(bets, target))
    row["years_with_bets"] = int(by_year["Year"].nunique()) if not by_year.empty else 0
    row["min_yearly_bets"] = int(by_year["bets"].min()) if not by_year.empty else 0
    row["year_roi_min"] = float(by_year["roi"].min()) if not by_year.empty else np.nan
    row["year_roi_mean"] = float(by_year["roi"].mean()) if not by_year.empty else np.nan
    row["year_roi_std"] = float(by_year["roi"].std(ddof=0)) if len(by_year) else np.nan
    row["roi_remove_top1"] = top_payout_removed_roi(bets, target, 1)
    row["roi_remove_top3"] = top_payout_removed_roi(bets, target, 3)
    row["roi_remove_top5"] = top_payout_removed_roi(bets, target, 5)
    row["roi_remove_top10"] = top_payout_removed_roi(bets, target, 10)
    row["top1_profit_share"] = profit_share(bets, target, 0.01)
    row["top5_profit_share"] = profit_share(bets, target, 0.05)
    ci = bootstrap_ci(bets, target, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
    row["bootstrap_roi_p025"], row["bootstrap_roi_p500"], row["bootstrap_roi_p975"] = ci
    return row


def summarize_group(df: pd.DataFrame, keys: list[str], cfg: dict[str, Any], label: dict[str, Any] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for name, g in df.groupby(keys, dropna=False):
        vals = name if isinstance(name, tuple) else (name,)
        target = str(g["target"].iloc[0])
        row = dict(label or {})
        row.update(dict(zip(keys, vals)))
        rows.append(enrich_summary(row, g, target, cfg))
    return pd.DataFrame(rows)


def add_place_edge_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fuku_low = pd.to_numeric(d["fuku_odds_low"], errors="coerce")
    d["break_even_probability"] = 1.0 / fuku_low
    d["place_edge_low"] = pd.to_numeric(d["conservative_probability"], errors="coerce") - d["break_even_probability"]
    d["place_ev_low"] = pd.to_numeric(d["conservative_probability"], errors="coerce") * fuku_low
    d["place_odds_band"] = pd.cut(
        fuku_low,
        [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 999.0],
        labels=["1.0-1.1", "1.1-1.2", "1.2-1.3", "1.3-1.5", "1.5-2.0", "2.0-3.0", "3.0+"],
        include_lowest=True,
        right=False,
    ).astype(str)
    return d


def place_low_odds_analysis(pred: pd.DataFrame, baseline: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for scope, df in [("all_predictions", pred[pred["target"] == "place"]), ("baseline_bets", baseline[baseline["target"] == "place"])]:
        d = add_place_edge_cols(df)
        grouped = summarize_group(d, ["eval_period", "place_odds_band"], cfg, {"scope": scope})
        rows.append(grouped)
        pop = pd.cut(pd.to_numeric(d["FukuNinki"], errors="coerce"), [0, 1, 3, 6, 999], labels=["1", "2-3", "4-6", "7+"], include_lowest=True).astype(str)
        d = d.assign(popularity_band=pop)
        rows.append(summarize_group(d, ["eval_period", "popularity_band"], cfg, {"scope": scope}))
    return pd.concat(rows, ignore_index=True)


def race_segment_analysis(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_cols = [c for c in cfg["segment_columns"] if c in pred.columns]
    rows = []
    yearly = []
    for target in TARGETS:
        d = pred[(pred["target"] == target) & (pred["eval_period"] == "validation_2020_2024")]
        for c in segment_cols:
            rows.append(summarize_group(d, [c], cfg, {"target": target, "segment": c, "axis_count": 1}))
            yearly.append(summarize_group(d, [c, "Year"], cfg, {"target": target, "segment": c, "axis_count": 1}))
        for c1, c2 in combinations(segment_cols, 2):
            s = summarize_group(d, [c1, c2], cfg, {"target": target, "segment": f"{c1}+{c2}", "axis_count": 2})
            if not s.empty:
                rows.append(s)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    yout = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame()
    if not out.empty:
        out["sample_filter_pass"] = (
            (out["bets"] >= int(cfg["min_total_bets"]))
            & (out["years_with_bets"] >= int(cfg["min_years_with_bets"]))
            & (out["min_yearly_bets"] >= int(cfg["min_yearly_bets"]))
        )
    return out, yout


def rule_mask(df: pd.DataFrame, rule: pd.Series) -> pd.Series:
    target = str(rule["target"])
    oc = odds_col(target)
    d = df[df["target"] == target]
    mask = pd.Series(True, index=d.index)
    if pd.notna(rule.get("strategy_type")):
        st = str(rule["strategy_type"])
        if st == "win_core":
            mask &= d["model_rank"].eq(1)
        elif st == "win_value":
            mask &= d["model_rank"].between(2, 3) & (d["market_rank"] >= 4)
        elif st == "win_rank_reversal":
            mask &= d["model_rank"].le(3) & (d["rank_gap"] >= 2)
        elif st == "win_longshot":
            mask &= d["model_rank"].le(3) & (pd.to_numeric(d["tan_odds"], errors="coerce") >= 20)
        elif st == "place_core":
            mask &= d["model_rank"].eq(1)
    for col, default in [
        ("model_rank_max", np.inf), ("market_rank_min", -np.inf), ("rank_gap_min", -np.inf),
        ("edge_min", -np.inf), ("ev_min", -np.inf), ("odds_min", -np.inf), ("odds_max", np.inf),
        ("margin_min", -np.inf), ("entropy_max", np.inf), ("place_edge_low_min", -np.inf), ("place_ev_low_min", -np.inf),
    ]:
        if col not in rule or pd.isna(rule[col]):
            continue
        val = float(rule[col])
        if col == "model_rank_max":
            mask &= d["model_rank"] <= val
        elif col == "market_rank_min":
            mask &= d["market_rank"] >= val
        elif col == "rank_gap_min":
            mask &= d["rank_gap"] >= val
        elif col == "edge_min":
            mask &= d["edge"] >= val
        elif col == "ev_min":
            mask &= d["ev"] >= val
        elif col == "odds_min":
            mask &= pd.to_numeric(d[oc], errors="coerce") >= val
        elif col == "odds_max":
            mask &= pd.to_numeric(d[oc], errors="coerce") < val
        elif col == "margin_min":
            mask &= d["top1_minus_top2_margin"] >= val
        elif col == "entropy_max":
            mask &= d["prediction_entropy"] <= val
        elif col == "place_edge_low_min":
            mask &= d["place_edge_low"] >= val
        elif col == "place_ev_low_min":
            mask &= d["place_ev_low"] >= val
    if rule.get("dynamic_low_odds_edge", False) is True:
        fuku_low = pd.to_numeric(d["fuku_odds_low"], errors="coerce")
        dyn_edge = np.select([fuku_low < 1.3, fuku_low < 1.8], [0.05, 0.03], default=0.02)
        mask &= d["place_edge_low"] >= dyn_edge
    for c in ["surface", "distance_group", "class_group", "field_size_band", "JyoCD", "month", "handicap_flag"]:
        rc = f"segment_{c}"
        if rc in rule and pd.notna(rule[rc]) and str(rule[rc]) != "":
            mask &= d[c].astype(str).eq(str(rule[rc]))
    return mask.reindex(df.index, fill_value=False)


def apply_rule(df: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    return df[rule_mask(df, rule)].copy()


def candidate_score(row: dict[str, Any]) -> float:
    return (
        np.nan_to_num(row.get("year_roi_min"), nan=0.0) * 10000
        + np.nan_to_num(row.get("roi_remove_top5"), nan=0.0) * 2500
        + np.nan_to_num(row.get("bootstrap_roi_p025"), nan=0.0) * 1500
        + min(float(row.get("bets", 0)), 5000.0)
        + np.nan_to_num(row.get("roi"), nan=0.0) * 100
        - np.nan_to_num(row.get("max_drawdown"), nan=0.0) * 0.01
    )


def build_rule_candidates(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pred[pred["eval_period"] == "validation_2020_2024"].copy()
    validation = add_place_edge_cols(validation)
    place_rows = []
    place = validation[validation["target"] == "place"]
    for edge_min in cfg["place_edge_mins"]:
        for ev_min in cfg["place_ev_mins"]:
            for mr in [1, 2, 3]:
                for omin, omax in [(1.0, 1.5), (1.0, 2.0), (1.0, 3.0), (1.0, 999.0), (1.1, 2.0), (1.2, 2.0), (1.3, 3.0), (1.5, 999.0)]:
                    rule = {
                        "target": "place", "strategy_type": "place_core", "model_rank_max": mr,
                        "place_edge_low_min": edge_min, "place_ev_low_min": ev_min,
                        "edge_min": np.nan, "ev_min": np.nan, "odds_min": omin, "odds_max": omax,
                        "market_rank_min": np.nan, "rank_gap_min": np.nan, "margin_min": 0.0, "entropy_max": 2.0,
                        "dynamic_low_odds_edge": False,
                    }
                    bets = apply_rule(place, pd.Series(rule))
                    if len(bets) == 0:
                        continue
                    row = enrich_summary(rule.copy(), bets, "place", cfg)
                    row["score"] = candidate_score(row)
                    place_rows.append(row)
    low_policy = {
        "target": "place", "strategy_type": "place_core", "model_rank_max": 3,
        "place_edge_low_min": np.nan, "place_ev_low_min": 0.8,
        "edge_min": np.nan, "ev_min": np.nan, "odds_min": 1.0, "odds_max": 999.0,
        "market_rank_min": np.nan, "rank_gap_min": np.nan, "margin_min": 0.0, "entropy_max": 2.0,
        "dynamic_low_odds_edge": True,
    }
    p = place.copy()
    dyn_edge = np.select(
        [pd.to_numeric(p["fuku_odds_low"], errors="coerce") < 1.3, pd.to_numeric(p["fuku_odds_low"], errors="coerce") < 1.8],
        [0.05, 0.03],
        default=0.02,
    )
    bets = p[(p["model_rank"] <= 3) & (p["place_edge_low"] >= dyn_edge) & (p["place_ev_low"] >= 0.8)]
    if len(bets):
        row = enrich_summary(low_policy.copy(), bets, "place", cfg)
        row["score"] = candidate_score(row)
        place_rows.append(row)

    win_rows = []
    win = validation[validation["target"] == "win"]
    strategies = [
        ("win_core", [1], [-999], [1]),
        ("win_value", [3], [1, 2, 3], [4, 5, 7]),
        ("win_rank_reversal", [3], [2, 3, 5], [1]),
        ("win_longshot", [3], [2, 3, 5], [1]),
    ]
    for st, mranks, gaps, market_mins in strategies:
        for model_rank_max in mranks:
            for gap in gaps:
                for market_min in market_mins:
                    for edge_min in cfg["win_edge_mins"]:
                        for ev_min in cfg["win_ev_mins"]:
                            for omin, omax in cfg["win_odds_bands"]:
                                if st == "win_longshot" and omin < 20:
                                    continue
                                rule = {
                                    "target": "win", "strategy_type": st, "model_rank_max": model_rank_max,
                                    "market_rank_min": market_min, "rank_gap_min": gap, "edge_min": edge_min,
                                    "ev_min": ev_min, "odds_min": omin, "odds_max": omax,
                                    "margin_min": 0.0, "entropy_max": 2.0,
                                    "place_edge_low_min": np.nan, "place_ev_low_min": np.nan,
                                }
                                bets = apply_rule(win, pd.Series(rule))
                                if len(bets) == 0:
                                    continue
                                row = enrich_summary(rule.copy(), bets, "win", cfg)
                                row["score"] = candidate_score(row)
                                win_rows.append(row)
    win_df = pd.DataFrame(win_rows)
    place_df = pd.DataFrame(place_rows)
    win_core = win_df[win_df["strategy_type"] == "win_core"].copy() if not win_df.empty else pd.DataFrame()
    win_value = win_df[win_df["strategy_type"] == "win_value"].copy() if not win_df.empty else pd.DataFrame()
    win_reversal = win_df[win_df["strategy_type"] == "win_rank_reversal"].copy() if not win_df.empty else pd.DataFrame()
    if not win_df.empty:
        win_df = filter_candidates(win_df, cfg, "win")
    if not place_df.empty:
        place_df = filter_candidates(place_df, cfg, "place")
    return win_df, place_df, win_core, pd.concat([win_value, win_reversal], ignore_index=True)


def filter_candidates(df: pd.DataFrame, cfg: dict[str, Any], target: str) -> pd.DataFrame:
    d = df.copy()
    d["sample_filter_pass"] = (
        (d["bets"] >= int(cfg["min_total_bets"]))
        & (d["years_with_bets"] >= int(cfg["min_years_with_bets"]))
        & (d["min_yearly_bets"] >= int(cfg["min_yearly_bets"]))
    )
    filtered = d[d["sample_filter_pass"]].copy()
    if filtered.empty:
        return d.sort_values("score", ascending=False).head(0)
    return filtered.sort_values("score", ascending=False).head(int(cfg["max_candidates_per_target"]))


def remove_similar_rules(rules: pd.DataFrame, validation: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = []
    sets: list[set[str]] = []
    overlap_rows = []
    threshold = float(cfg["jaccard_duplicate_threshold"])
    for idx, rule in rules.reset_index(drop=True).iterrows():
        s = set(apply_rule(validation, rule)["entry_id"].astype(str))
        duplicate = False
        for kept_idx, ks in enumerate(sets):
            denom = len(s | ks)
            sim = len(s & ks) / denom if denom else 0.0
            overlap_rows.append({"candidate_index": idx, "kept_index": kept_idx, "jaccard": sim, "duplicate": sim >= threshold})
            if sim >= threshold:
                duplicate = True
                break
        if not duplicate:
            row = rule.copy()
            row["selected_order"] = len(kept) + 1
            kept.append(row)
            sets.append(s)
    return pd.DataFrame(kept), pd.DataFrame(overlap_rows)


def select_rules(win_candidates: pd.DataFrame, place_candidates: pd.DataFrame, pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = add_place_edge_cols(pred[pred["eval_period"] == "validation_2020_2024"].copy())
    selected_all = []
    overlap_all = []
    for target, cands in [("win", win_candidates), ("place", place_candidates)]:
        if cands.empty:
            continue
        selected, overlap = remove_similar_rules(cands, validation[validation["target"] == target], cfg)
        selected = selected.head(int(cfg["max_selected_rules_per_target"]))
        selected_all.append(selected)
        if not overlap.empty:
            overlap["target"] = target
            overlap_all.append(overlap)
    return (
        pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame(),
        pd.concat(overlap_all, ignore_index=True) if overlap_all else pd.DataFrame(),
    )


def evaluate_rules(pred: pd.DataFrame, rules: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = add_place_edge_cols(pred)
    summaries = []
    details = []
    dependency = []
    drawdown = []
    boot = []
    for ridx, rule in rules.reset_index(drop=True).iterrows():
        target = str(rule["target"])
        rule_id = f"{target}_{rule['strategy_type']}_{ridx + 1:02d}"
        for period in EVAL_PERIODS:
            d = pred[(pred["target"] == target) & (pred["eval_period"] == period)]
            bets = apply_rule(d, rule)
            if target == "win" and str(rule["strategy_type"]) == "win_longshot" and len(bets):
                base_count = len(details[-1]) if details else len(d)
                max_bets = max(1, int(base_count * float(cfg["longshot_max_share"])))
                bets = bets.sort_values("ev", ascending=False).head(max_bets)
            bets = bets.copy()
            bets["rule_id"] = rule_id
            summaries.append(enrich_summary({"rule_id": rule_id, "target": target, "strategy_type": rule["strategy_type"], "eval_period": period}, bets, target, cfg))
            details.append(bets)
            for n in [0, 1, 3, 5, 10]:
                dependency.append({"rule_id": rule_id, "target": target, "eval_period": period, "removed_top_payouts": n, "roi": top_payout_removed_roi(bets, target, n)})
            dependency.append({"rule_id": rule_id, "target": target, "eval_period": period, "removed_top_payouts": "profit_top_share", "top1_profit_share": profit_share(bets, target, 0.01), "top5_profit_share": profit_share(bets, target, 0.05)})
            boot_ci = bootstrap_ci(bets, target, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
            boot.append({"rule_id": rule_id, "target": target, "eval_period": period, "roi_p025": boot_ci[0], "roi_p500": boot_ci[1], "roi_p975": boot_ci[2]})
            drawdown.append({"rule_id": rule_id, "target": target, "eval_period": period, "max_drawdown": summarize_bets(bets, target)["max_drawdown"], "max_losing_streak": summarize_bets(bets, target)["max_losing_streak"]})
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return pd.DataFrame(summaries), detail_df, pd.DataFrame(dependency), pd.DataFrame(drawdown), pd.DataFrame(boot)


def baseline_summary(baseline: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    return summarize_group(baseline, ["target", "eval_period", "rule_id"], cfg)


def write_docs(cfg: dict[str, Any], manifest: dict[str, Any], selected: pd.DataFrame, summaries: pd.DataFrame, place_low: pd.DataFrame, segments: pd.DataFrame) -> None:
    design = [
        "# ROI Strategy Refinement V1 Design",
        "",
        "- Input predictions: `outputs/final_odds_two_models_v1/oof_predictions.parquet` and `final_predictions.parquet`.",
        "- Models are not retrained; this task only analyzes existing predictions and refines betting rules.",
        "- Rule design and selection use only 2020-2024 walk-forward OOF predictions.",
        "- 2025 test and 2026 latest_holdout are fixed evaluation periods; thresholds are not changed there.",
        "- Place EV uses `conservative_place_probability * fuku_odds_low`; ROI always uses actual `fuku_pay`.",
        "- Win EV uses existing conservative probability and actual `tan_pay` for ROI.",
        "- Final-odds models are ideal-condition models and are not pre-race live operation models.",
    ]
    atomic_write_text(Path("docs/roi_strategy_refinement_v1_design.md"), "\n".join(design) + "\n")
    result = [
        "# ROI Strategy Refinement V1 Results",
        "",
        f"- Version: `{cfg['version']}`",
        f"- Elapsed seconds: `{manifest['elapsed_seconds']:.1f}`",
        f"- Source output root: `{cfg['source_output_root']}`",
        "",
        "## Selected Rules",
        selected.to_markdown(index=False) if not selected.empty else "(none)",
        "",
        "## Rule Evaluation",
        summaries.to_markdown(index=False) if not summaries.empty else "(none)",
        "",
        "## Place Low Odds Analysis",
        place_low.head(int(cfg["report_top_rows"])).to_markdown(index=False) if not place_low.empty else "(none)",
        "",
        "## Useful Segments",
        segments.sort_values("score", ascending=False).head(int(cfg["report_top_rows"])).to_markdown(index=False) if "score" in segments.columns and not segments.empty else "(none)",
    ]
    atomic_write_text(Path("docs/roi_strategy_refinement_v1_results.md"), "\n".join(result) + "\n")


def manifest_fingerprint(cfg: dict[str, Any], config_path: Path, db_validation: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(cfg["source_output_root"])
    files = ["oof_predictions.parquet", "final_predictions.parquet", "bet_details.parquet", "manifest.json"]
    return {
        "config_hash": sha256_json(cfg),
        "config_file_sha256": sha256_file(config_path),
        "code_sha256": sha256_file(Path(__file__)),
        "source_hashes": {f: sha256_file(root / f) for f in files if (root / f).exists()},
        "db_validation": db_validation,
    }


def required_outputs() -> list[str]:
    return [
        "baseline_summary.csv",
        "place_low_odds_analysis.csv",
        "place_edge_analysis.csv",
        "race_segment_summary.csv",
        "race_segment_yearly.csv",
        "win_core_analysis.csv",
        "win_value_analysis.csv",
        "win_rank_reversal_analysis.csv",
        "rule_candidates_win.csv",
        "rule_candidates_place.csv",
        "rule_overlap_matrix.csv",
        "selected_rules.json",
        "validation_rule_summary.csv",
        "test_2025_summary.csv",
        "latest_2026_summary.csv",
        "combined_2025_2026_summary.csv",
        "payout_dependency.csv",
        "drawdown_summary.csv",
        "bootstrap_ci.csv",
        "bet_details.parquet",
        "manifest.json",
    ]


def should_resume(out: Path, fingerprint: dict[str, Any], strict: bool) -> bool:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return False
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs_ok = all((out / name).exists() for name in required_outputs())
    if old.get("fingerprint") == fingerprint and outputs_ok:
        print("[roi-refine] resume: existing outputs match; skipped", flush=True)
        return True
    if strict:
        print("[roi-refine] strict resume mismatch; exit 2", flush=True)
        raise SystemExit(2)
    return False


def run(
    config_path: Path,
    resume: bool = False,
    strict_resume: bool = False,
    force: bool = False,
    db_validation_config: Path | str = "config/database_validation.yaml",
    force_integrity_check: bool = False,
    skip_db_validation: bool = False,
) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    db_path = Path(cfg.get("source_db_path", DEFAULT_DB_PATH))
    try:
        db_validation = validate_or_require_full(
            db_path,
            db_validation_config,
            force_integrity_check=force_integrity_check,
            skip=skip_db_validation,
        )
        if not skip_db_validation:
            db_validation = db_validation_fingerprint(db_path, db_validation_config)
    except DatabaseValidationError as exc:
        print(f"[roi-refine] DB validation failed: {exc}", flush=True)
        raise SystemExit(2)
    fingerprint = manifest_fingerprint(cfg, config_path, db_validation)
    if (resume or strict_resume) and not force and should_resume(out, fingerprint, strict_resume):
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    print("[roi-refine] preflight/load source predictions", flush=True)
    pred, baseline = load_source_predictions(cfg)
    meta = load_feature_meta(cfg)
    pred = add_segments(pred, meta)
    baseline = add_segments(baseline, meta)
    print("[roi-refine] baseline and low-odds analysis", flush=True)
    base_summary = baseline_summary(baseline, cfg)
    place_low = place_low_odds_analysis(pred, baseline, cfg)
    print("[roi-refine] race segment analysis", flush=True)
    segment_summary, segment_yearly = race_segment_analysis(pred, cfg)
    if not segment_summary.empty:
        segment_summary["score"] = segment_summary.apply(lambda r: candidate_score(r.to_dict()), axis=1)
    print("[roi-refine] candidate rule generation", flush=True)
    win_cands, place_cands, win_core, win_value = build_rule_candidates(pred, cfg)
    all_cands = pd.concat([win_cands, place_cands], ignore_index=True) if not win_cands.empty or not place_cands.empty else pd.DataFrame()
    print("[roi-refine] jaccard pruning and fixed evaluation", flush=True)
    selected, overlap = select_rules(win_cands, place_cands, pred, cfg)
    summaries, details, dependency, drawdown, boot = evaluate_rules(pred, selected, cfg)
    test_2025 = summaries[summaries["eval_period"] == "test_2025"].copy()
    latest_2026 = summaries[summaries["eval_period"] == "latest_holdout_2026"].copy()
    combined = summaries[summaries["eval_period"] == "test_latest_combined"].copy()
    outputs = {
        "baseline_summary.csv": base_summary,
        "place_low_odds_analysis.csv": place_low,
        "place_edge_analysis.csv": place_cands,
        "race_segment_summary.csv": segment_summary,
        "race_segment_yearly.csv": segment_yearly,
        "win_core_analysis.csv": win_core,
        "win_value_analysis.csv": win_value[win_value.get("strategy_type", pd.Series(dtype=str)).eq("win_value")] if not win_value.empty else win_value,
        "win_rank_reversal_analysis.csv": win_value[win_value.get("strategy_type", pd.Series(dtype=str)).eq("win_rank_reversal")] if not win_value.empty else win_value,
        "rule_candidates_win.csv": win_cands,
        "rule_candidates_place.csv": place_cands,
        "rule_overlap_matrix.csv": overlap,
        "validation_rule_summary.csv": summaries[summaries["eval_period"] == "validation_2020_2024"],
        "test_2025_summary.csv": test_2025,
        "latest_2026_summary.csv": latest_2026,
        "combined_2025_2026_summary.csv": combined,
        "payout_dependency.csv": dependency,
        "drawdown_summary.csv": drawdown,
        "bootstrap_ci.csv": boot,
    }
    hashes: dict[str, str] = {}
    for name, df in outputs.items():
        hashes[name] = atomic_write_csv(out / name, df)
    hashes["bet_details.parquet"] = atomic_write_parquet(out / "bet_details.parquet", details)
    selected_path = out / "selected_rules.json"
    atomic_write_json(selected_path, json.loads(selected.to_json(orient="records", force_ascii=False)) if not selected.empty else [])
    hashes["selected_rules.json"] = sha256_file(selected_path)
    manifest = {
        "version": cfg["version"],
        "fingerprint": fingerprint,
        "source_output_root": cfg["source_output_root"],
        "models_retrained": False,
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
    write_docs(cfg, manifest, selected, summaries, place_low, segment_summary)
    print("[roi-refine] done", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/roi_strategy_refinement_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-integrity-check", action="store_true")
    parser.add_argument("--skip-db-validation", action="store_true")
    parser.add_argument("--db-validation-config", default="config/database_validation.yaml")
    args = parser.parse_args()
    run(
        Path(args.config),
        resume=args.resume,
        strict_resume=args.strict_resume,
        force=args.force,
        db_validation_config=args.db_validation_config,
        force_integrity_check=args.force_integrity_check,
        skip_db_validation=args.skip_db_validation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
