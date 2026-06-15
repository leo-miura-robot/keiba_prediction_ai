from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


TARGETS = ["win", "place"]
FEATURE_SETS = ["market_free", "market_history", "market_aware"]
SPLITS = ["train", "validation", "test", "latest_holdout"]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, default=str))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    data = tmp.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    os.replace(tmp, path)
    return digest


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    data = tmp.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    os.replace(tmp, path)
    return digest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).replace("\\", "/").encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def git_info(root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"git_commit_sha": sha, "git_is_dirty": dirty}
    except Exception as exc:
        return {"git_commit_sha": "unknown", "git_is_dirty": None, "git_error": str(exc)}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expected_split(year: int, cfg: dict[str, Any]) -> str:
    for split, body in cfg["splits"].items():
        if year in body["years"]:
            return split
    raise ValueError(f"year {year} is not in split config")


def load_feature_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    base = Path(cfg["feature_dataset_dir"])
    cols = [
        "entry_id", "race_id", "race_date", "Year", "MonthDay", "JyoCD", "RaceNum",
        "Umaban", "Bamei", "KettoNum", "TrackCD", "Kyori", "Ninki", "TanNinki",
        "FukuNinki", "tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_pay",
        "fuku_pay", "target_win_paid", "target_place_paid", "eligible_for_win_training",
        "eligible_for_place_training", "race_is_finalized", "place_rank_limit",
    ]
    frames = []
    for year in range(2016, 2027):
        path = base / f"year={year}" / "data.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, columns=cols)
        df["expected_split"] = expected_split(year, cfg)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta["race_date"] = pd.to_datetime(meta["race_date"])
    meta["month"] = meta["race_date"].dt.month
    meta["distance_band"] = pd.cut(
        pd.to_numeric(meta["Kyori"], errors="coerce"),
        bins=[0, 1200, 1400, 1600, 1800, 2000, 2400, 10000],
        labels=["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"],
        include_lowest=True,
    ).astype(str)
    meta["popularity_band"] = pd.cut(
        pd.to_numeric(meta["Ninki"], errors="coerce"),
        bins=[0, 1, 3, 5, 10, 18, 999],
        labels=["1", "2-3", "4-5", "6-10", "11-18", "19+"],
        include_lowest=True,
    ).astype(str)
    return meta


def load_predictions(cfg: dict[str, Any], target: str, feature_set: str) -> pd.DataFrame:
    path = Path(cfg["prediction_root"]) / f"{target}_{feature_set}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = {"entry_id", "race_id", "race_date", "data_split", "actual", "pred_probability", "eligible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["target"] = target
    df["feature_set"] = feature_set
    return df


def attach_metadata(pred: pd.DataFrame, meta: pd.DataFrame, target: str) -> pd.DataFrame:
    keep = [
        "entry_id", "tan_pay", "fuku_pay", "Ninki", "TanNinki", "FukuNinki",
        "TrackCD", "Kyori", "JyoCD", "month", "distance_band", "popularity_band",
        "expected_split", "race_is_finalized", "place_rank_limit",
    ]
    out = pred.merge(meta[keep], on="entry_id", how="left", validate="one_to_one")
    if out["expected_split"].isna().any():
        raise ValueError(f"{target}: metadata join has missing rows")
    mismatch = out["data_split"] != out["expected_split"]
    if mismatch.any():
        raise ValueError(f"{target}: split mismatch rows={int(mismatch.sum())}")
    if target == "win":
        positives = out["actual"] == 1
        missing_pay = positives & (pd.to_numeric(out["tan_pay"], errors="coerce").fillna(0) <= 0)
    else:
        positives = out["actual"] == 1
        missing_pay = positives & (pd.to_numeric(out["fuku_pay"], errors="coerce").fillna(0) <= 0)
    if missing_pay.any():
        raise RuntimeError(f"{target}: positive target has missing payout rows={int(missing_pay.sum())}")
    return out


def ece_mce(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if not mask.any():
            continue
        gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
        ece += gap * (int(mask.sum()) / n)
        mce = max(mce, gap)
    return ece, mce


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


@dataclass
class Calibrator:
    method: str
    model: Any = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            return np.clip(self.model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        raise ValueError(self.method)


def fit_calibrators(df: pd.DataFrame) -> tuple[dict[str, Calibrator], pd.DataFrame, pd.DataFrame]:
    rows = []
    selected = []
    fitted: dict[str, Calibrator] = {}
    for target in TARGETS:
        for fs in FEATURE_SETS:
            key = f"{target}_{fs}"
            val = df[(df["target"] == target) & (df["feature_set"] == fs) & (df["data_split"] == "validation")]
            y = val["actual"].to_numpy(dtype=int)
            p = val["pred_probability"].to_numpy(dtype=float)
            candidates: list[Calibrator] = [Calibrator("none")]
            if len(np.unique(y)) == 2:
                lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                lr.fit(p.reshape(-1, 1), y)
                candidates.append(Calibrator("platt", lr))
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
                iso.fit(p, y)
                candidates.append(Calibrator("isotonic", iso))
            best = None
            for cal in candidates:
                cp = cal.transform(p)
                slope, intercept = calibration_slope_intercept(y, cp)
                ece, mce = ece_mce(y, cp)
                row = {
                    "target": target, "feature_set": fs, "method": cal.method,
                    "rows": len(val), "positives": int(y.sum()),
                    "logloss": log_loss(y, cp, labels=[0, 1]),
                    "brier": brier_score_loss(y, cp),
                    "ece": ece, "mce": mce,
                    "calibration_slope": slope, "calibration_intercept": intercept,
                }
                rows.append(row)
                if best is None or (row["logloss"], row["brier"], row["ece"], row["method"]) < (
                    best[0]["logloss"], best[0]["brier"], best[0]["ece"], best[0]["method"]
                ):
                    best = (row, cal)
            assert best is not None
            fitted[key] = best[1]
            selected.append({**best[0], "selected": True})
    return fitted, pd.DataFrame(rows), pd.DataFrame(selected)


def apply_calibration(df: pd.DataFrame, calibrators: dict[str, Calibrator]) -> pd.DataFrame:
    parts = []
    for (target, fs), g in df.groupby(["target", "feature_set"], sort=False):
        cal = calibrators[f"{target}_{fs}"]
        part = g.copy()
        part["calibration_method"] = cal.method
        part["calibrated_probability"] = cal.transform(part["pred_probability"].to_numpy(dtype=float))
        if target == "win":
            part["market_raw_probability"] = 1.0 / pd.to_numeric(part["tan_odds"], errors="coerce")
            sums = part.groupby("race_id")["market_raw_probability"].transform("sum")
            part["normalized_market_probability"] = part["market_raw_probability"] / sums
            part["market_gap"] = part["calibrated_probability"] - part["normalized_market_probability"]
            part["ev"] = part["calibrated_probability"] * pd.to_numeric(part["tan_odds"], errors="coerce")
        else:
            part["normalized_market_probability"] = np.nan
            part["market_gap"] = np.nan
            part["ev"] = part["calibrated_probability"] * pd.to_numeric(part["fuku_odds_low"], errors="coerce")
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def add_race_confidence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["target", "feature_set", "race_id"]
    out = out.sort_values(group_cols + ["calibrated_probability", "entry_id"], ascending=[True, True, True, False, True])
    out["model_rank_in_race"] = out.groupby(group_cols, sort=False).cumcount() + 1

    top_stats = (
        out[out["model_rank_in_race"].isin([1, 2, 3])]
        .pivot_table(
            index=group_cols,
            columns="model_rank_in_race",
            values="calibrated_probability",
            aggfunc="first",
        )
        .rename(columns={1: "top1_probability", 2: "top2_probability", 3: "top3_probability"})
        .reset_index()
    )
    for col in ["top1_probability", "top2_probability", "top3_probability"]:
        if col not in top_stats.columns:
            top_stats[col] = 0.0
    top_stats["top1_minus_top2_margin"] = top_stats["top1_probability"] - top_stats["top2_probability"].fillna(0.0)
    top_stats["top3_probability_sum"] = top_stats[["top1_probability", "top2_probability", "top3_probability"]].fillna(0.0).sum(axis=1)

    entropy_src = out[["target", "feature_set", "race_id", "calibrated_probability"]].copy()
    p = entropy_src["calibrated_probability"].clip(1e-12, 1.0)
    entropy_src["_entropy_part"] = -(p * np.log(p))
    entropy = entropy_src.groupby(group_cols, sort=False)["_entropy_part"].sum().reset_index(name="_entropy")
    sizes = out.groupby(group_cols, sort=False).size().reset_index(name="_race_size")
    entropy = entropy.merge(sizes, on=group_cols, how="left", validate="one_to_one")
    entropy["prediction_entropy"] = entropy["_entropy"] / np.log(np.maximum(entropy["_race_size"], 2))
    entropy = entropy[group_cols + ["prediction_entropy"]]

    top_entries = (
        out[out["model_rank_in_race"] == 1][["target", "feature_set", "race_id", "entry_id"]]
        .rename(columns={"entry_id": "top_entry_id"})
    )
    top_wide = top_entries.pivot(index=["target", "race_id"], columns="feature_set", values="top_entry_id").reset_index()
    agree_rows = top_entries.merge(top_wide, on=["target", "race_id"], how="left", validate="many_to_one")
    agree_rows["model_agreement_count"] = 0
    for fs in FEATURE_SETS:
        if fs in agree_rows.columns:
            agree_rows["model_agreement_count"] += (agree_rows["top_entry_id"] == agree_rows[fs]).astype(int)
    agree_rows = agree_rows[["target", "feature_set", "race_id", "top_entry_id", "model_agreement_count"]]

    out = out.merge(top_stats, on=group_cols, how="left", validate="many_to_one")
    out = out.merge(entropy, on=group_cols, how="left", validate="many_to_one")
    out = out.merge(agree_rows, on=group_cols, how="left", validate="many_to_one")
    out["is_model_top1"] = out["entry_id"] == out["top_entry_id"]
    return out.drop(columns=["top_entry_id"]).sort_index()


def summarize_roi(bets: pd.DataFrame, target: str, label: dict[str, Any] | None = None) -> dict[str, Any]:
    label = label or {}
    if bets.empty:
        base = {
            "bets": 0, "stake": 0, "return": 0, "profit": 0, "roi": np.nan,
            "hit_count": 0, "hit_rate": np.nan, "average_odds": np.nan, "median_odds": np.nan,
            "max_odds": np.nan, "average_payout": np.nan, "max_payout": np.nan,
            "max_losing_streak": 0, "max_drawdown": 0,
        }
        return {**label, **base}
    odds_col = "tan_odds" if target == "win" else "fuku_odds_low"
    pay_col = "tan_pay" if target == "win" else "fuku_pay"
    returns = pd.to_numeric(bets[pay_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    hits = returns > 0
    profit = returns - 100.0
    equity = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = peak - equity
    max_ls = 0
    cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        max_ls = max(max_ls, cur)
    base = {
        "bets": int(len(bets)),
        "stake": float(len(bets) * 100),
        "return": float(returns.sum()),
        "profit": float(profit.sum()),
        "roi": float(returns.sum() / (len(bets) * 100) * 100),
        "hit_count": int(hits.sum()),
        "hit_rate": float(hits.mean()),
        "average_odds": float(pd.to_numeric(bets[odds_col], errors="coerce").mean()),
        "median_odds": float(pd.to_numeric(bets[odds_col], errors="coerce").median()),
        "max_odds": float(pd.to_numeric(bets[odds_col], errors="coerce").max()),
        "average_payout": float(returns.mean()),
        "max_payout": float(returns.max()),
        "max_losing_streak": int(max_ls),
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
    }
    return {**label, **base}


def payout_dependency(bets: pd.DataFrame, target: str, label: dict[str, Any]) -> list[dict[str, Any]]:
    pay_col = "tan_pay" if target == "win" else "fuku_pay"
    rows = []
    for n in [0, 1, 3, 5, 10]:
        cut = bets.sort_values(pay_col, ascending=False).iloc[n:] if not bets.empty else bets
        row = summarize_roi(cut, target, {**label, "removed_top_payouts": n})
        rows.append(row)
    if bets.empty:
        top1_share = np.nan
        top1pct_share = np.nan
    else:
        returns = pd.to_numeric(bets[pay_col], errors="coerce").fillna(0).sort_values(ascending=False).to_numpy()
        total_profit = float((returns - 100).sum())
        positive_profit = np.maximum(returns - 100, 0)
        denom = float(positive_profit.sum())
        top1_share = float(positive_profit[:1].sum() / denom) if denom > 0 else np.nan
        top_n = max(1, int(math.ceil(len(returns) * 0.01)))
        top1pct_share = float(positive_profit[:top_n].sum() / denom) if denom > 0 else np.nan
    rows.append({**label, "removed_top_payouts": "dependency", "top1_profit_share": top1_share, "top1pct_profit_share": top1pct_share})
    return rows


def confidence_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runner_rows = []
    race_rows = []
    bin_defs = {
        "top1_probability": [-np.inf, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, np.inf],
        "top1_minus_top2_margin": [-np.inf, 0.02, 0.05, 0.10, 0.20, np.inf],
        "model_agreement_count": [0, 1, 2, 3, np.inf],
    }
    for (target, fs, split), g in df.groupby(["target", "feature_set", "data_split"]):
        pay_col = "tan_pay" if target == "win" else "fuku_pay"
        odds_col = "tan_odds" if target == "win" else "fuku_odds_low"
        top = g[g["is_model_top1"]].copy()
        for metric, bins in bin_defs.items():
            labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
            top["band"] = pd.cut(top[metric], bins=bins, labels=labels, include_lowest=True).astype(str)
            for band, bg in top.groupby("band", dropna=False):
                row = summarize_roi(bg, target, {
                    "target": target, "feature_set": fs, "data_split": split,
                    "metric": metric, "band": band,
                    "mean_predicted_probability": float(bg["calibrated_probability"].mean()) if len(bg) else np.nan,
                    "actual_rate": float(bg["actual"].mean()) if len(bg) else np.nan,
                    "calibration_gap": float(bg["actual"].mean() - bg["calibrated_probability"].mean()) if len(bg) else np.nan,
                    "average_odds_metric": float(pd.to_numeric(bg[odds_col], errors="coerce").mean()) if len(bg) else np.nan,
                })
                race_rows.append(row)
        for metric, q in [("prediction_entropy", 5), ("top3_probability_sum", 5)]:
            ranked = top.sort_values(metric).copy()
            ranked["band"] = pd.qcut(ranked[metric].rank(method="first"), q=min(q, max(1, len(ranked))), duplicates="drop").astype(str)
            for band, bg in ranked.groupby("band", dropna=False):
                race_rows.append(summarize_roi(bg, target, {
                    "target": target, "feature_set": fs, "data_split": split, "metric": metric,
                    "band": band, "mean_predicted_probability": float(bg["calibrated_probability"].mean()),
                    "actual_rate": float(bg["actual"].mean()),
                    "calibration_gap": float(bg["actual"].mean() - bg["calibrated_probability"].mean()),
                }))
        for metric, bins in [("calibrated_probability", [-np.inf, .02, .05, .1, .2, .4, .6, .8, np.inf])]:
            g2 = g.copy()
            g2["band"] = pd.cut(g2[metric], bins=bins, include_lowest=True).astype(str)
            for band, bg in g2.groupby("band", dropna=False):
                runner_rows.append({
                    "target": target, "feature_set": fs, "data_split": split, "metric": metric,
                    "band": band, "rows": len(bg),
                    "mean_predicted_probability": float(bg["calibrated_probability"].mean()),
                    "actual_rate": float(bg["actual"].mean()),
                    "calibration_gap": float(bg["actual"].mean() - bg["calibrated_probability"].mean()),
                    "mean_payout": float(pd.to_numeric(bg[pay_col], errors="coerce").mean()),
                })
    return pd.DataFrame(runner_rows), pd.DataFrame(race_rows)


def candidate_rules(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        odds_col = "tan_odds" if target == "win" else "fuku_odds_low"
        grid = cfg["rule_selection"][target]
        for fs in FEATURE_SETS:
            val = df[(df["target"] == target) & (df["feature_set"] == fs) & (df["data_split"] == "validation")]
            entropy_thresholds = [float(val["prediction_entropy"].quantile(q)) for q in cfg["rule_selection"]["entropy_max_quantiles"]]
            for top_n in cfg["rule_selection"]["top_n"]:
                base = val[val["model_rank_in_race"] <= top_n]
                for ev_min in grid["ev_min"]:
                    for prob_min in grid["probability_min"]:
                        for margin_min in cfg["rule_selection"]["margin_min"]:
                            for agree_min in cfg["rule_selection"]["model_agreement_min"]:
                                for entropy_max in entropy_thresholds:
                                    for low, high in grid["odds_bands"]:
                                        market_gaps = grid.get("market_gap_min", [-999.0])
                                        for market_gap_min in market_gaps:
                                            mask = (
                                                (base["ev"] >= ev_min)
                                                & (base["calibrated_probability"] >= prob_min)
                                                & (base[odds_col] >= low)
                                                & (base[odds_col] < high)
                                                & (base["top1_minus_top2_margin"] >= margin_min)
                                                & (base["model_agreement_count"] >= agree_min)
                                                & (base["prediction_entropy"] <= entropy_max)
                                            )
                                            if target == "win":
                                                mask &= base["market_gap"].fillna(-999.0) >= market_gap_min
                                            bets = base[mask].sort_values(["race_date", "race_id", "model_rank_in_race"])
                                            if len(bets) < cfg["rule_selection"]["min_validation_bets"]:
                                                continue
                                            row = summarize_roi(bets, target, {
                                                "target": target, "feature_set": fs, "data_split": "validation",
                                                "top_n": top_n, "ev_min": ev_min, "probability_min": prob_min,
                                                "odds_min": low, "odds_max": high, "margin_min": margin_min,
                                                "model_agreement_min": agree_min, "entropy_max": entropy_max,
                                                "market_gap_min": market_gap_min if target == "win" else np.nan,
                                            })
                                            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["passes_roi_goal"] = out["roi"] >= 90.0
    out["score"] = (
        out["passes_roi_goal"].astype(int) * 1_000_000
        + out["roi"].fillna(0) * 1000
        + np.minimum(out["bets"], 5000)
        - out["max_drawdown"].fillna(0) * 0.01
    )
    return out.sort_values(["target", "feature_set", "score"], ascending=[True, True, False])


def select_rules(candidates: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    selected = []
    n = cfg["rule_selection"]["max_rules_per_model"]
    for (_, _), g in candidates.groupby(["target", "feature_set"]):
        selected.append(g.sort_values("score", ascending=False).head(n))
    return pd.concat(selected, ignore_index=True)


def apply_rule(df: pd.DataFrame, rule: pd.Series, split: str) -> pd.DataFrame:
    target = rule["target"]
    odds_col = "tan_odds" if target == "win" else "fuku_odds_low"
    g = df[(df["target"] == target) & (df["feature_set"] == rule["feature_set"]) & (df["data_split"] == split)]
    mask = (
        (g["model_rank_in_race"] <= int(rule["top_n"]))
        & (g["ev"] >= float(rule["ev_min"]))
        & (g["calibrated_probability"] >= float(rule["probability_min"]))
        & (g[odds_col] >= float(rule["odds_min"]))
        & (g[odds_col] < float(rule["odds_max"]))
        & (g["top1_minus_top2_margin"] >= float(rule["margin_min"]))
        & (g["model_agreement_count"] >= int(rule["model_agreement_min"]))
        & (g["prediction_entropy"] <= float(rule["entropy_max"]))
    )
    if target == "win":
        mask &= g["market_gap"].fillna(-999.0) >= float(rule["market_gap_min"])
    sort_cols = [c for c in ["race_date", "race_id", "model_rank_in_race", "entry_id"] if c in g.columns]
    selected = g[mask]
    if sort_cols:
        selected = selected.sort_values(sort_cols)
    return selected.copy()


def evaluate_selected_rules(df: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = []
    detail_parts = []
    dependency_rows = []
    draw_rows = []
    for idx, rule in selected.reset_index(drop=True).iterrows():
        rule_id = f"{rule['target']}_{rule['feature_set']}_rule{idx+1:02d}"
        for split in ["validation", "test", "latest_holdout"]:
            bets = apply_rule(df, rule, split)
            bets["rule_id"] = rule_id
            detail_parts.append(bets)
            label = {"rule_id": rule_id, "target": rule["target"], "feature_set": rule["feature_set"], "data_split": split}
            row = summarize_roi(bets, rule["target"], label)
            summary.append(row)
            dependency_rows.extend(payout_dependency(bets, rule["target"], label))
            draw_rows.append({k: row[k] for k in ["rule_id", "target", "feature_set", "data_split", "max_losing_streak", "max_drawdown", "bets", "roi"]})
    details = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return pd.DataFrame(summary), details, pd.DataFrame(dependency_rows), pd.DataFrame(draw_rows)


def grouped_roi(details: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    rows = []
    for name, g in details.groupby(keys, dropna=False):
        vals = name if isinstance(name, tuple) else (name,)
        label = dict(zip(keys, vals))
        target = str(g["target"].iloc[0])
        rows.append(summarize_roi(g, target, label))
    return pd.DataFrame(rows)


def bootstrap_ci(details: pd.DataFrame, seed: int, iterations: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    if details.empty:
        return pd.DataFrame()
    for keys, g in details.groupby(["rule_id", "target", "feature_set", "data_split"], dropna=False):
        target = str(keys[1])
        pay_col = "tan_pay" if target == "win" else "fuku_pay"
        race_returns = (
            g.assign(_return=pd.to_numeric(g[pay_col], errors="coerce").fillna(0), _stake=100.0)
            .groupby("race_id", sort=False)[["_return", "_stake"]]
            .sum()
            .reset_index()
        )
        if race_returns.empty:
            continue
        returns = race_returns["_return"].to_numpy(dtype=float)
        stakes = race_returns["_stake"].to_numpy(dtype=float)
        race_count = len(race_returns)
        rois = []
        for _ in range(iterations):
            idx = rng.integers(0, race_count, size=race_count)
            stake_sum = stakes[idx].sum()
            roi = returns[idx].sum() / stake_sum * 100 if stake_sum > 0 else np.nan
            rois.append(roi)
        rows.append({
            "rule_id": keys[0], "target": keys[1], "feature_set": keys[2], "data_split": keys[3],
            "race_count": race_count, "bootstrap_iterations": iterations,
            "roi_point": summarize_roi(g, target)["roi"],
            "roi_p025": float(np.percentile(rois, 2.5)),
            "roi_p500": float(np.percentile(rois, 50)),
            "roi_p975": float(np.percentile(rois, 97.5)),
        })
    return pd.DataFrame(rows)


def docs(result: dict[str, Any], out: Path) -> None:
    design = [
        "# ROI Validation V2.1.2 V1",
        "",
        "- 入力は学習済み `catboost_baseline_v2_1_2_v1` の予測Parquetのみです。",
        "- モデル再学習は行っていません。",
        "- 補正方法と購入条件はvalidation 2024だけで選び、test 2025 / latest_holdout 2026には固定適用します。",
        "- 単勝EVは `calibrated_win_probability * tan_odds`、複勝EVは保守的に `calibrated_place_probability * fuku_odds_low` です。",
        "- ROIは100円均等買いで、実払戻 `tan_pay` / `fuku_pay` のみを使います。",
        "- `market_history` は発走前実運用候補、`market_aware` は確定オッズ入力の理想条件モデルとして別枠で扱います。",
        "- 自動購入、Kelly、資金配分最適化、Ability/ANA/Rankerは対象外です。",
    ]
    atomic_write_text(Path("docs/roi_validation_v2_1_2_v1_design.md"), "\n".join(design) + "\n")
    lines = [
        "# ROI Validation V2.1.2 V1 Results",
        "",
        f"- CSV hash一致: `{result['hashes_match']}`",
        f"- pytest: `{result.get('pytest_result', 'not run by script')}`",
        f"- 複勝払戻結合: `{result['payout_validation']['place_positive_missing_payout']}` positive rows missing payout",
        "",
        "## Selected Rules",
        "",
        result["selected_rules_markdown"],
        "",
        "## ROI Summary",
        "",
        result["roi_summary_markdown"],
    ]
    atomic_write_text(Path("docs/roi_validation_v2_1_2_v1_results.md"), "\n".join(lines) + "\n")


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    print("[roi] loading feature metadata", flush=True)
    meta = load_feature_metadata(cfg)
    prediction_files = sorted(Path(cfg["prediction_root"]).glob("*.parquet"))
    input_hash = hash_paths(prediction_files + sorted(Path(cfg["feature_dataset_dir"]).glob("year=*/data.parquet")))
    print("[roi] loading predictions and joining payouts", flush=True)
    frames = []
    payout_validation = {}
    for target in TARGETS:
        for fs in FEATURE_SETS:
            pred = load_predictions(cfg, target, fs)
            frames.append(attach_metadata(pred, meta, target))
    df = pd.concat(frames, ignore_index=True)
    payout_validation["win_positive_missing_payout"] = int(((df["target"] == "win") & (df["actual"] == 1) & (df["tan_pay"] <= 0)).sum())
    payout_validation["place_positive_missing_payout"] = int(((df["target"] == "place") & (df["actual"] == 1) & (df["fuku_pay"] <= 0)).sum())
    if payout_validation["place_positive_missing_payout"] > 0:
        raise RuntimeError("place ROI stopped: place payout cannot be joined for positive rows")
    print("[roi] calibration selection on validation", flush=True)
    calibrators, cal_metrics, cal_selected = fit_calibrators(df)
    df = apply_calibration(df, calibrators)
    df = add_race_confidence(df)
    print("[roi] confidence analysis", flush=True)
    conf_runner, conf_race = confidence_analysis(df)
    confidence_all = pd.concat([
        conf_runner.assign(level="runner"),
        conf_race.assign(level="race_top1"),
    ], ignore_index=True, sort=False)
    print("[roi] validation-only rule search", flush=True)
    candidates = candidate_rules(df, cfg)
    selected = select_rules(candidates, cfg)
    if selected.empty:
        raise RuntimeError("no validation candidate rules met the minimum bet count")
    print("[roi] fixed rule evaluation", flush=True)
    roi_summary, bet_details, dependency, drawdown = evaluate_selected_rules(df, selected)
    bet_details["odds_band"] = np.where(
        bet_details["target"] == "win",
        pd.cut(bet_details["tan_odds"], [0, 1.5, 3, 5, 10, 20, 999], labels=["<1.5", "1.5-3", "3-5", "5-10", "10-20", "20+"]).astype(str),
        pd.cut(bet_details["fuku_odds_low"], [0, 1.1, 1.5, 2, 3, 5, 999], labels=["<1.1", "1.1-1.5", "1.5-2", "2-3", "3-5", "5+"]).astype(str),
    )
    bet_details["confidence_band"] = pd.cut(
        bet_details["calibrated_probability"], [0, .05, .1, .2, .3, .5, .75, 1],
        labels=["<.05", ".05-.10", ".10-.20", ".20-.30", ".30-.50", ".50-.75", ".75+"],
        include_lowest=True,
    ).astype(str)
    print("[roi] stability and bootstrap", flush=True)
    roi_by_year = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "Year"])
    roi_by_month = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "month"])
    roi_by_track = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "JyoCD"])
    roi_by_odds = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "odds_band"])
    roi_by_pop = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "popularity_band"])
    roi_by_conf = grouped_roi(bet_details, ["rule_id", "target", "feature_set", "data_split", "confidence_band"])
    bootstrap = bootstrap_ci(bet_details, int(cfg["random_seed"]), int(cfg["bootstrap_iterations"]))
    print("[roi] writing outputs", flush=True)
    hashes = {}
    outputs = {
        "confidence_analysis.csv": confidence_all,
        "confidence_calibration.csv": conf_runner,
        "calibration_method_selection.csv": cal_selected,
        "calibration_metrics.csv": cal_metrics,
        "candidate_rules_validation.csv": candidates,
        "roi_summary_test.csv": roi_summary[roi_summary["data_split"] == "test"],
        "roi_summary_latest_holdout.csv": roi_summary[roi_summary["data_split"] == "latest_holdout"],
        "roi_by_year.csv": roi_by_year,
        "roi_by_month.csv": roi_by_month,
        "roi_by_track.csv": roi_by_track,
        "roi_by_odds_band.csv": roi_by_odds,
        "roi_by_popularity_band.csv": roi_by_pop,
        "roi_by_confidence_band.csv": roi_by_conf,
        "payout_dependency.csv": dependency,
        "drawdown_summary.csv": drawdown,
        "bootstrap_roi_ci.csv": bootstrap,
    }
    for name, table in outputs.items():
        hashes[name] = atomic_write_csv(out / name, table)
    hashes["bet_details.parquet"] = atomic_write_parquet(out / "bet_details.parquet", bet_details)
    selected_rules = selected.to_dict(orient="records")
    atomic_write_json(out / "selected_rules.json", selected_rules)
    for (target, fs), g in df.groupby(["target", "feature_set"], sort=False):
        keep = [
            "entry_id", "race_id", "race_date", "Year", "Umaban", "data_split", "target",
            "feature_set", "actual", "pred_probability", "calibrated_probability",
            "calibration_method", "tan_odds", "fuku_odds_low", "fuku_odds_high",
            "tan_pay", "fuku_pay", "ev", "top1_probability", "top1_minus_top2_margin",
            "prediction_entropy", "top3_probability_sum", "model_agreement_count",
            "model_rank_in_race", "market_gap",
        ]
        atomic_write_parquet(out / "calibrated_predictions" / f"{target}_{fs}.parquet", g[keep])
    manifest = {
        "version": cfg["version"],
        "model_version": cfg["model_version"],
        "input_hash": input_hash,
        "config_hash": sha256_file(config_path),
        "code_hash": sha256_file(Path(__file__)),
        "output_hashes": hashes,
        "payout_validation": payout_validation,
        "git": git_info(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__},
        "random_seed": cfg["random_seed"],
        "bootstrap_iterations": cfg["bootstrap_iterations"],
    }
    atomic_write_json(out / "manifest.json", manifest)
    selected_md = selected.head(18).to_markdown(index=False)
    roi_md = roi_summary.to_markdown(index=False)
    result = {
        "hashes_match": None,
        "payout_validation": payout_validation,
        "selected_rules_markdown": selected_md,
        "roi_summary_markdown": roi_md,
    }
    docs(result, out)
    print("[roi] done", flush=True)
    return {"manifest": manifest, "selected": selected, "roi_summary": roi_summary, "hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/roi_validation_v2_1_2_v1.yaml")
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
