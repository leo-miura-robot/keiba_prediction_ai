from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_sets_v2_1_2 import load_feature_set_yaml


TARGETS = ["win", "place"]


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


def gpu_name(devices: str | None) -> str:
    try:
        idx = (devices or "0").split(",")[0]
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={idx}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return out or "unknown"
    except Exception as exc:
        return f"unknown: {exc}"


def gpu_smoke(params: dict[str, Any]) -> dict[str, Any]:
    if str(params.get("task_type", "")).upper() != "GPU":
        raise RuntimeError("GPU is required; CPU fallback is disabled")
    x = np.random.default_rng(42).normal(size=(256, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = CatBoostClassifier(
        iterations=4,
        loss_function="Logloss",
        task_type="GPU",
        devices=params.get("devices", "0"),
        verbose=False,
        allow_writing_files=False,
        random_seed=42,
    )
    model.fit(x, y)
    return {
        "task_type": "GPU",
        "devices": params.get("devices", "0"),
        "gpu_name": gpu_name(params.get("devices")),
        "catboost_gpu_smoke": "ok",
        "cpu_fallback_used": False,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_params(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    params = dict(cfg["training_params"])
    if smoke:
        params.update({k: v for k, v in cfg.get("smoke_overrides", {}).items() if k in params})
    return params


def feature_columns(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    sets = load_feature_set_yaml(Path(cfg["feature_set_yaml"]))
    groups = sets[cfg["feature_set"]]
    return groups.get("numeric", []), groups.get("categorical", [])


def dataset_hash(cfg: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for year in range(2016, 2027):
        p = Path(cfg["input_dataset_dir"]) / f"year={year}" / "data.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        h.update(str(p).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def required_columns(cfg: dict[str, Any], numeric: list[str], cat: list[str]) -> list[str]:
    meta = [
        "race_id", "entry_id", "race_date", "Year", "MonthDay", "JyoCD", "RaceNum", "Umaban",
        "KettoNum", "Bamei", "TrackCD", "Kyori", "SyussoTosu", "Ninki", "TanNinki", "FukuNinki",
        "tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_pay", "fuku_pay",
        "target_win_paid", "target_place_paid", "eligible_for_win_training", "eligible_for_place_training",
        "race_is_finalized",
    ]
    return sorted(set(meta + numeric + cat))


def load_dataset(cfg: dict[str, Any], numeric: list[str], cat: list[str], smoke: bool) -> pd.DataFrame:
    cols = required_columns(cfg, numeric, cat)
    frames = []
    for year in range(2016, 2027):
        p = Path(cfg["input_dataset_dir"]) / f"year={year}" / "data.parquet"
        df = pd.read_parquet(p, columns=cols)
        if smoke:
            n = cfg.get("smoke_overrides", {}).get("train_rows_per_year", 800)
            if year >= 2020:
                n = cfg.get("smoke_overrides", {}).get("eval_rows_per_year", 400)
            df = df.head(int(n))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["race_date"] = pd.to_datetime(out["race_date"])
    out["month"] = out["race_date"].dt.month
    out["distance_band"] = pd.cut(
        pd.to_numeric(out["Kyori"], errors="coerce"),
        [0, 1200, 1400, 1600, 1800, 2000, 2400, 10000],
        labels=["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"],
        include_lowest=True,
    ).astype(str)
    out["field_size_band"] = pd.cut(
        pd.to_numeric(out["SyussoTosu"], errors="coerce"),
        [0, 8, 12, 16, 18, 99],
        labels=["<=8", "9-12", "13-16", "17-18", "19+"],
        include_lowest=True,
    ).astype(str)
    return out


def prepare_x(df: pd.DataFrame, numeric: list[str], cat: list[str]) -> pd.DataFrame:
    x = df[numeric + cat].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat:
        s = x[c].astype("object")
        x[c] = s.where(pd.notna(s), "__MISSING__").astype(str).replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    return x


def target_frame(df: pd.DataFrame, cfg: dict[str, Any], target: str) -> pd.DataFrame:
    tcfg = cfg["targets"][target]
    out = df[df[tcfg["eligible_column"]] == True].copy()
    out["actual"] = out[tcfg["target_column"]].astype(int)
    if target == "win":
        bad = (out["actual"].eq(1)) & (pd.to_numeric(out["tan_pay"], errors="coerce").fillna(0) <= 0)
    else:
        bad = (out["actual"].eq(1)) & (pd.to_numeric(out["fuku_pay"], errors="coerce").fillna(0) <= 0)
    if bad.any():
        raise RuntimeError(f"{target} positive rows with missing payout: {int(bad.sum())}")
    return out


def cat_indices(x: pd.DataFrame, cat: list[str]) -> list[int]:
    return [x.columns.get_loc(c) for c in cat]


def train_model(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], cat: list[str], params: dict[str, Any], model_path: Path) -> tuple[CatBoostClassifier, dict[str, Any]]:
    x_train = prepare_x(train, numeric, cat)
    x_valid = prepare_x(valid, numeric, cat)
    cats = cat_indices(x_train, cat)
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(
        Pool(x_train, train["actual"].to_numpy(), cat_features=cats),
        eval_set=Pool(x_valid, valid["actual"].to_numpy(), cat_features=cats),
        use_best_model=True,
    )
    elapsed = time.time() - started
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return model, {"training_seconds": elapsed, "best_iteration": int(model.get_best_iteration() or 0), "tree_count": int(model.tree_count_ or 0)}


def predict(model: CatBoostClassifier, df: pd.DataFrame, numeric: list[str], cat: list[str]) -> np.ndarray:
    x = prepare_x(df, numeric, cat)
    return model.predict_proba(Pool(x, cat_features=cat_indices(x, cat)))[:, 1]


def metric_row(y: np.ndarray, p: np.ndarray, label: dict[str, Any]) -> dict[str, Any]:
    out = dict(label)
    out.update({
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
    })
    return out


@dataclass
class Calibrator:
    method: str
    model: Any = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "platt":
            return np.clip(self.model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 1e-6, 1 - 1e-6)
        raise ValueError(self.method)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.any():
            total += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / len(y))
    return float(total)


def fit_calibration(oof: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Calibrator], pd.DataFrame]:
    rows = []
    selected: dict[str, Calibrator] = {}
    for target in TARGETS:
        d = oof[oof["target"] == target]
        best = None
        for method in cfg["calibration_methods"]:
            fold_rows = []
            for fold in sorted(d["fold"].unique()):
                val = d[d["fold"] == fold]
                train = d[d["fold"] != fold]
                y_train = train["actual"].to_numpy(int)
                p_train = train["raw_probability"].to_numpy(float)
                y_val = val["actual"].to_numpy(int)
                p_val = val["raw_probability"].to_numpy(float)
                if method == "none":
                    cal = Calibrator("none")
                elif method == "platt":
                    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                    lr.fit(p_train.reshape(-1, 1), y_train)
                    cal = Calibrator("platt", lr)
                else:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
                    iso.fit(p_train, y_train)
                    cal = Calibrator("isotonic", iso)
                cp = cal.transform(p_val)
                fold_rows.append({
                    "target": target,
                    "method": method,
                    "fold": fold,
                    "year": int(val["Year"].iloc[0]),
                    "logloss": log_loss(y_val, cp, labels=[0, 1]),
                    "brier": brier_score_loss(y_val, cp),
                    "ece": ece(y_val, cp),
                })
            df = pd.DataFrame(fold_rows)
            row = {
                "target": target,
                "method": method,
                "mean_logloss": float(df["logloss"].mean()),
                "mean_brier": float(df["brier"].mean()),
                "mean_ece": float(df["ece"].mean()),
                "worst_year_ece": float(df["ece"].max()),
                "std_logloss": float(df["logloss"].std(ddof=0)),
            }
            rows.append(row)
            score = (row["mean_logloss"], row["mean_brier"], row["worst_year_ece"], row["std_logloss"])
            if best is None or score < best[0]:
                best = (score, method)
        assert best is not None
        method = best[1]
        y = d["actual"].to_numpy(int)
        p = d["raw_probability"].to_numpy(float)
        if method == "none":
            selected[target] = Calibrator("none")
        elif method == "platt":
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(p.reshape(-1, 1), y)
            selected[target] = Calibrator("platt", lr)
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            iso.fit(p, y)
            selected[target] = Calibrator("isotonic", iso)
    out = pd.DataFrame(rows)
    out["selected"] = out.apply(lambda r: selected[r["target"]].method == r["method"], axis=1)
    return selected, out


def add_market_and_confidence(df: pd.DataFrame, target: str, alpha: float) -> pd.DataFrame:
    d = df.copy()
    if target == "win":
        raw = 1.0 / pd.to_numeric(d["tan_odds"], errors="coerce")
        d["normalized_market_probability"] = raw / raw.groupby(d["race_id"]).transform("sum")
        market_rank_col = "TanNinki"
        odds_col = "tan_odds"
    else:
        d["normalized_market_probability"] = np.nan
        market_rank_col = "FukuNinki"
        odds_col = "fuku_odds_low"
    d = d.sort_values(["race_id", "calibrated_probability", "entry_id"], ascending=[True, False, True])
    d["model_rank"] = d.groupby("race_id").cumcount() + 1
    d["market_rank"] = pd.to_numeric(d[market_rank_col], errors="coerce")
    d["rank_gap"] = d["market_rank"] - d["model_rank"]
    if target == "win":
        d["conservative_probability"] = alpha * d["calibrated_probability"] + (1 - alpha) * d["normalized_market_probability"]
        d["edge"] = d["calibrated_probability"] - d["normalized_market_probability"]
    else:
        d["conservative_probability"] = d["calibrated_probability"]
        d["edge"] = d["calibrated_probability"] - (1.0 / pd.to_numeric(d["fuku_odds_low"], errors="coerce"))
    d["ev"] = d["conservative_probability"] * pd.to_numeric(d[odds_col], errors="coerce")
    probs = d["calibrated_probability"].clip(1e-12, 1.0)
    d["_entropy_part"] = -(probs * np.log(probs))
    stats = d.groupby("race_id").agg(
        top1_probability=("calibrated_probability", "max"),
        prediction_entropy=("_entropy_part", "sum"),
        race_size=("entry_id", "size"),
    )
    top3 = d[d["model_rank"] <= 3].groupby("race_id")["calibrated_probability"].sum().rename("top3_probability_sum")
    second = d[d["model_rank"] == 2].set_index("race_id")["calibrated_probability"].rename("top2_probability")
    stats = stats.join(top3, how="left").join(second, how="left")
    stats["top1_minus_top2_margin"] = stats["top1_probability"] - stats["top2_probability"].fillna(0)
    stats["prediction_entropy"] = stats["prediction_entropy"] / np.log(np.maximum(stats["race_size"], 2))
    d = d.merge(stats.reset_index()[["race_id", "top1_probability", "top1_minus_top2_margin", "prediction_entropy", "top3_probability_sum"]], on="race_id", how="left")
    return d.drop(columns=["_entropy_part"])


def payout_col(target: str) -> str:
    return "tan_pay" if target == "win" else "fuku_pay"


def odds_col(target: str) -> str:
    return "tan_odds" if target == "win" else "fuku_odds_low"


def summarize_bets(bets: pd.DataFrame, target: str, label: dict[str, Any] | None = None) -> dict[str, Any]:
    label = label or {}
    if bets.empty:
        return {**label, "bets": 0, "races": 0, "stake": 0.0, "return": 0.0, "profit": 0.0, "roi": np.nan, "hit_count": 0, "hit_rate": np.nan, "average_odds": np.nan, "median_odds": np.nan, "average_payout": np.nan, "max_payout": np.nan, "max_losing_streak": 0, "max_drawdown": 0.0}
    pay = pd.to_numeric(bets[payout_col(target)], errors="coerce").fillna(0).to_numpy(float)
    hits = pay > 0
    profit = pay - 100
    equity = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_ls = 0
    cur = 0
    for h in hits:
        cur = 0 if h else cur + 1
        max_ls = max(max_ls, cur)
    return {
        **label,
        "bets": int(len(bets)),
        "races": int(bets["race_id"].nunique()),
        "stake": float(len(bets) * 100),
        "return": float(pay.sum()),
        "profit": float(profit.sum()),
        "roi": float(pay.sum() / (len(bets) * 100) * 100),
        "hit_count": int(hits.sum()),
        "hit_rate": float(hits.mean()),
        "average_odds": float(pd.to_numeric(bets[odds_col(target)], errors="coerce").mean()),
        "median_odds": float(pd.to_numeric(bets[odds_col(target)], errors="coerce").median()),
        "average_payout": float(pay.mean()),
        "max_payout": float(pay.max()),
        "max_losing_streak": int(max_ls),
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
    }


def threshold_for_odds(cfg: dict[str, Any], target: str, odds: pd.Series) -> pd.Series:
    out = pd.Series(np.inf, index=odds.index)
    for r in cfg["rule_selection"]["odds_thresholds"][target]:
        m = (odds >= float(r["min_odds"])) & (odds < float(r["max_odds"]))
        out.loc[m] = float(r["ev_min"])
    return out


def classify_candidates(df: pd.DataFrame, cfg: dict[str, Any], target: str) -> pd.DataFrame:
    d = df.copy()
    oc = odds_col(target)
    d["odds_ev_min"] = threshold_for_odds(cfg, target, pd.to_numeric(d[oc], errors="coerce"))
    d["odds_band"] = pd.cut(pd.to_numeric(d[oc], errors="coerce"), [0, 1.5, 3, 5, 10, 20, 999], labels=["<1.5", "1.5-3", "3-5", "5-10", "10-20", "20+"], include_lowest=True).astype(str)
    d["confidence_band"] = pd.cut(d["conservative_probability"], [0, .05, .1, .2, .3, .5, .75, 1], labels=["<.05", ".05-.10", ".10-.20", ".20-.30", ".30-.50", ".50-.75", ".75+"], include_lowest=True).astype(str)
    strategies = []
    core = (d["model_rank"] == 1) & (d["top1_minus_top2_margin"] >= 0.02) & (d["ev"] >= d["odds_ev_min"]) & (pd.to_numeric(d[oc], errors="coerce") < (10 if target == "win" else 3))
    value = (d["model_rank"] <= 3) & (d["market_rank"] >= 4) & (d["rank_gap"] >= 1) & (d["edge"] > 0) & (d["ev"] >= d["odds_ev_min"])
    longshot = (d["model_rank"] <= 3) & (pd.to_numeric(d[oc], errors="coerce") >= (20 if target == "win" else 5)) & (d["edge"] > 0.03) & (d["ev"] >= d["odds_ev_min"] + 0.15)
    for name, mask in [("core", core), ("value", value), ("longshot", longshot)]:
        c = d[mask].copy()
        c["strategy_type"] = name
        strategies.append(c)
    out = pd.concat(strategies, ignore_index=True) if strategies else pd.DataFrame(columns=list(d.columns) + ["strategy_type"])
    if out.empty:
        fallback = d[d["model_rank"] == 1].copy()
        fallback["strategy_type"] = "core"
        out = fallback
    return out.drop_duplicates(["entry_id", "strategy_type"])


def remove_similar_rules(rules: pd.DataFrame, validation_candidates: pd.DataFrame, threshold: float) -> pd.DataFrame:
    kept = []
    entry_sets: list[set[str]] = []
    for _, rule in rules.iterrows():
        bets = apply_rule(validation_candidates, rule)
        s = set(bets["entry_id"].astype(str))
        duplicate = False
        for ks in entry_sets:
            denom = len(s | ks)
            sim = len(s & ks) / denom if denom else 0
            if sim >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(rule)
            entry_sets.append(s)
    return pd.DataFrame(kept)


def apply_rule(candidates: pd.DataFrame, rule: pd.Series) -> pd.DataFrame:
    d = candidates[candidates["strategy_type"] == rule["strategy_type"]]
    m = (
        (d["model_rank"] <= int(rule["model_rank_max"]))
        & (d["ev"] >= float(rule["ev_min"]))
        & (d["edge"] >= float(rule["edge_min"]))
        & (d["top1_minus_top2_margin"] >= float(rule["margin_min"]))
        & (d[odds_col(rule["target"])] >= float(rule["odds_min"]))
        & (d[odds_col(rule["target"])] < float(rule["odds_max"]))
    )
    if "rank_gap_min" in rule:
        m &= d["rank_gap"] >= float(rule["rank_gap_min"])
    return d[m].sort_values(["race_date", "race_id", "model_rank", "entry_id"])


def top_payout_removed_roi(bets: pd.DataFrame, target: str, n: int) -> float:
    if bets.empty:
        return np.nan
    d = bets.sort_values(payout_col(target), ascending=False).iloc[n:]
    return summarize_bets(d, target)["roi"] if len(d) else np.nan


def profit_share(bets: pd.DataFrame, target: str, pct: float) -> float:
    if bets.empty:
        return np.nan
    pay = pd.to_numeric(bets[payout_col(target)], errors="coerce").fillna(0).to_numpy(float)
    positive_profit = np.maximum(pay - 100, 0)
    denom = positive_profit.sum()
    if denom <= 0:
        return np.nan
    n = max(1, int(math.ceil(len(pay) * pct)))
    return float(np.sort(positive_profit)[::-1][:n].sum() / denom)


def bootstrap_ci(bets: pd.DataFrame, target: str, iterations: int, seed: int) -> tuple[float, float, float]:
    if bets.empty:
        return np.nan, np.nan, np.nan
    pay_col = payout_col(target)
    race = bets.assign(_return=pd.to_numeric(bets[pay_col], errors="coerce").fillna(0), _stake=100.0).groupby("race_id")[["_return", "_stake"]].sum()
    returns = race["_return"].to_numpy(float)
    stakes = race["_stake"].to_numpy(float)
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(iterations):
        idx = rng.integers(0, len(race), size=len(race))
        rois.append(returns[idx].sum() / stakes[idx].sum() * 100)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 50)), float(np.percentile(rois, 97.5))


def select_alpha(oof: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    selected = {}
    for target in TARGETS:
        for alpha in (cfg["alpha_candidates"] if target == "win" else [1.0]):
            d = add_market_and_confidence(oof[oof["target"] == target], target, alpha)
            cand = classify_candidates(d, cfg, target)
            rows.append({"target": target, "alpha": alpha, **summarize_bets(cand, target)})
        best = pd.DataFrame([r for r in rows if r["target"] == target]).sort_values(["roi", "bets"], ascending=[False, False]).iloc[0]
        selected[target] = float(best["alpha"])
    return selected, pd.DataFrame(rows)


def select_rules(oof: pd.DataFrame, cfg: dict[str, Any], alpha_by_target: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_candidates = []
    rule_rows = []
    for target in TARGETS:
        prepared = add_market_and_confidence(oof[oof["target"] == target], target, alpha_by_target[target])
        candidates = classify_candidates(prepared, cfg, target)
        all_candidates.append(candidates)
        grids = []
        for strategy in cfg["rule_selection"]["strategies"]:
            for model_rank_max in ([1, 2, 3] if strategy != "core" else [1]):
                for ev_min in [1.00, 1.03, 1.08, 1.15, 1.25]:
                    for edge_min in [-1.0, 0.0, 0.02, 0.05]:
                        for margin_min in [0.0, 0.02, 0.05]:
                            for odds_min, odds_max in ([(1, 999), (1, 5), (5, 20), (20, 999)] if target == "win" else [(1, 999), (1, 2), (2, 5), (5, 999)]):
                                grids.append({
                                    "target": target, "strategy_type": strategy, "model_rank_max": model_rank_max,
                                    "ev_min": ev_min, "edge_min": edge_min, "margin_min": margin_min,
                                    "odds_min": odds_min, "odds_max": odds_max, "rank_gap_min": 0 if strategy == "core" else 1,
                                })
        min_bets = int(cfg["rule_selection"]["min_validation_bets"])
        for rule in grids:
            bets = apply_rule(candidates, pd.Series(rule))
            if len(bets) < min_bets:
                continue
            by_year = pd.DataFrame([summarize_bets(g, target, {"Year": y}) for y, g in bets.groupby("Year")])
            if by_year.empty:
                continue
            row = {
                **rule,
                **summarize_bets(bets, target),
                "year_roi_mean": float(by_year["roi"].mean()),
                "year_roi_min": float(by_year["roi"].min()),
                "year_roi_std": float(by_year["roi"].std(ddof=0)),
                "roi_remove_top1": top_payout_removed_roi(bets, target, 1),
                "roi_remove_top5": top_payout_removed_roi(bets, target, 5),
                "top1_profit_share": profit_share(bets, target, 0.01),
                "top5_profit_share": profit_share(bets, target, 0.05),
            }
            row["score"] = (
                row["year_roi_min"] * 10000
                + np.nan_to_num(row["roi_remove_top5"], nan=0) * 1000
                + row["roi"] * 100
                + min(row["bets"], 5000)
                - row["max_drawdown"] * 0.01
            )
            rule_rows.append(row)
        if not any(r["target"] == target for r in rule_rows):
            for relaxed_min_bets in [300, 100, 30]:
                fallback_rows = []
                relaxed_candidates = candidates
                if len(relaxed_candidates) < relaxed_min_bets:
                    relaxed_candidates = prepared[prepared["model_rank"] == 1].copy()
                    relaxed_candidates["strategy_type"] = "core"
                for strategy in cfg["rule_selection"]["strategies"]:
                    rule = {
                        "target": target,
                        "strategy_type": strategy,
                        "model_rank_max": 3 if strategy != "core" else 1,
                        "ev_min": 0.0,
                        "edge_min": -999.0,
                        "margin_min": 0.0,
                        "odds_min": 1.0,
                        "odds_max": 999.0,
                        "rank_gap_min": -999.0,
                    }
                    bets = apply_rule(relaxed_candidates, pd.Series(rule))
                    if len(bets) < relaxed_min_bets:
                        continue
                    by_year = pd.DataFrame([summarize_bets(g, target, {"Year": y}) for y, g in bets.groupby("Year")])
                    row = {
                        **rule,
                        **summarize_bets(bets, target),
                        "year_roi_mean": float(by_year["roi"].mean()),
                        "year_roi_min": float(by_year["roi"].min()),
                        "year_roi_std": float(by_year["roi"].std(ddof=0)),
                        "roi_remove_top1": top_payout_removed_roi(bets, target, 1),
                        "roi_remove_top5": top_payout_removed_roi(bets, target, 5),
                        "top1_profit_share": profit_share(bets, target, 0.01),
                        "top5_profit_share": profit_share(bets, target, 0.05),
                        "relaxed_min_validation_bets": relaxed_min_bets,
                    }
                    row["score"] = (
                        row["year_roi_min"] * 10000
                        + np.nan_to_num(row["roi_remove_top5"], nan=0) * 1000
                        + row["roi"] * 100
                        + min(row["bets"], 5000)
                        - row["max_drawdown"] * 0.01
                    )
                    fallback_rows.append(row)
                if fallback_rows:
                    rule_rows.extend(fallback_rows)
                    break
    candidates_all = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    rule_df = pd.DataFrame(rule_rows).sort_values(["target", "score"], ascending=[True, False]) if rule_rows else pd.DataFrame()
    selected = []
    if not rule_df.empty:
        for target, g in rule_df.groupby("target"):
            deduped = remove_similar_rules(g, candidates_all[candidates_all["target"] == target], float(cfg["rule_selection"]["jaccard_duplicate_threshold"]))
            selected.append(deduped.head(int(cfg["rule_selection"]["max_rules_per_target"])))
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return rule_df, selected_df, candidates_all


def evaluate_rules(prepared: pd.DataFrame, rules: pd.DataFrame, cfg: dict[str, Any], label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    dependency = []
    boot = []
    candidate_cache: dict[str, pd.DataFrame] = {}
    for idx, rule in rules.reset_index(drop=True).iterrows():
        target = rule["target"]
        rule_id = f"{target}_{rule['strategy_type']}_{idx+1:02d}"
        is_relaxed = "relaxed_min_validation_bets" in rule.index and pd.notna(rule.get("relaxed_min_validation_bets"))
        cache_key = f"{target}_relaxed" if is_relaxed else target
        if cache_key not in candidate_cache:
            base = prepared[prepared["target"] == target]
            if is_relaxed:
                c = base[base["model_rank"] == 1].copy()
                c["strategy_type"] = "core"
                candidate_cache[cache_key] = c
            else:
                candidate_cache[cache_key] = classify_candidates(base, cfg, target)
        bets = apply_rule(candidate_cache[cache_key], rule)
        if rule["strategy_type"] == "longshot" and len(bets):
            max_share = float(cfg["rule_selection"]["longshot_max_share"])
            max_bets = max(1, int(len(prepared[prepared["target"] == target]) * max_share))
            bets = bets.sort_values("ev", ascending=False).head(max_bets)
        bets = bets.copy()
        if "odds_band" not in bets.columns and not bets.empty:
            bets["odds_band"] = pd.cut(
                pd.to_numeric(bets[odds_col(target)], errors="coerce"),
                [0, 1.5, 3, 5, 10, 20, 999],
                labels=["<1.5", "1.5-3", "3-5", "5-10", "10-20", "20+"],
                include_lowest=True,
            ).astype(str)
        if "confidence_band" not in bets.columns and not bets.empty:
            bets["confidence_band"] = pd.cut(
                bets["conservative_probability"],
                [0, .05, .1, .2, .3, .5, .75, 1],
                labels=["<.05", ".05-.10", ".10-.20", ".20-.30", ".30-.50", ".50-.75", ".75+"],
                include_lowest=True,
            ).astype(str)
        bets["rule_id"] = rule_id
        bets["eval_period"] = label
        details.append(bets)
        summaries.append(summarize_bets(bets, target, {"rule_id": rule_id, "target": target, "strategy_type": rule["strategy_type"], "eval_period": label}))
        for n in [0, 1, 3, 5, 10]:
            dependency.append({"rule_id": rule_id, "target": target, "eval_period": label, "removed_top_payouts": n, "roi": top_payout_removed_roi(bets, target, n)})
        dependency.append({"rule_id": rule_id, "target": target, "eval_period": label, "removed_top_payouts": "dependency", "top1_profit_share": profit_share(bets, target, 0.01), "top5_profit_share": profit_share(bets, target, 0.05)})
        ci = bootstrap_ci(bets, target, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
        boot.append({"rule_id": rule_id, "target": target, "eval_period": label, "roi_p025": ci[0], "roi_p500": ci[1], "roi_p975": ci[2]})
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True) if details else pd.DataFrame(), pd.DataFrame(dependency), pd.DataFrame(boot)


def aggregate_strategy(details: pd.DataFrame, label: str) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in details.groupby(["target", "eval_period", "strategy_type"], dropna=False):
        rows.append(summarize_bets(g, keys[0], {"target": keys[0], "eval_period": keys[1], "strategy_type": keys[2]}))
    no_long = details[details["strategy_type"] != "longshot"]
    for keys, g in no_long.groupby(["target", "eval_period"], dropna=False):
        rows.append(summarize_bets(g, keys[0], {"target": keys[0], "eval_period": keys[1], "strategy_type": "without_longshot"}))
    return pd.DataFrame(rows)


def group_roi(details: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    rows = []
    for name, g in details.groupby(keys, dropna=False):
        vals = name if isinstance(name, tuple) else (name,)
        rows.append(summarize_bets(g, str(g["target"].iloc[0]), dict(zip(keys, vals))))
    return pd.DataFrame(rows)


def expected_fingerprint(cfg: dict[str, Any], params: dict[str, Any], code_hash: str) -> dict[str, Any]:
    return {
        "version": cfg["version"],
        "input_dataset_hash": dataset_hash(cfg),
        "feature_set_yaml_sha256": sha256_file(Path(cfg["feature_set_yaml"])),
        "config_hash": sha256_json(cfg),
        "params_hash": sha256_json(params),
        "code_hash": code_hash,
        "feature_set": cfg["feature_set"],
        "folds_hash": sha256_json(cfg["folds"]),
        "final_train_years": cfg["final_train_years"],
        "test_year": cfg["test_year"],
        "latest_holdout_year": cfg["latest_holdout_year"],
    }


def should_resume(out: Path, expected: dict[str, Any], strict: bool) -> bool:
    manifest = out / "manifest.json"
    if not manifest.exists():
        return False
    old = json.loads(manifest.read_text(encoding="utf-8"))
    if old.get("fingerprint") == expected and all((out / p).exists() for p in ["roi_summary.csv", "selected_rules.csv", "final_predictions.parquet"]):
        return True
    if strict:
        print("[resume] fingerprint mismatch; exit 2", flush=True)
        raise SystemExit(2)
    return False


def run(config_path: Path, smoke: bool, resume: bool, strict_resume: bool, force: bool) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["version"] = cfg["version"] + "_smoke"
        cfg["output_root"] = cfg["output_root"] + "_smoke"
        cfg["model_root"] = cfg["model_root"] + "_smoke"
        cfg["bootstrap_iterations"] = 100
        cfg["rule_selection"] = dict(cfg["rule_selection"])
        cfg["rule_selection"]["min_validation_bets"] = 10
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    params = resolve_params(cfg, smoke)
    code_files = [Path(__file__), config_path, Path(cfg["feature_set_yaml"])]
    code_hash = hashlib.sha256("".join(sha256_file(p) for p in code_files if p.exists()).encode()).hexdigest()
    fingerprint = expected_fingerprint(cfg, params, code_hash)
    if (resume or strict_resume) and not force and should_resume(out, fingerprint, strict_resume):
        print("[resume] existing outputs match; skipped", flush=True)
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    print("[final-odds] preflight/gpu", flush=True)
    gpu = gpu_smoke(params)
    numeric, cat = feature_columns(cfg)
    print("[final-odds] loading dataset", flush=True)
    df = load_dataset(cfg, numeric, cat, smoke)
    future_leakage = {"random_split_used": False, "years_min": int(df["Year"].min()), "years_max": int(df["Year"].max()), "test_used_for_selection": False}
    fold_meta = []
    oof_parts = []
    print("[final-odds] walk-forward training", flush=True)
    for target in TARGETS:
        tdf = target_frame(df, cfg, target)
        for fold in cfg["folds"]:
            train = tdf[tdf["Year"].isin(fold["train_years"])]
            valid = tdf[tdf["Year"] == fold["validation_year"]]
            print(f"[final-odds] train {target} {fold['name']} rows={len(train)}/{len(valid)}", flush=True)
            model, meta = train_model(train, valid, numeric, cat, params, model_root / "folds" / target / fold["name"] / "model.cbm")
            prob = predict(model, valid, numeric, cat)
            pred = valid[["entry_id", "race_id", "race_date", "Year", "month", "JyoCD", "TrackCD", "Kyori", "SyussoTosu", "Ninki", "TanNinki", "FukuNinki", "tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_pay", "fuku_pay", "distance_band", "field_size_band", "actual"]].copy()
            pred["raw_probability"] = prob
            pred["target"] = target
            pred["fold"] = fold["name"]
            oof_parts.append(pred)
            fold_meta.append({**meta, **metric_row(valid["actual"].to_numpy(int), prob, {"target": target, "fold": fold["name"], "validation_year": fold["validation_year"]})})
    oof = pd.concat(oof_parts, ignore_index=True)
    print("[final-odds] calibration/alpha/rules", flush=True)
    calibrators, cal_metrics = fit_calibration(oof, cfg)
    oof["calibrated_probability"] = np.nan
    for target in TARGETS:
        m = oof["target"] == target
        oof.loc[m, "calibrated_probability"] = calibrators[target].transform(oof.loc[m, "raw_probability"].to_numpy(float))
    alpha_by_target, alpha_metrics = select_alpha(oof, cfg)
    candidate_rules, selected_rules, validation_candidates = select_rules(oof, cfg, alpha_by_target)
    print("[final-odds] final train/predict 2025/2026", flush=True)
    final_parts = []
    final_meta = []
    for target in TARGETS:
        tdf = target_frame(df, cfg, target)
        train = tdf[tdf["Year"].isin(cfg["final_train_years"])]
        eval_df = tdf[tdf["Year"].isin([cfg["test_year"], cfg["latest_holdout_year"]])]
        valid_tail = tdf[tdf["Year"] == cfg["final_train_years"][-1]]
        model, meta = train_model(train, valid_tail, numeric, cat, params, model_root / "final" / target / "model.cbm")
        prob = predict(model, eval_df, numeric, cat)
        pred = eval_df[["entry_id", "race_id", "race_date", "Year", "month", "JyoCD", "TrackCD", "Kyori", "SyussoTosu", "Ninki", "TanNinki", "FukuNinki", "tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_pay", "fuku_pay", "distance_band", "field_size_band", "actual"]].copy()
        pred["raw_probability"] = prob
        pred["target"] = target
        pred["calibrated_probability"] = calibrators[target].transform(prob)
        pred["eval_period"] = np.where(pred["Year"] == cfg["test_year"], "test_2025", "latest_holdout_2026")
        final_parts.append(add_market_and_confidence(pred, target, alpha_by_target[target]))
        final_meta.append({**meta, "target": target, "train_years": cfg["final_train_years"], "eval_years": [cfg["test_year"], cfg["latest_holdout_year"]]})
    final_pred = pd.concat(final_parts, ignore_index=True)
    prepared_oof = pd.concat([add_market_and_confidence(oof[oof["target"] == t], t, alpha_by_target[t]) for t in TARGETS], ignore_index=True)
    validation_summary, validation_details, validation_dep, validation_boot = evaluate_rules(prepared_oof, selected_rules, cfg, "validation_2020_2024")
    test_summary, test_details, test_dep, test_boot = evaluate_rules(final_pred[final_pred["Year"] == cfg["test_year"]], selected_rules, cfg, "test_2025")
    latest_summary, latest_details, latest_dep, latest_boot = evaluate_rules(final_pred[final_pred["Year"] == cfg["latest_holdout_year"]], selected_rules, cfg, "latest_holdout_2026")
    combined_summary, combined_details, combined_dep, combined_boot = evaluate_rules(final_pred, selected_rules, cfg, "test_latest_combined")
    details = pd.concat([validation_details, test_details, latest_details, combined_details], ignore_index=True)
    roi_summary = pd.concat([validation_summary, test_summary, latest_summary, combined_summary], ignore_index=True)
    dependency = pd.concat([validation_dep, test_dep, latest_dep, combined_dep], ignore_index=True)
    bootstrap = pd.concat([validation_boot, test_boot, latest_boot, combined_boot], ignore_index=True)
    strategy_summary = aggregate_strategy(details, "all")
    outputs = {
        "fold_metrics.csv": pd.DataFrame(fold_meta),
        "final_model_metrics.csv": pd.DataFrame(final_meta),
        "calibration_metrics.csv": cal_metrics,
        "alpha_selection.csv": alpha_metrics,
        "candidate_rules.csv": candidate_rules,
        "selected_rules.csv": selected_rules,
        "roi_summary.csv": roi_summary,
        "strategy_summary.csv": strategy_summary,
        "payout_dependency.csv": dependency,
        "bootstrap_roi_ci.csv": bootstrap,
        "roi_by_year.csv": group_roi(details, ["target", "eval_period", "Year"]),
        "roi_by_month.csv": group_roi(details, ["target", "eval_period", "month"]),
        "roi_by_track.csv": group_roi(details, ["target", "eval_period", "JyoCD"]),
        "roi_by_odds_band.csv": group_roi(details, ["target", "eval_period", "odds_band"]),
        "roi_by_popularity_band.csv": group_roi(details, ["target", "eval_period", "Ninki"]),
        "roi_by_edge_band.csv": group_roi(details.assign(edge_band=pd.cut(details["edge"], [-np.inf, 0, .02, .05, .1, np.inf]).astype(str)) if not details.empty else details, ["target", "eval_period", "edge_band"]),
        "roi_by_rank_gap.csv": group_roi(details, ["target", "eval_period", "rank_gap"]),
        "roi_by_confidence_band.csv": group_roi(details, ["target", "eval_period", "confidence_band"]),
    }
    hashes = {}
    for name, table in outputs.items():
        hashes[name] = atomic_write_csv(out / name, table)
    hashes["oof_predictions.parquet"] = atomic_write_parquet(out / "oof_predictions.parquet", prepared_oof)
    hashes["final_predictions.parquet"] = atomic_write_parquet(out / "final_predictions.parquet", final_pred)
    hashes["bet_details.parquet"] = atomic_write_parquet(out / "bet_details.parquet", details)
    manifest = {
        "version": cfg["version"],
        "fingerprint": fingerprint,
        "ideal_condition_notice": cfg["ideal_condition_notice"],
        "gpu": gpu,
        "catboost_version": __import__("catboost").__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "feature_set": cfg["feature_set"],
        "folds": cfg["folds"],
        "alpha_by_target": alpha_by_target,
        "calibration_by_target": {k: v.method for k, v in calibrators.items()},
        "future_leakage_audit": future_leakage,
        "output_hashes": hashes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_docs(cfg, manifest, roi_summary, selected_rules, dependency, bootstrap, out)
    print("[final-odds] done", flush=True)
    return manifest


def write_docs(cfg: dict[str, Any], manifest: dict[str, Any], roi_summary: pd.DataFrame, selected_rules: pd.DataFrame, dependency: pd.DataFrame, bootstrap: pd.DataFrame, out: Path) -> None:
    design = [
        "# Final Odds Two Models V1 Design",
        "",
        cfg["ideal_condition_notice"],
        "",
        "- Input: `outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet`",
        "- Feature set: `market_aware`",
        "- Targets: `target_win_paid`, `target_place_paid`",
        "- Walk-forward: 2016-2019→2020, ..., 2016-2023→2024",
        "- Test/latest holdout: 2025 / 2026; not used for calibration or rule selection.",
        "- ROI uses actual `tan_pay` and `fuku_pay` with 100 yen flat stakes.",
    ]
    atomic_write_text(Path("docs/final_odds_two_models_v1_design.md"), "\n".join(design) + "\n")
    res = [
        "# Final Odds Two Models V1 Results",
        "",
        f"- Ideal condition: `{cfg['ideal_condition_notice']}`",
        f"- Feature set: `{cfg['feature_set']}`",
        f"- Calibration: `{manifest['calibration_by_target']}`",
        f"- Alpha: `{manifest['alpha_by_target']}`",
        f"- Elapsed seconds: `{manifest['elapsed_seconds']:.1f}`",
        "",
        "## Selected Rules",
        selected_rules.to_markdown(index=False) if not selected_rules.empty else "(none)",
        "",
        "## ROI Summary",
        roi_summary.to_markdown(index=False) if not roi_summary.empty else "(none)",
        "",
        "## Dependency",
        dependency.head(30).to_markdown(index=False) if not dependency.empty else "(none)",
        "",
        "## Bootstrap",
        bootstrap.to_markdown(index=False) if not bootstrap.empty else "(none)",
    ]
    atomic_write_text(Path("docs/final_odds_two_models_v1_results.md"), "\n".join(res) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_odds_two_models_v1.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(Path(args.config), args.smoke_test, args.resume, args.strict_resume, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
