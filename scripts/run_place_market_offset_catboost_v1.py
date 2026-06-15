from __future__ import annotations

import argparse
import hashlib
import json
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
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db_validation_cache import DEFAULT_DB_PATH, DatabaseValidationError, cache_paths, db_validation_fingerprint, load_manifest, load_validation_config, validate_or_require_full
from src.features.feature_sets_v2_1_2 import load_feature_set_yaml


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
        dirty_text = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return {"git_commit_sha": sha, "git_is_dirty": bool(dirty_text.strip()), "git_status_short": dirty_text.strip().splitlines()}
    except Exception as exc:
        return {"git_commit_sha": "unknown", "git_is_dirty": None, "git_error": str(exc)}


def gpu_name(devices: str | None) -> str:
    try:
        idx = (devices or "0").split(",")[0]
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={idx}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def gpu_smoke(params: dict[str, Any]) -> dict[str, Any]:
    if str(params.get("task_type", "")).upper() != "GPU":
        raise RuntimeError("GPU is required; CPU fallback is disabled")
    x = np.random.default_rng(42).normal(size=(128, 4))
    y = (x[:, 0] > 0).astype(int)
    model = CatBoostClassifier(
        iterations=3,
        loss_function="Logloss",
        task_type="GPU",
        devices=params.get("devices", "0"),
        verbose=False,
        allow_writing_files=False,
        random_seed=42,
    )
    model.fit(x, y)
    return {"task_type": "GPU", "devices": params.get("devices", "0"), "gpu_name": gpu_name(params.get("devices")), "catboost_gpu_smoke": "ok", "cpu_fallback_used": False}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_params(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    params = dict(cfg["training_params"])
    if smoke:
        params.update({k: v for k, v in cfg.get("smoke_overrides", {}).items() if k in params})
    return params


def dataset_hash(cfg: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for year in range(2016, 2027):
        p = Path(cfg["input_dataset_dir"]) / f"year={year}" / "data.parquet"
        h.update(str(p).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def feature_columns(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    sets = load_feature_set_yaml(Path(cfg["feature_set_yaml"]))
    mf = sets["market_free"]
    return list(mf.get("numeric", [])), list(mf.get("categorical", []))


def required_columns(cfg: dict[str, Any], numeric: list[str], cat: list[str]) -> list[str]:
    meta = [
        "race_id", "entry_id", "race_date", "Year", "JyoCD", "TrackCD", "Kyori", "SyussoTosu",
        "Ninki", "TanNinki", "FukuNinki", "tan_odds", "tan_ninki", "fuku_odds_low",
        "fuku_odds_high", "fuku_ninki", "tan_pay", "fuku_pay", "target_place_paid",
        "eligible_for_place_training", "race_is_finalized", "place_rank_limit",
    ]
    return sorted(set(meta + numeric + cat + cfg["market_baseline"]["features"]) - {"market_rank", "tan_rank", "fuku_odds_width", "log_tan_odds", "log_fuku_low", "log_fuku_high", "fuku_low_inverse", "fuku_mid_inverse", "fuku_low_to_race_min", "fuku_low_to_race_mean"})


def load_dataset(cfg: dict[str, Any], numeric: list[str], cat: list[str], smoke: bool) -> pd.DataFrame:
    cols = required_columns(cfg, numeric, cat)
    frames = []
    for year in range(2016, 2027):
        p = Path(cfg["input_dataset_dir"]) / f"year={year}" / "data.parquet"
        df = pd.read_parquet(p, columns=[c for c in cols if c])
        if smoke:
            n = cfg.get("smoke_overrides", {}).get("train_rows_per_year", 800 if year < 2020 else 400)
            if year >= 2020:
                n = cfg.get("smoke_overrides", {}).get("eval_rows_per_year", 400)
            df = df.head(int(n))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["race_date"] = pd.to_datetime(out["race_date"])
    out["actual_place"] = out[cfg["target_column"]].astype(int)
    return add_market_features(out)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["tan_odds", "tan_ninki", "fuku_odds_low", "fuku_odds_high", "fuku_ninki", "SyussoTosu", "place_rank_limit"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["market_rank"] = d["fuku_ninki"].fillna(d.groupby("race_id")["fuku_odds_low"].rank(method="first"))
    d["tan_rank"] = d["tan_ninki"].fillna(d.groupby("race_id")["tan_odds"].rank(method="first"))
    d["fuku_odds_width"] = d["fuku_odds_high"] - d["fuku_odds_low"]
    d["log_tan_odds"] = np.log(d["tan_odds"].clip(lower=1.0))
    d["log_fuku_low"] = np.log(d["fuku_odds_low"].clip(lower=1.0))
    d["log_fuku_high"] = np.log(d["fuku_odds_high"].clip(lower=1.0))
    mid = ((d["fuku_odds_low"] + d["fuku_odds_high"]) / 2).clip(lower=1.0)
    d["fuku_low_inverse"] = 1.0 / d["fuku_odds_low"].clip(lower=1.0)
    d["fuku_mid_inverse"] = 1.0 / mid
    race_min = d.groupby("race_id")["fuku_odds_low"].transform("min")
    race_mean = d.groupby("race_id")["fuku_odds_low"].transform("mean")
    d["fuku_low_to_race_min"] = d["fuku_odds_low"] / race_min.replace(0, np.nan)
    d["fuku_low_to_race_mean"] = d["fuku_odds_low"] / race_mean.replace(0, np.nan)
    return d


def target_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df[df[cfg["eligible_column"]] == True].copy()
    bad = out["actual_place"].eq(1) & (pd.to_numeric(out[cfg["payout_column"]], errors="coerce").fillna(0) <= 0)
    if bad.any():
        raise RuntimeError(f"positive place rows with missing payout: {int(bad.sum())}")
    return out


def prepare_x(df: pd.DataFrame, numeric: list[str], cat: list[str]) -> pd.DataFrame:
    x = df[numeric + cat].copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in cat:
        s = x[c].astype("object")
        x[c] = s.where(pd.notna(s), "__MISSING__").astype(str).replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    return x


def cat_indices(x: pd.DataFrame, cat: list[str]) -> list[int]:
    return [x.columns.get_loc(c) for c in cat]


def market_x(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x[c] = x[c].fillna(x[c].median())
    return x


def fit_market_model(train: pd.DataFrame, cfg: dict[str, Any]) -> Any:
    mb = cfg["market_baseline"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(mb.get("C", 1.0)), max_iter=int(mb.get("max_iter", 1000)), solver="lbfgs"),
    )
    model.fit(market_x(train, mb["features"]), train["actual_place"].to_numpy(int))
    return model


def heuristic_initial_market_probability(df: pd.DataFrame, eps: float) -> np.ndarray:
    slots = pd.to_numeric(df["place_rank_limit"], errors="coerce").fillna(3).clip(lower=1)
    field = pd.to_numeric(df["SyussoTosu"], errors="coerce").fillna(16).clip(lower=1)
    odds_p = 1.0 / pd.to_numeric(df["fuku_odds_low"], errors="coerce").fillna(99).clip(lower=1.0)
    base = (slots / field).clip(lower=eps, upper=1 - eps)
    return clip_prob(0.5 * odds_p + 0.5 * base, eps)


def expanding_market_predictions_for_train(train: pd.DataFrame, cfg: dict[str, Any], eps: float) -> pd.DataFrame:
    parts = []
    years = sorted(int(y) for y in train["Year"].unique())
    for year in years:
        cur = train[train["Year"] == year].copy()
        hist = train[train["Year"] < year]
        if hist.empty:
            p = heuristic_initial_market_probability(cur, eps)
            cur["baseline_source"] = "initial_market_heuristic_no_prior_year"
        else:
            model = fit_market_model(hist, cfg)
            p = clip_prob(model.predict_proba(market_x(cur, cfg["market_baseline"]["features"]))[:, 1], eps)
            cur["baseline_source"] = "time_series_train_oof"
        cur["p_market"] = p
        cur["market_logit"] = logit(p, eps)
        parts.append(cur)
    return pd.concat(parts, ignore_index=True)


def clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = clip_prob(p, eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def residual_features(cfg: dict[str, Any], mode: str, market_free_num: list[str], market_free_cat: list[str]) -> tuple[list[str], list[str]]:
    numeric = list(market_free_num) + ["p_market", "market_logit"]
    cat = list(market_free_cat)
    forbidden = {"tan_odds", "fuku_odds_low", "fuku_odds_high", "tan_ninki", "fuku_ninki", "TanNinki", "FukuNinki", "TanVote", "FukuVote"}
    numeric = [c for c in numeric if c not in forbidden]
    if mode == "limited_market":
        numeric += ["market_rank", "p_market_rank", "rank_gap", "SyussoTosu", "place_rank_limit"]
    return list(dict.fromkeys(numeric)), cat


def train_residual(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], cat: list[str], params: dict[str, Any], model_path: Path) -> tuple[CatBoostClassifier, dict[str, Any]]:
    x_train = prepare_x(train, numeric, cat)
    x_valid = prepare_x(valid, numeric, cat)
    cats = cat_indices(x_train, cat)
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(
        Pool(x_train, train["actual_place"].to_numpy(int), cat_features=cats, baseline=train["market_logit"].to_numpy(float)),
        eval_set=Pool(x_valid, valid["actual_place"].to_numpy(int), cat_features=cats, baseline=valid["market_logit"].to_numpy(float)),
        use_best_model=True,
    )
    elapsed = time.time() - started
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return model, {"training_seconds": elapsed, "best_iteration": int(model.get_best_iteration() or 0), "tree_count": int(model.tree_count_ or 0)}


def predict_residual(model: CatBoostClassifier, df: pd.DataFrame, numeric: list[str], cat: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = prepare_x(df, numeric, cat)
    pool = Pool(x, cat_features=cat_indices(x, cat), baseline=df["market_logit"].to_numpy(float))
    raw_with_baseline = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=float)
    prob = clip_prob(sigmoid(raw_with_baseline), 1e-6)
    return raw_with_baseline, prob


def raw_without_baseline(model: CatBoostClassifier, df: pd.DataFrame, numeric: list[str], cat: list[str]) -> np.ndarray:
    x = prepare_x(df, numeric, cat)
    return np.asarray(model.predict(Pool(x, cat_features=cat_indices(x, cat)), prediction_type="RawFormulaVal"), dtype=float)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.any():
            total += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / len(y))
    return float(total)


def mce(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    gaps = []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.any():
            gaps.append(abs(float(y[m].mean()) - float(p[m].mean())))
    return float(max(gaps) if gaps else np.nan)


def calibration_line(y: np.ndarray, p: np.ndarray, eps: float) -> tuple[float, float]:
    z = logit(p, eps).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(z, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def metric_row(df: pd.DataFrame, prob_col: str, label: dict[str, Any], eps: float) -> dict[str, Any]:
    y = df["actual_place"].to_numpy(int)
    p = clip_prob(df[prob_col].to_numpy(float), eps)
    slope, intercept = calibration_line(y, p, eps)
    return {
        **label,
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "ece": ece(y, p),
        "mce": mce(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


@dataclass
class Calibrator:
    method: str
    model: Any = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return clip_prob(p, 1e-6)
        if self.method == "platt":
            return clip_prob(self.model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6)
        if self.method == "isotonic":
            return clip_prob(self.model.predict(p), 1e-6)
        raise ValueError(self.method)


def fit_one_calibrator(method: str, train: pd.DataFrame, prob_col: str) -> Calibrator:
    y = train["actual_place"].to_numpy(int)
    p = train[prob_col].to_numpy(float)
    if method == "none":
        return Calibrator("none")
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(p.reshape(-1, 1), y)
        return Calibrator("platt", lr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(p, y)
    return Calibrator("isotonic", iso)


def nested_calibration(oof: pd.DataFrame, model_key: str, cfg: dict[str, Any]) -> tuple[str, pd.DataFrame, Calibrator]:
    rows = []
    prob_col = "probability"
    d = oof[oof["model_key"] == model_key].copy()
    for method in cfg["calibration_methods"]:
        for year in [2021, 2022, 2023, 2024]:
            train = d[d["Year"].between(2020, year - 1)]
            valid = d[d["Year"] == year]
            if train.empty or valid.empty:
                continue
            cal = fit_one_calibrator(method, train, prob_col)
            cp = cal.transform(valid[prob_col].to_numpy(float))
            tmp = valid.copy()
            tmp["calibrated_probability"] = cp
            rows.append(metric_row(tmp, "calibrated_probability", {"model_key": model_key, "method": method, "year": year}, float(cfg["epsilon"])))
    if not rows:
        return "none", pd.DataFrame(), Calibrator("none")
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("method", as_index=False).agg(mean_logloss=("logloss", "mean"), mean_brier=("brier", "mean"), mean_ece=("ece", "mean"), high_mce=("mce", "max"))
    summary["score"] = summary["mean_logloss"] + summary["mean_brier"] + summary["mean_ece"]
    selected = str(summary.sort_values(["score", "high_mce"]).iloc[0]["method"])
    final_cal = fit_one_calibrator(selected, d[d["Year"].between(2020, 2024)], prob_col)
    metrics = metrics.merge(summary[["method", "score"]], on="method", how="left")
    metrics["selected"] = metrics["method"].eq(selected)
    return selected, metrics, final_cal


ODDS_BINS = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, np.inf]
ODDS_LABELS = ["1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0+"]
EV_BINS = [-np.inf, .85, .90, .95, 1.00, 1.02, 1.05, 1.10, np.inf]
EV_LABELS = ["<0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00", "1.00-1.02", "1.02-1.05", "1.05-1.10", "1.10+"]


def add_eval_columns(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    d = df.copy()
    d["adjusted_place_ev"] = d[prob_col] * pd.to_numeric(d["fuku_odds_low"], errors="coerce")
    d["odds_band"] = pd.cut(pd.to_numeric(d["fuku_odds_low"], errors="coerce"), ODDS_BINS, labels=ODDS_LABELS, right=False).astype(str)
    d["ev_band"] = pd.cut(d["adjusted_place_ev"], EV_BINS, labels=EV_LABELS, right=False).astype(str)
    return d


def roi_of(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    return float(pd.to_numeric(df["fuku_pay"], errors="coerce").fillna(0).sum() / (len(df) * 100) * 100)


def summarize_bets(df: pd.DataFrame, label: dict[str, Any]) -> dict[str, Any]:
    pay = pd.to_numeric(df["fuku_pay"], errors="coerce").fillna(0).to_numpy(float) if len(df) else np.array([])
    hits = pay > 0
    profit = pay - 100
    equity = profit.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] if len(equity) else np.array([])
    cur = max_ls = 0
    for h in hits:
        cur = 0 if h else cur + 1
        max_ls = max(max_ls, cur)
    return {
        **label,
        "bets": int(len(df)),
        "return": float(pay.sum()) if len(pay) else 0.0,
        "roi": roi_of(df),
        "hit_rate": float(hits.mean()) if len(hits) else np.nan,
        "max_losing_streak": int(max_ls),
        "max_drawdown": float((peak - equity).max()) if len(equity) else 0.0,
    }


def odds_band_calibration(pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    d = add_eval_columns(pred, prob_col)
    for keys, g in d.groupby(["model_key", "period", "odds_band"], dropna=False):
        if g.empty:
            continue
        rows.append({
            "model_key": keys[0], "period": keys[1], "odds_band": keys[2], "rows": len(g),
            "mean_predicted_probability": g[prob_col].mean(),
            "actual_place_rate": g["actual_place"].mean(),
            "calibration_gap": g[prob_col].mean() - g["actual_place"].mean(),
            "logloss": log_loss(g["actual_place"], clip_prob(g[prob_col], 1e-6), labels=[0, 1]),
            "brier": brier_score_loss(g["actual_place"], g[prob_col]),
        })
    return pd.DataFrame(rows)


def ev_band_roi(pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    d = add_eval_columns(pred, prob_col)
    for keys, g in d.groupby(["model_key", "period", "ev_band"], dropna=False):
        if g.empty:
            continue
        rows.append({
            "model_key": keys[0], "period": keys[1], "ev_band": keys[2], "ev_band_order": EV_LABELS.index(str(keys[2])) if str(keys[2]) in EV_LABELS else -1,
            "bets": len(g), "actual_roi": roi_of(g), "hit_rate": g["actual_place"].mean(),
            "calibration_gap": g[prob_col].mean() - g["actual_place"].mean(),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ev_roi_spearman"] = out.groupby(["model_key", "period"])["actual_roi"].transform(lambda s: spearmanr(out.loc[s.index, "ev_band_order"], s, nan_policy="omit").statistic if len(s.dropna()) >= 2 else np.nan)
    return out


def threshold_roi(pred: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    d = add_eval_columns(pred, prob_col)
    for keys, g in d.groupby(["model_key", "period", "Year"], dropna=False):
        for th in [1.0, 1.05]:
            b = g[g["adjusted_place_ev"] >= th]
            rows.append(summarize_bets(b, {"model_key": keys[0], "period": keys[1], "Year": int(keys[2]), "threshold": f"EV>={th:.2f}"}))
    return pd.DataFrame(rows)


def strategy_roi(pred: pd.DataFrame, prob_col: str, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    d = add_eval_columns(pred, prob_col)
    for keys, g in d.groupby(["model_key", "period"], dropna=False):
        for rule in cfg["strategy_rules"]:
            odds = pd.to_numeric(g["fuku_odds_low"], errors="coerce")
            b = g[(odds >= float(rule["min_odds"])) & (odds < float(rule["max_odds"])) & (g["adjusted_place_ev"] >= float(rule["min_ev"]))]
            rows.append(summarize_bets(b, {"model_key": keys[0], "period": keys[1], "strategy": rule["name"]}))
    return pd.DataFrame(rows)


def top_removed_roi(df: pd.DataFrame, n: int) -> float:
    if len(df) <= n:
        return np.nan
    return roi_of(df.assign(_pay=pd.to_numeric(df["fuku_pay"], errors="coerce").fillna(0)).sort_values("_pay", ascending=False).iloc[n:])


def bootstrap_ci(df: pd.DataFrame, iterations: int, seed: int) -> tuple[float, float, float]:
    if df.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    races = df["race_id"].drop_duplicates().to_numpy()
    race_returns = df.groupby("race_id")["fuku_pay"].sum().reindex(races).fillna(0).to_numpy(float)
    race_bets = df.groupby("race_id").size().reindex(races).to_numpy(int)
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(races), len(races))
        stake = race_bets[idx].sum() * 100
        vals.append(race_returns[idx].sum() / stake * 100 if stake else np.nan)
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 50, 97.5]))


def comparison(pred: pd.DataFrame, prob_col: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    comp_rows = []
    evaled = add_eval_columns(pred, prob_col)
    ev_tbl = ev_band_roi(pred, prob_col)
    strat = strategy_roi(pred, prob_col, cfg)
    for keys, g in evaled.groupby(["model_key", "period"], dropna=False):
        metric_rows.append(metric_row(g, prob_col, {"model_key": keys[0], "period": keys[1]}, float(cfg["epsilon"])))
        sp = ev_tbl[(ev_tbl["model_key"] == keys[0]) & (ev_tbl["period"] == keys[1])]["ev_roi_spearman"]
        ev1 = g[g["adjusted_place_ev"] >= 1.0]
        ev105 = g[g["adjusted_place_ev"] >= 1.05]
        high = g[pd.to_numeric(g["fuku_odds_low"], errors="coerce") >= float(cfg["selection"]["high_odds_min"])]
        all_bets = g[g["adjusted_place_ev"] >= 1.0]
        ci = bootstrap_ci(all_bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
        comp_rows.append({
            "model_key": keys[0], "period": keys[1],
            "validation logloss": metric_rows[-1]["logloss"],
            "validation Brier": metric_rows[-1]["brier"],
            "validation ECE": metric_rows[-1]["ece"],
            "validation calibration slope": metric_rows[-1]["calibration_slope"],
            "high-odds calibration gap": high[prob_col].mean() - high["actual_place"].mean() if len(high) else np.nan,
            "EV-ROI Spearman": float(sp.iloc[0]) if len(sp) else np.nan,
            "EV>=1 count": int(len(ev1)),
            "EV>=1 ROI": roi_of(ev1),
            "EV>=1.05 count": int(len(ev105)),
            "EV>=1.05 ROI": roi_of(ev105),
            "2025 ROI": roi_of(g[(g["Year"] == 2025) & (g["adjusted_place_ev"] >= 1.0)]),
            "2026 ROI": roi_of(g[(g["Year"] == 2026) & (g["adjusted_place_ev"] >= 1.0)]),
            "combined ROI": roi_of(g[(g["Year"].isin([2025, 2026])) & (g["adjusted_place_ev"] >= 1.0)]),
            "top5 removed ROI": top_removed_roi(all_bets, 5),
            "bootstrap CI": f"{ci[0]:.2f},{ci[1]:.2f},{ci[2]:.2f}",
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(comp_rows)


def load_current_predictions(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    parts = []
    for p, period in [("oof_predictions.parquet", "validation_2020_2024"), ("final_predictions.parquet", "test_latest")]:
        df = pd.read_parquet(Path(cfg["current_predictions_dir"]) / p)
        df = df[df["target"] == "place"].copy()
        if p.startswith("oof"):
            df = df[df["Year"].between(2020, 2024)]
        else:
            df["period"] = np.where(df["Year"].eq(2025), "test_2025", "latest_holdout_2026")
        if "period" not in df.columns:
            df["period"] = period
        parts.append(df)
    cur = pd.concat(parts, ignore_index=True)
    if smoke:
        cur = cur.groupby("Year", group_keys=False).head(400)
    cur = cur.rename(columns={"calibrated_probability": "final_probability"})
    cur["actual_place"] = cur["actual"].astype(int)
    cur["model_key"] = "A_current_market_aware"
    keep = ["entry_id", "race_id", "race_date", "Year", "period", "actual_place", "final_probability", "fuku_odds_low", "fuku_pay", "model_key"]
    return cur[keep]


def make_market_predictions(tdf: pd.DataFrame, cfg: dict[str, Any], out: Path, model_root: Path, smoke: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    del smoke, out, model_root
    rows = []
    meta = []
    eps = float(cfg["epsilon"])
    for fold in cfg["folds"]:
        train = tdf[tdf["Year"].isin(fold["train_years"])].copy()
        valid = tdf[tdf["Year"] == fold["validation_year"]].copy()
        model = fit_market_model(train, cfg)
        train = expanding_market_predictions_for_train(train, cfg, eps)
        train["fold"] = fold["name"]
        train["baseline_scope"] = fold["name"]
        train["period"] = "residual_train"
        p = clip_prob(model.predict_proba(market_x(valid, cfg["market_baseline"]["features"]))[:, 1], eps)
        valid["p_market"] = p
        valid["market_logit"] = logit(p, eps)
        valid["fold"] = fold["name"]
        valid["baseline_scope"] = fold["name"]
        valid["period"] = "validation_2020_2024"
        valid["baseline_source"] = "time_series_oof"
        rows.extend([train, valid])
        meta.append(metric_row(valid, "p_market", {"model_key": "B_market_baseline", "fold": fold["name"], "validation_year": fold["validation_year"]}, eps))
    final_train = tdf[tdf["Year"].isin(cfg["final_train_years"])]
    final_eval = tdf[tdf["Year"].isin([cfg["test_year"], cfg["latest_holdout_year"]])].copy()
    final_model = fit_market_model(final_train, cfg)
    p = clip_prob(final_model.predict_proba(market_x(final_eval, cfg["market_baseline"]["features"]))[:, 1], eps)
    final_eval["p_market"] = p
    final_eval["market_logit"] = logit(p, eps)
    final_eval["fold"] = "final"
    final_eval["baseline_scope"] = "final"
    final_eval["period"] = np.where(final_eval["Year"].eq(cfg["test_year"]), "test_2025", "latest_holdout_2026")
    final_eval["baseline_source"] = "time_series_future_holdout"
    final_train = expanding_market_predictions_for_train(final_train.copy(), cfg, eps)
    final_train["fold"] = "final"
    final_train["baseline_scope"] = "final"
    final_train["period"] = "final_residual_train"
    rows.append(final_train)
    rows.append(final_eval)
    all_pred = pd.concat(rows, ignore_index=True)
    all_pred["p_market_rank"] = all_pred.groupby("race_id")["p_market"].rank(ascending=False, method="first")
    all_pred["rank_gap"] = all_pred["market_rank"] - all_pred["p_market_rank"]
    all_pred["probability"] = all_pred["p_market"]
    all_pred["final_probability"] = all_pred["p_market"]
    all_pred["model_key"] = "B_market_baseline"
    return all_pred, {"market_fold_metrics": meta}


def train_offset_models(market_pred: pd.DataFrame, cfg: dict[str, Any], params: dict[str, Any], market_free_num: list[str], market_free_cat: list[str], model_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_parts = []
    fold_rows = []
    consistency: dict[str, Any] = {}
    eps = float(cfg["epsilon"])
    for key, rcfg in cfg["residual_models"].items():
        numeric, cat = residual_features(cfg, rcfg["feature_mode"], market_free_num, market_free_cat)
        for fold in cfg["folds"]:
            scoped = market_pred[market_pred["baseline_scope"] == fold["name"]]
            train = scoped[scoped["Year"].isin(fold["train_years"])]
            valid = scoped[scoped["Year"] == fold["validation_year"]].copy()
            model, meta = train_residual(train, valid, numeric, cat, params, model_root / key / "folds" / fold["name"] / "model.cbm")
            raw, p = predict_residual(model, valid, numeric, cat)
            residual_raw = raw_without_baseline(model, valid, numeric, cat)
            valid["probability"] = p
            valid["final_probability_raw"] = raw
            valid["catboost_residual_score"] = residual_raw
            valid["final_probability"] = p
            valid["model_key"] = f"{key}_{rcfg['name']}"
            all_parts.append(valid)
            fold_rows.append({**meta, **metric_row(valid, "probability", {"model_key": valid["model_key"].iloc[0], "fold": fold["name"], "validation_year": fold["validation_year"]}, eps)})
            consistency[f"{key}_{fold['name']}"] = float(np.max(np.abs(raw - (valid["market_logit"].to_numpy(float) + residual_raw))))
        scoped = market_pred[market_pred["baseline_scope"] == "final"]
        train = scoped[scoped["Year"].isin(cfg["final_train_years"])]
        eval_df = scoped[scoped["Year"].isin([cfg["test_year"], cfg["latest_holdout_year"]])].copy()
        valid_tail = scoped[scoped["Year"] == cfg["final_train_years"][-1]]
        model, meta = train_residual(train, valid_tail, numeric, cat, params, model_root / key / "final" / "model.cbm")
        raw, p = predict_residual(model, eval_df, numeric, cat)
        residual_raw = raw_without_baseline(model, eval_df, numeric, cat)
        eval_df["probability"] = p
        eval_df["final_probability_raw"] = raw
        eval_df["catboost_residual_score"] = residual_raw
        eval_df["final_probability"] = p
        eval_df["model_key"] = f"{key}_{rcfg['name']}"
        all_parts.append(eval_df)
        fold_rows.append({**meta, "model_key": eval_df["model_key"].iloc[0], "fold": "final", "validation_year": cfg["test_year"]})
        consistency[f"{key}_final"] = float(np.max(np.abs(raw - (eval_df["market_logit"].to_numpy(float) + residual_raw))))
    return pd.concat(all_parts, ignore_index=True), pd.DataFrame(fold_rows), {"baseline_raw_consistency_max_abs": consistency}


def apply_calibration(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    parts = []
    cal_rows = []
    selected = {}
    for model_key in pred["model_key"].unique():
        if model_key == "A_current_market_aware":
            d = pred[pred["model_key"] == model_key].copy()
            d["probability"] = d["final_probability"]
            parts.append(d)
            selected[model_key] = "existing"
            continue
        sel, metrics, cal = nested_calibration(pred[pred["Year"].between(2020, 2024)], model_key, cfg)
        selected[model_key] = sel
        cal_rows.append(metrics)
        d = pred[pred["model_key"] == model_key].copy()
        d["final_probability"] = cal.transform(d["probability"].to_numpy(float))
        parts.append(d)
    return pd.concat(parts, ignore_index=True), pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame(), selected


def select_model(model_comparison: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    val = model_comparison[model_comparison["period"] == "validation_2020_2024"].copy()
    val = val[val["model_key"].str.startswith(("C1_", "C2_"))]
    val["spearman_score"] = val["EV-ROI Spearman"].fillna(-2)
    val["high_gap_abs"] = val["high-odds calibration gap"].abs()
    val["score"] = val["validation logloss"] + val["validation Brier"] + val["validation ECE"] + val["high_gap_abs"] - (val["spearman_score"] * 0.01)
    row = val.sort_values(["score", "validation logloss"]).iloc[0].to_dict()
    return {"selected_model_key": row["model_key"], "selection_years": cfg["selection"]["years"], "selection_basis": "2020-2024 probability metrics, high-odds calibration gap, EV-ROI Spearman; ROI not sole criterion", "row": row}


def expected_fingerprint(cfg: dict[str, Any], params: dict[str, Any], code_hash: str, db_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": cfg["version"],
        "input_feature_hash": dataset_hash(cfg),
        "split_hash": sha256_json(cfg["folds"]),
        "market_baseline_config_hash": sha256_json(cfg["market_baseline"]),
        "residual_feature_hash": sha256_json({"feature_set_yaml": sha256_file(Path(cfg["feature_set_yaml"])), "residual_models": cfg["residual_models"]}),
        "catboost_config_hash": sha256_json(params),
        "calibration_config_hash": sha256_json(cfg["calibration_methods"]),
        "payout_source_hash": dataset_hash(cfg),
        "db_validation_manifest_hash": sha256_json(db_validation),
        "code_hash": code_hash,
    }


def reusable_db_manifest_fallback(db_path: Path, db_validation_config: Path | str, exc: DatabaseValidationError) -> dict[str, Any]:
    cfg = load_validation_config(db_validation_config)
    manifest = load_manifest(db_path, cfg)
    if not manifest:
        raise exc
    paths = cache_paths(db_path, cfg)
    return {
        "status": "cache_manifest_reused_without_db_read",
        "cache_hit": False,
        "strict_cache_hit_blocked": True,
        "blocked_reasons": exc.reasons,
        "reason": "pipeline uses existing parquet outputs only; DB was not read because validation cache strict HIT was blocked",
        "manifest_path": str(paths["manifest"]),
        "db_path_hash": manifest.get("db_path_hash"),
        "db_light_fingerprint": manifest.get("light_fingerprint"),
        "db_full_sha256": manifest.get("full_file_sha256"),
        "integrity_checked_at": manifest.get("integrity_checked_at"),
        "validator_manifest_version": manifest.get("manifest_version"),
    }


def should_resume(out: Path, expected: dict[str, Any], strict: bool) -> bool:
    manifest = out / "manifest.json"
    required = ["market_baseline_oof.parquet", "residual_oof_predictions.parquet", "final_predictions_2025.parquet", "final_predictions_2026.parquet", "model_comparison.csv", "selected_model.json"]
    if not manifest.exists():
        return False
    old = json.loads(manifest.read_text(encoding="utf-8"))
    if old.get("fingerprint") == expected and all((out / p).exists() for p in required):
        return True
    if strict:
        print("[place-offset] strict resume fingerprint mismatch", flush=True)
        raise SystemExit(2)
    return False


def write_docs(cfg: dict[str, Any], manifest: dict[str, Any], model_comp: pd.DataFrame, roi_comp: pd.DataFrame, selected: dict[str, Any]) -> None:
    atomic_write_text(Path("docs/place_market_offset_catboost_v1_design.md"), "\n".join([
        "# Place Market Offset CatBoost V1 Design",
        "",
        "- Target: `target_place_paid` / all eligible runners; no 1.2-2.5 training filter.",
        "- Market baseline: time-series OOF logistic regression using final odds/rank market features.",
        "- Residual models: CatBoost Logloss with `market_logit` passed as Pool baseline for train, validation, and inference.",
        "- C1 excludes raw odds and uses market-free features plus `p_market` and `market_logit`.",
        "- C2 adds limited rank/field-size market deviation features.",
        "- Model/calibration selection uses only 2020-2024. 2025 and 2026 are fixed evaluation only.",
    ]) + "\n")
    lines = [
        "# Place Market Offset CatBoost V1 Results",
        "",
        f"- Selected model: `{selected['selected_model_key']}`",
        f"- Selection: {selected['selection_basis']}",
        f"- DB cache status: `{manifest['db_validation'].get('status')}`",
        f"- GPU: `{manifest['gpu'].get('gpu_name')}`",
        f"- CatBoost: `{manifest['catboost_version']}`",
        f"- Elapsed seconds: `{manifest['elapsed_seconds']:.1f}`",
        "",
        "## Model Comparison",
        model_comp.to_markdown(index=False) if not model_comp.empty else "(none)",
        "",
        "## ROI Comparison",
        roi_comp.to_markdown(index=False) if not roi_comp.empty else "(none)",
    ]
    atomic_write_text(Path("docs/place_market_offset_catboost_v1_results.md"), "\n".join(lines) + "\n")


def run(config_path: Path, smoke: bool, resume: bool, strict_resume: bool, force: bool, db_validation_config: Path | str, skip_db_validation: bool) -> dict[str, Any]:
    started = time.time()
    cfg = load_config(config_path)
    if smoke:
        cfg = dict(cfg)
        cfg["version"] += "_smoke"
        cfg["output_root"] += "_smoke"
        cfg["model_root"] += "_smoke"
        cfg["bootstrap_iterations"] = 100
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    params = resolve_params(cfg, smoke)
    db_path = Path(cfg.get("source_db_path", DEFAULT_DB_PATH))
    try:
        db_validation = validate_or_require_full(db_path, db_validation_config, skip=skip_db_validation)
        if not skip_db_validation:
            db_validation = db_validation_fingerprint(db_path, db_validation_config)
    except DatabaseValidationError as exc:
        print(f"[place-offset] DB validation strict HIT unavailable: {exc}", flush=True)
        db_validation = reusable_db_manifest_fallback(db_path, db_validation_config, exc)
        print("[place-offset] reusing existing DB validation manifest; DB will not be read", flush=True)
    code_hash = hashlib.sha256("".join(sha256_file(p) for p in [Path(__file__), config_path, Path(cfg["feature_set_yaml"])] if p.exists()).encode()).hexdigest()
    fingerprint = expected_fingerprint(cfg, params, code_hash, db_validation)
    if (resume or strict_resume) and not force and should_resume(out, fingerprint, strict_resume):
        print("[place-offset] resume hit", flush=True)
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    print("[place-offset] gpu preflight", flush=True)
    gpu = gpu_smoke(params)
    market_free_num, market_free_cat = feature_columns(cfg)
    print("[place-offset] loading dataset", flush=True)
    df = load_dataset(cfg, market_free_num, market_free_cat, smoke)
    tdf = target_frame(df, cfg)
    print("[place-offset] market baseline OOF/final", flush=True)
    market_pred, market_meta = make_market_predictions(tdf, cfg, out, model_root, smoke)
    print("[place-offset] residual CatBoost C1/C2", flush=True)
    residual_pred, residual_fold_metrics, residual_meta = train_offset_models(market_pred, cfg, params, market_free_num, market_free_cat, model_root)
    current = load_current_predictions(cfg, smoke)
    base_keep = ["entry_id", "race_id", "race_date", "Year", "period", "actual_place", "fuku_odds_low", "fuku_pay", "model_key", "probability", "final_probability"]
    market_eval = market_pred[market_pred["baseline_source"].isin(["time_series_oof", "time_series_future_holdout"])][base_keep + ["p_market", "market_logit", "fold", "baseline_source"]].copy()
    residual_eval = residual_pred[[c for c in residual_pred.columns if c in set(base_keep + ["p_market", "market_logit", "fold", "final_probability_raw", "catboost_residual_score"])]].copy()
    all_pred = pd.concat([current, market_eval, residual_eval], ignore_index=True, sort=False)
    print("[place-offset] nested calibration/evaluation", flush=True)
    all_pred, cal_metrics, selected_cal = apply_calibration(all_pred, cfg)
    metrics, model_comp = comparison(all_pred, "final_probability", cfg)
    odds_cal = odds_band_calibration(all_pred, "final_probability")
    ev_roi = ev_band_roi(all_pred, "final_probability")
    threshold = threshold_roi(all_pred, "final_probability")
    roi_comp = strategy_roi(all_pred, "final_probability", cfg)
    selected = select_model(model_comp, cfg)
    selected_model_key = selected["selected_model_key"]
    selected_preds = all_pred[all_pred["model_key"] == selected_model_key]
    dep_rows = []
    boot_rows = []
    for period, g in selected_preds.groupby("period"):
        bets = add_eval_columns(g, "final_probability")
        bets = bets[bets["adjusted_place_ev"] >= 1.0]
        for n in [1, 3, 5, 10]:
            dep_rows.append({"model_key": selected_model_key, "period": period, "removed_top_payouts": n, "roi": top_removed_roi(bets, n)})
        ci = bootstrap_ci(bets, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"]))
        boot_rows.append({"model_key": selected_model_key, "period": period, "roi_p025": ci[0], "roi_p500": ci[1], "roi_p975": ci[2]})
    fold_metrics = pd.concat([pd.DataFrame(market_meta["market_fold_metrics"]), residual_fold_metrics], ignore_index=True, sort=False)
    hashes = {}
    hashes["market_baseline_oof.parquet"] = atomic_write_parquet(out / "market_baseline_oof.parquet", market_pred[(market_pred["Year"].between(2020, 2024)) & (market_pred["baseline_source"].eq("time_series_oof"))])
    hashes["residual_oof_predictions.parquet"] = atomic_write_parquet(out / "residual_oof_predictions.parquet", residual_pred[residual_pred["Year"].between(2020, 2024)])
    hashes["final_predictions_2025.parquet"] = atomic_write_parquet(out / "final_predictions_2025.parquet", all_pred[all_pred["Year"] == 2025])
    hashes["final_predictions_2026.parquet"] = atomic_write_parquet(out / "final_predictions_2026.parquet", all_pred[all_pred["Year"] == 2026])
    for name, table in {
        "fold_metrics.csv": fold_metrics,
        "calibration_metrics.csv": cal_metrics,
        "odds_band_calibration.csv": odds_cal,
        "ev_band_roi.csv": ev_roi,
        "model_comparison.csv": model_comp,
        "roi_comparison.csv": roi_comp,
        "bootstrap_ci.csv": pd.DataFrame(boot_rows),
        "threshold_yearly.csv": threshold,
        "metrics.csv": metrics,
        "payout_dependency.csv": pd.DataFrame(dep_rows),
    }.items():
        hashes[name] = atomic_write_csv(out / name, table)
    atomic_write_json(out / "selected_model.json", selected)
    hashes["selected_model.json"] = sha256_file(out / "selected_model.json")
    manifest = {
        "version": cfg["version"],
        "fingerprint": fingerprint,
        "input_feature_hash": fingerprint["input_feature_hash"],
        "split_hash": fingerprint["split_hash"],
        "market_baseline_config_hash": fingerprint["market_baseline_config_hash"],
        "residual_feature_hash": fingerprint["residual_feature_hash"],
        "catboost_config_hash": fingerprint["catboost_config_hash"],
        "calibration_config_hash": fingerprint["calibration_config_hash"],
        "prediction_hash": sha256_json(hashes),
        "payout_source_hash": fingerprint["payout_source_hash"],
        "db_validation": db_validation,
        "git": git_info(),
        "gpu": gpu,
        "catboost_version": __import__("catboost").__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": cfg["random_seed"],
        "market_baseline_features": cfg["market_baseline"]["features"],
        "C1_features": residual_features(cfg, "fundamental", market_free_num, market_free_cat),
        "C2_features": residual_features(cfg, "limited_market", market_free_num, market_free_cat),
        "calibration_by_model": selected_cal,
        "selected_model": selected,
        "leakage_audit": {"random_split_used": False, "market_baseline_oof": True, "selection_years": [2020, 2021, 2022, 2023, 2024], "test_2025_used_for_selection": False, "latest_2026_used_for_selection": False},
        "catboost_baseline_checks": residual_meta,
        "output_hashes": hashes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "manifest.json", manifest)
    write_docs(cfg, manifest, model_comp, roi_comp, selected)
    print("[place-offset] done", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_catboost_v1.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-db-validation", action="store_true")
    parser.add_argument("--db-validation-config", default="config/database_validation.yaml")
    args = parser.parse_args()
    run(Path(args.config), args.smoke_test, args.resume, args.strict_resume, args.force, args.db_validation_config, args.skip_db_validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
