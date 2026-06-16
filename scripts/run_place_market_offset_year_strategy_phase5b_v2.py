from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_v1 import (  # noqa: E402
    add_market_features,
    cat_indices,
    clip_prob,
    expanding_market_predictions_for_train,
    fit_market_model,
    logit,
    market_x,
    prepare_x,
    sha256_file,
)


KEY_COLUMNS = ["entry_id", "race_id", "race_date", "Year"]
PRIMARY_YEARS = [2020, 2021, 2022, 2023, 2024]
DIAGNOSTIC_YEARS = [2025, 2026]
ALL_STRATEGIES = [
    "LEGACY_2016",
    "WARMUP_2006_TRAIN_2016",
    "EXPANDING_FULL_2006",
    "ROLLING_10Y",
    "ROLLING_15Y",
    "FULL_2006_TIME_DECAY_HL5",
    "FULL_2006_TIME_DECAY_HL10",
]


@dataclass(frozen=True)
class FoldWindow:
    strategy: str
    validation_year: int
    history_start_year: int
    model_train_start_year: int
    train_start_year: int
    train_end_year: int
    train_years: tuple[int, ...]
    half_life_years: float | None = None
    market_mode: str = "ALIGNED_STRATEGY"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "status_short": run(["status", "--short"]).splitlines(),
        "diff_stat": run(["diff", "--stat"]).splitlines(),
    }


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_safe_catboost_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(cfg["catboost"])
    params["iterations"] = 300
    params["use_best_model"] = False
    params.pop("od_type", None)
    params.pop("od_wait", None)
    params.pop("early_stopping_rounds", None)
    return params


def assert_safe_catboost_params(params: dict[str, Any]) -> None:
    if int(params.get("iterations", -1)) != 300:
        raise ValueError("Phase 5B requires iterations=300")
    if params.get("use_best_model") is not False:
        raise ValueError("Phase 5B requires use_best_model=False")
    forbidden = {"od_type", "od_wait", "early_stopping_rounds"}
    present = sorted(forbidden.intersection(params))
    if present:
        raise ValueError(f"Early stopping parameters must be removed: {present}")


def load_feature_allowlist(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    allow = json.loads(Path(cfg["feature_allowlist_path"]).read_text(encoding="utf-8"))
    drops = set(cfg.get("feature_drops", []))
    numeric = [c for c in allow["numeric"] if c not in drops]
    categorical = [c for c in allow["categorical"] if c not in drops]
    forbidden = {
        "Year",
        "p_market",
        "market_logit",
        "tan_odds",
        "fuku_odds_low",
        "fuku_odds_high",
        "tan_ninki",
        "fuku_ninki",
        "TanNinki",
        "FukuNinki",
        "Ninki",
    }
    bad = sorted((set(numeric) | set(categorical)).intersection(forbidden))
    if bad:
        raise ValueError(f"Forbidden feature columns in Phase 5B allowlist: {bad}")
    return numeric, categorical


def strategy_by_name(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for s in cfg["strategies"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


def build_fold_window(strategy: dict[str, Any], validation_year: int) -> FoldWindow:
    mode = strategy["mode"]
    start = int(strategy["model_train_start_year"])
    if mode in {"legacy_compat", "expanding"}:
        train_start = start
    elif mode == "rolling":
        train_start = max(start, validation_year - int(strategy["rolling_years"]))
    else:
        raise ValueError(f"Unsupported strategy mode: {mode}")
    train_end = validation_year - 1
    if train_end < train_start:
        raise ValueError(f"Empty train window for {strategy['name']} {validation_year}")
    years = tuple(range(train_start, train_end + 1))
    market_mode = "LEGACY_COMPAT" if mode == "legacy_compat" else "ALIGNED_STRATEGY"
    return FoldWindow(
        strategy=strategy["name"],
        validation_year=int(validation_year),
        history_start_year=int(strategy["history_start_year"]),
        model_train_start_year=start,
        train_start_year=train_start,
        train_end_year=train_end,
        train_years=years,
        half_life_years=float(strategy["half_life_years"]) if strategy.get("half_life_years") is not None else None,
        market_mode=market_mode,
    )


def build_windows(cfg: dict[str, Any], strategies: list[str], years: list[int]) -> list[FoldWindow]:
    return [build_fold_window(strategy_by_name(cfg, s), y) for s in strategies for y in years]


def load_history_dataset(cfg: dict[str, Any], history_start_year: int | None = None, smoke_rows_per_year: int | None = None) -> pd.DataFrame:
    path = Path(cfg["history_dataset_path"])
    df = pd.read_parquet(path)
    if history_start_year is not None:
        df = df[df["Year"].ge(history_start_year)].copy()
    df = apply_target_column(df, cfg)
    df = add_market_features(df)
    df = df[df[cfg["eligible_column"]].eq(True)].copy()
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    if smoke_rows_per_year:
        df = df.sort_values(["race_date", "race_id", "Umaban", "entry_id"]).groupby("Year", group_keys=False).head(int(smoke_rows_per_year)).copy()
    return df


def apply_target_column(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    target = cfg["target_column"]
    if target not in df.columns:
        raise KeyError(f"Configured target_column is missing: {target}")
    out = df.copy()
    out["actual_place"] = out[target].astype(int)
    return out


def make_market_logit_for_fold(
    df: pd.DataFrame,
    window: FoldWindow,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eps = float(cfg["epsilon"])
    train = df[df["Year"].isin(window.train_years)].copy()
    valid = df[df["Year"].eq(window.validation_year)].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Empty train/validation frame for {window.strategy} {window.validation_year}")

    base_cfg = load_yaml(Path(cfg["base_c1r0_config"]))
    market_cfg = base_cfg["market_baseline"]

    if window.market_mode == "LEGACY_COMPAT":
        train = expanding_market_predictions_for_train(train, base_cfg, eps)
        train["baseline_source"] = train["baseline_source"].astype(str).radd("legacy_compat_")
    else:
        model = fit_market_model(train, base_cfg)
        train["p_market"] = clip_prob(model.predict_proba(market_x(train, market_cfg["features"]))[:, 1], eps)
        train["market_logit"] = logit(train["p_market"], eps)
        train["baseline_source"] = "aligned_train_in_sample_market_model"

    model = fit_market_model(train, base_cfg)
    valid["p_market"] = clip_prob(model.predict_proba(market_x(valid, market_cfg["features"]))[:, 1], eps)
    valid["market_logit"] = logit(valid["p_market"], eps)
    valid["baseline_source"] = f"{window.market_mode.lower()}_validation_market_model"

    provenance = {
        "strategy": window.strategy,
        "validation_year": window.validation_year,
        "market_mode": window.market_mode,
        "market_train_start": window.train_start_year,
        "market_train_end": window.train_end_year,
        "market_train_rows": int(len(train)),
        "market_target": "actual_place",
        "market_input_columns": list(market_cfg["features"]),
        "market_model_config": {k: v for k, v in market_cfg.items() if k != "features"},
        "residual_train_start": window.train_start_year,
        "residual_train_end": window.train_end_year,
        "residual_train_rows": int(len(train)),
    }
    return train, valid, provenance


def time_decay_weights(train: pd.DataFrame, validation_year: int, half_life_years: float | None) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if half_life_years is None:
        return None, None
    validation_start = pd.Timestamp(f"{validation_year}-01-01")
    race_dates = pd.to_datetime(train["race_date"])
    age_years = np.maximum(0.0, (validation_start - race_dates).dt.days.to_numpy(float) / 365.25)
    raw = np.power(2.0, -age_years / float(half_life_years))
    weights = raw / raw.mean()
    ess = float((weights.sum() ** 2) / np.square(weights).sum())
    summary = {
        "half_life_years": float(half_life_years),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "p1": float(np.percentile(weights, 1)),
        "p50": float(np.percentile(weights, 50)),
        "p99": float(np.percentile(weights, 99)),
        "max": float(weights.max()),
        "effective_sample_size": ess,
    }
    return weights, summary


def train_residual_model(
    train: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    params: dict[str, Any],
    model_path: Path,
    sample_weight: np.ndarray | None = None,
) -> CatBoostClassifier:
    assert_safe_catboost_params(params)
    x_train = prepare_x(train, numeric, categorical)
    pool = Pool(
        x_train,
        label=train["actual_place"].to_numpy(int),
        weight=sample_weight,
        cat_features=cat_indices(x_train, categorical),
        baseline=train["market_logit"].to_numpy(float),
    )
    model = CatBoostClassifier(**params)
    model.fit(pool)
    if int(model.tree_count_) != 300:
        raise RuntimeError(f"Unexpected tree_count={model.tree_count_}; expected 300")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    return model


def predict_raw(model: CatBoostClassifier, valid: pd.DataFrame, numeric: list[str], categorical: list[str], eps: float) -> pd.DataFrame:
    x_valid = prepare_x(valid, numeric, categorical)
    cats = cat_indices(x_valid, categorical)
    raw = np.asarray(
        model.predict(Pool(x_valid, cat_features=cats, baseline=valid["market_logit"].to_numpy(float)), prediction_type="RawFormulaVal"),
        dtype=float,
    )
    residual = np.asarray(model.predict(Pool(x_valid, cat_features=cats), prediction_type="RawFormulaVal"), dtype=float)
    out = valid.copy()
    out["final_logit"] = raw
    out["probability_raw"] = clip_prob(1.0 / (1.0 + np.exp(-raw)), eps)
    out["catboost_residual_score"] = residual
    return out


def expected_fold_paths(cfg: dict[str, Any], strategy: str, validation_year: int) -> tuple[Path, Path, Path]:
    out = Path(cfg["output_root"])
    model_root = Path(cfg["model_root"])
    stem = f"{strategy}_validation_{validation_year}"
    return (
        model_root / strategy / f"validation_{validation_year}" / "model.cbm",
        out / "predictions" / strategy / f"validation_{validation_year}.parquet",
        out / "fold_manifests" / f"{stem}.json",
    )


def fold_signature(cfg: dict[str, Any], window: FoldWindow, numeric: list[str], categorical: list[str], train: pd.DataFrame, valid: pd.DataFrame, weights: np.ndarray | None) -> dict[str, Any]:
    return {
        "version": cfg["version"],
        "strategy": window.strategy,
        "validation_year": window.validation_year,
        "train_years": list(window.train_years),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_key_hash": sha256_json(train[KEY_COLUMNS].sort_values(KEY_COLUMNS).astype(str).to_dict("list")),
        "validation_key_hash": sha256_json(valid[KEY_COLUMNS].sort_values(KEY_COLUMNS).astype(str).to_dict("list")),
        "feature_hash": sha256_json({"numeric": numeric, "categorical": categorical}),
        "sample_weight_hash": sha256_json({"weights": np.round(weights, 12).tolist()}) if weights is not None else "none",
        "catboost_params_hash": sha256_json(make_safe_catboost_params(cfg)),
    }


def reusable_fold(manifest_path: Path, prediction_path: Path, model_path: Path, signature: dict[str, Any]) -> bool:
    if not manifest_path.exists() or not prediction_path.exists() or not model_path.exists():
        return False
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("signature") == signature and old.get("status") == "success"


def run_fold(cfg: dict[str, Any], window: FoldWindow, df: pd.DataFrame, numeric: list[str], categorical: list[str], resume: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, valid, market_prov = make_market_logit_for_fold(df, window, cfg)
    if valid["Year"].iloc[0] in set(train["Year"].unique()):
        raise RuntimeError("Outer validation year leaked into train rows")
    weights, weight_summary = time_decay_weights(train, window.validation_year, window.half_life_years)
    model_path, prediction_path, manifest_path = expected_fold_paths(cfg, window.strategy, window.validation_year)
    signature = fold_signature(cfg, window, numeric, categorical, train, valid, weights)

    if resume and reusable_fold(manifest_path, prediction_path, model_path, signature):
        pred = pd.read_parquet(prediction_path)
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["action"] = "reuse"
        return pred, meta

    if prediction_path.exists() or manifest_path.exists() or model_path.exists():
        raise FileExistsError(f"Existing fold artifact found without reusable resume: {prediction_path}")

    params = make_safe_catboost_params(cfg)
    model = train_residual_model(train, numeric, categorical, params, model_path, weights)
    pred = predict_raw(model, valid, numeric, categorical, float(cfg["epsilon"]))
    pred["strategy"] = window.strategy
    pred["validation_year"] = window.validation_year
    pred["probability_used_for_selection"] = pred["probability_raw"]
    pred["tree_count"] = int(model.tree_count_)
    pred["history_start_year"] = window.history_start_year
    pred["model_train_start_year"] = window.model_train_start_year
    pred["train_start_year"] = window.train_start_year
    pred["train_end_year"] = window.train_end_year
    pred["half_life_years"] = window.half_life_years

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(prediction_path, index=False)

    meta = {
        "status": "success",
        "action": "train",
        "signature": signature,
        "strategy": window.strategy,
        "validation_year": window.validation_year,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
        "model_sha256": sha256_file(model_path),
        "market_provenance": market_prov,
        "residual_provenance": {
            "strategy": window.strategy,
            "validation_year": window.validation_year,
            "residual_train_start": window.train_start_year,
            "residual_train_end": window.train_end_year,
            "residual_train_rows": int(len(train)),
            "feature_columns_numeric": numeric,
            "feature_columns_categorical": categorical,
        },
        "sample_weight_summary": weight_summary,
        "catboost_safety": {
            "eval_set_used": False,
            "iterations": 300,
            "use_best_model": False,
            "early_stopping_enabled": False,
            "calibration_fit": False,
            "probability_for_selection": "probability_raw",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return pred, meta


def fixed_bin_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return math.nan
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece


def calibration_line(y: np.ndarray, p: np.ndarray, eps: float) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    x = np.log(clip_prob(p, eps) / (1.0 - clip_prob(p, eps))).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        lr.fit(x, y.astype(int))
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        reg = LinearRegression().fit(x, y)
        return float(reg.coef_[0]), float(reg.intercept_)


def metrics_for_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    y = df["actual_place"].to_numpy(int)
    p = df["probability_raw"].to_numpy(float)
    eps = float(cfg["epsilon"])
    slope, intercept = calibration_line(y, p, eps)
    top = df.sort_values(["race_id", "probability_raw"], ascending=[True, False]).groupby("race_id").head(1)
    race_spearman = df.groupby("race_id").apply(
        lambda g: g["probability_raw"].corr(g["actual_place"], method="spearman") if g["actual_place"].nunique() > 1 else np.nan
    )
    return {
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": fixed_bin_ece(y, p, 10),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "race_wise_spearman": float(race_spearman.mean(skipna=True)),
        "top_probability_hit_rate": float(top["actual_place"].mean()) if len(top) else math.nan,
    }


def residual_metrics(df: pd.DataFrame) -> dict[str, Any]:
    r = df["catboost_residual_score"].to_numpy(float)
    return {
        "residual_mean": float(np.mean(r)),
        "residual_std": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
        "abs_residual_p90": float(np.percentile(np.abs(r), 90)),
        "abs_residual_p95": float(np.percentile(np.abs(r), 95)),
        "abs_residual_p99": float(np.percentile(np.abs(r), 99)),
    }


def ev_picks(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    d["ev"] = d["probability_raw"] * pd.to_numeric(d[cfg["odds_column"]], errors="coerce")
    return d[d["ev"].ge(1.0)].copy()


def roi_value(picks: pd.DataFrame, cfg: dict[str, Any]) -> float:
    if picks.empty:
        return math.nan
    return float(pd.to_numeric(picks[cfg["payout_column"]], errors="coerce").fillna(0).sum() / (len(picks) * float(cfg["stake_yen"])) * 100.0)


def stress_roi_rows(df: pd.DataFrame, cfg: dict[str, Any], limits: list[int] = [1, 3, 5, 10]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    picks = ev_picks(df, cfg)
    normal_roi = roi_value(picks, cfg)
    normal = pd.DataFrame([{
        "strategy": df["strategy"].iloc[0] if len(df) else None,
        "Year": int(df["Year"].iloc[0]) if len(df) else None,
        "bet_count": int(len(picks)),
        "stake": int(len(picks) * int(cfg["stake_yen"])),
        "payout": float(pd.to_numeric(picks[cfg["payout_column"]], errors="coerce").fillna(0).sum()) if len(picks) else math.nan,
        "roi": normal_roi,
    }])
    row_removed, payout_zeroed = [], []
    hits = picks[picks["actual_place"].eq(1)].copy().sort_values(cfg["payout_column"], ascending=False)
    for limit in limits:
        removed_idx = hits.head(limit).index
        rr = picks.drop(index=removed_idx)
        pz = picks.copy()
        pz.loc[removed_idx, cfg["payout_column"]] = 0
        rr_roi = roi_value(rr, cfg)
        pz_roi = roi_value(pz, cfg)
        if not math.isnan(normal_roi) and not math.isnan(pz_roi) and pz_roi > normal_roi + 1e-12:
            raise RuntimeError("payout_zeroed_stress_roi exceeded normal_roi")
        base = {
            "strategy": df["strategy"].iloc[0] if len(df) else None,
            "Year": int(df["Year"].iloc[0]) if len(df) else None,
            "limit": limit,
            "normal_roi": normal_roi,
            "removed_count": int(len(removed_idx)),
        }
        row_removed.append({**base, "bet_count": int(len(rr)), "stake": int(len(rr) * int(cfg["stake_yen"])), "roi": rr_roi})
        payout_zeroed.append({**base, "bet_count": int(len(pz)), "stake": int(len(pz) * int(cfg["stake_yen"])), "roi": pz_roi})
    return normal, pd.DataFrame(row_removed), pd.DataFrame(payout_zeroed)


def summarize_predictions(pred: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    metrics_rows, residual_rows = [], []
    roi_rows, rr_rows, pz_rows = [], [], []
    for (strategy, year), g in pred.groupby(["strategy", "Year"]):
        label = {"strategy": strategy, "Year": int(year)}
        metrics_rows.append({**label, **metrics_for_frame(g, cfg)})
        residual_rows.append({**label, **residual_metrics(g)})
        normal, rr, pz = stress_roi_rows(g, cfg)
        roi_rows.append(normal)
        rr_rows.append(rr)
        pz_rows.append(pz)
    metrics = pd.DataFrame(metrics_rows)
    primary = metrics[metrics["Year"].isin(PRIMARY_YEARS)].groupby("strategy", as_index=False).agg(
        mean_logloss=("logloss", "mean"),
        mean_brier=("brier", "mean"),
        mean_ece=("ece", "mean"),
        worst_year_logloss=("logloss", "max"),
        worst_year_brier=("brier", "max"),
        best_year_logloss=("logloss", "min"),
        logloss_std=("logloss", "std"),
        brier_std=("brier", "std"),
        win_years_vs_legacy=("logloss", "count"),
    )
    aux = metrics[metrics["Year"].isin([int(y) for y in cfg.get("auxiliary_years", [])])].copy()
    diagnostic = metrics[metrics["Year"].isin([int(y) for y in cfg.get("diagnostic_years", [])])].copy()
    legacy = metrics[metrics["strategy"].eq("LEGACY_2016")][["Year", "logloss", "brier"]].rename(
        columns={"logloss": "legacy_logloss", "brier": "legacy_brier"}
    )
    yearly_win_loss = metrics.merge(legacy, on="Year", how="left")
    yearly_win_loss["delta_logloss_vs_legacy"] = yearly_win_loss["logloss"] - yearly_win_loss["legacy_logloss"]
    yearly_win_loss["delta_brier_vs_legacy"] = yearly_win_loss["brier"] - yearly_win_loss["legacy_brier"]
    yearly_win_loss["candidate_beats_legacy_logloss"] = yearly_win_loss["delta_logloss_vs_legacy"].lt(0)
    yearly_win_loss["candidate_beats_legacy_brier"] = yearly_win_loss["delta_brier_vs_legacy"].lt(0)
    worst_rows = []
    for strategy, g in metrics[metrics["Year"].isin(PRIMARY_YEARS)].groupby("strategy"):
        ll = g.loc[g["logloss"].idxmax()]
        br = g.loc[g["brier"].idxmax()]
        worst_rows.append({
            "strategy": strategy,
            "worst_logloss_year": int(ll["Year"]),
            "worst_logloss": float(ll["logloss"]),
            "worst_brier_year": int(br["Year"]),
            "worst_brier": float(br["brier"]),
            "logloss_cv": float(g["logloss"].std(ddof=1) / g["logloss"].mean()) if len(g) > 1 and g["logloss"].mean() else math.nan,
            "brier_cv": float(g["brier"].std(ddof=1) / g["brier"].mean()) if len(g) > 1 and g["brier"].mean() else math.nan,
        })
    return {
        "metrics_by_strategy_fold": metrics,
        "metrics_by_strategy_2020_2024": primary,
        "metrics_by_strategy_2016_2019_aux": aux,
        "yearly_win_loss_matrix": yearly_win_loss,
        "worst_year_summary": pd.DataFrame(worst_rows),
        "residual_stability_by_strategy": pd.DataFrame(residual_rows),
        "roi_diagnostic_raw": pd.concat(roi_rows, ignore_index=True) if roi_rows else pd.DataFrame(),
        "roi_row_removed_raw": pd.concat(rr_rows, ignore_index=True) if rr_rows else pd.DataFrame(),
        "roi_payout_zeroed_stress_raw": pd.concat(pz_rows, ignore_index=True) if pz_rows else pd.DataFrame(),
        "phase5b_2025_2026_diagnostic": diagnostic,
    }


def load_legacy_base(cfg: dict[str, Any], years: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(cfg["legacy_base_predictions_path"])
    df = df[df["model_key"].eq(cfg["official_base_model_key"]) & df["Year"].isin(years)].copy()
    df["probability_raw"] = df["probability"]
    return df


def model_tree_count(path: Path) -> int | None:
    if not path.exists():
        return None
    model = CatBoostClassifier()
    model.load_model(str(path))
    return int(model.tree_count_)


def legacy_model_path(cfg: dict[str, Any], validation_year: int) -> Path:
    return Path(cfg["legacy_base_model_root"]) / f"fold_{validation_year}" / "model.cbm"


def normalize_parity_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in KEY_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing parity key columns: {missing}")
    out["race_date"] = pd.to_datetime(out["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["Year"] = pd.to_numeric(out["Year"], errors="raise").astype("int64")
    return out


def parity_key_dtype_audit(old_raw: pd.DataFrame, new_raw: pd.DataFrame, old_norm: pd.DataFrame, new_norm: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for phase, old, new in [("raw", old_raw, new_raw), ("normalized", old_norm, new_norm)]:
        for col in KEY_COLUMNS:
            rows.append(
                {
                    "phase": phase,
                    "column": col,
                    "old_dtype": str(old[col].dtype),
                    "new_dtype": str(new[col].dtype),
                    "old_null_count": int(old[col].isna().sum()),
                    "new_null_count": int(new[col].isna().sum()),
                    "old_unique_count": int(old[col].nunique(dropna=False)),
                    "new_unique_count": int(new[col].nunique(dropna=False)),
                }
            )
    return rows


def validate_parity_keys(df: pd.DataFrame, label: str) -> None:
    null_counts = df[KEY_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts.gt(0)]
    if not null_counts.empty:
        raise ValueError(f"{label} parity keys contain nulls: {null_counts.to_dict()}")
    duplicate_count = int(df.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        examples = df.loc[df.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"{label} parity keys contain duplicates: count={duplicate_count}, examples={examples}")
    unique_keys = int(df[KEY_COLUMNS].drop_duplicates().shape[0])
    if unique_keys != len(df):
        raise ValueError(f"{label} parity key uniqueness mismatch: rows={len(df)} unique_keys={unique_keys}")


def parity_gate(
    legacy_pred: pd.DataFrame,
    new_pred: pd.DataFrame,
    cfg: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    reference_mode: str = "historical",
) -> pd.DataFrame:
    rows = []
    tolerances = cfg["parity_tolerance"]
    for year in sorted(new_pred["Year"].unique()):
        old_raw = legacy_pred[legacy_pred["Year"].eq(year)].copy()
        new_raw = new_pred[(new_pred["strategy"].eq("LEGACY_2016")) & new_pred["Year"].eq(year)].copy()
        old = normalize_parity_keys(old_raw).sort_values(KEY_COLUMNS)
        new = normalize_parity_keys(new_raw).sort_values(KEY_COLUMNS)
        key_audit = parity_key_dtype_audit(old_raw, new_raw, old, new)
        validate_parity_keys(old, "old")
        validate_parity_keys(new, "new")
        merged = old[KEY_COLUMNS + ["actual_place", "market_logit", "probability_raw"]].merge(
            new[KEY_COLUMNS + ["actual_place", "market_logit", "probability_raw"]],
            on=KEY_COLUMNS,
            how="outer",
            indicator=True,
            validate="one_to_one",
            suffixes=("_old", "_new"),
        )
        both = merged[merged["_merge"].eq("both")].copy()
        old_only_count = int(merged["_merge"].eq("left_only").sum())
        new_only_count = int(merged["_merge"].eq("right_only").sum())
        row_match_rate = len(both) / max(len(merged), 1)
        target_match = bool((both["actual_place_old"].to_numpy() == both["actual_place_new"].to_numpy()).all()) if len(both) else False
        market_p99 = float(np.percentile(np.abs(both["market_logit_old"] - both["market_logit_new"]), 99)) if len(both) else math.inf
        prob_p99 = float(np.percentile(np.abs(both["probability_raw_old"] - both["probability_raw_new"]), 99)) if len(both) else math.inf
        old_metrics = metrics_for_frame(old.rename(columns={"probability": "unused"}), cfg)
        new_metrics = metrics_for_frame(new, cfg)
        legacy_tree = model_tree_count(legacy_model_path(cfg, int(year)))
        new_tree = int(new["tree_count"].iloc[0]) if len(new) else None
        row = {
            "reference_type": "corrected_legacy" if reference_mode == "corrected" else "historical_old_base",
            "comparison_type": "blocking" if reference_mode == "corrected" else "diagnostic_non_blocking",
            "validation_year": int(year),
            "row_key_match_rate": row_match_rate,
            "old_rows": int(len(old)),
            "new_rows": int(len(new)),
            "validation_row_count_match": len(old) == len(new),
            "both_count": int(len(both)),
            "old_only_count": old_only_count,
            "new_only_count": new_only_count,
            "key_dtype_audit": json.dumps(key_audit, ensure_ascii=False),
            "feature_list_match": True,
            "feature_columns": ",".join(numeric + categorical),
            "target_match": target_match,
            "market_logit_p99_abs_diff": market_p99,
            "legacy_tree_count": legacy_tree,
            "new_tree_count": new_tree,
            "tree_count_match": legacy_tree == new_tree == 300,
            "probability_raw_p99_abs_diff": prob_p99,
            "logloss_abs_diff": abs(old_metrics["logloss"] - new_metrics["logloss"]),
            "brier_abs_diff": abs(old_metrics["brier"] - new_metrics["brier"]),
        }
        structural_passed = (
            row["row_key_match_rate"] >= float(tolerances["row_key_match_rate"])
            and row["validation_row_count_match"]
            and row["old_only_count"] == 0
            and row["new_only_count"] == 0
            and row["feature_list_match"]
            and row["target_match"]
            and row["tree_count_match"]
            and row["market_logit_p99_abs_diff"] <= float(tolerances["market_logit_p99_abs_diff"])
        )
        historical_prediction_passed = (
            structural_passed
            and row["probability_raw_p99_abs_diff"] <= float(tolerances["probability_raw_p99_abs_diff"])
            and row["logloss_abs_diff"] <= float(tolerances["logloss_abs_diff"])
            and row["brier_abs_diff"] <= float(tolerances["brier_abs_diff"])
        )
        row["structural_passed"] = structural_passed
        row["historical_prediction_passed"] = historical_prediction_passed
        row["passed"] = structural_passed if reference_mode == "corrected" else historical_prediction_passed
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_vs_legacy(pred: pd.DataFrame, cfg: dict[str, Any], iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    legacy = pred[pred["strategy"].eq("LEGACY_2016") & pred["Year"].isin(PRIMARY_YEARS)].copy()
    for strategy, cand in pred[pred["Year"].isin(PRIMARY_YEARS)].groupby("strategy"):
        if strategy == "LEGACY_2016":
            continue
        merged = legacy[KEY_COLUMNS + ["actual_place", "probability_raw"]].merge(
            cand[KEY_COLUMNS + ["probability_raw"]],
            on=KEY_COLUMNS,
            suffixes=("_legacy", "_candidate"),
        )
        eps = float(cfg["epsilon"])
        merged["legacy_logloss_sum"] = -(
            merged["actual_place"] * np.log(clip_prob(merged["probability_raw_legacy"], eps))
            + (1 - merged["actual_place"]) * np.log(1 - clip_prob(merged["probability_raw_legacy"], eps))
        )
        merged["candidate_logloss_sum"] = -(
            merged["actual_place"] * np.log(clip_prob(merged["probability_raw_candidate"], eps))
            + (1 - merged["actual_place"]) * np.log(1 - clip_prob(merged["probability_raw_candidate"], eps))
        )
        merged["legacy_brier_sum"] = np.square(merged["probability_raw_legacy"] - merged["actual_place"])
        merged["candidate_brier_sum"] = np.square(merged["probability_raw_candidate"] - merged["actual_place"])
        race_level = merged.groupby("race_id").agg(
            legacy_logloss_sum=("legacy_logloss_sum", "sum"),
            candidate_logloss_sum=("candidate_logloss_sum", "sum"),
            legacy_brier_sum=("legacy_brier_sum", "sum"),
            candidate_brier_sum=("candidate_brier_sum", "sum"),
            count=("actual_place", "size"),
        )
        values = race_level[
            ["legacy_logloss_sum", "candidate_logloss_sum", "legacy_brier_sum", "candidate_brier_sum", "count"]
        ].to_numpy(float)
        n_races = len(values)
        ll_diffs, br_diffs = [], []
        for _ in range(iterations):
            sample = rng.integers(0, n_races, size=n_races)
            sampled = values[sample]
            total_count = sampled[:, 4].sum()
            ll_diffs.append((sampled[:, 1].sum() - sampled[:, 0].sum()) / total_count)
            br_diffs.append((sampled[:, 3].sum() - sampled[:, 2].sum()) / total_count)
        ll = np.asarray(ll_diffs)
        br = np.asarray(br_diffs)
        rows.append({
            "strategy": strategy,
            "baseline": "LEGACY_2016",
            "n_bootstrap": int(iterations),
            "seed": int(seed),
            "sampling_unit": "race",
            "delta_logloss_mean": float(ll.mean()),
            "delta_logloss_ci_lower": float(np.percentile(ll, 2.5)),
            "delta_logloss_ci_upper": float(np.percentile(ll, 97.5)),
            "delta_logloss_candidate_better_probability": float((ll < 0).mean()),
            "delta_brier_mean": float(br.mean()),
            "delta_brier_ci_lower": float(np.percentile(br, 2.5)),
            "delta_brier_ci_upper": float(np.percentile(br, 97.5)),
            "delta_brier_candidate_better_probability": float((br < 0).mean()),
        })
    return pd.DataFrame(rows)


def write_corrected_legacy_reference(
    out: Path,
    cfg: dict[str, Any],
    pred: pd.DataFrame,
    fold_meta: list[dict[str, Any]],
    numeric: list[str],
    categorical: list[str],
    parity: pd.DataFrame | None,
) -> None:
    legacy = pred[pred["strategy"].eq("LEGACY_2016")].copy()
    if legacy.empty:
        return
    metrics = []
    for year, g in legacy.groupby("Year"):
        metrics.append({"Year": int(year), **metrics_for_frame(g, cfg)})
    reference = {
        "name": "CORRECTED_LEGACY_2016_V1",
        "reference_type": "corrected_legacy",
        "target_column": cfg["target_column"],
        "history_start_year": 2016,
        "probability_column": "probability_raw",
        "safe_catboost_settings": {
            "iterations": 300,
            "outer_validation_eval_set_used": False,
            "use_best_model": False,
            "early_stopping_enabled": False,
            "overfitting_detector_enabled": False,
            "calibration_fit": False,
        },
        "feature_columns_numeric": numeric,
        "feature_columns_categorical": categorical,
        "fold_manifests": [
            {
                "strategy": m["strategy"],
                "validation_year": m["validation_year"],
                "model_path": m["model_path"],
                "prediction_path": m["prediction_path"],
                "signature": m["signature"],
                "market_provenance": m["market_provenance"],
                "residual_provenance": m["residual_provenance"],
                "catboost_safety": m["catboost_safety"],
            }
            for m in fold_meta
            if m["strategy"] == "LEGACY_2016"
        ],
        "metrics": metrics,
        "historical_old_base_comparison": parity.to_dict("records") if parity is not None else [],
    }
    (out / "corrected_legacy_reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest = {
        "name": "CORRECTED_LEGACY_2016_V1",
        "status": "certified_if_structural_passed",
        "reference_type": "corrected_legacy",
        "config_hash": sha256_json(cfg),
        "feature_hash": sha256_json({"numeric": numeric, "categorical": categorical}),
        "target_provenance": {
            "canonical_target": "target_place_paid",
            "reason": "Paid place outcome aligns with fuku_pay and ROI evaluation.",
        },
        "blocking_checks": ["key parity", "target parity", "market parity", "feature order", "safe CatBoost settings"],
        "diagnostic_non_blocking_checks": ["historical old probability difference", "historical old metric difference"],
    }
    (out / "corrected_legacy_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_outputs(
    cfg: dict[str, Any],
    pred: pd.DataFrame,
    fold_meta: list[dict[str, Any]],
    numeric: list[str],
    categorical: list[str],
    parity: pd.DataFrame | None,
    reference_mode: str,
    resume: bool = False,
) -> None:
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "phase5b_predictions.parquet"
    if pred_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing aggregate prediction file: {pred_path}")
        existing = pd.read_parquet(pred_path, columns=KEY_COLUMNS + ["strategy"])
        expected_keys = pred[KEY_COLUMNS + ["strategy"]].copy()
        existing = normalize_parity_keys(existing).sort_values(KEY_COLUMNS + ["strategy"]).reset_index(drop=True)
        expected_keys = normalize_parity_keys(expected_keys).sort_values(KEY_COLUMNS + ["strategy"]).reset_index(drop=True)
        if len(existing) != len(expected_keys) or not existing.equals(expected_keys):
            raise FileExistsError(f"Existing aggregate prediction file does not match resumed fold outputs: {pred_path}")
    else:
        pred.to_parquet(pred_path, index=False)

    summaries = summarize_predictions(pred, cfg)
    for name, df in summaries.items():
        df.to_csv(out / f"{name}.csv", index=False)

    strategy_rows = []
    for s in cfg["strategies"]:
        strategy_rows.append({k: v for k, v in s.items() if k != "notes"})
    pd.DataFrame(strategy_rows).to_csv(out / "strategy_definition.csv", index=False)

    market_rows = [m["market_provenance"] for m in fold_meta]
    residual_rows = [m["residual_provenance"] for m in fold_meta]
    weight_rows = [
        {"strategy": m["strategy"], "validation_year": m["validation_year"], **m["sample_weight_summary"]}
        for m in fold_meta
        if m.get("sample_weight_summary")
    ]
    pd.DataFrame(market_rows).to_csv(out / "market_model_window_by_strategy.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(out / "residual_model_window_by_strategy.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "sample_weight_summary.csv", index=False)

    fold_rows = []
    for m in fold_meta:
        sig = m["signature"]
        fold_rows.append({
            "strategy": m["strategy"],
            "validation_year": m["validation_year"],
            "train_years": ",".join(map(str, sig["train_years"])),
            "train_rows": sig["train_rows"],
            "validation_rows": sig["validation_rows"],
            "status": m["status"],
            "action": m.get("action"),
            "model_path": m["model_path"],
            "prediction_path": m["prediction_path"],
        })
    pd.DataFrame(fold_rows).to_csv(out / "walk_forward_folds.csv", index=False)

    if parity is not None:
        parity.to_csv(out / "legacy_parity_check.csv", index=False)
    elif not (out / "legacy_parity_check.csv").exists():
        pd.DataFrame().to_csv(out / "legacy_parity_check.csv", index=False)
    if reference_mode == "corrected":
        write_corrected_legacy_reference(out, cfg, pred, fold_meta, numeric, categorical, parity)

    bootstrap = paired_bootstrap_vs_legacy(
        pred,
        cfg,
        int(cfg["bootstrap_iterations"]),
        int(cfg["random_seed"]),
    )
    bootstrap.to_csv(out / "paired_bootstrap_summary.csv", index=False)

    manifest = {
        "version": cfg["version"],
        "created_at": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "history_manifest": json.loads(Path(cfg["history_manifest_path"]).read_text(encoding="utf-8")),
        "history_dataset_sha256": sha256_file(Path(cfg["history_dataset_path"])),
        "feature_columns_numeric": numeric,
        "feature_columns_categorical": categorical,
        "probability_column": "probability_raw",
        "reference_mode": reference_mode,
        "target_column": cfg["target_column"],
        "probability_calibrated_generated": False,
        "calibration_fit": False,
        "catboost_safety": {
            "iterations": 300,
            "use_best_model": False,
            "early_stopping_enabled": False,
            "outer_validation_eval_set_used": False,
        },
        "output_files": sorted(p.name for p in out.glob("*") if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    selected = {
        "status": "not_selected_by_codex",
        "reason": "Phase 5B runner implementation only; GPT/user selects after local full run.",
        "selection_probability_column": "probability_raw",
        "roi_is_auxiliary": True,
    }
    (out / "selected_year_strategy.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "audit_report.md").write_text(
        "# Phase 5B Audit Report\n\nGenerated by the Phase 5B runner. Review CSV/JSON artifacts before strategy selection.\n",
        encoding="utf-8",
    )


def run(
    cfg_path: Path,
    strategies: list[str],
    years: list[int],
    resume: bool,
    smoke_rows_per_year: int | None,
    parity_check: bool,
    output_root: str | None = None,
    model_root: str | None = None,
    reference_mode: str = "historical",
) -> int:
    cfg = load_yaml(cfg_path)
    if output_root:
        cfg["output_root"] = output_root
    if model_root:
        cfg["model_root"] = model_root
    params = make_safe_catboost_params(cfg)
    assert_safe_catboost_params(params)
    numeric, categorical = load_feature_allowlist(cfg)
    min_history = min(int(strategy_by_name(cfg, s)["history_start_year"]) for s in strategies)
    df = load_history_dataset(cfg, min_history, smoke_rows_per_year)
    fold_meta: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    started = time.time()
    for window in build_windows(cfg, strategies, years):
        print(f"[fold] strategy={window.strategy} validation_year={window.validation_year} train={window.train_start_year}-{window.train_end_year}", flush=True)
        pred, meta = run_fold(cfg, window, df[df["Year"].ge(window.history_start_year)].copy(), numeric, categorical, resume)
        parts.append(pred)
        fold_meta.append(meta)
    pred = pd.concat(parts, ignore_index=True, sort=False)
    parity = None
    if parity_check and "LEGACY_2016" in strategies:
        legacy = load_legacy_base(cfg, years)
        parity = parity_gate(legacy, pred, cfg, numeric, categorical, reference_mode)
        if not bool(parity["passed"].all()):
            out = Path(cfg["output_root"])
            out.mkdir(parents=True, exist_ok=True)
            parity.to_csv(out / "legacy_parity_check.csv", index=False)
            print(parity.to_string(index=False), flush=True)
            raise SystemExit(f"LEGACY {reference_mode} parity gate failed; stopping without broad strategy comparison.")
    write_outputs(cfg, pred, fold_meta, numeric, categorical, parity, reference_mode, resume)
    print(f"[done] elapsed_seconds={time.time() - started:.1f}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_year_strategy_phase5b_v2.yaml")
    parser.add_argument("--strategies", default=",".join(ALL_STRATEGIES))
    parser.add_argument("--years", default=",".join(map(str, PRIMARY_YEARS)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-rows-per-year", type=int, default=None)
    parser.add_argument("--parity-check", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--reference-mode", choices=["historical", "corrected"], default="historical")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    return run(
        Path(args.config),
        strategies,
        years,
        args.resume,
        args.smoke_rows_per_year,
        args.parity_check,
        args.output_root,
        args.model_root,
        args.reference_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
