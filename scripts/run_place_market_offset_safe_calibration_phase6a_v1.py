from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402

STRATEGIES = ["ROLLING_10Y", "ROLLING_15Y"]
METHODS = ["RAW_IDENTITY", "TEMPERATURE_SCALING", "PLATT_SCALING", "ISOTONIC"]
KEYS = ["entry_id", "race_id", "race_date", "Year", "strategy"]
STRESS_LIMITS = [1, 3, 5, 10]


@dataclass
class Calibrator:
    method: str
    params: dict[str, Any]
    model: Any = None

    def transform(self, p: np.ndarray, eps: float) -> np.ndarray:
        p = clip_prob(np.asarray(p, dtype=float), eps)
        if self.method == "RAW_IDENTITY":
            return p
        z = logit(p, eps)
        if self.method == "TEMPERATURE_SCALING":
            return sigmoid(z / float(self.params["temperature"]), eps)
        if self.method == "PLATT_SCALING":
            return clip_prob(self.model.predict_proba(z.reshape(-1, 1))[:, 1], eps)
        if self.method == "ISOTONIC":
            return clip_prob(self.model.predict(p), eps)
        raise ValueError(f"Unknown calibrator method: {self.method}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame_keys(df: pd.DataFrame) -> str:
    d = canonicalize_keys(df[KEYS]).sort_values(KEYS).astype(str)
    payload = json.dumps(d.to_dict("list"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clip_prob(p: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    q = clip_prob(p, eps)
    return np.log(q / (1.0 - q))


def sigmoid(z: np.ndarray, eps: float) -> np.ndarray:
    return clip_prob(1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float))), eps)


def canonicalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["Year"] = pd.to_numeric(out["Year"], errors="raise").astype(int)
    return out


def fit_calibrator(method: str, train: pd.DataFrame, eps: float) -> Calibrator:
    if method not in METHODS:
        raise ValueError(method)
    y = train["actual_place"].to_numpy(int)
    p = clip_prob(train["probability_raw"].to_numpy(float), eps)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("actual_place must be binary target_place_paid values")
    if method == "RAW_IDENTITY":
        return Calibrator(method, {"fit_rows": int(len(train))})
    z = logit(p, eps)
    if method == "TEMPERATURE_SCALING":
        def objective(log_t: float) -> float:
            t = float(np.exp(log_t))
            return float(log_loss(y, sigmoid(z / t, eps), labels=[0, 1]))

        res = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
        return Calibrator(method, {"temperature": float(np.exp(res.x)), "fit_rows": int(len(train)), "success": bool(res.success)})
    if method == "PLATT_SCALING":
        model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        model.fit(z.reshape(-1, 1), y)
        return Calibrator(
            method,
            {
                "coef": float(model.coef_[0][0]),
                "intercept": float(model.intercept_[0]),
                "fit_rows": int(len(train)),
            },
            model,
        )
    model = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1.0 - eps)
    model.fit(p, y)
    return Calibrator(method, {"fit_rows": int(len(train))}, model)


def fixed_bin_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.sum() / total * abs(p[mask].mean() - y[mask].mean()))
    return out


def calibration_line(y: np.ndarray, p: np.ndarray, eps: float) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    z = logit(p, eps).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        model.fit(z, y.astype(int))
        return float(model.coef_[0][0]), float(model.intercept_[0])
    except Exception:
        return math.nan, math.nan


def metrics(g: pd.DataFrame, prob_col: str, cfg: dict[str, Any]) -> dict[str, Any]:
    y = g["actual_place"].to_numpy(int)
    p = clip_prob(g[prob_col].to_numpy(float), float(cfg["epsilon"]))
    slope, intercept = calibration_line(y, p, float(cfg["epsilon"]))
    return {
        "rows": int(len(g)),
        "races": int(g["race_id"].nunique()),
        "positives": int(y.sum()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": fixed_bin_ece(y, p, 10),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def reliability_table(pred: pd.DataFrame, prob_col: str, label: dict[str, Any], bins: int = 10) -> pd.DataFrame:
    d = pred.copy()
    d["_bin"] = pd.cut(d[prob_col], np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    rows = []
    for i, (_bin, g) in enumerate(d.groupby("_bin", observed=False)):
        if g.empty:
            continue
        rows.append(
            {
                **label,
                "bin_id": i,
                "count": int(len(g)),
                "mean_probability": float(g[prob_col].mean()),
                "actual_rate": float(g["actual_place"].mean()),
                "gap_pred_minus_actual": float(g[prob_col].mean() - g["actual_place"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_new_8fold(cfg_path: Path, cfg: dict[str, Any], resume: bool) -> None:
    phase5b.run(
        cfg_path,
        STRATEGIES,
        [2016, 2017, 2018, 2019],
        resume=resume,
        smoke_rows_per_year=None,
        parity_check=False,
        output_root=cfg["output_root"],
        model_root=cfg["model_root"],
        reference_mode="corrected",
    )


def validate_prediction_artifact(path: Path, expected_years: set[int], strategies: set[str]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    cols = KEYS + ["actual_place", "probability_raw", "target_place_paid"]
    d = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path, columns=None).columns])
    d = d[d["strategy"].isin(strategies) & d["Year"].isin(expected_years)].copy()
    if set(d["Year"].unique()) != expected_years:
        raise ValueError(f"{path} missing years: expected={expected_years} got={set(d['Year'].unique())}")
    if set(d["strategy"].unique()) != strategies:
        raise ValueError(f"{path} missing strategies: expected={strategies} got={set(d['strategy'].unique())}")
    return {"path": str(path), "rows": int(len(d)), "key_hash": sha256_frame_keys(d), "file_sha256": sha256_file(path)}


def load_combined_predictions(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = Path(cfg["output_root"])
    paths = [
        (out / "phase5b_predictions.parquet", set(cfg["new_training_years"])),
        (Path(cfg["phase5b_output_root"]) / "phase5b_predictions.parquet", set(cfg["selection_years"])),
        (Path(cfg["phase5c_output_root"]) / "phase5c_predictions.parquet", set(cfg["diagnostic_years"])),
    ]
    manifests = []
    parts = []
    for path, years in paths:
        manifests.append(validate_prediction_artifact(path, years, set(STRATEGIES)))
        cols = pd.read_parquet(path, columns=None).columns
        needed = [c for c in cols if c in {
            *KEYS,
            "actual_place",
            "target_place_paid",
            "probability_raw",
            "catboost_residual_score",
            "fuku_odds_low",
            "fuku_pay",
            "market_logit",
            "train_start_year",
            "train_end_year",
            "tree_count",
            "JyoCD",
            "Kyori",
            "SyussoTosu",
        }]
        d = pd.read_parquet(path, columns=needed)
        d = d[d["strategy"].isin(STRATEGIES) & d["Year"].isin(years)].copy()
        parts.append(d)
    pred = canonicalize_keys(pd.concat(parts, ignore_index=True, sort=False))
    if pred.duplicated(KEYS).any():
        raise ValueError("Duplicate combined prediction keys")
    if "target_place_paid" in pred.columns and not pred["actual_place"].astype(int).equals(pred["target_place_paid"].astype(int)):
        raise ValueError("actual_place does not match target_place_paid")
    if not set(pred["actual_place"].dropna().unique()) <= {0, 1}:
        raise ValueError("actual_place must be binary; no rank conversion is allowed")
    return pred, manifests


def walk_forward_calibration(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, prov, reliability = [], [], []
    years = sorted(cfg["selection_years"] + cfg["diagnostic_years"])
    for strategy in STRATEGIES:
        s = pred[pred["strategy"].eq(strategy)].copy()
        for year in years:
            fit = s[(s["Year"] >= int(cfg["calibration_fit_start_year"])) & (s["Year"] < year)].copy()
            eval_df = s[s["Year"].eq(year)].copy()
            if fit.empty or eval_df.empty:
                raise ValueError(f"Missing fit/eval data for {strategy} {year}")
            for method in METHODS:
                cal = fit_calibrator(method, fit, float(cfg["epsilon"]))
                p = cal.transform(eval_df["probability_raw"].to_numpy(float), float(cfg["epsilon"]))
                e = eval_df.copy()
                e["probability_calibrated"] = p
                e["calibration_method"] = method
                rows.append(e)
                prov.append(
                    {
                        "strategy": strategy,
                        "evaluation_year": int(year),
                        "calibration_method": method,
                        "fit_start_year": int(fit["Year"].min()),
                        "fit_end_year": int(fit["Year"].max()),
                        "fit_rows": int(len(fit)),
                        "fit_races": int(fit["race_id"].nunique()),
                        "target": "target_place_paid",
                        "uses_only_prior_years": bool(fit["Year"].max() < year),
                        "params": json.dumps(cal.params, ensure_ascii=False),
                    }
                )
                reliability.append(reliability_table(e, "probability_calibrated", {"strategy": strategy, "Year": int(year), "calibration_method": method}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(prov), pd.concat(reliability, ignore_index=True)


def metric_tables(calibrated: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (strategy, method, year), g in calibrated.groupby(["strategy", "calibration_method", "Year"]):
        rows.append({"strategy": strategy, "calibration_method": method, "Year": int(year), **metrics(g, "probability_calibrated", cfg)})
    by_year = pd.DataFrame(rows)
    year_mean = by_year[by_year["Year"].isin(cfg["selection_years"])].groupby(["strategy", "calibration_method"], as_index=False).agg(
        mean_logloss=("logloss", "mean"),
        mean_brier=("brier", "mean"),
        mean_ece=("ece", "mean"),
        worst_logloss=("logloss", "max"),
        worst_brier=("brier", "max"),
    )
    pooled_rows = []
    for (strategy, method), g in calibrated[calibrated["Year"].isin(cfg["selection_years"])].groupby(["strategy", "calibration_method"]):
        pooled_rows.append({"strategy": strategy, "calibration_method": method, **metrics(g, "probability_calibrated", cfg)})
    selection = pd.DataFrame(pooled_rows).rename(
        columns={"logloss": "pooled_logloss", "brier": "pooled_brier", "ece": "pooled_ece"}
    )
    selection = selection.merge(year_mean, on=["strategy", "calibration_method"], how="left", validate="one_to_one")
    diagnostic = by_year[by_year["Year"].isin(cfg["diagnostic_years"])].copy()
    return by_year, selection, diagnostic


def select_calibrators(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in selection.groupby("strategy"):
        rank = g.sort_values(["pooled_logloss", "pooled_brier", "pooled_ece", "calibration_method"]).iloc[0]
        rows.append(
            {
                "strategy": strategy,
                "selected_calibration_method": rank["calibration_method"],
                "selection_years": "2020,2021,2022,2023,2024",
                "selection_basis": "2020-2024 pooled-row Logloss, then pooled Brier/ECE; ROI and 2025/2026 excluded",
                "pooled_logloss": float(rank["pooled_logloss"]),
                "pooled_brier": float(rank["pooled_brier"]),
                "pooled_ece": float(rank["pooled_ece"]),
                "operationally_activated": False,
            }
        )
    return pd.DataFrame(rows)


def paired_raw_vs_calibrated_bootstrap(calibrated: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["random_seed"]))
    rows = []
    chosen = calibrated.merge(selected[["strategy", "selected_calibration_method"]], on="strategy")
    chosen = chosen[chosen["calibration_method"].eq(chosen["selected_calibration_method"])].copy()
    for strategy, sd in chosen.groupby("strategy"):
        for label, g in list(sd.groupby("Year")) + [("2025_2026", sd[sd["Year"].isin(cfg["diagnostic_years"])])]:
            if g.empty:
                continue
            y = g["actual_place"].to_numpy(int)
            praw = phase5b.clip_prob(g["probability_raw"].to_numpy(float), float(cfg["epsilon"]))
            pcal = phase5b.clip_prob(g["probability_calibrated"].to_numpy(float), float(cfg["epsilon"]))
            ll = (-(y * np.log(pcal) + (1 - y) * np.log(1 - pcal))) - (-(y * np.log(praw) + (1 - y) * np.log(1 - praw)))
            br = (pcal - y) ** 2 - (praw - y) ** 2
            races = np.array(sorted(g["race_id"].unique()))
            idx_map = {r: i for i, r in enumerate(races)}
            idx = np.array([idx_map[r] for r in g["race_id"]], dtype=np.int64)
            count = np.bincount(idx, minlength=len(races))
            sums = {"logloss": np.bincount(idx, weights=ll, minlength=len(races)), "brier": np.bincount(idx, weights=br, minlength=len(races))}
            for metric_name, race_sum in sums.items():
                draws = np.empty(int(cfg["bootstrap_iterations"]), dtype=float)
                for i in range(len(draws)):
                    sample = rng.integers(0, len(races), len(races))
                    draws[i] = race_sum[sample].sum() / count[sample].sum()
                rows.append(
                    {
                        "strategy": strategy,
                        "Year": label,
                        "metric": metric_name,
                        "delta_calibrated_minus_raw": float(race_sum.sum() / count.sum()),
                        "bootstrap_mean": float(draws.mean()),
                        "ci95_lower": float(np.percentile(draws, 2.5)),
                        "ci95_upper": float(np.percentile(draws, 97.5)),
                        "calibrated_better_probability": float((draws < 0).mean()),
                        "races": int(len(races)),
                        "rows": int(len(g)),
                        "n_bootstrap": int(cfg["bootstrap_iterations"]),
                    }
                )
    return pd.DataFrame(rows)


def roi_tables(calibrated: pd.DataFrame, selected: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen = calibrated.merge(selected[["strategy", "selected_calibration_method"]], on="strategy")
    chosen = chosen[chosen["calibration_method"].eq(chosen["selected_calibration_method"])].copy()
    chosen["ev_calibrated"] = chosen["probability_calibrated"] * pd.to_numeric(chosen[cfg["odds_column"]], errors="coerce")
    rows, rr_rows, pz_rows = [], [], []
    for (strategy, year), g in chosen.groupby(["strategy", "Year"]):
        picks = g[g["ev_calibrated"].ge(1.0)].copy()
        payout = pd.to_numeric(picks[cfg["payout_column"]], errors="coerce").fillna(0)
        stake = len(picks) * int(cfg["stake_yen"])
        rows.append(
            {
                "strategy": strategy,
                "Year": int(year),
                "bet_count": int(len(picks)),
                "race_count_with_bet": int(picks["race_id"].nunique()),
                "stake": int(stake),
                "payout": float(payout.sum()),
                "roi": float(payout.sum() / stake * 100.0) if stake else math.nan,
                "hit_count": int((payout > 0).sum()),
                "hit_rate": float((payout > 0).mean()) if len(picks) else math.nan,
                "average_probability_calibrated": float(picks["probability_calibrated"].mean()) if len(picks) else math.nan,
                "average_ev_calibrated": float(picks["ev_calibrated"].mean()) if len(picks) else math.nan,
            }
        )
        hits = picks[picks["actual_place"].eq(1)].copy().sort_values(cfg["payout_column"], ascending=False)
        normal_roi = rows[-1]["roi"]
        for limit in STRESS_LIMITS:
            removed_idx = hits.head(limit).index
            rr = picks.drop(index=removed_idx)
            pz = picks.copy()
            pz.loc[removed_idx, cfg["payout_column"]] = 0
            for kind, frame, table in [("row_removed", rr, rr_rows), ("payout_zeroed", pz, pz_rows)]:
                pay = pd.to_numeric(frame[cfg["payout_column"]], errors="coerce").fillna(0).sum() if len(frame) else 0.0
                st = len(frame) * int(cfg["stake_yen"])
                roi = float(pay / st * 100.0) if st else math.nan
                if kind == "payout_zeroed" and not math.isnan(normal_roi) and not math.isnan(roi) and roi > normal_roi + 1e-12:
                    raise RuntimeError("payout_zeroed_stress_roi exceeded normal_roi")
                table.append(
                    {
                        "strategy": strategy,
                        "Year": int(year),
                        "limit": limit,
                        "normal_roi": normal_roi,
                        "removed_count": int(len(removed_idx)),
                        "bet_count": int(len(frame)),
                        "stake": int(st),
                        "roi": roi,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(rr_rows), pd.DataFrame(pz_rows)


def write_report(out: Path) -> None:
    selection = pd.read_csv(out / "calibrator_selection.csv")
    metrics_df = pd.read_csv(out / "calibration_metrics_by_year.csv")
    boot = pd.read_csv(out / "raw_vs_calibrated_bootstrap.csv")
    roi = pd.read_csv(out / "roi_ev_ge_1_calibrated.csv")
    text = "\n".join(
        [
            "# Phase 6A Safe Walk-Forward Calibration Results",
            "",
            "Champion remains `ROLLING_10Y`; Challenger remains `ROLLING_15Y`. `operationally_activated=false`.",
            "",
            "## Calibrator Selection",
            selection.to_markdown(index=False),
            "",
            "## Metrics By Year",
            metrics_df.to_markdown(index=False),
            "",
            "## Raw vs Calibrated Bootstrap",
            boot.to_markdown(index=False),
            "",
            "## EV>=1.00 ROI",
            roi.to_markdown(index=False),
            "",
        ]
    )
    Path("docs/place_market_offset_safe_calibration_phase6a_v1_results.md").write_text(text, encoding="utf-8")


def postprocess(cfg: dict[str, Any]) -> None:
    out = Path(cfg["output_root"])
    pred, artifact_manifest = load_combined_predictions(cfg)
    pred.to_parquet(out / "phase6a_combined_raw_predictions.parquet", index=False)
    calibrated, provenance, reliability = walk_forward_calibration(pred, cfg)
    calibrated.to_parquet(out / "phase6a_calibrated_predictions.parquet", index=False)
    provenance.to_csv(out / "calibrator_fit_provenance.csv", index=False)
    reliability.to_csv(out / "reliability_table.csv", index=False)
    by_year, selection_metrics, diagnostic = metric_tables(calibrated, cfg)
    by_year.to_csv(out / "calibration_metrics_by_year.csv", index=False)
    selection_metrics.to_csv(out / "calibrator_comparison_2020_2024.csv", index=False)
    diagnostic.to_csv(out / "diagnostic_2025_2026.csv", index=False)
    selected = select_calibrators(selection_metrics)
    selected.to_csv(out / "calibrator_selection.csv", index=False)
    paired_raw_vs_calibrated_bootstrap(calibrated, selected, cfg).to_csv(out / "raw_vs_calibrated_bootstrap.csv", index=False)
    roi, rr, pz = roi_tables(calibrated, selected, cfg)
    roi.to_csv(out / "roi_ev_ge_1_calibrated.csv", index=False)
    rr.to_csv(out / "roi_row_removed_calibrated.csv", index=False)
    pz.to_csv(out / "roi_payout_zeroed_stress_calibrated.csv", index=False)
    target_audit = {
        "target_column": "target_place_paid",
        "actual_place_binary": bool(set(pred["actual_place"].dropna().unique()) <= {0, 1}),
        "actual_place_matches_target_place_paid": bool(pred["actual_place"].astype(int).equals(pred["target_place_paid"].astype(int))) if "target_place_paid" in pred.columns else None,
        "rank_transform_used": False,
        "le_3_transform_used": False,
        "rows": int(len(pred)),
        "strategies": sorted(pred["strategy"].unique()),
        "years": sorted(map(int, pred["Year"].unique())),
    }
    (out / "target_integrity_audit.json").write_text(json.dumps(target_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "version": cfg["version"],
        "champion": "ROLLING_10Y",
        "challenger": "ROLLING_15Y",
        "champion_changed": False,
        "operationally_activated": False,
        "selection_years": cfg["selection_years"],
        "diagnostic_years": cfg["diagnostic_years"],
        "calibration_methods": METHODS,
        "calibrator_selection_uses_2025_2026": False,
        "roi_used_for_selection": False,
        "ev_threshold": 1.0,
        "ensemble_created": False,
        "artifact_manifest": artifact_manifest,
        "target_integrity_audit": target_audit,
        "output_files": sorted(p.name for p in out.glob("*") if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(out)


def run(config: Path, resume: bool, skip_train: bool = False) -> int:
    cfg = load_yaml(config)
    Path(cfg["output_root"]).mkdir(parents=True, exist_ok=True)
    if not skip_train:
        run_new_8fold(config, cfg, resume)
    postprocess(cfg)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_safe_calibration_phase6a_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.config), args.resume, args.skip_train)


if __name__ == "__main__":
    raise SystemExit(main())
