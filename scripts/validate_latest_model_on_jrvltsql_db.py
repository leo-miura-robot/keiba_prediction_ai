from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_full_runner_dataset_o1_fixed as base_builder  # noqa: E402
from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402
from src.calibration.official_calibrator_loader import (  # noqa: E402
    apply_official_platt_calibrator,
    load_official_platt_calibrator,
    sha256_file as sha256_official_file,
)
from src.features.history_builder_v2_1_2 import build_pre_day_history_features_v2_1, new_state  # noqa: E402
from src.features.target_builder import add_target_columns  # noqa: E402


STRATEGY_CALIBRATION = {
    "ROLLING_10Y": "PLATT_SCALING",
    "ROLLING_15Y": "BLOCKED_MISSING_ISOTONIC_THRESHOLDS",
}

DEFAULT_OFFICIAL_PLATT_ARTIFACT = Path(
    "outputs/place_market_offset_official_calibrators_phase6a_v1/rolling_10y_platt_phase6a_v1.json"
)
DEFAULT_INVALID_REFIT_PREDICTIONS = Path(
    "outputs/latest_model_validation_on_jrvltsql_20260608/latest_model_predictions.parquet"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_new_base_rows(db_path: Path) -> pl.DataFrame:
    base_builder.DB_PATH = db_path

    class Logger:
        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    df = base_builder.fetch_year(2026, Logger())
    if df.height == 0:
        raise RuntimeError(f"No 2026 rows exported from DB: {db_path}")
    return df.unique(subset=["entry_id"], keep="last").sort(["race_date", "race_id", "Umaban"])


def build_incremental_history(new_base: pl.DataFrame, history_dir: Path) -> pl.DataFrame:
    canonical_path = history_dir / "canonical_race_rows_2006_2026.parquet"
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    min_new_date = new_base["race_date"].min()
    old = pl.read_parquet(canonical_path).filter(pl.col("race_date") < min_new_date)
    combined = pl.concat([old, new_base], how="diagonal_relaxed").sort(
        ["race_date", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban", "entry_id"]
    )
    labeled = add_target_columns(combined)
    features, _state, audit_counts, _audit_samples = build_pre_day_history_features_v2_1(labeled, None, new_state())
    bad_audit = []
    for store, counts in audit_counts.items():
        bad = int(counts.get("same_race", 0)) + int(counts.get("same_day", 0)) + int(counts.get("future", 0))
        if bad:
            bad_audit.append({"store_name": store, **counts})
    if bad_audit:
        raise RuntimeError(f"History leakage audit failed: {bad_audit[:5]}")
    new_ids = new_base.select("entry_id").to_series().to_list()
    out = features.filter(pl.col("entry_id").is_in(new_ids))
    if out.height != len(set(new_ids)):
        raise RuntimeError(f"Incremental history row mismatch: features={out.height} new_ids={len(set(new_ids))}")
    return out


def load_train_frame(cfg: dict[str, Any], train_years: tuple[int, ...]) -> pd.DataFrame:
    df = phase5b.load_history_dataset(cfg, min(train_years), smoke_rows_per_year=None)
    return df[df["Year"].isin(train_years)].copy()


def predict_strategy(
    cfg: dict[str, Any],
    strategy: str,
    new_features: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    strategy_cfg = phase5b.strategy_by_name(cfg, strategy)
    window = phase5b.build_fold_window(strategy_cfg, 2026)
    train = load_train_frame(cfg, window.train_years)
    valid = new_features.copy()
    valid["Year"] = pd.to_numeric(valid["Year"], errors="raise").astype(int)
    valid = valid[valid["Year"].eq(2026)].copy()
    valid = phase5b.apply_target_column(valid, cfg)
    valid = phase5b.add_market_features(valid)
    valid = valid[valid[cfg["eligible_column"]].eq(True)].copy()
    valid = valid[pd.to_numeric(valid[cfg["odds_column"]], errors="coerce").gt(0)].copy()
    if valid.empty:
        raise RuntimeError(f"No eligible validation rows for {strategy}")

    df_for_market = pd.concat([train, valid], ignore_index=True, sort=False)
    _train_with_market, valid_with_market, market_prov = phase5b.make_market_logit_for_fold(df_for_market, window, cfg)
    model_path = Path(cfg["model_root"]) / strategy / "validation_2026" / "model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    pred = phase5b.predict_raw(model, valid_with_market, numeric, categorical, float(cfg["epsilon"]))
    pred["strategy"] = strategy
    pred["validation_year"] = 2026
    pred["tree_count"] = int(model.tree_count_)
    pred["train_start_year"] = window.train_start_year
    pred["train_end_year"] = window.train_end_year
    pred["model_sha256"] = sha256_file(model_path)
    pred["market_train_rows"] = market_prov["market_train_rows"]
    pred["market_train_start"] = market_prov["market_train_start"]
    pred["market_train_end"] = market_prov["market_train_end"]
    return pred


def sigmoid_array(z: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float))), eps, 1.0 - eps)


def load_invalid_refit_platt(pred: pd.DataFrame, path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(np.nan, index=pred.index, dtype=float)
    old = pd.read_parquet(path)
    old = old[old["strategy"].eq("ROLLING_10Y")].copy()
    required = {"entry_id", "race_id", "race_date", "probability_calibrated"}
    if not required.issubset(old.columns):
        return pd.Series(np.nan, index=pred.index, dtype=float)
    left = pred[["entry_id", "race_id", "race_date"]].copy()
    left["_row_order"] = np.arange(len(left))
    left["race_date"] = pd.to_datetime(left["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    right = old[["entry_id", "race_id", "race_date", "probability_calibrated"]].copy()
    right["race_date"] = pd.to_datetime(right["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    merged = left.merge(
        right.rename(columns={"probability_calibrated": "probability_invalid_refit_platt"}),
        on=["entry_id", "race_id", "race_date"],
        how="left",
        validate="one_to_one",
    ).sort_values("_row_order")
    return pd.Series(merged["probability_invalid_refit_platt"].to_numpy(float), index=pred.index, dtype=float)


def apply_official_calibration(
    pred: pd.DataFrame,
    official_artifact_path: Path,
    invalid_refit_predictions_path: Path,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact = load_official_platt_calibrator(official_artifact_path)
    out = pred.copy()
    out["probability_market"] = sigmoid_array(out["market_logit"].to_numpy(float), eps)
    out["official_calibration_status"] = "RAW_DIAGNOSTIC_ONLY"
    out["calibration_method"] = "RAW_DIAGNOSTIC_ONLY"
    out["probability_official_platt"] = np.nan
    out["probability_invalid_refit_platt"] = np.nan
    out["calibrator_params"] = ""

    mask_10y = out["strategy"].eq("ROLLING_10Y")
    if not mask_10y.any():
        raise RuntimeError("No ROLLING_10Y rows for official Platt validation")
    out.loc[mask_10y, "calibration_method"] = "PLATT_SCALING"
    out.loc[mask_10y, "official_calibration_status"] = "OFFICIAL_10Y_PLATT_LOADED_READ_ONLY"
    out.loc[mask_10y, "probability_official_platt"] = apply_official_platt_calibrator(
        artifact,
        out.loc[mask_10y, "probability_raw"].to_numpy(float),
    )
    out.loc[mask_10y, "probability_invalid_refit_platt"] = load_invalid_refit_platt(
        out.loc[mask_10y].copy(),
        invalid_refit_predictions_path,
    ).to_numpy(float)
    out.loc[mask_10y, "calibrator_params"] = json.dumps(
        {
            "coef": artifact["coef"],
            "intercept": artifact["intercept"],
            "clip_min": artifact["clip_min"],
            "clip_max": artifact["clip_max"],
            "input_space": artifact["input_space"],
            "source": "official_read_only_artifact",
        },
        ensure_ascii=False,
    )

    mask_15y = out["strategy"].eq("ROLLING_15Y")
    out.loc[mask_15y, "calibration_method"] = "BLOCKED_MISSING_ISOTONIC_THRESHOLDS"
    out.loc[mask_15y, "official_calibration_status"] = "BLOCKED_MISSING_ISOTONIC_THRESHOLDS"
    audit = {
        "official_10y_platt_loaded": True,
        "official_platt_artifact_path": str(official_artifact_path.resolve()),
        "official_platt_artifact_file_sha256": sha256_official_file(official_artifact_path),
        "official_platt_artifact_payload_sha256": artifact.get("artifact_payload_sha256"),
        "refit_performed": False,
        "oof_parameter_generation_performed": False,
        "invalid_refit_predictions_loaded_for_comparison": bool(invalid_refit_predictions_path.exists()),
        "invalid_refit_predictions_path": str(invalid_refit_predictions_path.resolve()),
        "shadow_15y_status": "BLOCKED_MISSING_ISOTONIC_THRESHOLDS",
        "shadow_15y_official_calibration_available": False,
        "usable_for_roi_judgement": False,
    }
    return out, audit


def metrics_for(d: pd.DataFrame, probability_col: str) -> dict[str, Any]:
    y = d["actual_place"].to_numpy(int)
    p = d[probability_col].to_numpy(float)
    return {
        "rows": int(len(d)),
        "races": int(d["race_id"].nunique()),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def fixed_bin_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.sum() / total * abs(float(p[mask].mean()) - float(y[mask].mean())))
    return out


def calibration_line(y: np.ndarray, p: np.ndarray, eps: float) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    z = np.log(np.clip(p, eps, 1.0 - eps) / (1.0 - np.clip(p, eps, 1.0 - eps))).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    model.fit(z, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def race_wise_spearman(d: pd.DataFrame, probability_col: str) -> float:
    vals = []
    for _race_id, g in d.groupby("race_id"):
        if len(g) < 2 or g[probability_col].nunique(dropna=True) < 2 or g["actual_place"].nunique(dropna=True) < 2:
            continue
        corr = g[probability_col].corr(g["actual_place"], method="spearman")
        if pd.notna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else math.nan


def rich_metrics_for(d: pd.DataFrame, probability_col: str, eps: float) -> dict[str, Any]:
    work = d.dropna(subset=[probability_col]).copy()
    base = metrics_for(work, probability_col)
    y = work["actual_place"].to_numpy(int)
    p = work[probability_col].to_numpy(float)
    slope, intercept = calibration_line(y, p, eps)
    base.update(
        {
            "ece": fixed_bin_ece(y, p, 10),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "mean_predicted_probability": float(np.mean(p)),
            "actual_positive_rate": float(np.mean(y)),
            "calibration_gap": float(np.mean(p) - np.mean(y)),
            "race_wise_spearman": race_wise_spearman(work, probability_col),
        }
    )
    return base


def roi_summary(d: pd.DataFrame, probability_col: str, threshold: float = 1.0) -> dict[str, Any]:
    work = d.copy()
    work["ev"] = work[probability_col].astype(float) * pd.to_numeric(work["fuku_odds_low"], errors="coerce")
    picks = work[work["ev"].ge(threshold)].copy()
    stake = len(picks) * 100
    payout = float(pd.to_numeric(picks["fuku_pay"], errors="coerce").fillna(0).sum())
    return {
        "threshold": threshold,
        "bet_count": int(len(picks)),
        "race_count_with_bet": int(picks["race_id"].nunique()) if len(picks) else 0,
        "stake": int(stake),
        "payout": payout,
        "roi": float(payout / stake * 100.0) if stake else math.nan,
        "hit_count": int((pd.to_numeric(picks["fuku_pay"], errors="coerce").fillna(0) > 0).sum()),
        "average_ev": float(picks["ev"].mean()) if len(picks) else math.nan,
    }


def tier_counts(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w = d.copy()
    w["ev"] = w["probability_official_platt"] * pd.to_numeric(w["fuku_odds_low"], errors="coerce")
    for threshold, tier in [(1.00, "CORE"), (1.05, "MARGIN"), (1.10, "HIGH"), (1.15, "VERY_HIGH")]:
        picks = w[w["ev"].ge(threshold)]
        rows.append({"threshold_tier": tier, "threshold": threshold, "bet_count": int(len(picks)), "race_count": int(picks["race_id"].nunique()) if len(picks) else 0})
    return pd.DataFrame(rows)


def make_phase6c_prediction_input(pred: pd.DataFrame) -> pd.DataFrame:
    official = pred[pred["strategy"].eq("ROLLING_10Y")].copy()
    return pd.DataFrame(
        {
            "strategy": official["strategy"],
            "calibration_method": official["calibration_method"],
            "entry_id": official["entry_id"],
            "race_id": official["race_id"],
            "race_date": pd.to_datetime(official["race_date"], errors="raise").dt.strftime("%Y-%m-%d"),
            "horse_no": official["Umaban"].astype(str),
            "probability_raw": official["probability_raw"],
            "probability_calibrated": official["probability_official_platt"],
            "market_logit": official["market_logit"],
            "residual_raw": official["catboost_residual_score"],
            "fuku_odds_low_at_prediction": official["fuku_odds_low"],
        }
    )


def make_settlement(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[["entry_id", "race_id", "race_date", "target_place_paid", "fuku_pay"]].drop_duplicates(["entry_id", "race_id", "race_date"]).copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-db", required=True)
    parser.add_argument("--output-root", default="outputs/latest_model_validation_on_jrvltsql_20260608_official_10y_platt_v1")
    parser.add_argument("--official-platt-artifact", default=str(DEFAULT_OFFICIAL_PLATT_ARTIFACT))
    parser.add_argument("--invalid-refit-predictions", default=str(DEFAULT_INVALID_REFIT_PREDICTIONS))
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    cfg = phase5b.load_yaml(Path("config/place_market_offset_champion_challenger_phase5c_v1.yaml"))
    numeric, categorical = phase5b.load_feature_allowlist(cfg)
    history_dir = Path("data/derived/history_extension_2006_phase5_v1")

    new_base = load_new_base_rows(Path(args.sqlite_db))
    new_base.write_parquet(out / "new_base_rows_2026.parquet", compression="zstd")
    new_features_pl = build_incremental_history(new_base, history_dir)
    new_features_pl.write_parquet(out / "new_history_features_2026.parquet", compression="zstd")
    new_features = new_features_pl.to_pandas()
    new_features["race_date"] = pd.to_datetime(new_features["race_date"], errors="raise").dt.strftime("%Y-%m-%d")

    parts = []
    for strategy in ["ROLLING_10Y", "ROLLING_15Y"]:
        parts.append(predict_strategy(cfg, strategy, new_features, numeric, categorical))
    pred = pd.concat(parts, ignore_index=True, sort=False)
    pred, calibration_audit = apply_official_calibration(
        pred,
        Path(args.official_platt_artifact),
        Path(args.invalid_refit_predictions),
        float(cfg["epsilon"]),
    )
    pred.to_parquet(out / "latest_model_predictions.parquet", index=False)
    pred.to_parquet(out / "predictions.parquet", index=False)

    metric_rows = []
    roi_rows = []
    tier_rows = []
    g10 = pred[pred["strategy"].eq("ROLLING_10Y")].copy()
    for label, col in [
        ("market_only", "probability_market"),
        ("raw_c1r0", "probability_raw"),
        ("official_platt", "probability_official_platt"),
        ("invalid_refit_platt", "probability_invalid_refit_platt"),
    ]:
        metric_rows.append(
            {
                "strategy": "ROLLING_10Y",
                "calibration_method": "PLATT_SCALING" if label == "official_platt" else label,
                "probability_type": label,
                "probability_column": col,
                **rich_metrics_for(g10, col, float(cfg["epsilon"])),
            }
        )
    g15 = pred[pred["strategy"].eq("ROLLING_15Y")].copy()
    if not g15.empty:
        metric_rows.append(
            {
                "strategy": "ROLLING_15Y",
                "calibration_method": "BLOCKED_MISSING_ISOTONIC_THRESHOLDS",
                "probability_type": "raw_diagnostic_only",
                "probability_column": "probability_raw",
                **rich_metrics_for(g15, "probability_raw", float(cfg["epsilon"])),
            }
        )
    roi_rows.append({"strategy": "ROLLING_10Y", "calibration_method": "PLATT_SCALING", "probability_column": "probability_official_platt", **roi_summary(g10, "probability_official_platt", 1.0), "usable_for_roi_judgement": False})
    tc = tier_counts(g10)
    tc.insert(0, "calibration_method", "PLATT_SCALING")
    tc.insert(0, "strategy", "ROLLING_10Y")
    tier_rows.append(tc)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "metrics_market_raw_official_platt.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(roi_rows).to_csv(out / "roi_ev_ge_1_auxiliary.csv", index=False)
    pd.DataFrame(roi_rows).to_csv(out / "roi_ev_ge_1.csv", index=False)
    pd.concat(tier_rows, ignore_index=True).to_csv(out / "tier_counts.csv", index=False)

    official = metrics_df[metrics_df["probability_type"].eq("official_platt")].iloc[0].to_dict()
    raw = metrics_df[metrics_df["probability_type"].eq("raw_c1r0")].iloc[0].to_dict()
    invalid = metrics_df[metrics_df["probability_type"].eq("invalid_refit_platt")].iloc[0].to_dict()
    comparison = pd.DataFrame(
        [
            {"comparison": "official_platt_minus_raw", "logloss_delta": official["logloss"] - raw["logloss"], "brier_delta": official["brier"] - raw["brier"]},
            {"comparison": "official_platt_minus_invalid_refit_platt", "logloss_delta": official["logloss"] - invalid["logloss"], "brier_delta": official["brier"] - invalid["brier"]},
        ]
    )
    comparison.to_csv(out / "comparison_with_invalid_refit.csv", index=False)
    metrics_df.to_csv(out / "calibration_diagnostics.csv", index=False)
    write_json(out / "official_calibrator_manifest_snapshot.json", load_official_platt_calibrator(Path(args.official_platt_artifact)))

    make_phase6c_prediction_input(pred).to_csv(out / "phase6c_prediction_input.csv", index=False)
    make_settlement(pred).to_csv(out / "settlement.csv", index=False)

    manifest = {
        "sqlite_db": str(Path(args.sqlite_db).resolve()),
        "output_root": str(out.resolve()),
        "source_history_manifest": read_json(history_dir / "manifest.json"),
        "catboost_new_training": False,
        "market_model_refit_for_validation_year": True,
        "calibrator_reconstructed_from_prior_oof_predictions": False,
        "official_10y_platt_loaded": True,
        "refit_performed": False,
        "invalid_refit_platt_from_existing_predictions_only": calibration_audit["invalid_refit_predictions_loaded_for_comparison"],
        "usable_for_probability_diagnostic": True,
        "usable_for_model_limit_judgement": False,
        "usable_for_roi_judgement": False,
        "calibration_audit": calibration_audit,
        "validation_dates": sorted(pred["race_date"].unique().tolist()),
        "prediction_rows": int(len(pred)),
        "prediction_races": int(pred["race_id"].nunique()),
        "strategies": STRATEGY_CALIBRATION,
        "metrics_file": str((out / "metrics_market_raw_official_platt.csv").resolve()),
        "roi_file": str((out / "roi_ev_ge_1_auxiliary.csv").resolve()),
        "final_status": "OFFICIAL_10Y_PLATT_VALIDATION_PASSED",
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "artifact_audit.json", calibration_audit)
    report = [
        "# Latest Model Validation with Official 10Y Platt",
        "",
        f"- final_status: `{manifest['final_status']}`",
        f"- official_10y_platt_loaded: `{manifest['official_10y_platt_loaded']}`",
        f"- refit_performed: `{manifest['refit_performed']}`",
        f"- shadow_15y_status: `{calibration_audit['shadow_15y_status']}`",
        f"- usable_for_roi_judgement: `{manifest['usable_for_roi_judgement']}`",
        "",
        "## Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False),
    ]
    (out / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
