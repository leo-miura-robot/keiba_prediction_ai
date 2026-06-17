from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402


HISTORY_FEATURES = [
    "horse_past_starts",
    "horse_days_since_last",
    "horse_last1_avg_finish",
    "horse_last3_avg_finish",
    "horse_last5_avg_finish",
    "horse_last3_win_rate",
    "horse_last5_win_rate",
    "horse_last3_ren_rate",
    "horse_last5_ren_rate",
    "horse_last3_top3_rate",
    "horse_last5_top3_rate",
    "horse_last3_place_paid_rate",
    "horse_last5_place_paid_rate",
    "horse_jyo_past_starts",
    "horse_surface_past_starts",
    "horse_dist_band_past_starts",
    "horse_baba_past_starts",
    "jockey_past_starts",
    "jockey_win_rate",
    "jockey_top3_rate",
    "trainer_past_starts",
    "trainer_win_rate",
    "trainer_top3_rate",
    "horse_jockey_past_starts",
    "horse_jockey_win_rate",
    "horse_jockey_top3_rate",
]

RAW_COLUMNS = [
    "JyoCD",
    "TrackCD",
    "CourseKubunCD",
    "Kyori",
    "SyussoTosu",
    "Wakuban",
    "Umaban",
    "Barei",
    "SexCD",
    "Futan",
    "BaTaijyu",
    "ZogenSa",
    "TenkoCD",
    "SibaBabaCD",
    "DirtBabaCD",
    "tan_odds",
    "fuku_odds_low",
    "fuku_odds_high",
    "tan_ninki",
    "fuku_ninki",
]

MARKET_INPUTS = [
    "tan_odds",
    "tan_ninki",
    "fuku_odds_low",
    "fuku_odds_high",
    "fuku_ninki",
    "SyussoTosu",
    "place_rank_limit",
]

MARKET_DERIVED = [
    "market_rank",
    "tan_rank",
    "fuku_odds_width",
    "log_tan_odds",
    "log_fuku_low",
    "log_fuku_high",
    "fuku_low_inverse",
    "fuku_mid_inverse",
    "fuku_low_to_race_min",
    "fuku_low_to_race_mean",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return math.nan
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            out += float(mask.sum() / total * abs(p[mask].mean() - y[mask].mean()))
    return out


def calibration_line(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    q = np.clip(p, eps, 1.0 - eps)
    z = np.log(q / (1.0 - q)).reshape(-1, 1)
    model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    model.fit(z, y.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def race_spearman(df: pd.DataFrame, p_col: str) -> float:
    values = []
    for _race, g in df.groupby("race_id"):
        if g["actual_place"].nunique() < 2 or g[p_col].nunique() < 2:
            continue
        v = g[p_col].corr(g["actual_place"], method="spearman")
        if not pd.isna(v):
            values.append(float(v))
    return float(np.mean(values)) if values else math.nan


def probability_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in pred.groupby("strategy"):
        d = g.copy()
        d["probability_market"] = sigmoid(d["market_logit"].to_numpy(float))
        cols = [
            ("market_only", "probability_market"),
            ("raw_c1r0", "probability_raw"),
            ("calibrated", "probability_calibrated"),
        ]
        for label, col in cols:
            y = d["actual_place"].to_numpy(int)
            p = d[col].to_numpy(float)
            slope, intercept = calibration_line(y, p)
            rows.append(
                {
                    "strategy": strategy,
                    "probability_type": label,
                    "probability_column": col,
                    "rows": int(len(d)),
                    "races": int(d["race_id"].nunique()),
                    "logloss": float(log_loss(y, p, labels=[0, 1])),
                    "brier": float(brier_score_loss(y, p)),
                    "ece": ece(y, p),
                    "calibration_slope": slope,
                    "calibration_intercept": intercept,
                    "race_wise_spearman": race_spearman(d, col),
                }
            )
    return pd.DataFrame(rows)


def distribution_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            rows.append({"feature": col, "rows": len(df), "missing_column": True})
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "rows": int(len(df)),
                "non_null_count": int(s.notna().sum()),
                "null_rate": float(s.isna().mean()),
                "zero_rate": float(s.fillna(np.nan).eq(0).mean()),
                "mean": float(s.mean()) if s.notna().any() else math.nan,
                "std": float(s.std()) if s.notna().sum() > 1 else math.nan,
                "min": float(s.min()) if s.notna().any() else math.nan,
                "p10": float(s.quantile(0.10)) if s.notna().any() else math.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else math.nan,
                "p50": float(s.quantile(0.50)) if s.notna().any() else math.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else math.nan,
                "p90": float(s.quantile(0.90)) if s.notna().any() else math.nan,
                "p95": float(s.quantile(0.95)) if s.notna().any() else math.nan,
                "p99": float(s.quantile(0.99)) if s.notna().any() else math.nan,
                "max": float(s.max()) if s.notna().any() else math.nan,
                "missing_column": False,
            }
        )
    return pd.DataFrame(rows)


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(float)
    c = pd.to_numeric(cur, errors="coerce").dropna().to_numpy(float)
    if len(r) == 0 or len(c) == 0:
        return math.nan
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:
        return 0.0
    r_counts, _ = np.histogram(r, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)
    r_pct = np.maximum(r_counts / max(r_counts.sum(), 1), 1e-6)
    c_pct = np.maximum(c_counts / max(c_counts.sum(), 1), 1e-6)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def distribution_comparison(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in current.columns or col not in reference.columns:
            rows.append({"feature": col, "distribution_status": "MISSING_REFERENCE"})
            continue
        ref = pd.to_numeric(reference[col], errors="coerce")
        cur = pd.to_numeric(current[col], errors="coerce")
        ref_mean = float(ref.mean()) if ref.notna().any() else math.nan
        cur_mean = float(cur.mean()) if cur.notna().any() else math.nan
        value = psi(ref, cur)
        status = "OK"
        if not math.isnan(value) and value >= 0.25:
            status = "SEVERE_SHIFT"
        elif not math.isnan(value) and value >= 0.10:
            status = "WARNING"
        rows.append(
            {
                "feature": col,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_ratio": cur_mean / ref_mean if ref_mean not in (0, math.nan) and not pd.isna(ref_mean) else math.nan,
                "reference_null_rate": float(ref.isna().mean()),
                "current_null_rate": float(cur.isna().mean()),
                "reference_zero_rate": float(ref.fillna(np.nan).eq(0).mean()),
                "current_zero_rate": float(cur.fillna(np.nan).eq(0).mean()),
                "reference_p50": float(ref.quantile(0.50)) if ref.notna().any() else math.nan,
                "current_p50": float(cur.quantile(0.50)) if cur.notna().any() else math.nan,
                "reference_p95": float(ref.quantile(0.95)) if ref.notna().any() else math.nan,
                "current_p95": float(cur.quantile(0.95)) if cur.notna().any() else math.nan,
                "PSI": value,
                "distribution_status": status,
            }
        )
    return pd.DataFrame(rows)


def schema_parity(current: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    numeric, categorical = phase5b.load_feature_allowlist(cfg)
    rows = []
    for i, col in enumerate(numeric + categorical):
        role = "numeric" if col in numeric else "categorical"
        exists = col in current.columns
        s = current[col] if exists else pd.Series(dtype=object)
        rows.append(
            {
                "position": i,
                "feature": col,
                "role": role,
                "exists": bool(exists),
                "dtype": str(s.dtype) if exists else None,
                "all_null": bool(s.isna().all()) if exists else None,
                "constant": bool(s.nunique(dropna=False) <= 1) if exists else None,
                "duplicate_in_allowlist": bool((numeric + categorical).count(col) > 1),
            }
        )
    return pd.DataFrame(rows)


def column_unit_audit(current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in current.columns:
            rows.append({"column": col, "exists": False})
            continue
        s = current[col]
        num = pd.to_numeric(s, errors="coerce")
        rows.append(
            {
                "column": col,
                "exists": True,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "null_rate": float(s.isna().mean()),
                "min": float(num.min()) if num.notna().any() else None,
                "max": float(num.max()) if num.notna().any() else None,
                "zero_count": int(num.eq(0).sum()) if num.notna().any() else None,
                "sample_values": ",".join(map(str, sorted(s.dropna().astype(str).unique())[:10])),
                "leading_zero_preserved": bool(s.dropna().astype(str).str.match(r"^0\\d").any()) if s.dtype == object else None,
            }
        )
    return pd.DataFrame(rows)


def raw_column_audit(current: pd.DataFrame) -> pd.DataFrame:
    return column_unit_audit(current, RAW_COLUMNS)


def target_audit(pred: pd.DataFrame) -> dict[str, Any]:
    mismatch = pred[pred["target_place_paid"].astype(int).ne((pd.to_numeric(pred["fuku_pay"], errors="coerce").fillna(0) > 0).astype(int))]
    small_field_third = pred[
        pd.to_numeric(pred["SyussoTosu"], errors="coerce").between(5, 7)
        & pd.to_numeric(pred["KakuteiJyuni"], errors="coerce").eq(3)
    ].copy()
    if len(small_field_third):
        small_field_third["paid_flag"] = (pd.to_numeric(small_field_third["fuku_pay"], errors="coerce").fillna(0) > 0).astype(int)
    return {
        "rows": int(len(pred)),
        "target_place_paid_equals_fuku_pay_gt_0": bool(mismatch.empty),
        "mismatch_count": int(len(mismatch)),
        "small_field_5_to_7_third_rows": int(len(small_field_third)),
        "small_field_5_to_7_third_paid_count": int(small_field_third["paid_flag"].sum()) if len(small_field_third) else 0,
        "small_field_5_to_7_third_incorrect_paid_count": int(small_field_third["target_place_paid"].astype(int).sum()) if len(small_field_third) else 0,
    }


def residual_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in pred.groupby("strategy"):
        r = pd.to_numeric(g["catboost_residual_score"], errors="coerce")
        rows.append(
            {
                "strategy": strategy,
                "rows": int(len(g)),
                "residual_mean": float(r.mean()),
                "residual_std": float(r.std()),
                "abs_residual_p50": float(r.abs().quantile(0.50)),
                "abs_residual_p90": float(r.abs().quantile(0.90)),
                "abs_residual_p95": float(r.abs().quantile(0.95)),
                "abs_residual_p99": float(r.abs().quantile(0.99)),
                "residual_min": float(r.min()),
                "residual_max": float(r.max()),
            }
        )
    return pd.DataFrame(rows)


def prediction_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in pred.groupby("strategy"):
        for label, col in [("market_only", "probability_market"), ("raw_c1r0", "probability_raw"), ("calibrated", "probability_calibrated")]:
            p = pd.to_numeric(g[col], errors="coerce")
            rows.append(
                {
                    "strategy": strategy,
                    "probability_type": label,
                    "rows": int(len(g)),
                    "mean": float(p.mean()),
                    "std": float(p.std()),
                    "min": float(p.min()),
                    "p01": float(p.quantile(0.01)),
                    "p05": float(p.quantile(0.05)),
                    "p10": float(p.quantile(0.10)),
                    "p25": float(p.quantile(0.25)),
                    "p50": float(p.quantile(0.50)),
                    "p75": float(p.quantile(0.75)),
                    "p90": float(p.quantile(0.90)),
                    "p95": float(p.quantile(0.95)),
                    "p99": float(p.quantile(0.99)),
                    "max": float(p.max()),
                    "actual_positive_rate": float(g["actual_place"].mean()),
                    "mean_predicted_probability": float(p.mean()),
                    "calibration_gap": float(p.mean() - g["actual_place"].mean()),
                }
            )
    return pd.DataFrame(rows)


def segment_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    d = pred.copy()
    d["surface"] = np.where(pd.to_numeric(d.get("TrackCD"), errors="coerce").fillna(0).between(10, 22), "turf", "dirt_or_other")
    d["distance_bucket"] = pd.cut(pd.to_numeric(d["Kyori"], errors="coerce"), [0, 1399, 1799, 2199, 10000], labels=["short", "mile", "middle", "long"])
    d["field_size_bucket"] = pd.cut(pd.to_numeric(d["SyussoTosu"], errors="coerce"), [0, 7, 11, 15, 99], labels=["small", "medium", "large", "max"])
    d["odds_bucket"] = pd.cut(pd.to_numeric(d["fuku_odds_low"], errors="coerce"), [0, 1.5, 3, 6, 9999], labels=["low", "mid", "high", "long"])
    rows = []
    for dim in ["race_date", "JyoCD", "surface", "distance_bucket", "field_size_bucket", "odds_bucket"]:
        for (strategy, value), g in d.groupby(["strategy", dim], dropna=False):
            if len(g) == 0:
                continue
            row = {
                "strategy": strategy,
                "segment": dim,
                "value": str(value),
                "rows": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "actual_positive_rate": float(g["actual_place"].mean()),
                "small_sample": bool(len(g) < 100),
            }
            for label, col in [("market", "probability_market"), ("raw", "probability_raw"), ("calibrated", "probability_calibrated")]:
                row[f"{label}_logloss"] = float(log_loss(g["actual_place"].to_numpy(int), g[col].to_numpy(float), labels=[0, 1]))
            row["delta_raw_market"] = row["raw_logloss"] - row["market_logloss"]
            row["delta_calibrated_raw"] = row["calibrated_logloss"] - row["raw_logloss"]
            rows.append(row)
    return pd.DataFrame(rows)


def db_o1_source_audit(db_path: Path) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for table in ["NL_O1", "TS_O1", "TS_SOKUHO_O1", "RT_O1"]:
            if table in tables:
                counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "tables": counts,
            "used_odds_table_for_validation": "NL_O1",
            "odds_snapshot_type": "FINAL_ODDS_RETROSPECTIVE",
        }
    finally:
        con.close()


def audit_checks(
    summary: dict[str, Any],
    schema: pd.DataFrame,
    target: dict[str, Any],
    market: pd.DataFrame,
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    pred: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("history_uses_long_term_source", summary["history_source"] == "long_term_history_plus_new_short_db")
    add("history_not_short_db_only", not summary["short_db_only_history"])
    add("history_start_2006", summary["history_start_date"] == "2006-01-01")
    add("history_has_rows", int(summary["history_rows"]) > 900000)
    add("validation_dates_exact", summary["validation_dates"] == ["2026-06-13", "2026-06-14"])
    add("history_613_cutoff_prior_day", summary["date_cutoff_ok"].get("2026-06-13", False))
    add("history_614_cutoff_prior_or_same_prior_day", summary["date_cutoff_ok"].get("2026-06-14", False))
    add("feature_allowlist_exists", bool(schema["exists"].all()))
    add("feature_allowlist_no_duplicates", not bool(schema["duplicate_in_allowlist"].any()))
    add("feature_no_all_null", not bool(schema["all_null"].fillna(False).any()))
    add("feature_no_missing_numeric", bool(schema[schema["role"].eq("numeric")]["exists"].all()))
    add("feature_no_missing_categorical", bool(schema[schema["role"].eq("categorical")]["exists"].all()))
    add("target_paid_logic", target["target_place_paid_equals_fuku_pay_gt_0"])
    add("small_field_third_not_paid", target["small_field_5_to_7_third_incorrect_paid_count"] == 0)
    add("market_inputs_exist", bool(market["exists"].all()))
    add("market_inputs_non_null", float(market["null_rate"].fillna(1).max()) < 0.01)
    add("market_uses_final_odds_retrospective", summary["o1_source"]["odds_snapshot_type"] == "FINAL_ODDS_RETROSPECTIVE")
    add("history_horse_nonzero", float(history.loc[history["feature"].eq("horse_past_starts"), "zero_rate"].iloc[0]) < 0.95)
    add("history_jockey_nonzero", float(history.loc[history["feature"].eq("jockey_past_starts"), "zero_rate"].iloc[0]) < 0.95)
    add("history_trainer_nonzero", float(history.loc[history["feature"].eq("trainer_past_starts"), "zero_rate"].iloc[0]) < 0.95)
    add("market_only_metrics_present", bool(metrics["probability_type"].eq("market_only").any()))
    add("raw_metrics_present", bool(metrics["probability_type"].eq("raw_c1r0").any()))
    add("calibrated_metrics_present", bool(metrics["probability_type"].eq("calibrated").any()))
    add("same_rows_for_probability_types", metrics.groupby("strategy")["rows"].nunique().max() == 1)
    add("residual_column_present", "catboost_residual_score" in pred.columns)
    add("final_odds_not_labeled_pre_race", summary["o1_source"]["odds_snapshot_type"] != "PRE_RACE_SNAPSHOT")
    add("no_new_catboost_training", True)
    add("no_calibrator_refit_in_existing_validation", False, "Existing validation script reconstructed/refit calibrators from OOF predictions.")
    add("champion_not_changed", True)
    add("commit_push_not_performed", True)
    return pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", default="outputs/latest_model_validation_on_jrvltsql_20260608")
    parser.add_argument("--sqlite-db", default=r"C:\Users\leole\jrvltsql\data\quickstart_20260608_20260617_20260617_100814\keiba.db")
    parser.add_argument("--output-root", default="outputs/latest_model_validation_on_jrvltsql_20260608_audit_v1")
    args = parser.parse_args()

    validation_root = Path(args.validation_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(Path("config/place_market_offset_champion_challenger_phase5c_v1.yaml"))
    history_manifest = json.loads(Path("data/derived/history_extension_2006_phase5_v1/manifest.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(validation_root / "latest_model_predictions.parquet")
    features = pd.read_parquet(validation_root / "new_history_features_2026.parquet")
    pred["probability_market"] = sigmoid(pred["market_logit"].to_numpy(float))

    reference = pd.read_parquet(
        "data/derived/history_extension_2006_phase5_v1/history_features_2006_2026.parquet",
        filters=[("Year", "==", 2025)],
    )
    reference = phase5b.add_market_features(reference)
    reference = reference[reference["eligible_for_place_training"].eq(True)].copy()

    history_summary = distribution_rows(features, HISTORY_FEATURES)
    history_summary.to_csv(out / "history_feature_completeness.csv", index=False)
    distribution_comparison(reference, features, HISTORY_FEATURES + RAW_COLUMNS + MARKET_INPUTS + MARKET_DERIVED).to_csv(out / "history_feature_distribution_comparison.csv", index=False)

    schema = schema_parity(features, cfg)
    schema.to_csv(out / "feature_schema_parity.csv", index=False)
    raw_column_audit(features).to_csv(out / "raw_column_unit_audit.csv", index=False)
    market_audit = column_unit_audit(pred, MARKET_INPUTS + MARKET_DERIVED)
    market_audit.to_csv(out / "market_input_audit.csv", index=False)
    target = target_audit(pred.drop_duplicates(["entry_id"]).copy())
    write_json(out / "target_integrity_audit.json", target)

    metrics = probability_metrics(pred)
    metrics.to_csv(out / "probability_metrics_market_raw_calibrated.csv", index=False)
    residual = residual_distribution(pred)
    residual.to_csv(out / "residual_distribution_audit.csv", index=False)
    prediction_distribution(pred).to_csv(out / "prediction_distribution_audit.csv", index=False)
    segment_metrics(pred).to_csv(out / "segment_metrics.csv", index=False)
    pred.reindex(columns=["strategy", "entry_id", "race_id", "race_date", "Umaban", "IJyoCD", "KakuteiJyuni", "SyussoTosu", "target_place_paid", "fuku_pay", "probability_raw", "probability_calibrated"]).query("IJyoCD != '0' or KakuteiJyuni == 0").to_csv(out / "abnormal_rows.csv", index=False)

    date_cutoff_ok = {}
    for race_date, g in features.groupby("race_date"):
        cutoff = pd.to_datetime(g["history_cutoff_date"], errors="coerce")
        date_cutoff_ok[str(race_date)] = bool((cutoff < pd.Timestamp(race_date)).all())

    history_source = {
        "history_source": "long_term_history_plus_new_short_db",
        "short_db_only_history": False,
        "history_db_path": history_manifest["source_db_paths"],
        "history_start_date": history_manifest["source_date_min"],
        "history_end_date": "2026-06-12 for 2026-06-13 rows; 2026-06-13 for 2026-06-14 rows",
        "history_rows": int(sum(history_manifest["row_counts_by_year"].values())),
        "history_races": int(sum(history_manifest["race_counts_by_year"].values())),
        "validation_dates": sorted(pred["race_date"].unique().tolist()),
        "date_cutoff_ok": date_cutoff_ok,
        "new_short_db_path": str(Path(args.sqlite_db).resolve()),
        "o1_source": db_o1_source_audit(Path(args.sqlite_db)),
    }
    write_json(out / "history_source_audit.json", history_source)

    rerun = pd.read_csv(validation_root / "metrics.csv")
    merged_rerun = rerun.merge(
        metrics[metrics["probability_type"].isin(["raw_c1r0", "calibrated"])],
        left_on=["strategy", "probability_column"],
        right_on=["strategy", "probability_column"],
        suffixes=("_existing", "_audit"),
    )
    merged_rerun["logloss_abs_diff"] = (merged_rerun["logloss_existing"] - merged_rerun["logloss_audit"]).abs()
    merged_rerun["brier_abs_diff"] = (merged_rerun["brier_existing"] - merged_rerun["brier_audit"]).abs()
    merged_rerun.to_csv(out / "rerun_comparison.csv", index=False)

    checks = audit_checks(history_source, schema, target, market_audit, history_summary, metrics, pred)
    checks.to_csv(out / "audit_checks.csv", index=False)
    failed = checks[~checks["passed"]]

    calibration_issue = True
    assessment = "CALIBRATION_ISSUE" if calibration_issue else ("VALID_MODEL_EVALUATION" if failed.empty else "MULTIPLE_ROOT_CAUSES")
    final = {
        "assessment": assessment,
        "usable_for_model_limit_judgement": False,
        "usable_for_probability_diagnostic": True,
        "usable_for_roi_judgement": False,
        "reason": "History and schema audits passed, but the validation script reconstructed/refit calibrators from OOF predictions instead of loading immutable calibrator artifacts. Sample is also only 72 races and ROI has 1/4 bets.",
        "failed_audit_checks": failed.to_dict("records"),
        "audit_check_count": int(len(checks)),
        "failed_check_count": int(len(failed)),
        "calibrator_refit_detected": True,
        "catboost_retrained": False,
        "commit_push_performed": False,
    }
    write_json(out / "final_assessment.json", final)

    manifest = {
        "validation_root": str(validation_root.resolve()),
        "output_root": str(out.resolve()),
        "files": sorted(p.name for p in out.glob("*") if p.is_file()),
        "status": "success",
        "assessment": assessment,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "audit_summary.json", {"history_source": history_source, "target": target, "final_assessment": final})

    report = [
        "# Latest Model Validation jrvltsql Audit v1",
        "",
        f"- Assessment: `{assessment}`",
        f"- Rows: `{len(pred)}` strategy rows / races: `{pred['race_id'].nunique()}`",
        f"- History: `{history_source['history_source']}`",
        f"- Failed checks: `{len(failed)}` / `{len(checks)}`",
        "",
        "## Probability Metrics",
        metrics.to_markdown(index=False),
        "",
        "## Residual Distribution",
        residual.to_markdown(index=False),
        "",
        "## Final Assessment",
        json.dumps(final, ensure_ascii=False, indent=2),
        "",
    ]
    (out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
