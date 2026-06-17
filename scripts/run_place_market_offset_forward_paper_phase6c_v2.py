from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config/place_market_offset_forward_paper_phase6c_v2.yaml"


SCHEMA = {
    "prediction_runs": """
        CREATE TABLE IF NOT EXISTS prediction_runs (
            prediction_run_id TEXT PRIMARY KEY,
            prediction_generated_at TEXT NOT NULL,
            race_date TEXT NOT NULL,
            data_cutoff_at TEXT NOT NULL,
            odds_observed_at TEXT NOT NULL,
            source_data_latest_at TEXT NOT NULL,
            code_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            feature_hash TEXT NOT NULL,
            threshold_policy_version TEXT NOT NULL,
            is_fixture INTEGER NOT NULL DEFAULT 0,
            champion_strategy TEXT,
            model_artifact_path TEXT,
            model_artifact_sha256 TEXT,
            calibrator_artifact_path TEXT,
            calibrator_artifact_sha256 TEXT,
            calibrator_type TEXT,
            calibrator_input_space TEXT,
            calibrator_refit_performed INTEGER,
            ev_definition TEXT,
            odds_snapshot_type TEXT,
            retrospective_only INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE(race_date, is_fixture)
        )
    """,
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            calibration_method TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            race_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            horse_no TEXT NOT NULL,
            probability_raw REAL NOT NULL,
            probability_calibrated REAL NOT NULL,
            market_logit REAL,
            residual_raw REAL,
            fuku_odds_low_at_prediction REAL NOT NULL,
            ev_at_prediction REAL NOT NULL,
            probability_market REAL,
            expected_value REAL,
            model_artifact_path TEXT,
            model_artifact_sha256 TEXT,
            calibrator_artifact_path TEXT,
            calibrator_artifact_sha256 TEXT,
            calibrator_type TEXT,
            calibrator_input_space TEXT,
            calibrator_refit_performed INTEGER,
            odds_snapshot_type TEXT,
            odds_observed_at TEXT,
            prediction_created_at TEXT,
            retrospective_only INTEGER,
            model_version TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            calibrator_version TEXT NOT NULL,
            calibrator_hash TEXT NOT NULL,
            PRIMARY KEY(prediction_run_id, strategy, entry_id),
            FOREIGN KEY(prediction_run_id) REFERENCES prediction_runs(prediction_run_id)
        )
    """,
    "prediction_tiers": """
        CREATE TABLE IF NOT EXISTS prediction_tiers (
            prediction_run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            race_id TEXT NOT NULL,
            threshold REAL NOT NULL,
            threshold_tier TEXT NOT NULL,
            paper_bet_flag INTEGER NOT NULL,
            paper_stake_if_bet INTEGER NOT NULL,
            PRIMARY KEY(prediction_run_id, strategy, entry_id, threshold),
            FOREIGN KEY(prediction_run_id) REFERENCES prediction_runs(prediction_run_id)
        )
    """,
    "settlements": """
        CREATE TABLE IF NOT EXISTS settlements (
            settlement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_run_id TEXT NOT NULL,
            strategy TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            race_id TEXT NOT NULL,
            race_date TEXT NOT NULL,
            threshold REAL NOT NULL,
            threshold_tier TEXT NOT NULL,
            settled_at TEXT NOT NULL,
            target_place_paid INTEGER NOT NULL,
            fuku_pay REAL NOT NULL,
            paper_stake INTEGER NOT NULL,
            paper_payout REAL NOT NULL,
            paper_profit REAL NOT NULL,
            settlement_status TEXT NOT NULL,
            FOREIGN KEY(prediction_run_id) REFERENCES prediction_runs(prediction_run_id)
        )
    """,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def db_path(output_root: Path) -> Path:
    return output_root / "forward_paper.sqlite"


def connect(output_root: Path) -> sqlite3.Connection:
    output_root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path(output_root))
    con.row_factory = sqlite3.Row
    for ddl in SCHEMA.values():
        con.execute(ddl)
    ensure_schema_columns(con)
    con.commit()
    return con


def ensure_schema_columns(con: sqlite3.Connection) -> None:
    expected = {
        "prediction_runs": {
            "champion_strategy": "TEXT",
            "model_artifact_path": "TEXT",
            "model_artifact_sha256": "TEXT",
            "calibrator_artifact_path": "TEXT",
            "calibrator_artifact_sha256": "TEXT",
            "calibrator_type": "TEXT",
            "calibrator_input_space": "TEXT",
            "calibrator_refit_performed": "INTEGER",
            "ev_definition": "TEXT",
            "odds_snapshot_type": "TEXT",
            "retrospective_only": "INTEGER",
        },
        "predictions": {
            "probability_market": "REAL",
            "expected_value": "REAL",
            "model_artifact_path": "TEXT",
            "model_artifact_sha256": "TEXT",
            "calibrator_artifact_path": "TEXT",
            "calibrator_artifact_sha256": "TEXT",
            "calibrator_type": "TEXT",
            "calibrator_input_space": "TEXT",
            "calibrator_refit_performed": "INTEGER",
            "odds_snapshot_type": "TEXT",
            "odds_observed_at": "TEXT",
            "prediction_created_at": "TEXT",
            "retrospective_only": "INTEGER",
        },
    }
    for table, columns in expected.items():
        existing = {row["name"] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in columns.items():
            if name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def fixture_predictions(race_date: str) -> pd.DataFrame:
    rows = []
    now = utc_now()
    for i, (p, odds, pay) in enumerate([(0.55, 2.2, 220), (0.30, 3.8, 0), (0.20, 6.1, 610), (0.12, 8.0, 0)], start=1):
        rows.append(
            {
                "strategy": "ROLLING_10Y",
                "calibration_method": "PLATT_SCALING",
                "entry_id": f"{race_date}_{i}",
                "race_id": f"{race_date}_R01",
                "race_date": race_date,
                "horse_no": str(i),
                "Umaban": str(i),
                "probability_market": max(0.001, p - 0.02),
                "probability_raw": max(0.001, p - 0.01),
                "probability_calibrated": min(0.999, p),
                "market_logit": 0.0,
                "residual_raw": 0.0,
                "fuku_odds_low": odds,
                "fuku_odds_low_at_prediction": odds,
                "expected_value": min(0.999, p) * odds,
                "ev_at_prediction": min(0.999, p) * odds,
                "model_artifact_path": "fixture_model.cbm",
                "model_artifact_sha256": "fixture_model_hash",
                "calibrator_artifact_path": "fixture_calibrator.json",
                "calibrator_artifact_sha256": "fixture_calibrator_hash",
                "calibrator_type": "PLATT_SCALING",
                "calibrator_input_space": "logit_probability_raw",
                "calibrator_refit_performed": False,
                "odds_snapshot_type": "FINAL_ODDS",
                "odds_observed_at": now,
                "prediction_created_at": now,
                "retrospective_only": True,
                "fuku_pay": pay,
                "target_place_paid": int(pay > 0),
            }
        )
    return pd.DataFrame(rows)


def read_prediction_input(args: argparse.Namespace, cfg: dict[str, Any]) -> pd.DataFrame:
    if args.fixture and not args.input_csv:
        return fixture_predictions(args.race_date)
    if not args.input_csv:
        raise ValueError("--input-csv is required without --fixture")
    df = pd.read_csv(args.input_csv)
    required = {
        "strategy",
        "calibration_method",
        "entry_id",
        "race_id",
        "race_date",
        "probability_market",
        "probability_raw",
        "probability_calibrated",
        "fuku_odds_low",
        "expected_value",
        "model_artifact_path",
        "model_artifact_sha256",
        "calibrator_artifact_path",
        "calibrator_artifact_sha256",
        "calibrator_type",
        "calibrator_input_space",
        "calibrator_refit_performed",
        "odds_snapshot_type",
        "odds_observed_at",
        "prediction_created_at",
        "retrospective_only",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing prediction input columns: {missing}")
    if "horse_no" not in df.columns and "Umaban" in df.columns:
        df["horse_no"] = df["Umaban"].astype(str)
    if "horse_no" not in df.columns:
        raise ValueError("Missing prediction input columns: ['horse_no']")
    if "fuku_odds_low_at_prediction" not in df.columns:
        df["fuku_odds_low_at_prediction"] = df["fuku_odds_low"]
    allowed = {(cfg["champion"]["strategy"], cfg["champion"]["calibration_method"])}
    actual = set(map(tuple, df[["strategy", "calibration_method"]].drop_duplicates().to_numpy()))
    if not actual.issubset(allowed):
        raise ValueError(f"Unexpected strategy/calibration pairs: {actual - allowed}")
    if df["calibrator_refit_performed"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("calibrator_refit_performed must be false")
    for col in ["probability_market", "probability_raw", "probability_calibrated", "fuku_odds_low_at_prediction", "expected_value"]:
        vals = pd.to_numeric(df[col], errors="raise").to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"NaN/inf in prediction input column: {col}")
    if pd.to_numeric(df["fuku_odds_low_at_prediction"], errors="raise").le(0).any():
        raise ValueError("odds missing or <= 0")
    recalculated = df["probability_calibrated"].astype(float) * df["fuku_odds_low_at_prediction"].astype(float)
    if not np.allclose(recalculated.to_numpy(float), df["expected_value"].astype(float).to_numpy(float), rtol=0, atol=1e-12):
        raise ValueError("expected_value does not match probability_calibrated * fuku_odds_low")
    return df


def prediction_run_id(race_date: str, is_fixture: bool, cfg_hash: str) -> str:
    return hashlib.sha256(f"{race_date}|{int(is_fixture)}|{cfg_hash}".encode()).hexdigest()[:24]


def tier_rows(df: pd.DataFrame, cfg: dict[str, Any], run_id: str) -> pd.DataFrame:
    rows = []
    stake = int(cfg["paper_stake_yen"])
    for _, r in df.iterrows():
        ev = float(r["probability_calibrated"]) * float(r["fuku_odds_low_at_prediction"])
        for tier in cfg["threshold_tiers"]:
            flag = ev >= float(tier["threshold"])
            rows.append(
                {
                    "prediction_run_id": run_id,
                    "strategy": r["strategy"],
                    "entry_id": r["entry_id"],
                    "race_id": r["race_id"],
                    "threshold": float(tier["threshold"]),
                    "threshold_tier": tier["tier"],
                    "paper_bet_flag": int(flag),
                    "paper_stake_if_bet": stake if flag else 0,
                }
            )
    return pd.DataFrame(rows)


def audit_tier_inclusion(tiers: pd.DataFrame) -> tuple[bool, str]:
    for (run_id, strategy), g in tiers.groupby(["prediction_run_id", "strategy"]):
        sets = {tier: set(g[(g["threshold_tier"].eq(tier)) & (g["paper_bet_flag"].eq(1))]["entry_id"]) for tier in ["CORE", "MARGIN", "HIGH", "VERY_HIGH"]}
        if not sets["VERY_HIGH"].issubset(sets["HIGH"]):
            return False, f"VERY_HIGH not subset HIGH for {run_id} {strategy}"
        if not sets["HIGH"].issubset(sets["MARGIN"]):
            return False, f"HIGH not subset MARGIN for {run_id} {strategy}"
        if not sets["MARGIN"].issubset(sets["CORE"]):
            return False, f"MARGIN not subset CORE for {run_id} {strategy}"
    return True, "ok"


def predict(args: argparse.Namespace) -> int:
    cfg = load_yaml(Path(args.config))
    output_root = Path(args.output_root)
    con = connect(output_root)
    cfg_hash = sha256_json(cfg)
    df = read_prediction_input(args, cfg).copy()
    csv_run_ids = sorted(df["prediction_run_id"].dropna().astype(str).unique().tolist()) if "prediction_run_id" in df.columns else []
    run_id = csv_run_ids[0] if len(csv_run_ids) == 1 else prediction_run_id(args.race_date, bool(args.fixture), cfg_hash)
    if len(csv_run_ids) > 1:
        raise SystemExit("prediction input contains multiple prediction_run_id values")
    if con.execute("SELECT 1 FROM prediction_runs WHERE prediction_run_id=?", (run_id,)).fetchone():
        raise SystemExit(f"Duplicate prediction run rejected: {run_id}")
    now = args.prediction_generated_at or utc_now()
    data_cutoff = args.data_cutoff_at or now
    odds_time = args.odds_observed_at or now
    source_latest = args.source_data_latest_at or data_cutoff
    if pd.Timestamp(data_cutoff) > pd.Timestamp(now) or pd.Timestamp(odds_time) > pd.Timestamp(now):
        raise SystemExit("timestamp violation: cutoff/odds cannot be after prediction_generated_at")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    df["ev_at_prediction"] = df["expected_value"].astype(float)
    con.execute(
        """
        INSERT INTO prediction_runs (
            prediction_run_id, prediction_generated_at, race_date, data_cutoff_at, odds_observed_at,
            source_data_latest_at, code_version, config_hash, feature_hash, threshold_policy_version,
            is_fixture, champion_strategy, model_artifact_path, model_artifact_sha256,
            calibrator_artifact_path, calibrator_artifact_sha256, calibrator_type,
            calibrator_input_space, calibrator_refit_performed, ev_definition,
            odds_snapshot_type, retrospective_only, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            now,
            args.race_date,
            data_cutoff,
            odds_time,
            source_latest,
            cfg["version"],
            cfg_hash,
            cfg["feature_hash"],
            cfg["threshold_policy_version"],
            int(args.fixture),
            cfg["champion"]["strategy"],
            str(df["model_artifact_path"].iloc[0]),
            str(df["model_artifact_sha256"].iloc[0]),
            str(df["calibrator_artifact_path"].iloc[0]),
            str(df["calibrator_artifact_sha256"].iloc[0]),
            str(df["calibrator_type"].iloc[0]),
            str(df["calibrator_input_space"].iloc[0]),
            int(False),
            cfg.get("ev_definition", "probability_calibrated * fuku_odds_low"),
            str(df["odds_snapshot_type"].iloc[0]),
            int(str(df["retrospective_only"].iloc[0]).lower() in {"true", "1"}),
            utc_now(),
        ),
    )
    pred_rows = []
    for _, r in df.iterrows():
        pred_rows.append(
            (
                run_id,
                r["strategy"],
                r["calibration_method"],
                r["entry_id"],
                r["race_id"],
                r["race_date"],
                str(r["horse_no"]),
                float(r["probability_raw"]),
                float(r["probability_calibrated"]),
                float(r.get("market_logit", 0.0)),
                float(r.get("residual_raw", 0.0)),
                float(r["fuku_odds_low_at_prediction"]),
                float(r["ev_at_prediction"]),
                float(r["probability_market"]),
                float(r["expected_value"]),
                str(r["model_artifact_path"]),
                str(r["model_artifact_sha256"]),
                str(r["calibrator_artifact_path"]),
                str(r["calibrator_artifact_sha256"]),
                str(r["calibrator_type"]),
                str(r["calibrator_input_space"]),
                int(False),
                str(r["odds_snapshot_type"]),
                str(r["odds_observed_at"]),
                str(r["prediction_created_at"]),
                int(str(r["retrospective_only"]).lower() in {"true", "1"}),
                cfg["model_version"],
                cfg["model_hash"],
                cfg["calibrator_version"],
                cfg["calibrator_hash"],
            )
        )
    con.executemany(
        """
        INSERT INTO predictions (
            prediction_run_id, strategy, calibration_method, entry_id, race_id, race_date, horse_no,
            probability_raw, probability_calibrated, market_logit, residual_raw,
            fuku_odds_low_at_prediction, ev_at_prediction, probability_market, expected_value,
            model_artifact_path, model_artifact_sha256, calibrator_artifact_path,
            calibrator_artifact_sha256, calibrator_type, calibrator_input_space,
            calibrator_refit_performed, odds_snapshot_type, odds_observed_at,
            prediction_created_at, retrospective_only, model_version, model_hash,
            calibrator_version, calibrator_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )
    tiers = tier_rows(df, cfg, run_id)
    ok, detail = audit_tier_inclusion(tiers)
    if not ok:
        raise SystemExit(detail)
    con.executemany(
        "INSERT INTO prediction_tiers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [tuple(x) for x in tiers[["prediction_run_id", "strategy", "entry_id", "race_id", "threshold", "threshold_tier", "paper_bet_flag", "paper_stake_if_bet"]].to_numpy()],
    )
    con.commit()
    report(args.output_root, include_fixture=True)
    print(json.dumps({"prediction_run_id": run_id, "rows": len(df), "tier_rows": len(tiers), "fixture": bool(args.fixture)}, ensure_ascii=False))
    return 0


def read_settlement_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.fixture:
        return fixture_predictions(args.race_date)[["entry_id", "race_id", "race_date", "target_place_paid", "fuku_pay"]].drop_duplicates(
            ["entry_id", "race_id", "race_date"]
        ).copy()
    if not args.settlement_csv:
        raise ValueError("--settlement-csv is required without --fixture")
    df = pd.read_csv(args.settlement_csv)
    required = {"entry_id", "race_id", "race_date", "target_place_paid", "fuku_pay"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing settlement columns: {missing}")
    return df.drop_duplicates(["entry_id", "race_id", "race_date"]).copy()


def settle(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    con = connect(output_root)
    runs = pd.read_sql_query("SELECT * FROM prediction_runs WHERE race_date=? AND is_fixture=?", con, params=(args.race_date, int(args.fixture)))
    if runs.empty:
        raise SystemExit("No prediction run found for settlement")
    settled_at = args.settled_at or utc_now()
    settlements = read_settlement_input(args)
    rows = []
    for _, run in runs.iterrows():
        if pd.Timestamp(settled_at) <= pd.Timestamp(run["prediction_generated_at"]):
            raise SystemExit("timestamp violation: settled_at must be after prediction_generated_at")
        pred = pd.read_sql_query("SELECT * FROM predictions WHERE prediction_run_id=?", con, params=(run["prediction_run_id"],))
        tiers = pd.read_sql_query("SELECT * FROM prediction_tiers WHERE prediction_run_id=?", con, params=(run["prediction_run_id"],))
        merged = tiers.merge(pred[["prediction_run_id", "strategy", "entry_id", "race_id", "race_date"]], on=["prediction_run_id", "strategy", "entry_id", "race_id"])
        merged = merged.merge(settlements, on=["entry_id", "race_id", "race_date"], how="left", validate="many_to_one")
        for _, r in merged.iterrows():
            if pd.isna(r["target_place_paid"]):
                status = "pending"
                target = 0
                fuku_pay = 0.0
            else:
                status = "settled"
                target = int(r["target_place_paid"])
                fuku_pay = float(r["fuku_pay"])
            stake = int(r["paper_stake_if_bet"])
            payout = fuku_pay if stake else 0.0
            rows.append(
                (
                    r["prediction_run_id"],
                    r["strategy"],
                    r["entry_id"],
                    r["race_id"],
                    r["race_date"],
                    float(r["threshold"]),
                    r["threshold_tier"],
                    settled_at,
                    target,
                    fuku_pay,
                    stake,
                    payout,
                    payout - stake,
                    status,
                )
            )
    con.executemany(
        """
        INSERT INTO settlements (
            prediction_run_id, strategy, entry_id, race_id, race_date, threshold, threshold_tier, settled_at,
            target_place_paid, fuku_pay, paper_stake, paper_payout, paper_profit, settlement_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    report(args.output_root, include_fixture=True)
    print(json.dumps({"settlement_rows_appended": len(rows), "fixture": bool(args.fixture)}, ensure_ascii=False))
    return 0


def roi_frame(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["races", "entries", "paper_bets", "stake", "payout", "profit", "ROI", "hit_count", "hit_rate", "average_odds", "median_odds", "average_probability", "average_EV", "median_EV"])
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        bets = g[g["paper_stake"] > 0]
        stake = float(bets["paper_stake"].sum())
        payout = float(bets["paper_payout"].sum())
        rows.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "races": int(g["race_id"].nunique()),
                "entries": int(g["entry_id"].nunique()),
                "paper_bets": int(len(bets)),
                "stake": stake,
                "payout": payout,
                "profit": payout - stake,
                "ROI": float(payout / stake * 100.0) if stake else np.nan,
                "hit_count": int((bets["paper_payout"] > 0).sum()),
                "hit_rate": float((bets["paper_payout"] > 0).mean()) if len(bets) else np.nan,
                "average_odds": float(bets["fuku_odds_low_at_prediction"].mean()) if len(bets) else np.nan,
                "median_odds": float(bets["fuku_odds_low_at_prediction"].median()) if len(bets) else np.nan,
                "average_probability": float(bets["probability_calibrated"].mean()) if len(bets) else np.nan,
                "average_EV": float(bets["ev_at_prediction"].mean()) if len(bets) else np.nan,
                "median_EV": float(bets["ev_at_prediction"].median()) if len(bets) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def report(output_root_arg: str, include_fixture: bool = False) -> int:
    output_root = Path(output_root_arg)
    con = connect(output_root)
    runs = pd.read_sql_query("SELECT * FROM prediction_runs", con)
    preds = pd.read_sql_query("SELECT * FROM predictions", con)
    tiers = pd.read_sql_query("SELECT * FROM prediction_tiers", con)
    settles = pd.read_sql_query("SELECT * FROM settlements", con)
    runs.to_parquet(output_root / "prediction_runs_export.parquet", index=False)
    preds.to_parquet(output_root / "predictions_export.parquet", index=False)
    tiers.to_parquet(output_root / "prediction_tiers_export.parquet", index=False)
    settles.to_parquet(output_root / "settlements_export.parquet", index=False)
    if not settles.empty:
        d = settles.merge(preds, on=["prediction_run_id", "strategy", "entry_id", "race_id", "race_date"], how="left").merge(runs[["prediction_run_id", "is_fixture"]], on="prediction_run_id")
        d = d[d["is_fixture"].eq(1 if include_fixture else 0)].copy()
    else:
        d = pd.DataFrame()
    if not d.empty:
        d["day"] = pd.to_datetime(d["race_date"]).dt.strftime("%Y-%m-%d")
        d["month"] = pd.to_datetime(d["race_date"]).dt.strftime("%Y-%m")
    roi_frame(d, ["day", "strategy", "calibration_method", "threshold", "threshold_tier"]).to_csv(output_root / "daily_summary.csv", index=False)
    roi_frame(d, ["month", "strategy", "calibration_method", "threshold", "threshold_tier"]).to_csv(output_root / "monthly_summary.csv", index=False)
    roi_frame(d, ["strategy", "calibration_method", "threshold", "threshold_tier"]).to_csv(output_root / "cumulative_summary.csv", index=False)
    threshold_comparison(d).to_csv(output_root / "threshold_comparison.csv", index=False)
    stress(d).to_csv(output_root / "high_payout_stress.csv", index=False)
    overlap(d).to_csv(output_root / "champion_challenger_overlap.csv", index=False)
    manifest = {"paper_trade_enabled": True, "real_money_betting": False, "operationally_activated": False, "include_fixture_in_reports": include_fixture, "db_path": str(db_path(output_root))}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def threshold_comparison(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    rows = []
    for strategy, g in d.groupby("strategy"):
        sets = {tier: set(g[(g["threshold_tier"].eq(tier)) & (g["paper_stake"].gt(0))]["entry_id"]) for tier in ["CORE", "MARGIN", "HIGH", "VERY_HIGH"]}
        for lower, higher in [("CORE", "MARGIN"), ("MARGIN", "HIGH"), ("HIGH", "VERY_HIGH")]:
            inc = sets[lower] - sets[higher]
            rows.append({"strategy": strategy, "lower_tier": lower, "higher_tier": higher, "lower_bets": len(sets[lower]), "higher_bets": len(sets[higher]), "common_bets": len(sets[lower] & sets[higher]), "lower_only_incremental_bets": len(inc), "jaccard": len(sets[lower] & sets[higher]) / len(sets[lower] | sets[higher]) if sets[lower] | sets[higher] else np.nan})
    return pd.DataFrame(rows)


def stress(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, tier), g in d.groupby(["strategy", "threshold_tier"]):
        bets = g[g["paper_stake"].gt(0)].copy()
        stake = bets["paper_stake"].sum()
        payout = bets["paper_payout"].sum()
        normal = payout / stake * 100.0 if stake else np.nan
        hits = bets[bets["paper_payout"].gt(0)].sort_values("paper_payout", ascending=False)
        total_hit_payout = hits["paper_payout"].sum()
        for limit in [1, 3, 5, 10]:
            idx = hits.head(limit).index
            for kind in ["row_removed", "payout_zeroed"]:
                frame = bets.drop(index=idx) if kind == "row_removed" else bets.copy()
                if kind == "payout_zeroed":
                    frame.loc[idx, "paper_payout"] = 0.0
                st = frame["paper_stake"].sum()
                pay = frame["paper_payout"].sum()
                roi = pay / st * 100.0 if st else np.nan
                rows.append({"strategy": strategy, "threshold_tier": tier, "stress_type": kind, "limit": limit, "normal_roi": normal, "stress_roi": roi, "roi_drop_point": normal - roi if not np.isnan(normal) and not np.isnan(roi) else np.nan, "remaining_bet_count": len(frame), "removed_or_zeroed_payout_share": hits.head(limit)["paper_payout"].sum() / total_hit_payout if total_hit_payout else np.nan})
    return pd.DataFrame(rows)


def overlap(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    rows = []
    for tier, g in d[d["paper_stake"].gt(0)].groupby("threshold_tier"):
        by = {s: set(x["entry_id"]) for s, x in g.groupby("strategy")}
        a = by.get("ROLLING_10Y", set())
        b = by.get("ROLLING_15Y", set())
        rows.append({"threshold_tier": tier, "common_bets": len(a & b), "10Y_only_bets": len(a - b), "15Y_only_bets": len(b - a), "jaccard": len(a & b) / len(a | b) if a | b else np.nan})
    return pd.DataFrame(rows)


def smoke(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    if db_path(output_root).exists():
        db_path(output_root).unlink()
    race_date = "2099-01-01"
    base = ["--config", args.config, "--output-root", str(output_root), "--race-date", race_date, "--fixture"]
    predict(parse_args(["predict", *base, "--prediction-generated-at", "2099-01-01T09:00:00+00:00", "--data-cutoff-at", "2099-01-01T08:00:00+00:00", "--odds-observed-at", "2099-01-01T08:30:00+00:00"]))
    duplicate_rejected = False
    try:
        predict(parse_args(["predict", *base, "--prediction-generated-at", "2099-01-01T09:00:00+00:00"]))
    except SystemExit:
        duplicate_rejected = True
    timestamp_rejected = False
    try:
        settle(parse_args(["settle", *base, "--settled-at", "2099-01-01T08:59:00+00:00"]))
    except SystemExit:
        timestamp_rejected = True
    pred_before = pd.read_sql_query("SELECT * FROM predictions", connect(output_root)).to_json()
    settle(parse_args(["settle", *base, "--settled-at", "2099-01-01T10:00:00+00:00"]))
    settle(parse_args(["settle", *base, "--settled-at", "2099-01-01T10:05:00+00:00"]))
    con = connect(output_root)
    pred_after = pd.read_sql_query("SELECT * FROM predictions", con).to_json()
    settlements = pd.read_sql_query("SELECT * FROM settlements", con)
    report(str(output_root), include_fixture=False)
    forward_summary = pd.read_csv(output_root / "cumulative_summary.csv")
    audit = {
        "predict_success": True,
        "four_tiers_generated": int(pd.read_sql_query("SELECT COUNT(DISTINCT threshold_tier) AS n FROM prediction_tiers", con)["n"].iloc[0]) == 4,
        "tier_inclusion": audit_tier_inclusion(pd.read_sql_query("SELECT * FROM prediction_tiers", con))[0],
        "duplicate_prediction_rejected": duplicate_rejected,
        "timestamp_violation_rejected": timestamp_rejected,
        "prediction_immutable": pred_before == pred_after,
        "settlement_append_only_rows": int(len(settlements)),
        "fixture_excluded_from_forward_reports": forward_summary.empty,
        "daily_report_generated": (output_root / "daily_summary.csv").exists(),
        "monthly_report_generated": (output_root / "monthly_summary.csv").exists(),
        "cumulative_report_generated": (output_root / "cumulative_summary.csv").exists(),
        "threshold_comparison_generated": (output_root / "threshold_comparison.csv").exists(),
    }
    (output_root / "fixture_smoke_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if all(v is True or k == "settlement_append_only_rows" and v > 0 for k, v in audit.items()) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["predict", "settle"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--race-date", required=True)
        p.add_argument("--output-root", default="outputs/place_market_offset_forward_paper_phase6c_v2")
        p.add_argument("--fixture", action="store_true")
        p.add_argument("--prediction-generated-at", default=None)
        p.add_argument("--data-cutoff-at", default=None)
        p.add_argument("--odds-observed-at", default=None)
        p.add_argument("--source-data-latest-at", default=None)
        p.add_argument("--input-csv", default=None)
        p.add_argument("--settlement-csv", default=None)
        p.add_argument("--settled-at", default=None)
    r = sub.add_parser("report")
    r.add_argument("--config", default=str(DEFAULT_CONFIG))
    r.add_argument("--output-root", default="outputs/place_market_offset_forward_paper_phase6c_v2")
    r.add_argument("--include-fixture", action="store_true")
    s = sub.add_parser("smoke-fixture")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--output-root", default="outputs/place_market_offset_forward_paper_phase6c_v2_fixture")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.command == "predict":
        return predict(args)
    if args.command == "settle":
        return settle(args)
    if args.command == "report":
        return report(args.output_root, include_fixture=args.include_fixture)
    if args.command == "smoke-fixture":
        return smoke(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
