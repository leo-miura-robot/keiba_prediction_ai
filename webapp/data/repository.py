from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from webapp.data.normalization import normalize_prediction_frame


def load_config(path: str | Path = "config/current_model_webapp_mvp_v1.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_parquet_sources(config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for pattern in config.get("data_sources", {}).get("prediction_parquet_globs", []):
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(set(p for p in paths if p.exists()))


def sqlite_tables_readonly(path: Path) -> dict[str, Any]:
    uri = "file:" + str(path.resolve()).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name").fetchall()]
        details: dict[str, Any] = {}
        for table in tables:
            columns = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
            rows = con.execute(f"select count(*) from {table}").fetchone()[0]
            details[table] = {"row_count": rows, "columns": columns}
        return details
    finally:
        con.close()


def load_phase6c_sqlite(path: Path, stake_per_bet_yen: int) -> pd.DataFrame:
    uri = "file:" + str(path.resolve()).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
        if not {"predictions", "prediction_runs", "prediction_tiers"}.issubset(set(tables)):
            return pd.DataFrame()
        predictions = pd.read_sql_query("select * from predictions", con)
        if predictions.empty:
            return pd.DataFrame()
        runs = pd.read_sql_query("select * from prediction_runs", con)
        tiers = pd.read_sql_query("select * from prediction_tiers where threshold_tier = 'CORE'", con)
        settlements = pd.read_sql_query("select * from settlements", con) if "settlements" in tables else pd.DataFrame()
    finally:
        con.close()
    df = predictions.merge(runs, on="prediction_run_id", how="left", suffixes=("", "_run"))
    if not tiers.empty:
        df = df.merge(tiers[["prediction_run_id", "strategy", "entry_id", "race_id", "threshold_tier", "paper_bet_flag"]], on=["prediction_run_id", "strategy", "entry_id", "race_id"], how="left")
    if not settlements.empty:
        keep = ["prediction_run_id", "strategy", "entry_id", "race_id", "target_place_paid", "fuku_pay", "paper_stake", "paper_payout", "paper_profit"]
        df = df.merge(settlements[[c for c in keep if c in settlements.columns]], on=["prediction_run_id", "strategy", "entry_id", "race_id"], how="left", suffixes=("", "_settled"))
    df = df.rename(columns={
        "horse_no": "Umaban",
        "fuku_odds_low_at_prediction": "fuku_odds_low",
        "ev_at_prediction": "expected_value",
        "is_fixture": "fixture",
    })
    df = df.loc[:, ~df.columns.duplicated()]
    if "paper_bet_flag" in df.columns and "expected_value" in df.columns:
        df["expected_value"] = pd.to_numeric(df.get("expected_value"), errors="coerce")
    return normalize_prediction_frame(
        df,
        path,
        stake_per_bet_yen=stake_per_bet_yen,
        source_type="FIXTURE" if "fixture" in str(path).lower() else "FORWARD_PAPER",
        fixture="fixture" in str(path).lower(),
    )


def source_inventory_for_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    date_min = str(pd.to_datetime(df["race_date"]).min().date()) if "race_date" in df.columns and len(df) else None
    date_max = str(pd.to_datetime(df["race_date"]).max().date()) if "race_date" in df.columns and len(df) else None
    return {
        "source_path": str(path),
        "source_type": "parquet",
        "row_count": int(len(df)),
        "date_range": [date_min, date_max],
        "available_columns": list(df.columns),
        "strategy": "ROLLING_10Y",
        "validation_year": int(pd.to_datetime(df["race_date"]).dt.year.max()) if "race_date" in df.columns and len(df) else None,
        "fixture_flag": "fixture" in str(path).lower(),
        "retrospective_flag": True,
        "settlement_availability": {"fuku_pay": "fuku_pay" in df.columns, "target_place_paid": "target_place_paid" in df.columns},
        "horse_name_availability": {"horse_name": "horse_name" in df.columns, "Bamei": "Bamei" in df.columns},
    }


def build_inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for p in discover_parquet_sources(config):
        inventory.append(source_inventory_for_parquet(p))
    for s in config.get("data_sources", {}).get("phase6c_sqlite_paths", []):
        p = Path(s)
        if not p.exists():
            inventory.append({"source_path": str(p), "source_type": "sqlite", "exists": False})
            continue
        tables = sqlite_tables_readonly(p)
        inventory.append({
            "source_path": str(p),
            "source_type": "sqlite",
            "exists": True,
            "tables": tables,
            "row_count": int(sum(v["row_count"] for k, v in tables.items() if k in {"predictions", "settlements"})),
            "fixture_flag": "fixture" in str(p).lower(),
            "retrospective_flag": False,
            "settlement_availability": {"settlements_table": "settlements" in tables},
            "horse_name_availability": {"horse_name": False},
        })
    return inventory


def write_inventory(config: dict[str, Any]) -> Path:
    out = Path(config["data_sources"]["output_inventory_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_inventory(config), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_all_normalized(config: dict[str, Any]) -> pd.DataFrame:
    stake = int(config.get("betting", {}).get("stake_per_bet_yen", 100))
    frames: list[pd.DataFrame] = []
    for p in discover_parquet_sources(config):
        df = pd.read_parquet(p)
        frames.append(normalize_prediction_frame(df, p, stake_per_bet_yen=stake))
    for s in config.get("data_sources", {}).get("phase6c_sqlite_paths", []):
        p = Path(s)
        if p.exists():
            frame = load_phase6c_sqlite(p, stake)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["source_path", "strategy", "race_id", "entry_id"])
    return out.reset_index(drop=True)
