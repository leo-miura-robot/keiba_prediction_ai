from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from webapp.data.schema import NORMALIZED_COLUMNS

JYO_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

TIER_THRESHOLDS = {
    "CORE": 1.00,
    "MARGIN": 1.05,
    "HIGH": 1.10,
    "VERY_HIGH": 1.15,
}


def racecourse_name(jyo_cd: Any) -> str:
    if pd.isna(jyo_cd):
        return "競馬場不明"
    key = str(jyo_cd).zfill(2)
    return JYO_MAP.get(key, f"競馬場コード{key}")


def classify_source(path: str | Path, df: pd.DataFrame | None = None) -> str:
    text = str(path).replace("\\", "/").lower()
    if "fixture" in text:
        return "FIXTURE"
    if df is not None and "fixture" in df.columns and bool(pd.Series(df["fixture"]).fillna(False).astype(bool).any()):
        return "FIXTURE"
    if "forward_paper" in text:
        return "FORWARD_PAPER"
    if "validation" in text or "champion_challenger" in text or "latest_model_validation" in text:
        return "RETROSPECTIVE_VALIDATION"
    if "backtest" in text or "phase5b" in text:
        return "BACKTEST"
    return "UNKNOWN"


def coalesce(df: pd.DataFrame, names: list[str], default: Any = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for name in names:
        if name in df.columns:
            out = out.where(out.notna(), df[name])
    out = out.fillna(default)
    return out


def tier_for_ev(ev: Any) -> str:
    try:
        value = float(ev)
    except (TypeError, ValueError):
        return "NONE"
    if not math.isfinite(value) or value < TIER_THRESHOLDS["CORE"]:
        return "NONE"
    if value >= TIER_THRESHOLDS["VERY_HIGH"]:
        return "VERY_HIGH"
    if value >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if value >= TIER_THRESHOLDS["MARGIN"]:
        return "MARGIN"
    return "CORE"


def is_place_paid(df: pd.DataFrame) -> pd.Series:
    if "target_place_paid" in df.columns:
        return pd.to_numeric(df["target_place_paid"], errors="coerce").fillna(0).astype(int).eq(1)
    if "fuku_pay" in df.columns:
        return pd.to_numeric(df["fuku_pay"], errors="coerce").fillna(0).gt(0)
    return pd.Series(False, index=df.index)


def normalize_prediction_frame(
    df: pd.DataFrame,
    source_path: str | Path,
    *,
    strategy: str = "ROLLING_10Y",
    stake_per_bet_yen: int = 100,
    source_type: str | None = None,
    fixture: bool | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["race_date"] = pd.to_datetime(coalesce(df, ["race_date"]), errors="raise").dt.date.astype(str)
    out["race_id"] = coalesce(df, ["race_id"]).astype(str)
    out["entry_id"] = coalesce(df, ["entry_id"], "").astype(str)
    out["JyoCD"] = coalesce(df, ["JyoCD"]).astype(str).str.zfill(2)
    out["racecourse"] = out["JyoCD"].map(racecourse_name)
    out["RaceNum"] = pd.to_numeric(coalesce(df, ["RaceNum"]), errors="coerce").astype("Int64")
    out["Umaban"] = pd.to_numeric(coalesce(df, ["Umaban", "horse_no"]), errors="coerce").astype("Int64")
    out["horse_name"] = coalesce(df, ["horse_name", "Bamei", "馬名"])
    fallback_name = "馬番" + out["Umaban"].astype(str) + " / " + coalesce(df, ["KettoNum"], "").astype(str)
    out["horse_name"] = out["horse_name"].fillna("").astype(str).replace("", np.nan).fillna(fallback_name)
    out["KettoNum"] = coalesce(df, ["KettoNum"], "").astype(str)
    out["strategy"] = coalesce(df, ["strategy"], strategy).fillna(strategy).astype(str)
    out["source_type"] = source_type or classify_source(source_path, df)
    if fixture is None:
        fixture = out["source_type"].eq("FIXTURE").any()
    out["fixture"] = bool(fixture)
    out["retrospective_only"] = coalesce(df, ["retrospective_only"], out["source_type"].isin(["RETROSPECTIVE_VALIDATION", "BACKTEST"]).any()).fillna(False).astype(bool)
    out["odds_snapshot_type"] = coalesce(df, ["odds_snapshot_type"], "saved")
    out["prediction_created_at"] = coalesce(df, ["prediction_created_at", "prediction_generated_at"], "")
    out["odds_observed_at"] = coalesce(df, ["odds_observed_at"], "")
    out["probability_market"] = pd.to_numeric(coalesce(df, ["probability_market", "p_market"]), errors="coerce")
    out["probability_raw"] = pd.to_numeric(coalesce(df, ["probability_raw", "probability_used_for_selection"]), errors="coerce")
    out["probability_calibrated"] = pd.to_numeric(coalesce(df, ["probability_calibrated", "probability_official_platt"]), errors="coerce")
    ev_probability = out["probability_calibrated"].where(out["probability_calibrated"].notna(), out["probability_raw"])
    out["ev_probability_source"] = np.where(out["probability_calibrated"].notna(), "probability_calibrated", "probability_raw")
    out["fuku_odds_low"] = pd.to_numeric(coalesce(df, ["fuku_odds_low", "FukuOddsLow", "fuku_odds_low_at_prediction"]), errors="coerce")
    out["fuku_odds_high"] = pd.to_numeric(coalesce(df, ["fuku_odds_high", "FukuOddsHigh"]), errors="coerce")
    out["expected_value"] = pd.to_numeric(coalesce(df, ["expected_value", "ev_at_prediction"]), errors="coerce")
    out["expected_value"] = out["expected_value"].where(out["expected_value"].notna(), ev_probability * out["fuku_odds_low"])
    out["tier"] = out["expected_value"].map(tier_for_ev)
    out["selected_for_bet"] = out["expected_value"].ge(TIER_THRESHOLDS["CORE"]).fillna(False)
    out["tan_ninki"] = pd.to_numeric(coalesce(df, ["tan_ninki", "TanNinki", "Ninki"]), errors="coerce")
    out["fuku_ninki"] = pd.to_numeric(coalesce(df, ["fuku_ninki", "FukuNinki"]), errors="coerce")
    out["Kyori"] = pd.to_numeric(coalesce(df, ["Kyori"]), errors="coerce")
    out["TrackCD"] = coalesce(df, ["TrackCD"], "").astype(str)
    out["SyussoTosu"] = pd.to_numeric(coalesce(df, ["SyussoTosu"]), errors="coerce")
    out["actual_finish_position"] = pd.to_numeric(coalesce(df, ["actual_finish_position", "KakuteiJyuni", "NyusenJyuni"]), errors="coerce")
    out["target_place_paid"] = is_place_paid(df).astype(int)
    out["fuku_pay"] = pd.to_numeric(coalesce(df, ["fuku_pay"]), errors="coerce").fillna(0)
    out["stake_yen"] = np.where(out["selected_for_bet"], stake_per_bet_yen, 0)
    out["payout_yen"] = np.where(out["selected_for_bet"] & out["target_place_paid"].eq(1), out["fuku_pay"], 0)
    out["profit_yen"] = out["payout_yen"] - out["stake_yen"]
    out["source_path"] = str(source_path)
    year = pd.to_datetime(out["race_date"], errors="coerce").dt.year
    out["validation_year"] = year.astype("Int64")
    out = out[NORMALIZED_COLUMNS].drop_duplicates(subset=["source_path", "strategy", "race_id", "entry_id"])
    return out.reset_index(drop=True)


def summarize_bets(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "total_stake_yen": 0,
            "total_payout_yen": 0,
            "total_profit_yen": 0,
            "bets": 0,
            "hits": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "races": 0,
            "date_min": None,
            "date_max": None,
        }
    bets = df[df["selected_for_bet"]].copy()
    stake = float(bets["stake_yen"].sum())
    payout = float(bets["payout_yen"].sum())
    hits = int((bets["payout_yen"] > 0).sum())
    bet_count = int(len(bets))
    return {
        "total_stake_yen": stake,
        "total_payout_yen": payout,
        "total_profit_yen": payout - stake,
        "bets": bet_count,
        "hits": hits,
        "hit_rate": hits / bet_count if bet_count else np.nan,
        "roi": payout / stake * 100 if stake else np.nan,
        "races": int(df["race_id"].nunique()),
        "date_min": str(df["race_date"].min()) if len(df) else None,
        "date_max": str(df["race_date"].max()) if len(df) else None,
    }


def grouped_roi(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "bets", "hits", "stake_yen", "payout_yen", "profit_yen", "roi"])
    bets = df[df["selected_for_bet"]].copy()
    if bets.empty:
        return pd.DataFrame(columns=[group_col, "bets", "hits", "stake_yen", "payout_yen", "profit_yen", "roi"])
    grouped = bets.groupby(group_col, dropna=False).agg(
        bets=("entry_id", "count"),
        hits=("payout_yen", lambda s: int((s > 0).sum())),
        stake_yen=("stake_yen", "sum"),
        payout_yen=("payout_yen", "sum"),
        profit_yen=("profit_yen", "sum"),
        average_ev=("expected_value", "mean"),
        average_odds=("fuku_odds_low", "mean"),
    ).reset_index()
    grouped["hit_rate"] = grouped["hits"] / grouped["bets"]
    grouped["roi"] = np.where(grouped["stake_yen"] > 0, grouped["payout_yen"] / grouped["stake_yen"] * 100, np.nan)
    return grouped.sort_values(group_col)


def race_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["race_date", "race_id", "racecourse", "RaceNum"], dropna=False).agg(
        runners=("entry_id", "count"),
        selected_horses=("selected_for_bet", "sum"),
        actual_place_horses=("target_place_paid", "sum"),
        hits=("payout_yen", lambda s: int((s > 0).sum())),
        stake_yen=("stake_yen", "sum"),
        payout_yen=("payout_yen", "sum"),
        profit_yen=("profit_yen", "sum"),
        Kyori=("Kyori", "max"),
    ).reset_index()
