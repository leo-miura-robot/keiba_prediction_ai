from __future__ import annotations

import pickle
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.features.target_builder import is_valid_id


HISTORY_WINDOWS = (1, 3, 5)

HIST_FEATURE_COLUMNS = [
    "feature_snapshot_mode", "history_cutoff_date", "horse_days_since_last", "horse_past_starts",
    "horse_last1_avg_finish", "horse_last3_avg_finish", "horse_last5_avg_finish",
    "horse_last3_win_rate", "horse_last5_win_rate", "horse_last3_ren_rate", "horse_last5_ren_rate",
    "horse_last3_top3_rate", "horse_last5_top3_rate", "horse_last3_place_paid_rate", "horse_last5_place_paid_rate",
    "horse_last3_avg_ninki", "horse_last5_avg_ninki", "horse_last3_avg_haron_l3", "horse_last5_avg_haron_l3",
    "horse_last3_avg_time", "horse_last5_avg_time", "horse_distance_diff_last", "horse_futan_diff_last",
    "horse_body_weight_diff_last", "horse_jyo_past_starts", "horse_jyo_win_rate", "horse_jyo_top3_rate",
    "horse_surface_past_starts", "horse_surface_win_rate", "horse_surface_top3_rate",
    "horse_dist_band_past_starts", "horse_dist_band_win_rate", "horse_dist_band_top3_rate",
    "horse_baba_past_starts", "horse_baba_win_rate", "horse_baba_top3_rate",
    "jockey_past_starts", "jockey_win_rate", "jockey_ren_rate", "jockey_top3_rate",
    "trainer_past_starts", "trainer_win_rate", "trainer_ren_rate", "trainer_top3_rate",
    "jockey_jyo_past_starts", "jockey_jyo_win_rate", "jockey_jyo_top3_rate",
    "jockey_dist_band_past_starts", "jockey_dist_band_win_rate", "jockey_dist_band_top3_rate",
    "horse_jockey_past_starts", "horse_jockey_win_rate", "horse_jockey_top3_rate",
]


def date_from_str(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def avg(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def rate(values: list[int | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def distance_band(kyori: Any) -> int | None:
    value = to_float(kyori)
    if value is None:
        return None
    return int(value // 200 * 200)


def surface_code(track_cd: Any) -> str | None:
    if track_cd is None or str(track_cd).strip() == "":
        return None
    return str(track_cd)[0]


def baba_code(row: dict[str, Any]) -> str | None:
    track = str(row.get("TrackCD") or "")
    if track.startswith("1"):
        return row.get("SibaBabaCD")
    if track.startswith("2"):
        return row.get("DirtBabaCD")
    return row.get("SibaBabaCD") or row.get("DirtBabaCD")


def stats_empty() -> dict[str, int]:
    return {"starts": 0, "wins": 0, "rens": 0, "top3": 0, "place_paid": 0}


def update_stats(stats: dict[str, int], row: dict[str, Any]) -> None:
    stats["starts"] += 1
    stats["wins"] += int(row.get("target_win_rank") == 1)
    stats["rens"] += int(row.get("target_ren_rank") == 1)
    stats["top3"] += int(row.get("target_top3_rank") == 1)
    stats["place_paid"] += int(row.get("target_place_paid") == 1)


def stats_features(stats: dict[str, int], prefix: str, include_ren: bool = False) -> dict[str, Any]:
    starts = stats["starts"]
    out = {
        f"{prefix}_past_starts": starts,
        f"{prefix}_win_rate": stats["wins"] / starts if starts else None,
        f"{prefix}_top3_rate": stats["top3"] / starts if starts else None,
    }
    if include_ren:
        out[f"{prefix}_ren_rate"] = stats["rens"] / starts if starts else None
    return out


def new_state() -> dict[str, Any]:
    return {
        "horse_recent": defaultdict(deque),
        "horse_total_starts": defaultdict(int),
        "stats": defaultdict(stats_empty),
        "last_completed_date": None,
        "excluded_history_rows": defaultdict(int),
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def is_history_update_row(row: dict[str, Any], state: dict[str, Any]) -> bool:
    if not row.get("race_has_result"):
        state["excluded_history_rows"]["no_result"] += 1
        return False
    if row.get("IJyoCD") != "0":
        state["excluded_history_rows"]["abnormal_or_cancelled"] += 1
        return False
    if not is_valid_id(row.get("KettoNum")):
        state["excluded_history_rows"]["invalid_horse_id"] += 1
        return False
    if not row.get("KakuteiJyuni") or row.get("KakuteiJyuni") <= 0:
        state["excluded_history_rows"]["missing_rank"] += 1
        return False
    return True


def feature_for_row(row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    horse = str(row.get("KettoNum") or "").strip()
    jockey = str(row.get("KisyuCode") or "").strip()
    trainer = str(row.get("ChokyosiCode") or "").strip()
    jyo = str(row.get("JyoCD") or "").strip()
    surf = surface_code(row.get("TrackCD"))
    dist = distance_band(row.get("Kyori"))
    baba = baba_code(row)
    current_date = date_from_str(row["race_date"])

    hist = list(state["horse_recent"].get(horse, [])) if is_valid_id(horse) else []
    previous = hist[-1] if hist else None
    feature: dict[str, Any] = {
        "entry_id": row["entry_id"],
        "feature_snapshot_mode": "pre_day",
        "history_cutoff_date": row["race_date"],
        "horse_past_starts": state["horse_total_starts"].get(horse, 0) if is_valid_id(horse) else None,
        "horse_days_since_last": (current_date - previous["race_date_obj"]).days if previous else None,
        "historical_source_race_date": previous["race_date"] if previous else None,
    }
    feature["horse_distance_diff_last"] = to_float(row.get("Kyori")) - to_float(previous.get("Kyori")) if previous and to_float(row.get("Kyori")) is not None and to_float(previous.get("Kyori")) is not None else None
    feature["horse_futan_diff_last"] = to_float(row.get("Futan")) - to_float(previous.get("Futan")) if previous and to_float(row.get("Futan")) is not None and to_float(previous.get("Futan")) is not None else None
    feature["horse_body_weight_diff_last"] = to_float(row.get("BaTaijyu")) - to_float(previous.get("BaTaijyu")) if previous and to_float(row.get("BaTaijyu")) is not None and to_float(previous.get("BaTaijyu")) is not None else None

    for n in HISTORY_WINDOWS:
        recent = hist[-n:]
        feature[f"horse_last{n}_avg_finish"] = avg([to_float(r.get("KakuteiJyuni")) for r in recent])
        if n in (3, 5):
            feature[f"horse_last{n}_win_rate"] = rate([r.get("target_win_rank") for r in recent])
            feature[f"horse_last{n}_ren_rate"] = rate([r.get("target_ren_rank") for r in recent])
            feature[f"horse_last{n}_top3_rate"] = rate([r.get("target_top3_rank") for r in recent])
            feature[f"horse_last{n}_place_paid_rate"] = rate([r.get("target_place_paid") for r in recent])
            feature[f"horse_last{n}_avg_ninki"] = avg([to_float(r.get("Ninki")) for r in recent])
            feature[f"horse_last{n}_avg_haron_l3"] = avg([to_float(r.get("HaronTimeL3")) for r in recent])
            feature[f"horse_last{n}_avg_time"] = avg([to_float(r.get("Time")) for r in recent])

    stats = state["stats"]
    feature.update(stats_features(stats[("horse_jyo", horse, jyo)], "horse_jyo"))
    feature.update(stats_features(stats[("horse_surface", horse, surf)], "horse_surface"))
    feature.update(stats_features(stats[("horse_dist_band", horse, dist)], "horse_dist_band"))
    feature.update(stats_features(stats[("horse_baba", horse, baba)], "horse_baba"))
    feature.update(stats_features(stats[("jockey", jockey)], "jockey", include_ren=True) if is_valid_id(jockey) else stats_features(stats_empty(), "jockey", include_ren=True))
    feature.update(stats_features(stats[("trainer", trainer)], "trainer", include_ren=True) if is_valid_id(trainer) else stats_features(stats_empty(), "trainer", include_ren=True))
    feature.update(stats_features(stats[("jockey_jyo", jockey, jyo)], "jockey_jyo") if is_valid_id(jockey) else stats_features(stats_empty(), "jockey_jyo"))
    feature.update(stats_features(stats[("jockey_dist_band", jockey, dist)], "jockey_dist_band") if is_valid_id(jockey) else stats_features(stats_empty(), "jockey_dist_band"))
    feature.update(stats_features(stats[("horse_jockey", horse, jockey)], "horse_jockey") if is_valid_id(horse) and is_valid_id(jockey) else stats_features(stats_empty(), "horse_jockey"))
    return feature


def update_state_with_day(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    for row in rows:
        if not is_history_update_row(row, state):
            continue
        horse = str(row.get("KettoNum") or "").strip()
        jockey = str(row.get("KisyuCode") or "").strip()
        trainer = str(row.get("ChokyosiCode") or "").strip()
        jyo = str(row.get("JyoCD") or "").strip()
        surf = surface_code(row.get("TrackCD"))
        dist = distance_band(row.get("Kyori"))
        baba = baba_code(row)
        record = dict(row)
        record["race_date_obj"] = date_from_str(row["race_date"])
        state["horse_total_starts"][horse] += 1
        state["horse_recent"][horse].append(record)
        if len(state["horse_recent"][horse]) > 20:
            state["horse_recent"][horse].popleft()
        for key in [
            ("horse_jyo", horse, jyo),
            ("horse_surface", horse, surf),
            ("horse_dist_band", horse, dist),
            ("horse_baba", horse, baba),
        ]:
            update_stats(state["stats"][key], row)
        if is_valid_id(jockey):
            for key in [("jockey", jockey), ("jockey_jyo", jockey, jyo), ("jockey_dist_band", jockey, dist), ("horse_jockey", horse, jockey)]:
                update_stats(state["stats"][key], row)
        if is_valid_id(trainer):
            update_stats(state["stats"][("trainer", trainer)], row)
    if rows:
        state["last_completed_date"] = rows[0]["race_date"]


def build_pre_day_history_features(df: pl.DataFrame, logger: Any | None = None, initial_state: dict[str, Any] | None = None) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    state = initial_state or new_state()
    sort_cols = ["race_date", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban", "entry_id"]
    rows = df.sort(sort_cols).to_dicts()
    features: list[dict[str, Any]] = []
    leakage_samples: list[dict[str, Any]] = []
    started = time.time()
    processed = 0

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["race_date"]].append(row)

    for race_date, day_rows in sorted(by_date.items()):
        for row in day_rows:
            feature = feature_for_row(row, state)
            features.append(feature)
            if feature.get("historical_source_race_date") and len(leakage_samples) < 500:
                leakage_samples.append({
                    "entry_id": row["entry_id"],
                    "race_id": row["race_id"],
                    "race_date": row["race_date"],
                    "horse_id": row.get("KettoNum"),
                    "historical_source_race_date": feature["historical_source_race_date"],
                    "history_cutoff_date": feature["history_cutoff_date"],
                    "source_before_current": feature["historical_source_race_date"] < row["race_date"],
                    "same_race_reference": False,
                    "same_day_reference": feature["historical_source_race_date"] == row["race_date"],
                })
            processed += 1
        update_state_with_day(day_rows, state)
        if logger and processed and processed % 50000 < len(day_rows):
            logger.info("history_v2 progress rows=%s date=%s elapsed=%.1fs", processed, race_date, time.time() - started)

    features_df = pl.DataFrame(features, infer_schema_length=10000)
    out = df.join(features_df, on="entry_id", how="left")
    return out, state, leakage_samples

