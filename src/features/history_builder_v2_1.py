from __future__ import annotations

import pickle
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.features.target_builder import is_valid_id


STORE_NAMES = [
    "horse", "jockey", "trainer", "horse_jockey", "jockey_jyo", "jockey_dist_band",
    "horse_jyo", "horse_surface", "horse_dist_band", "horse_baba",
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
        "history_state_version": "v2_1_pre_day_audit",
        "horse_recent": defaultdict(deque),
        "horse_total_starts": defaultdict(int),
        "stats": defaultdict(stats_empty),
        "sources": {},
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


def store_keys(row: dict[str, Any]) -> dict[str, tuple[Any, ...] | None]:
    horse = str(row.get("KettoNum") or "").strip()
    jockey = str(row.get("KisyuCode") or "").strip()
    trainer = str(row.get("ChokyosiCode") or "").strip()
    jyo = str(row.get("JyoCD") or "").strip()
    surf = surface_code(row.get("TrackCD"))
    dist = distance_band(row.get("Kyori"))
    baba = baba_code(row)
    return {
        "horse": ("horse", horse) if is_valid_id(horse) else None,
        "jockey": ("jockey", jockey) if is_valid_id(jockey) else None,
        "trainer": ("trainer", trainer) if is_valid_id(trainer) else None,
        "horse_jockey": ("horse_jockey", horse, jockey) if is_valid_id(horse) and is_valid_id(jockey) else None,
        "jockey_jyo": ("jockey_jyo", jockey, jyo) if is_valid_id(jockey) else None,
        "jockey_dist_band": ("jockey_dist_band", jockey, dist) if is_valid_id(jockey) and dist is not None else None,
        "horse_jyo": ("horse_jyo", horse, jyo) if is_valid_id(horse) else None,
        "horse_surface": ("horse_surface", horse, surf) if is_valid_id(horse) and surf is not None else None,
        "horse_dist_band": ("horse_dist_band", horse, dist) if is_valid_id(horse) and dist is not None else None,
        "horse_baba": ("horse_baba", horse, baba) if is_valid_id(horse) and baba is not None else None,
    }


def audit_store(row: dict[str, Any], state: dict[str, Any], store: str, key: tuple[Any, ...] | None) -> dict[str, Any]:
    if key is None:
        return {"store_name": store, "status": "no_history", "last_source_race_id": None, "last_source_race_date": None}
    source = state["sources"].get(key)
    if source is None:
        return {"store_name": store, "status": "no_history", "last_source_race_id": None, "last_source_race_date": None}
    source_race_id = source["race_id"]
    source_date = source["race_date"]
    if source_race_id == row["race_id"]:
        status = "same_race"
    elif source_date == row["race_date"]:
        status = "same_day"
    elif source_date > row["race_date"]:
        status = "future"
    else:
        status = "ok"
    return {"store_name": store, "status": status, "last_source_race_id": source_race_id, "last_source_race_date": source_date}


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


def feature_for_row(row: dict[str, Any], state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    horse = str(row.get("KettoNum") or "").strip()
    hist = list(state["horse_recent"].get(horse, [])) if is_valid_id(horse) else []
    previous = hist[-1] if hist else None
    current_date = date_from_str(row["race_date"])
    keys = store_keys(row)
    audits = [audit_store(row, state, store, key) for store, key in keys.items()]
    feature: dict[str, Any] = {
        "entry_id": row["entry_id"],
        "feature_snapshot_mode": "pre_day",
        "history_cutoff_date": state.get("last_completed_date"),
        "historical_source_race_date": previous["race_date"] if previous else None,
        "horse_past_starts": state["horse_total_starts"].get(horse, 0) if is_valid_id(horse) else None,
        "horse_days_since_last": (current_date - previous["race_date_obj"]).days if previous else None,
    }
    feature["horse_distance_diff_last"] = to_float(row.get("Kyori")) - to_float(previous.get("Kyori")) if previous and to_float(row.get("Kyori")) is not None and to_float(previous.get("Kyori")) is not None else None
    feature["horse_futan_diff_last"] = to_float(row.get("Futan")) - to_float(previous.get("Futan")) if previous and to_float(row.get("Futan")) is not None and to_float(previous.get("Futan")) is not None else None
    feature["horse_body_weight_diff_last"] = to_float(row.get("BaTaijyu")) - to_float(previous.get("BaTaijyu")) if previous and to_float(row.get("BaTaijyu")) is not None and to_float(previous.get("BaTaijyu")) is not None else None
    for n in (1, 3, 5):
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
    feature.update(stats_features(stats[keys["horse_jyo"]], "horse_jyo") if keys["horse_jyo"] else stats_features(stats_empty(), "horse_jyo"))
    feature.update(stats_features(stats[keys["horse_surface"]], "horse_surface") if keys["horse_surface"] else stats_features(stats_empty(), "horse_surface"))
    feature.update(stats_features(stats[keys["horse_dist_band"]], "horse_dist_band") if keys["horse_dist_band"] else stats_features(stats_empty(), "horse_dist_band"))
    feature.update(stats_features(stats[keys["horse_baba"]], "horse_baba") if keys["horse_baba"] else stats_features(stats_empty(), "horse_baba"))
    feature.update(stats_features(stats[keys["jockey"]], "jockey", include_ren=True) if keys["jockey"] else stats_features(stats_empty(), "jockey", include_ren=True))
    feature.update(stats_features(stats[keys["trainer"]], "trainer", include_ren=True) if keys["trainer"] else stats_features(stats_empty(), "trainer", include_ren=True))
    feature.update(stats_features(stats[keys["jockey_jyo"]], "jockey_jyo") if keys["jockey_jyo"] else stats_features(stats_empty(), "jockey_jyo"))
    feature.update(stats_features(stats[keys["jockey_dist_band"]], "jockey_dist_band") if keys["jockey_dist_band"] else stats_features(stats_empty(), "jockey_dist_band"))
    feature.update(stats_features(stats[keys["horse_jockey"]], "horse_jockey") if keys["horse_jockey"] else stats_features(stats_empty(), "horse_jockey"))
    return feature, audits


def update_state_with_day(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    for row in rows:
        if not is_history_update_row(row, state):
            continue
        horse = str(row.get("KettoNum") or "").strip()
        record = dict(row)
        record["race_date_obj"] = date_from_str(row["race_date"])
        state["horse_total_starts"][horse] += 1
        state["horse_recent"][horse].append(record)
        if len(state["horse_recent"][horse]) > 20:
            state["horse_recent"][horse].popleft()
        keys = store_keys(row)
        source = {"race_id": row["race_id"], "race_date": row["race_date"]}
        for store, key in keys.items():
            if key is None:
                continue
            update_stats(state["stats"][key], row)
            state["sources"][key] = source
    if rows:
        state["last_completed_date"] = rows[0]["race_date"]


def empty_audit_counts() -> dict[str, int]:
    return {"validation_rows": 0, "ok": 0, "no_history": 0, "same_race": 0, "same_day": 0, "future": 0}


def build_pre_day_history_features_v2_1(df: pl.DataFrame, logger: Any | None = None, initial_state: dict[str, Any] | None = None) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, int]], list[dict[str, Any]]]:
    state = initial_state or new_state()
    rows = df.sort(["race_date", "Year", "MonthDay", "JyoCD", "Kaiji", "Nichiji", "RaceNum", "Umaban", "entry_id"]).to_dicts()
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["race_date"]].append(row)
    features: list[dict[str, Any]] = []
    audit_counts = {store: empty_audit_counts() for store in STORE_NAMES}
    audit_samples: list[dict[str, Any]] = []
    started = time.time()
    processed = 0
    for race_date, day_rows in sorted(by_date.items()):
        for row in day_rows:
            feature, audits = feature_for_row(row, state)
            features.append(feature)
            for audit in audits:
                store = audit["store_name"]
                status = audit["status"]
                audit_counts[store]["validation_rows"] += 1
                audit_counts[store][status] += 1
                if len(audit_samples) < 1000 or status in {"same_race", "same_day", "future"}:
                    audit_samples.append({
                        "entry_id": row["entry_id"],
                        "race_id": row["race_id"],
                        "race_date": row["race_date"],
                        "store_name": store,
                        "status": status,
                        "last_source_race_id": audit["last_source_race_id"],
                        "last_source_race_date": audit["last_source_race_date"],
                        "history_cutoff_date": feature["history_cutoff_date"],
                    })
            processed += 1
        update_state_with_day(day_rows, state)
        if logger and processed and processed % 50000 < len(day_rows):
            logger.info("history_v2_1 progress rows=%s date=%s elapsed=%.1fs", processed, race_date, time.time() - started)
    features_df = pl.DataFrame(features, infer_schema_length=10000)
    return df.join(features_df, on="entry_id", how="left"), state, audit_counts, audit_samples
