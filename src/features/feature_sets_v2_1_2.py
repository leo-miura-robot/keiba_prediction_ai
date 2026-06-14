from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.features.feature_sets_v2_1 import (
    CURRENT_MARKET_COLUMNS,
    LEAKAGE_COLUMNS,
    MARKET_HISTORY_COLUMNS,
    RAW_MARKET_COLUMNS,
    feature_sets as default_feature_sets,
)


def feature_sets() -> dict[str, dict[str, list[str]]]:
    return default_feature_sets()


def write_feature_set_yaml(path: Path) -> None:
    lines: list[str] = []
    for set_name, groups in feature_sets().items():
        lines.append(f"{set_name}:")
        for group_name, columns in groups.items():
            lines.append(f"  {group_name}:")
            for column in columns:
                lines.append(f"    - {column}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_feature_set_yaml(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        write_feature_set_yaml(path)
    result: dict[str, dict[str, list[str]]] = {}
    current_set: str | None = None
    current_group: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current_set = stripped[:-1]
            result[current_set] = {}
            current_group = None
            continue
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            if current_set is None:
                raise ValueError(f"group outside feature set: {line}")
            current_group = stripped[:-1]
            result[current_set][current_group] = []
            continue
        if line.startswith("    - "):
            if current_set is None or current_group is None:
                raise ValueError(f"column outside feature group: {line}")
            result[current_set][current_group].append(stripped[2:].strip())
            continue
        raise ValueError(f"unsupported feature yaml line: {line}")
    return result


def canonical_feature_set_text(path: Path) -> str:
    return json.dumps(load_feature_set_yaml(path), ensure_ascii=False, sort_keys=True)


def validate_feature_sets_from_file(path: Path) -> list[dict[str, str]]:
    sets = load_feature_set_yaml(path)
    rows: list[dict[str, str]] = []
    required_sets = {"market_free", "market_history", "market_aware"}
    missing_sets = sorted(required_sets - set(sets))
    if missing_sets:
        rows.append({"check_name": "required_sets", "status": "fail", "details": ",".join(missing_sets)})
    for set_name, groups in sets.items():
        columns = groups.get("numeric", []) + groups.get("categorical", [])
        duplicated = sorted({c for c in columns if columns.count(c) > 1})
        leaked = sorted(set(columns) & LEAKAGE_COLUMNS)
        raw_market = sorted(set(columns) & RAW_MARKET_COLUMNS)
        if duplicated:
            rows.append({"check_name": f"{set_name}_duplicates", "status": "fail", "details": ",".join(duplicated)})
        if leaked:
            rows.append({"check_name": f"{set_name}_leakage", "status": "fail", "details": ",".join(leaked)})
        if raw_market:
            rows.append({"check_name": f"{set_name}_raw_market", "status": "fail", "details": ",".join(raw_market)})
    free_cols = set(sets.get("market_free", {}).get("numeric", []) + sets.get("market_free", {}).get("categorical", []))
    hist_cols = set(sets.get("market_history", {}).get("numeric", []) + sets.get("market_history", {}).get("categorical", []))
    aware_cols = set(sets.get("market_aware", {}).get("numeric", []) + sets.get("market_aware", {}).get("categorical", []))
    free_market = sorted(free_cols & (CURRENT_MARKET_COLUMNS | MARKET_HISTORY_COLUMNS | RAW_MARKET_COLUMNS))
    hist_current = sorted(hist_cols & (CURRENT_MARKET_COLUMNS | RAW_MARKET_COLUMNS))
    if free_market:
        rows.append({"check_name": "market_free_no_market", "status": "fail", "details": ",".join(free_market)})
    if hist_current:
        rows.append({"check_name": "market_history_no_current_market", "status": "fail", "details": ",".join(hist_current)})
    if hist_cols and not hist_cols <= aware_cols:
        rows.append({"check_name": "market_aware_contains_market_history", "status": "fail", "details": ",".join(sorted(hist_cols - aware_cols))})
    if not rows:
        rows.append({"check_name": "all_feature_set_checks", "status": "pass", "details": ""})
    return rows


def feature_inventory_rows(path: Path, dataset_columns: set[str], null_rates: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for set_name, groups in load_feature_set_yaml(path).items():
        for kind, columns in groups.items():
            for column in columns:
                rows.append({
                    "feature_set": set_name,
                    "kind": kind,
                    "column_name": column,
                    "exists_in_dataset": column in dataset_columns,
                    "null_rate": null_rates.get(column),
                })
    return rows
