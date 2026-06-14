from __future__ import annotations

from src.features.history_builder_v2_1 import *  # noqa: F403


STATE_VERSION = "v2_1_2_o1_fixed_pre_day_audit"


def new_state() -> dict:  # type: ignore[override]
    from src.features.history_builder_v2_1 import new_state as base_new_state

    state = base_new_state()
    state["history_state_version"] = STATE_VERSION
    state["completed_year"] = None
    return state


def mark_completed_year(state: dict, year: int) -> None:
    state["history_state_version"] = STATE_VERSION
    state["completed_year"] = year
