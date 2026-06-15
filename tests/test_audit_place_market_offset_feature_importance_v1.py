from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_place_market_offset_feature_importance_v1 import (
    distribution_by_year,
    group_for_feature,
    permute_group,
)


def test_group_for_feature_classifies_core_groups() -> None:
    assert group_for_feature("market_logit") == "market_baseline"
    assert group_for_feature("horse_jyo_win_rate") == "horse_course_suitability"
    assert group_for_feature("jockey_dist_band_top3_rate") == "jockey_course_suitability"
    assert group_for_feature("JyoCD") == "venue_identity"


def test_distribution_reproduces_ev_counts() -> None:
    df = pd.DataFrame({
        "model_key": ["C1_market_offset_fundamental"] * 2,
        "Year": [2024, 2024],
        "race_id": ["r1", "r1"],
        "final_probability": [0.5, 0.2],
        "p_market": [0.4, 0.3],
        "market_logit": [0.0, 0.0],
        "catboost_residual_score": [0.0, 0.0],
        "fuku_odds_low": [2.0, 4.0],
    })
    _, counts, crossing, _ = distribution_by_year(df)
    assert int(counts.loc[0, "ev_ge_1_count"]) == 1
    assert "ev_lt_1_to_ge_1_by_residual" in crossing.columns


def test_market_permutation_only_changes_baseline() -> None:
    df = pd.DataFrame({"race_id": ["r1", "r2", "r3"], "market_logit": [1.0, 2.0, 3.0], "x": [10, 20, 30]})
    out = permute_group(df, "market_baseline", [], 1)
    assert sorted(out["market_logit"]) == [1.0, 2.0, 3.0]
    assert list(out["x"]) == [10, 20, 30]
