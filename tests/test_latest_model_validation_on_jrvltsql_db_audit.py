from __future__ import annotations

import math

import pandas as pd

from scripts.audit_latest_model_validation_on_jrvltsql_db import (
    probability_metrics,
    target_audit,
)


def test_target_audit_uses_paid_place_not_rank_three() -> None:
    df = pd.DataFrame(
        {
            "target_place_paid": [0, 1],
            "fuku_pay": [0, 220],
            "SyussoTosu": [7, 8],
            "KakuteiJyuni": [3, 3],
        }
    )

    result = target_audit(df)

    assert result["target_place_paid_equals_fuku_pay_gt_0"] is True
    assert result["small_field_5_to_7_third_rows"] == 1
    assert result["small_field_5_to_7_third_incorrect_paid_count"] == 0


def test_probability_metrics_separates_market_raw_calibrated() -> None:
    df = pd.DataFrame(
        {
            "strategy": ["S"] * 4,
            "race_id": ["r1", "r1", "r2", "r2"],
            "actual_place": [1, 0, 1, 0],
            "market_logit": [1.0, -1.0, 0.5, -0.5],
            "probability_raw": [0.8, 0.2, 0.7, 0.3],
            "probability_calibrated": [0.75, 0.25, 0.65, 0.35],
        }
    )

    out = probability_metrics(df)

    assert set(out["probability_type"]) == {"market_only", "raw_c1r0", "calibrated"}
    assert out["rows"].eq(4).all()
    assert out["races"].eq(2).all()
    assert out["logloss"].map(math.isfinite).all()
    assert out["brier"].map(math.isfinite).all()
