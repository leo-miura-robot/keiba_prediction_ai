import pytest
from pathlib import Path

def test_audit_artifacts_exist():
    out = Path("outputs/place_market_offset_catboost_c1r0_metric_consistency_audit_v2")
    assert out.exists()
    assert (out / "manifest.json").exists()
    assert (out / "raw_vs_clip_final_decision.json").exists()
