import pytest
from pathlib import Path

def test_phase4_output_exists():
    out = Path("outputs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1_smoke")
    assert out.exists()
    assert (out / "manifest.json").exists()
    assert (out / "selected_rate_smoothing.json").exists()
