from __future__ import annotations

from src.features.feature_sets import validate_feature_sets


def test_feature_sets_do_not_include_leakage_or_duplicates() -> None:
    assert validate_feature_sets() == []
