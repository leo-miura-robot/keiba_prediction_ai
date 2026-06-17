from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.audit_and_recover_official_market_baseline_phase5c_v1 as recovery
from src.market.official_market_baseline_loader import load_official_market_baseline


def test_recovery_inventory_is_partial_without_parameters(tmp_path: Path) -> None:
    code = recovery.run(tmp_path)

    assert code == 2
    inventory = json.loads((tmp_path / "market_parameter_inventory.json").read_text(encoding="utf-8"))
    assert inventory["classification"] == "PARTIAL_PARAMETERS_FOUND"
    assert inventory["market_feature_count"] == 17
    assert inventory["standard_scaler"]["mean_"] == "NOT_FOUND"
    assert inventory["logistic_regression"]["coef_"] == "NOT_FOUND"
    blocked = json.loads((tmp_path / "blocked_market_artifact_recovery.json").read_text(encoding="utf-8"))
    assert blocked["refit_performed"] is False
    assert blocked["parameter_generation_performed"] is False
    statuses = {item["status"] for item in blocked["attempted_recovery_methods"]}
    assert "REJECTED_PROHIBITED" in statuses
    assert blocked["migration_plan"]


def test_incomplete_official_market_artifact_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "official_market_manifest.json").write_text(
        json.dumps({"final_status": "BLOCKED_MISSING_MARKET_PARAMETERS"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete"):
        load_official_market_baseline(tmp_path)
