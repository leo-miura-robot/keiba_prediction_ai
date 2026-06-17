from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_phase6c_raw_to_official_champion as bridge


def raw_fixture(path: Path, include_forbidden: bool = False) -> Path:
    rows = []
    for i in range(1, 5):
        row = {
            "race_id": "2099-01-08_R01",
            "entry_id": f"2099-01-08_{i}",
            "race_date": "2099-01-08",
            "JyoCD": "06",
            "RaceNum": "01",
            "Umaban": str(i),
            "Wakuban": str(i),
            "KettoNum": f"horse{i}",
            "TrackCD": "24",
            "Kyori": 1200,
            "SyussoTosu": 4,
            "Barei": 4,
            "SexCD": "1",
            "Futan": 560,
            "BaTaijyu": 480,
            "ZogenSa": 0,
            "KisyuCode": "00001",
            "ChokyosiCode": "00002",
            "tan_odds": 5.0,
            "tan_ninki": i,
            "fuku_odds_low": 2.0 + i / 10,
            "fuku_odds_high": 3.0 + i / 10,
            "fuku_ninki": i,
            "odds_observed_at": "2099-01-08T08:30:00+00:00",
            "odds_snapshot_type": "FINAL_ODDS",
            "retrospective_only": True,
        }
        if include_forbidden:
            row["target_place_paid"] = 1
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_prepare_contract_is_model_ready_only() -> None:
    audit = bridge.audit_prepare_contract()

    assert audit["contract"] == "MODEL_READY_INPUT_ONLY"
    assert audit["raw_input_supported"] is False
    assert audit["requires_market_logit"] is True


def test_raw_bridge_fails_closed_without_market_artifact(tmp_path: Path) -> None:
    raw = raw_fixture(tmp_path / "raw.csv")
    ns = bridge.parse_args(
        [
            "--race-date",
            "2099-01-08",
            "--raw-pre-race-csv",
            str(raw),
            "--output-root",
            str(tmp_path / "bridge"),
            "--fixture",
        ]
    )

    assert bridge.run_pipeline(ns) == 2
    manifest = json.loads(next((tmp_path / "bridge").glob("run_manifest_*.json")).read_text(encoding="utf-8"))
    assert manifest["final_status"] == "BLOCKED_MARKET_ARTIFACT"
    assert manifest["phase6c_registration_performed"] is False
    assert manifest["refit_performed"] is False


def test_raw_bridge_rejects_result_columns(tmp_path: Path) -> None:
    raw = raw_fixture(tmp_path / "raw_bad.csv", include_forbidden=True)

    with pytest.raises(ValueError, match="Raw input audit failed"):
        bridge.audit_raw_input(raw, "2099-01-08", tmp_path)
