from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.prepare_place_forward_predictions_phase6c_v2 as prepare
import scripts.run_place_market_offset_forward_paper_phase6c_v2 as phase6c
from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b


def make_pre_race_csv(path: Path, race_date: str = "2099-01-06") -> Path:
    cfg = phase5b.load_yaml(Path("config/place_market_offset_champion_challenger_phase5c_v1.yaml"))
    numeric, categorical = phase5b.load_feature_allowlist(cfg)
    rows = []
    for i, odds in enumerate([2.2, 3.5, 5.0, 8.0], start=1):
        row = {
            "race_id": f"{race_date}_R01",
            "entry_id": f"{race_date}_{i}",
            "race_date": race_date,
            "Umaban": str(i),
            "market_logit": -1.2 + i * 0.1,
            "fuku_odds_low": odds,
        }
        for col in numeric:
            row[col] = 0.0
        for col in categorical:
            row[col] = "0"
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_prepare_official_champion_predictions_and_register(tmp_path: Path) -> None:
    pre = make_pre_race_csv(tmp_path / "pre.csv")
    out_csv = tmp_path / "paper" / "input" / "pre_race_predictions_20990106.csv"

    manifest = prepare.prepare_predictions(
        race_date="2099-01-06",
        pre_race_feature_csv=pre,
        output_csv=out_csv,
        fixture=True,
    )

    pred = pd.read_csv(out_csv)
    assert manifest["refit_performed"] is False
    assert manifest["raw_fallback_used"] is False
    assert set(pred["strategy"]) == {"ROLLING_10Y"}
    assert set(pred["calibrator_type"]) == {"PLATT_SCALING"}
    assert pred["probability_raw"].between(0, 1).all()
    assert pred["probability_calibrated"].between(0, 1).all()
    np.testing.assert_allclose(pred["expected_value"], pred["probability_calibrated"] * pred["fuku_odds_low"])

    ns = phase6c.parse_args(
        [
            "predict",
            "--race-date",
            "2099-01-06",
            "--input-csv",
            str(out_csv),
            "--output-root",
            str(tmp_path / "paper"),
            "--fixture",
            "--prediction-generated-at",
            "2099-01-06T09:00:00+00:00",
            "--data-cutoff-at",
            "2099-01-06T08:00:00+00:00",
            "--odds-observed-at",
            "2099-01-06T08:30:00+00:00",
        ]
    )
    assert phase6c.predict(ns) == 0
    con = sqlite3.connect(tmp_path / "paper" / "forward_paper.sqlite")
    runs = pd.read_sql_query("SELECT * FROM prediction_runs", con)
    db_pred = pd.read_sql_query("SELECT * FROM predictions", con)
    assert runs["calibrator_refit_performed"].eq(0).all()
    assert set(db_pred["calibrator_type"]) == {"PLATT_SCALING"}
    assert db_pred["calibrator_input_space"].eq("logit_probability_raw").all()
    assert db_pred["expected_value"].notna().all()
    with pytest.raises(SystemExit):
        phase6c.predict(ns)


def test_official_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    pre = make_pre_race_csv(tmp_path / "pre.csv")
    cfg = phase6c.load_yaml(Path("config/place_market_offset_forward_paper_phase6c_v2.yaml"))
    cfg["champion"]["calibration"]["required_artifact_sha256"] = "bad"
    bad_cfg = tmp_path / "bad.yml"
    bad_cfg.write_text(__import__("yaml").safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        prepare.prepare_predictions(
            race_date="2099-01-06",
            pre_race_feature_csv=pre,
            output_csv=tmp_path / "out.csv",
            forward_config=bad_cfg,
            fixture=True,
        )
