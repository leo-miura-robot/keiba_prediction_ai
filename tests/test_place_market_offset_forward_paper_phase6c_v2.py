from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_place_market_offset_forward_paper_phase6c_v2 as phase6c


def run_cli(args: list[str]) -> int:
    return phase6c.parse_args(args) and phase6c.main_from_args if False else 0


def call(args: list[str]) -> int:
    ns = phase6c.parse_args(args)
    if ns.command == "predict":
        return phase6c.predict(ns)
    if ns.command == "settle":
        return phase6c.settle(ns)
    if ns.command == "report":
        return phase6c.report(ns.output_root, include_fixture=ns.include_fixture)
    if ns.command == "smoke-fixture":
        return phase6c.smoke(ns)
    raise AssertionError(ns.command)


def test_tier_rows_are_nested() -> None:
    cfg = phase6c.load_yaml(phase6c.DEFAULT_CONFIG)
    df = phase6c.fixture_predictions("2099-01-02")
    df["ev_at_prediction"] = df["probability_calibrated"] * df["fuku_odds_low_at_prediction"]
    tiers = phase6c.tier_rows(df, cfg, "run")
    ok, detail = phase6c.audit_tier_inclusion(tiers)
    assert ok, detail
    assert set(tiers["threshold_tier"]) == {"CORE", "MARGIN", "HIGH", "VERY_HIGH"}


def test_predict_rejects_duplicate_and_timestamp_violation(tmp_path: Path) -> None:
    out = tmp_path / "paper"
    base = ["predict", "--output-root", str(out), "--race-date", "2099-01-03", "--fixture"]
    assert call(base + ["--prediction-generated-at", "2099-01-03T09:00:00+00:00", "--data-cutoff-at", "2099-01-03T08:00:00+00:00"]) == 0
    with pytest.raises(SystemExit):
        call(base + ["--prediction-generated-at", "2099-01-03T09:00:00+00:00"])
    with pytest.raises(SystemExit):
        call(["predict", "--output-root", str(tmp_path / "bad"), "--race-date", "2099-01-04", "--fixture", "--prediction-generated-at", "2099-01-04T09:00:00+00:00", "--data-cutoff-at", "2099-01-04T10:00:00+00:00"])


def test_settle_append_only_and_prediction_immutable(tmp_path: Path) -> None:
    out = tmp_path / "paper"
    base = ["--output-root", str(out), "--race-date", "2099-01-05", "--fixture"]
    call(["predict", *base, "--prediction-generated-at", "2099-01-05T09:00:00+00:00"])
    con = sqlite3.connect(out / "forward_paper.sqlite")
    before = pd.read_sql_query("SELECT * FROM predictions ORDER BY strategy, entry_id", con).to_json()
    with pytest.raises(SystemExit):
        call(["settle", *base, "--settled-at", "2099-01-05T08:00:00+00:00"])
    call(["settle", *base, "--settled-at", "2099-01-05T10:00:00+00:00"])
    call(["settle", *base, "--settled-at", "2099-01-05T10:05:00+00:00"])
    after = pd.read_sql_query("SELECT * FROM predictions ORDER BY strategy, entry_id", con).to_json()
    settlements = pd.read_sql_query("SELECT * FROM settlements", con)
    assert before == after
    assert len(settlements) > 0


def test_fixture_smoke_excludes_forward_reports(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    assert call(["smoke-fixture", "--output-root", str(out)]) == 0
    audit = json.loads((out / "fixture_smoke_audit.json").read_text(encoding="utf-8"))
    assert audit["four_tiers_generated"]
    assert audit["tier_inclusion"]
    assert audit["fixture_excluded_from_forward_reports"]
    summary = pd.read_csv(out / "cumulative_summary.csv")
    assert summary.empty
