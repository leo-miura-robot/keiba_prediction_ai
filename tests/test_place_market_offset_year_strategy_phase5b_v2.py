from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

import scripts.run_place_market_offset_year_strategy_phase5b_v2 as runner
from src.features.history_builder_v2_1 import build_pre_day_history_features_v2_1
from src.features.target_builder import add_target_columns


def cfg() -> dict:
    return {
        "version": "test",
        "epsilon": 1e-6,
        "stake_yen": 100,
        "odds_column": "fuku_odds_low",
        "payout_column": "fuku_pay",
        "target_column": "target_place_paid",
        "eligible_column": "eligible_for_place_training",
        "random_seed": 42,
        "bootstrap_iterations": 10,
        "catboost": {"iterations": 99, "use_best_model": True, "od_type": "Iter", "od_wait": 20, "early_stopping_rounds": 10},
        "feature_drops": ["KisyuCode", "ChokyosiCode"],
        "parity_tolerance": {
            "row_key_match_rate": 1.0,
            "market_logit_p99_abs_diff": 1e-8,
            "probability_raw_p99_abs_diff": 1e-4,
            "logloss_abs_diff": 1e-5,
            "brier_abs_diff": 1e-5,
        },
        "auxiliary_years": [2016, 2017, 2018, 2019],
        "diagnostic_years": [2025, 2026],
        "strategies": [
            {"name": "LEGACY_2016", "mode": "legacy_compat", "history_start_year": 2016, "model_train_start_year": 2016},
            {"name": "WARMUP_2006_TRAIN_2016", "mode": "expanding", "history_start_year": 2006, "model_train_start_year": 2016},
            {"name": "EXPANDING_FULL_2006", "mode": "expanding", "history_start_year": 2006, "model_train_start_year": 2006},
            {"name": "ROLLING_10Y", "mode": "rolling", "history_start_year": 2006, "model_train_start_year": 2006, "rolling_years": 10},
            {"name": "ROLLING_15Y", "mode": "rolling", "history_start_year": 2006, "model_train_start_year": 2006, "rolling_years": 15},
            {"name": "FULL_2006_TIME_DECAY_HL5", "mode": "expanding", "history_start_year": 2006, "model_train_start_year": 2006, "half_life_years": 5},
            {"name": "FULL_2006_TIME_DECAY_HL10", "mode": "expanding", "history_start_year": 2006, "model_train_start_year": 2006, "half_life_years": 10},
        ],
    }


def test_catboost_safety_removes_outer_validation_controls() -> None:
    params = runner.make_safe_catboost_params(cfg())
    assert params["iterations"] == 300
    assert params["use_best_model"] is False
    assert "od_type" not in params
    assert "od_wait" not in params
    assert "early_stopping_rounds" not in params
    runner.assert_safe_catboost_params(params)
    assert "eval_set" not in inspect.getsource(runner.train_residual_model)


def test_year_windows_and_history_train_start_separation() -> None:
    c = cfg()
    windows = {w.strategy: w for w in runner.build_windows(c, runner.ALL_STRATEGIES, [2024])}
    assert windows["LEGACY_2016"].train_years == tuple(range(2016, 2024))
    assert windows["LEGACY_2016"].market_mode == "LEGACY_COMPAT"
    assert windows["WARMUP_2006_TRAIN_2016"].history_start_year == 2006
    assert windows["WARMUP_2006_TRAIN_2016"].model_train_start_year == 2016
    assert windows["WARMUP_2006_TRAIN_2016"].train_years == tuple(range(2016, 2024))
    assert windows["EXPANDING_FULL_2006"].train_years == tuple(range(2006, 2024))
    assert windows["ROLLING_10Y"].train_years == tuple(range(2014, 2024))
    assert windows["ROLLING_15Y"].train_years == tuple(range(2009, 2024))
    assert windows["FULL_2006_TIME_DECAY_HL5"].half_life_years == 5.0
    assert all(w.validation_year not in w.train_years for w in windows.values())


def test_market_and_residual_window_provenance_is_recorded(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Year": [2022, 2022, 2023, 2024, 2024],
            "actual_place": [1, 0, 1, 0, 1],
            "race_id": ["a", "a", "b", "c", "c"],
            "race_date": ["2022-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2024-01-01"],
            "fuku_odds_low": [1.2, 2.0, 1.4, 1.6, 1.1],
            "SyussoTosu": [8, 8, 8, 8, 8],
            "place_rank_limit": [3, 3, 3, 3, 3],
            "fuku_ninki": [1, 2, 1, 2, 1],
            "tan_ninki": [1, 2, 1, 2, 1],
        }
    )

    class Model:
        def predict_proba(self, x):
            return np.column_stack([np.full(len(x), 0.7), np.full(len(x), 0.3)])

    monkeypatch.setattr(runner, "load_yaml", lambda _p: {"market_baseline": {"features": ["fuku_odds_low"], "C": 1.0, "max_iter": 100}})
    monkeypatch.setattr(runner, "fit_market_model", lambda _train, _cfg: Model())
    window = runner.build_fold_window({"name": "EXPANDING_FULL_2006", "mode": "expanding", "history_start_year": 2006, "model_train_start_year": 2022}, 2024)
    train, valid, prov = runner.make_market_logit_for_fold(frame, window, {**cfg(), "base_c1r0_config": "unused"})
    assert set(train["Year"]) == {2022, 2023}
    assert set(valid["Year"]) == {2024}
    assert prov["market_train_start"] == 2022
    assert prov["market_train_end"] == 2023
    assert prov["market_train_rows"] == len(train)
    assert prov["residual_train_start"] == prov["market_train_start"]
    assert prov["residual_train_rows"] == prov["market_train_rows"]
    assert prov["market_input_columns"] == ["fuku_odds_low"]


def test_apply_target_column_uses_paid_place_and_does_not_reconvert_binary_actual() -> None:
    d = pd.DataFrame(
        {
            "target_place": [1, 1, 0],
            "target_place_paid": [1, 0, 0],
            "actual_place": [0, 0, 1],
            "KakuteiJyuni": [1, 3, 4],
            "SyussoTosu": [7, 7, 8],
            "place_rank_limit": [2, 2, 3],
        }
    )
    out = runner.apply_target_column(d, cfg())
    assert list(out["actual_place"]) == [1, 0, 0]
    assert set(out["actual_place"]) <= {0, 1}


def test_canonical_paid_place_handles_two_slot_race_and_cancellations() -> None:
    d = pd.DataFrame(
        {
            "target_place": [1, 1, 0, 0],
            "target_place_paid": [1, 0, 0, 0],
            "KakuteiJyuni": [2, 3, 0, 0],
            "SyussoTosu": [7, 7, 7, 7],
            "place_rank_limit": [2, 2, 2, 2],
            "IJyoCD": ["0", "0", "2", "5"],
        }
    )
    out = runner.apply_target_column(d, cfg())
    assert list(out["actual_place"]) == [1, 0, 0, 0]


def test_feature_allowlist_exact_and_forbidden_columns(tmp_path) -> None:
    allow = tmp_path / "allow.json"
    allow.write_text('{"numeric":["a","Year","market_logit"],"categorical":["b","KisyuCode","ChokyosiCode"]}', encoding="utf-8")
    c = {**cfg(), "feature_allowlist_path": str(allow)}
    with pytest.raises(ValueError):
        runner.load_feature_allowlist(c)
    allow.write_text('{"numeric":["a"],"categorical":["b","KisyuCode","ChokyosiCode"]}', encoding="utf-8")
    assert runner.load_feature_allowlist(c) == (["a"], ["b"])


def test_time_decay_formula_normalization_and_ess() -> None:
    train = pd.DataFrame({"race_date": ["2023-01-01", "2022-01-01", "2020-01-01"]})
    weights, summary = runner.time_decay_weights(train, 2024, 5.0)
    age = np.array([(pd.Timestamp("2024-01-01") - pd.Timestamp(d)).days / 365.25 for d in train["race_date"]])
    expected = np.power(2.0, -age / 5.0)
    expected = expected / expected.mean()
    assert np.allclose(weights, expected)
    assert summary["half_life_years"] == 5.0
    assert weights.mean() == pytest.approx(1.0)
    assert summary["effective_sample_size"] <= len(train)


def test_stress_roi_population_and_na_behavior() -> None:
    d = pd.DataFrame(
        {
            "strategy": ["s"] * 4,
            "Year": [2024] * 4,
            "probability_raw": [0.9, 0.8, 0.1, 0.2],
            "fuku_odds_low": [1.2, 2.0, 1.0, 1.0],
            "fuku_pay": [120, 500, 0, 0],
            "actual_place": [1, 1, 0, 0],
        }
    )
    normal, rr, pz = runner.stress_roi_rows(d, cfg(), [1, 3])
    assert normal["bet_count"].iloc[0] == 2
    assert pz["stake"].iloc[0] == normal["stake"].iloc[0]
    assert rr["stake"].iloc[0] < normal["stake"].iloc[0]
    assert (pz["roi"].fillna(-math.inf) <= pz["normal_roi"].fillna(math.inf)).all()

    no_picks = d.assign(probability_raw=0.01)
    normal, _rr, _pz = runner.stress_roi_rows(no_picks, cfg(), [1])
    assert normal["bet_count"].iloc[0] == 0
    assert math.isnan(normal["roi"].iloc[0])


def test_summary_combined_roi_uses_total_stake_not_year_mean() -> None:
    d = pd.DataFrame(
        {
            "strategy": ["s"] * 4,
            "Year": [2020, 2020, 2021, 2021],
            "race_id": ["r1", "r1", "r2", "r2"],
            "entry_id": ["e1", "e2", "e3", "e4"],
            "race_date": ["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01"],
            "actual_place": [1, 0, 1, 0],
            "probability_raw": [0.9, 0.9, 0.9, 0.01],
            "catboost_residual_score": [0.1, 0.2, 0.3, 0.4],
            "fuku_odds_low": [2.0, 2.0, 2.0, 1.0],
            "fuku_pay": [100, 0, 400, 0],
        }
    )
    summaries = runner.summarize_predictions(d, cfg())
    roi = summaries["roi_diagnostic_raw"].sort_values("Year")
    assert list(roi["roi"]) == [50.0, 400.0]
    combined = d[d["probability_raw"].mul(d["fuku_odds_low"]).ge(1.0)]
    assert runner.roi_value(combined, cfg()) == pytest.approx((100 + 400) / 300 * 100)


def test_parity_gate_passes_exact_match_and_fails_probability(monkeypatch) -> None:
    base = pd.DataFrame(
        {
            "entry_id": ["e1", "e2"],
            "race_id": ["r1", "r1"],
            "race_date": ["2024-01-01", "2024-01-01"],
            "Year": [2024, 2024],
            "actual_place": [1, 0],
            "market_logit": [0.1, -0.2],
            "probability": [0.6, 0.4],
            "probability_raw": [0.6, 0.4],
        }
    )
    new = base.drop(columns=["probability"]).assign(strategy="LEGACY_2016", tree_count=300)
    monkeypatch.setattr(runner, "model_tree_count", lambda _p: 300)
    monkeypatch.setattr(runner, "legacy_model_path", lambda _cfg, _year: "unused")
    result = runner.parity_gate(base, new, cfg(), ["a"], ["b"])
    assert result["passed"].iloc[0]
    bad = new.assign(probability_raw=[0.7, 0.3])
    result = runner.parity_gate(base, bad, cfg(), ["a"], ["b"])
    assert not result["passed"].iloc[0]
    assert not result["historical_prediction_passed"].iloc[0]
    corrected = runner.parity_gate(base, bad, cfg(), ["a"], ["b"], reference_mode="corrected")
    assert corrected["passed"].iloc[0]
    assert corrected["structural_passed"].iloc[0]
    assert not corrected["historical_prediction_passed"].iloc[0]
    assert corrected["reference_type"].iloc[0] == "corrected_legacy"
    assert corrected["comparison_type"].iloc[0] == "blocking"


def test_parity_key_normalization_matches_string_and_timestamp(monkeypatch) -> None:
    old = pd.DataFrame(
        {
            "entry_id": ["e1"],
            "race_id": ["r1"],
            "race_date": ["2024-01-06"],
            "Year": [2024],
            "actual_place": [1],
            "market_logit": [0.1],
            "probability": [0.6],
            "probability_raw": [0.6],
        }
    )
    new = old.drop(columns=["probability"]).assign(race_date=[pd.Timestamp("2024-01-06")], strategy="LEGACY_2016", tree_count=300)
    monkeypatch.setattr(runner, "model_tree_count", lambda _p: 300)
    monkeypatch.setattr(runner, "legacy_model_path", lambda _cfg, _year: "unused")
    result = runner.parity_gate(old, new, cfg(), ["a"], ["b"])
    assert result["passed"].iloc[0]
    assert result["both_count"].iloc[0] == 1
    assert result["old_only_count"].iloc[0] == 0
    assert result["new_only_count"].iloc[0] == 0


def test_parity_key_normalization_matches_datetime_us_and_ns(monkeypatch) -> None:
    old = pd.DataFrame(
        {
            "entry_id": ["e1"],
            "race_id": ["r1"],
            "race_date": pd.Series(np.array(["2024-01-06"], dtype="datetime64[us]")),
            "Year": [2024],
            "actual_place": [1],
            "market_logit": [0.1],
            "probability": [0.6],
            "probability_raw": [0.6],
        }
    )
    new = old.drop(columns=["probability"]).copy()
    new["race_date"] = pd.Series(np.array(["2024-01-06"], dtype="datetime64[ns]"))
    new = new.assign(strategy="LEGACY_2016", tree_count=300)
    monkeypatch.setattr(runner, "model_tree_count", lambda _p: 300)
    monkeypatch.setattr(runner, "legacy_model_path", lambda _cfg, _year: "unused")
    result = runner.parity_gate(old, new, cfg(), ["a"], ["b"])
    assert result["passed"].iloc[0]
    audit = pd.DataFrame(json.loads(result["key_dtype_audit"].iloc[0]))
    raw_race_date = audit[(audit["phase"].eq("raw")) & (audit["column"].eq("race_date"))].iloc[0]
    normalized_race_date = audit[(audit["phase"].eq("normalized")) & (audit["column"].eq("race_date"))].iloc[0]
    assert raw_race_date["old_dtype"] != raw_race_date["new_dtype"]
    assert normalized_race_date["old_dtype"] == normalized_race_date["new_dtype"]


def test_parity_key_normalization_rejects_invalid_date() -> None:
    bad = pd.DataFrame({"entry_id": ["e1"], "race_id": ["r1"], "race_date": ["not-a-date"], "Year": [2024]})
    with pytest.raises(Exception):
        runner.normalize_parity_keys(bad)


def test_parity_key_duplicates_fail(monkeypatch) -> None:
    old = pd.DataFrame(
        {
            "entry_id": ["e1", "e1"],
            "race_id": ["r1", "r1"],
            "race_date": ["2024-01-06", "2024-01-06"],
            "Year": [2024, 2024],
            "actual_place": [1, 1],
            "market_logit": [0.1, 0.1],
            "probability": [0.6, 0.6],
            "probability_raw": [0.6, 0.6],
        }
    )
    new = old.drop(columns=["probability"]).assign(strategy="LEGACY_2016", tree_count=300)
    monkeypatch.setattr(runner, "model_tree_count", lambda _p: 300)
    monkeypatch.setattr(runner, "legacy_model_path", lambda _cfg, _year: "unused")
    with pytest.raises(ValueError, match="duplicates"):
        runner.parity_gate(old, new, cfg(), ["a"], ["b"])


def test_parity_key_normalization_does_not_change_prediction_target_or_market_values() -> None:
    d = pd.DataFrame(
        {
            "entry_id": ["e1"],
            "race_id": ["r1"],
            "race_date": [pd.Timestamp("2024-01-06 10:30:00")],
            "Year": [2024],
            "actual_place": [1],
            "market_logit": [0.123],
            "probability_raw": [0.456],
        }
    )
    normalized = runner.normalize_parity_keys(d)
    assert normalized["race_date"].iloc[0] == "2024-01-06"
    assert normalized["actual_place"].iloc[0] == d["actual_place"].iloc[0]
    assert normalized["market_logit"].iloc[0] == d["market_logit"].iloc[0]
    assert normalized["probability_raw"].iloc[0] == d["probability_raw"].iloc[0]


def test_parity_gate_fails_when_old_only_or_new_only_exists(monkeypatch) -> None:
    old = pd.DataFrame(
        {
            "entry_id": ["e1", "e2"],
            "race_id": ["r1", "r1"],
            "race_date": ["2024-01-06", "2024-01-06"],
            "Year": [2024, 2024],
            "actual_place": [1, 0],
            "market_logit": [0.1, -0.2],
            "probability": [0.6, 0.4],
            "probability_raw": [0.6, 0.4],
        }
    )
    new = old.iloc[[0]].drop(columns=["probability"]).assign(strategy="LEGACY_2016", tree_count=300)
    monkeypatch.setattr(runner, "model_tree_count", lambda _p: 300)
    monkeypatch.setattr(runner, "legacy_model_path", lambda _cfg, _year: "unused")
    result = runner.parity_gate(old, new, cfg(), ["a"], ["b"])
    assert not result["passed"].iloc[0]
    assert result["old_only_count"].iloc[0] == 1
    assert result["new_only_count"].iloc[0] == 0

    extra = old.iloc[[0]].drop(columns=["probability"]).assign(entry_id=["e3"], strategy="LEGACY_2016", tree_count=300)
    result = runner.parity_gate(old.iloc[[0]], extra, cfg(), ["a"], ["b"])
    assert not result["passed"].iloc[0]
    assert result["old_only_count"].iloc[0] == 1
    assert result["new_only_count"].iloc[0] == 1


def test_probability_raw_only_and_no_calibration_fit_in_runner_source() -> None:
    source = inspect.getsource(runner)
    assert "IsotonicRegression" not in source
    assert 'pred["probability_calibrated"]' not in source
    assert "probability_calibrated =" not in source
    assert "probability_used_for_selection" in source
    assert "probability_raw" in source


def test_bootstrap_is_race_unit_and_seeded() -> None:
    source = inspect.getsource(runner.paired_bootstrap_vs_legacy)
    assert 'groupby("race_id")' in source
    assert "np.random.default_rng(seed)" in source


def test_history_builder_excludes_current_race_and_same_day_future() -> None:
    def row(race_id: str, entry_id: str, race_date: str, horse: str = "h1") -> dict:
        return {
            "race_id": race_id,
            "entry_id": entry_id,
            "race_date": race_date,
            "Year": int(race_date[:4]),
            "MonthDay": int(race_date[5:7] + race_date[8:10]),
            "JyoCD": "06",
            "Kaiji": 1,
            "Nichiji": 1,
            "RaceNum": 1,
            "Wakuban": 1,
            "Umaban": 1,
            "KettoNum": horse,
            "KisyuCode": "01001",
            "ChokyosiCode": "02001",
            "TrackCD": "10",
            "Kyori": 1600,
            "Futan": 560,
            "BaTaijyu": 480,
            "SibaBabaCD": "1",
            "DirtBabaCD": "0",
            "IJyoCD": "0",
            "KakuteiJyuni": 1,
            "SyussoTosu": 8,
            "Ninki": 1,
            "HaronTimeL3": 35.0,
            "Time": 95.0,
            "tan_pay": 100,
            "fuku_pay": 100,
            "is_win_paid": 1,
            "is_place_paid": 1,
            "tan_odds": 2.0,
            "tan_ninki": 1,
            "fuku_odds_low": 1.1,
            "fuku_odds_high": 1.3,
            "fuku_ninki": 1,
            "TanVote": 100,
            "FukuVote": 200,
        }

    df = add_target_columns(pl.DataFrame([
        row("r1", "e1", "2020-01-01"),
        row("r2", "e2", "2020-01-01"),
        row("r3", "e3", "2020-01-02"),
    ]))
    out, _state, audit, _samples = build_pre_day_history_features_v2_1(df)
    by_entry = {r["entry_id"]: r for r in out.to_dicts()}
    assert by_entry["e1"]["history_cutoff_date"] is None
    assert by_entry["e2"]["history_cutoff_date"] is None
    assert by_entry["e3"]["history_cutoff_date"] == "2020-01-01"
    assert all(v["same_race"] == 0 and v["same_day"] == 0 and v["future"] == 0 for v in audit.values())
