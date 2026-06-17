from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.run_place_market_offset_safe_calibration_phase6a_v1 as phase6a
import scripts.run_place_market_offset_year_strategy_phase5b_v2 as phase5b


def test_new_8fold_windows_match_phase6a_spec() -> None:
    cfg = phase6a.load_yaml(phase6a.ROOT / "config/place_market_offset_safe_calibration_phase6a_v1.yaml")
    windows = phase5b.build_windows(cfg, ["ROLLING_10Y", "ROLLING_15Y"], [2016, 2017, 2018, 2019])
    by_key = {(w.strategy, w.validation_year): w for w in windows}
    assert by_key[("ROLLING_10Y", 2016)].train_years == tuple(range(2006, 2016))
    assert by_key[("ROLLING_10Y", 2019)].train_years == tuple(range(2009, 2019))
    assert by_key[("ROLLING_15Y", 2016)].train_years == tuple(range(2006, 2016))
    assert by_key[("ROLLING_15Y", 2019)].train_years == tuple(range(2006, 2019))
    assert all(w.validation_year not in w.train_years for w in windows)


def sample_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for strategy in ["ROLLING_10Y", "ROLLING_15Y"]:
        for year in range(2016, 2027):
            for i in range(20):
                p = 0.1 + 0.7 * rng.random()
                y = int(rng.random() < p)
                rows.append(
                    {
                        "entry_id": f"{strategy}_{year}_{i}",
                        "race_id": f"r{year}_{i // 5}",
                        "race_date": f"{year}-01-01",
                        "Year": year,
                        "strategy": strategy,
                        "actual_place": y,
                        "target_place_paid": y,
                        "probability_raw": p,
                        "fuku_odds_low": 2.0,
                        "fuku_pay": 200 if y else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_walk_forward_calibration_uses_prior_years_only() -> None:
    cfg = {
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "diagnostic_years": [2025, 2026],
        "calibration_fit_start_year": 2016,
        "epsilon": 1e-6,
    }
    pred = sample_frame()
    calibrated, prov, _rel = phase6a.walk_forward_calibration(pred, cfg)
    assert set(calibrated["Year"].unique()) == set(range(2020, 2027))
    assert prov["uses_only_prior_years"].all()
    assert set(prov["calibration_method"].unique()) == set(phase6a.METHODS)
    assert prov.loc[prov["evaluation_year"].eq(2025), "fit_end_year"].max() == 2024
    assert prov.loc[prov["evaluation_year"].eq(2026), "fit_end_year"].max() == 2025


def test_target_is_binary_paid_place_not_rank_transform() -> None:
    d = sample_frame()
    assert set(d["actual_place"].unique()) <= {0, 1}
    assert d["actual_place"].astype(int).equals(d["target_place_paid"].astype(int))


def test_calibrator_outputs_are_bounded_and_raw_identity_is_exact() -> None:
    eps = 1e-6
    train = sample_frame()
    p = train["probability_raw"].to_numpy(float)
    for method in phase6a.METHODS:
        cal = phase6a.fit_calibrator(method, train, eps)
        out = cal.transform(p, eps)
        assert np.all(out >= eps)
        assert np.all(out <= 1 - eps)
        if method == "RAW_IDENTITY":
            assert np.allclose(out, p)


def test_selection_excludes_2025_2026_and_roi() -> None:
    selection = pd.DataFrame(
        {
            "strategy": ["ROLLING_10Y"] * 2,
            "calibration_method": ["A", "B"],
            "pooled_logloss": [0.2, 0.1],
            "pooled_brier": [0.1, 0.1],
            "pooled_ece": [0.1, 0.1],
            "mean_logloss": [0.2, 0.1],
            "mean_brier": [0.1, 0.1],
            "mean_ece": [0.1, 0.1],
            "worst_logloss": [0.2, 0.1],
            "worst_brier": [0.1, 0.1],
        }
    )
    out = phase6a.select_calibrators(selection)
    assert out["selected_calibration_method"].iloc[0] == "B"
    assert bool(out["operationally_activated"].iloc[0]) is False
    assert "2025" not in out["selection_years"].iloc[0]
    assert "ROI" in out["selection_basis"].iloc[0]


def test_selection_primary_uses_pooled_not_year_mean() -> None:
    selection = pd.DataFrame(
        {
            "strategy": ["ROLLING_10Y", "ROLLING_10Y"],
            "calibration_method": ["YEAR_MEAN_WINNER", "POOLED_WINNER"],
            "pooled_logloss": [0.20, 0.10],
            "pooled_brier": [0.10, 0.10],
            "pooled_ece": [0.10, 0.10],
            "mean_logloss": [0.10, 0.20],
            "mean_brier": [0.10, 0.10],
            "mean_ece": [0.10, 0.10],
            "worst_logloss": [0.20, 0.20],
            "worst_brier": [0.10, 0.10],
        }
    )
    out = phase6a.select_calibrators(selection)
    assert out["selected_calibration_method"].iloc[0] == "POOLED_WINNER"
    assert "pooled-row Logloss" in out["selection_basis"].iloc[0]
