from __future__ import annotations

import math

import numpy as np
import pandas as pd

import scripts.run_place_market_offset_ev_threshold_phase6b_v1 as phase6b


def cfg() -> dict:
    return {
        "threshold_min": 1.00,
        "threshold_max": 1.30,
        "threshold_step": 0.01,
        "stake_yen": 100,
        "probability_column": "probability_calibrated",
        "odds_column": "fuku_odds_low",
        "payout_column": "fuku_pay",
        "selection_years": [2020, 2021, 2022, 2023, 2024],
        "diagnostic_years": [2025, 2026],
        "bootstrap_iterations": 50,
        "random_seed": 42,
        "champion_strategy": "ROLLING_10Y",
        "challenger_strategy": "ROLLING_15Y",
        "eligibility": {
            "combined_bet_count_min": 1,
            "bet_years_min": 1,
            "min_yearly_bet_count": 1,
            "roi_ge_90_years_min": 1,
            "combined_roi_min": 0.0,
            "probability_roi_ge_90_min": 0.0,
            "top3_payout_zeroed_roi_min": 0.0,
            "top5_payout_zeroed_roi_min": 0.0,
            "nested_validation_combined_roi_min": 0.0,
        },
    }


def sample() -> pd.DataFrame:
    rows = []
    for strategy in ["ROLLING_10Y", "ROLLING_15Y"]:
        for year in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            for i in range(4):
                rows.append(
                    {
                        "strategy": strategy,
                        "calibration_method": "PLATT_SCALING" if strategy == "ROLLING_10Y" else "ISOTONIC",
                        "Year": year,
                        "race_id": f"r{year}_{i//2}",
                        "entry_id": f"{strategy}_{year}_{i}",
                        "race_date": f"{year}-01-01",
                        "probability_calibrated": 0.6 if i == 0 else 0.3,
                        "fuku_odds_low": 2.0,
                        "fuku_pay": 200 if i == 0 else 0,
                        "payout": 200 if i == 0 else 0,
                        "ev": 1.2 if i == 0 else 0.6,
                        "TrackCD": 10,
                        "Kyori": 1600,
                        "SyussoTosu": 12,
                    }
                )
    return pd.DataFrame(rows)


def test_threshold_grid_exact() -> None:
    th = phase6b.threshold_grid(cfg())
    assert len(th) == 31
    assert th[0] == 1.0
    assert th[-1] == 1.3


def test_roi_uses_total_stake_not_year_mean() -> None:
    c = cfg()
    p = sample()
    yearly, combined = phase6b.grid_tables(p, c)
    row = combined[(combined["strategy"].eq("ROLLING_10Y")) & (combined["threshold"].eq(1.0))].iloc[0]
    assert row["stake"] == 5 * 100
    assert row["payout"] == 5 * 200
    assert row["roi"] == 200.0
    assert set(yearly["Year"].unique()) == {2020, 2021, 2022, 2023, 2024}


def test_payout_zeroed_never_exceeds_normal() -> None:
    rr, pz = phase6b.stress_tables(sample(), cfg(), [2020, 2021], [("x", 1.0)])
    assert not rr.empty
    assert np.all(pz["payout_zeroed_roi"].fillna(-math.inf) <= pz["normal_roi"].fillna(math.inf) + 1e-12)


def test_bootstrap_includes_no_bet_races() -> None:
    out = phase6b.race_bootstrap(sample(), cfg(), [2020, 2021], [1.3], "ROLLING_10Y")
    assert out["races"].iloc[0] == 4
    assert math.isnan(out["point_roi"].iloc[0])


def test_selected_threshold_remains_not_activated() -> None:
    c = cfg()
    p = sample()
    yearly, combined = phase6b.grid_tables(p, c)
    rr, pz = phase6b.stress_tables(p[p["strategy"].eq("ROLLING_10Y")], c, [2020, 2021, 2022, 2023, 2024], [("grid", 1.0)])
    boot = phase6b.race_bootstrap(p, c, [2020, 2021, 2022, 2023, 2024], phase6b.threshold_grid(c), "ROLLING_10Y")
    elig = phase6b.eligibility(combined, yearly, boot, pz, c, "ROLLING_10Y")
    nested = pd.DataFrame({"validation_year": [2021], "validation_payout": [100], "validation_stake": [100]})
    selected = phase6b.choose_threshold(elig, nested, c, "ROLLING_10Y")
    assert selected["operationally_activated"] is False
