# CatBoost Baseline V1 Design

Input is the V2.1.1 feature dataset only. Feature columns are read from `config/feature_sets_v2_1_1.yaml`.

Targets:

- win: `target_win_paid`, eligible rows only
- place: `target_place_paid`, eligible rows only

Splits:

- train: 2016-2023
- validation: 2024
- test: 2025
- latest_holdout: 2026

CatBoost uses GPU with fixed baseline parameters. No class weights, resampling, Optuna, calibration application, ROI, EV, or betting strategy is used.
