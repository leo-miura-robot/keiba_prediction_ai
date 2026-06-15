# CatBoost Baseline V2.1.2 V1 Design

Input: `outputs/model_feature_dataset_v2_1_2`
Feature sets: `config/feature_sets_v2_1_2.yaml`

Six new CatBoost GPU binary classifiers are trained. V1.0.2 weights are not reused.

The `market_aware` feature set uses final `NL_O1` odds for an ideal-condition final-odds model. It is not a pre-race live operation model.

Phase 1 future ROI goals are documented only: win ROI >= 90%, place ROI >= 90%. ROI, EV, bet generation, bankroll allocation, calibration application, Ability, ANA, and Ranker are not implemented here.
