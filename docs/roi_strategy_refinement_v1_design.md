# ROI Strategy Refinement V1 Design

- Input predictions: `outputs/final_odds_two_models_v1/oof_predictions.parquet` and `final_predictions.parquet`.
- Models are not retrained; this task only analyzes existing predictions and refines betting rules.
- Rule design and selection use only 2020-2024 walk-forward OOF predictions.
- 2025 test and 2026 latest_holdout are fixed evaluation periods; thresholds are not changed there.
- Place EV uses `conservative_place_probability * fuku_odds_low`; ROI always uses actual `fuku_pay`.
- Win EV uses existing conservative probability and actual `tan_pay` for ROI.
- Final-odds models are ideal-condition models and are not pre-race live operation models.
