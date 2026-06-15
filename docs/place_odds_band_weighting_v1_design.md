# Place Odds Band Weighting V1 Design

- Existing `final_odds_two_models_v1` place predictions are used.
- Models are not retrained and feature datasets are not regenerated.
- Odds bands are exclusive left-closed/right-open intervals based on `fuku_odds_low`.
- Rule selection uses only 2020-2024 validation predictions.
- 2025 and 2026 are fixed evaluations with no threshold adjustment.
- Official evaluation uses equal 100 yen stakes; weighted stakes are reference diagnostics only.
- DB was not accessed by this script; existing validation manifest was reused.
