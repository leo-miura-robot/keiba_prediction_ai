# Place Odds EV Surface V1 Design

- Existing `final_odds_two_models_v1` place predictions are used.
- Models are not retrained and feature datasets are not regenerated.
- Selection uses only 2020-2024; 2025/2026 are fixed holdout evaluations.
- ROI uses actual `fuku_pay`, not estimated odds payout.
- DB was not accessed; existing DB validation manifest was reused.
- Odds ranges use lower-inclusive / upper-exclusive boundaries.