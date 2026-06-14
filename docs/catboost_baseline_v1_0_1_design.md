# CatBoost Baseline V1.0.1 Design

V1.0.1 keeps the V2.1.1 feature dataset unchanged and adds YAML-resolved training configuration, artifact fingerprint resume, idempotent analysis outputs, fixed-width and quantile calibration diagnostics, and same-sample market comparison.

Phase 1 future ROI goal is documented only: win ROI >= 0.90 and place ROI >= 0.90. ROI, EV, bet generation, bankroll allocation, probability calibration application, Ability, ANA, LightGBM Ranker, Optuna, and walk-forward redesign are not implemented in this phase.

A future ROI pass must not be judged successful from a single high payout or one long-shot dependency; it must consider sufficient bet count, validation-only threshold decisions, no test tuning, odds-band ROI, period stability, and top-payout exclusion robustness.
