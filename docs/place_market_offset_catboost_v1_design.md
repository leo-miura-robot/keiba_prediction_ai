# Place Market Offset CatBoost V1 Design

- Target: `target_place_paid` / all eligible runners; no 1.2-2.5 training filter.
- Market baseline: time-series OOF logistic regression using final odds/rank market features.
- Residual models: CatBoost Logloss with `market_logit` passed as Pool baseline for train, validation, and inference.
- C1 excludes raw odds and uses market-free features plus `p_market` and `market_logit`.
- C2 adds limited rank/field-size market deviation features.
- Model/calibration selection uses only 2020-2024. 2025 and 2026 are fixed evaluation only.
