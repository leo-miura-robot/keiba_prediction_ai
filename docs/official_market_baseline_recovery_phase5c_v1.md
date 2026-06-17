# Official Market Baseline Recovery Phase5C v1

Target:

- Strategy: `ROLLING_10Y`
- Validation year: `2026`
- Training period: `2016-2025`
- CatBoost: `models/place_market_offset_champion_challenger_phase5c_v1/ROLLING_10Y/validation_2026/model.cbm`

Recovery result:

- Market feature names/order are available in the Phase 5C fold manifest.
- Market preprocessing code is available.
- Fitted `StandardScaler` parameters are not saved.
- Fitted `LogisticRegression` parameters are not saved.
- No official market artifact was materialized.
- Final status: `BLOCKED_MISSING_MARKET_PARAMETERS`

No scaler/logistic refit, parameter generation, CatBoost training, Platt refit, raw fallback, commit, or push was performed.
