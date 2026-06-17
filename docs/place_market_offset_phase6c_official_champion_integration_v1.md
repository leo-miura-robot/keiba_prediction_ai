# Phase 6C Official Champion Integration v1

Official Champion:

- Strategy: `ROLLING_10Y`
- Calibrator: `PLATT_SCALING`
- Calibrator artifact: `outputs/place_market_offset_official_calibrators_phase6a_v1/rolling_10y_platt_phase6a_v1.json`
- EV: `probability_calibrated * fuku_odds_low`
- Official threshold: `EV >= 1.00`
- Shadow tiers: `1.05`, `1.10`, `1.15`
- 15Y status: `BLOCKED_MISSING_ISOTONIC_THRESHOLDS`

Prediction wrapper:

```powershell
.\scripts\run_forward_predict_official_champion_phase6c_v2.ps1 `
  -RaceDate 2026-06-20 `
  -PreRaceFeatureCsv "inputs\forward\pre_race_20260620.csv" `
  -OutputRoot "outputs\place_market_offset_forward_paper_phase6c_v2"
```

The pre-race feature CSV must already contain the official feature allowlist columns and `market_logit`. The script does not fit a market model, CatBoost model, Platt calibrator, or Isotonic calibrator.

Fail-closed checks include model/calibrator existence, hash, strategy, calibrator type, input space, refit flag, feature schema, finite probabilities, odds presence, EV recalculation, duplicate prediction, and timestamp ordering.
