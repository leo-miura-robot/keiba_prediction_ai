# Phase 6B EV Threshold Runbook

Phase 6B uses only certified Phase 6A calibrated predictions.

- Champion: `ROLLING_10Y + PLATT_SCALING`
- Shadow: `ROLLING_15Y + ISOTONIC`
- Selection years: 2020-2024 only
- Diagnostic years: 2025-2026 only
- EV: `probability_calibrated * fuku_odds_low`
- Stake: 100 yen flat per selected entry
- Threshold grid: `1.00` to `1.30` by `0.01`

`operationally_activated` remains `false`. A threshold candidate, even if certified, requires user approval before any operational use.
