# Phase 5C Champion-Challenger Forward Runbook

Champion is `ROLLING_10Y`; Challenger is `ROLLING_15Y`.

2025/2026 diagnostics do not change the champion, features, EV threshold, or hyperparameters. Future champion changes require a forward-only comparison after the freeze date with all of these minimum conditions:

- At least 6 months of forward data.
- At least 1000 races.
- At least 200 `EV >= 1.00` candidates for each strategy.
- Race-paired bootstrap Logloss improvement with 95% CI upper bound below 0.
- Brier is not worse.
- Worst-month behavior is not materially worse.
- Residual p99 is not more than 10% worse.

ROI is auxiliary only. A strategy is not promoted by ROI alone.
