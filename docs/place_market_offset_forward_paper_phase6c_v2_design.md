# Phase 6C v2 Forward Paper Trading Design

Phase 6C v2 records forward paper predictions before results and appends settlements after results. It does not train CatBoost, refit calibration, change Champion, create ensembles, or place real bets.

Fixed tiers:

- `CORE`: EV >= 1.00
- `MARGIN`: EV >= 1.05
- `HIGH`: EV >= 1.10
- `VERY_HIGH`: EV >= 1.15

SQLite is the source of record. Parquet and CSV exports are report artifacts. Predictions and tier rows are immutable; settlement rows are append-only.
