# Current Model Webapp MVP v1

Read-only Streamlit dashboard for saved `ROLLING_10Y` C1R0 prediction outputs.

## Start

```powershell
.\scripts\start_current_model_webapp.ps1
```

or:

```powershell
streamlit run webapp\app.py
```

## Data

The app reads saved Parquet predictions and Phase 6C SQLite files in read-only mode. Fixture rows are excluded by default. It does not train models, refit calibrators, overwrite predictions, update settlements, or place bets.

Place success is based on `target_place_paid == 1`. If that column is absent, the normalization fallback is `fuku_pay > 0`; it never uses `finish_position <= 3` as the primary rule.

ROI is `total payout / total stake * 100`, with a fixed 100 yen stake per selected row.

Current live raw prediction limitation: `BLOCKED_MISSING_MARKET_PARAMETERS`.
