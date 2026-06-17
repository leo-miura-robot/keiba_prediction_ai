# Current Model Webapp MVP v1

## Scope

This MVP visualizes already-saved prediction outputs for the current `ROLLING_10Y` C1R0 model. It does not generate new raw predictions because the official market baseline parameters are still blocked by `BLOCKED_MISSING_MARKET_PARAMETERS`.

## Read-only behavior

- Parquet files are read only.
- SQLite is opened with `mode=ro`.
- No model training, calibrator refit, source overwrite, settlement overwrite, commit, push, or real betting is performed.

## Place target and ROI

Place-paid status uses `target_place_paid == 1` when present. Only if that column is absent does the app use `fuku_pay > 0`. It does not classify 5-7 runner race third place by `finish_position <= 3`.

ROI is:

```text
total payout / total stake * 100
```

with 100 yen per selected row.

## Data sources

Configured in `config/current_model_webapp_mvp_v1.yaml`.

The inventory report is written to:

```text
outputs/current_model_webapp_mvp_v1/data_source_inventory.json
```

Fixture data is labeled `FIXTURE` and excluded by default.

## Start

```powershell
.\scripts\start_current_model_webapp.ps1
```

or:

```powershell
streamlit run webapp\app.py
```
