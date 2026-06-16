# Phase 5B Year Strategy Runbook

This runbook is for local execution. Codex implemented the runner and tests only; long CatBoost training should be started from the local terminal.

## Preconditions

- Do not overwrite existing artifacts unless you intentionally delete or archive the Phase 5B output/model directories first.
- Do not use calibrated outputs for Phase 5B selection.
- Use `probability_raw` only.
- Stop if `LEGACY_2016` parity fails.

## 1. LEGACY_2016 2024 Parity Smoke

This is the gate before broad strategy comparison.

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016 `
  --years 2024 `
  --parity-check `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_parity_2024 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_parity_2024
```

Check the parity file:

```powershell
Import-Csv outputs\place_market_offset_year_strategy_phase5b_v2_parity_2024\legacy_parity_check.csv | Format-List
```

If any `passed` value is not `True`, stop and inspect the row-level differences. Do not relax tolerance.

## 2. 2024 All-Strategy Smoke

Use this only after parity passes. This command exercises all seven strategies on one validation year.

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2024 `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2_smoke_2024 `
  --model-root models\place_market_offset_year_strategy_phase5b_v2_smoke_2024
```

## 3. 2020-2024 Full Run

Run the full primary comparison after the parity smoke succeeds.

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2020,2021,2022,2023,2024 `
  --parity-check
```

## 4. Resume

Resume reuses only fold artifacts whose manifest signatures match.

```powershell
python scripts\run_place_market_offset_year_strategy_phase5b_v2.py `
  --config config\place_market_offset_year_strategy_phase5b_v2.yaml `
  --strategies LEGACY_2016,WARMUP_2006_TRAIN_2016,EXPANDING_FULL_2006,ROLLING_10Y,ROLLING_15Y,FULL_2006_TIME_DECAY_HL5,FULL_2006_TIME_DECAY_HL10 `
  --years 2020,2021,2022,2023,2024 `
  --parity-check `
  --resume
```

## 5. Artifact Audit

```powershell
python scripts\audit_place_market_offset_year_strategy_phase5b_v2.py `
  --output-root outputs\place_market_offset_year_strategy_phase5b_v2
```

## 6. Artifact Checks

```powershell
Get-ChildItem outputs\place_market_offset_year_strategy_phase5b_v2
Get-ChildItem models\place_market_offset_year_strategy_phase5b_v2 -Recurse -Filter model.cbm
Import-Csv outputs\place_market_offset_year_strategy_phase5b_v2\metrics_by_strategy_2020_2024.csv | Format-Table
Import-Csv outputs\place_market_offset_year_strategy_phase5b_v2\worst_year_summary.csv | Format-Table
Import-Csv outputs\place_market_offset_year_strategy_phase5b_v2\paired_bootstrap_summary.csv | Format-Table
Import-Csv outputs\place_market_offset_year_strategy_phase5b_v2\roi_diagnostic_raw.csv | Format-Table
```

## Output Directories

- `outputs/place_market_offset_year_strategy_phase5b_v2/`
- `models/place_market_offset_year_strategy_phase5b_v2/`
