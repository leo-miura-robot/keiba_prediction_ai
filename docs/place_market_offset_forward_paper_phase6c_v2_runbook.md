# Phase 6C v2 Forward Paper Trading Runbook

PowerShell examples:

```powershell
python scripts\run_place_market_offset_forward_paper_phase6c_v2.py predict `
  --race-date 2026-06-20 `
  --input-csv path\to\pre_race_predictions.csv `
  --output-root outputs\place_market_offset_forward_paper_phase6c_v2
```

```powershell
python scripts\run_place_market_offset_forward_paper_phase6c_v2.py settle `
  --race-date 2026-06-20 `
  --settlement-csv path\to\results.csv `
  --output-root outputs\place_market_offset_forward_paper_phase6c_v2
```

```powershell
python scripts\run_place_market_offset_forward_paper_phase6c_v2.py report `
  --output-root outputs\place_market_offset_forward_paper_phase6c_v2
```

Use `--fixture` only for smoke tests. Fixture runs are excluded from forward production reports.
