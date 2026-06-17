# Phase 6C Raw To Official Champion Bridge v1

Existing prepare script contract:

- `scripts/prepare_place_forward_predictions_phase6c_v2.py` is `MODEL_READY_INPUT_ONLY`.
- It requires `market_logit` and the official 79 feature allowlist columns.
- It does not convert raw jrvltsql pre-race rows into model-ready features.

Raw wrapper:

```powershell
.\scripts\run_phase6c_raw_to_official_champion.ps1 `
  -RaceDate 2026-06-20 `
  -RawPreRaceCsv "inputs\forward\pre_race_20260620.csv" `
  -OutputRoot "outputs\place_market_offset_forward_paper_phase6c_v2"
```

Current status:

- The raw input and D-1 history cutoff audits are implemented.
- The official 79-feature allowlist contract is audited.
- The CatBoost and official Platt artifacts are verified by hash.
- The pipeline fails closed with `BLOCKED_MARKET_ARTIFACT` because the training-time market baseline LogisticRegression/scaler artifact is not persisted.

No market model is refit or guessed. No raw probability fallback is used.
