# CatBoost Baseline V1.0.2 Design

V1.0.2 keeps V1, V1.0.1, the existing six model weights, and the V2.1.1 feature dataset intact. It adds integrity fixes for analysis and model reuse before probability calibration.

Key changes:

- Market comparison uses only complete races where every win-eligible runner has valid odds and all three CatBoost predictions.
- Reused model weights are always applied to the current V2.1.1 dataset to regenerate full predictions.
- Split definition is resolved from `config/catboost_baseline_v1_0_2.yaml`.
- Quantile calibration uses tie-preserving bins and may drop duplicate bin edges.
- Analysis CSVs are fully regenerated and atomically replaced.
- Scripts write machine summaries under `outputs/model_training/catboost_baseline_v1_0_2/` and do not overwrite this document.

Phase 1 future ROI goal remains win ROI >= 0.90 and place ROI >= 0.90. ROI is not computed in V1.0.2. A future ROI pass must not treat one long-shot or one high payout as sufficient evidence of success.
