# Final Odds Two Models V1 Design

This is a final-odds ideal-condition model. It uses final current-race odds as model inputs and is not a pre-race live operation model.

- Input: `outputs/model_feature_dataset_v2_1_2/year=YYYY/data.parquet`
- Feature set: `market_aware`
- Targets: `target_win_paid`, `target_place_paid`
- Walk-forward: 2016-2019→2020, ..., 2016-2023→2024
- Test/latest holdout: 2025 / 2026; not used for calibration or rule selection.
- ROI uses actual `tan_pay` and `fuku_pay` with 100 yen flat stakes.
