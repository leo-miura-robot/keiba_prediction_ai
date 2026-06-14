# Model Feature Design V2

V2 is a Phase 1 pre-day dataset. It does not use time-series odds, model training, backtesting, Optuna, or betting strategy optimization.

Outputs are written separately from V1 under `outputs/model_feature_dataset_v2/year=YYYY/data.parquet`.

- rows: 505,881
- races: 36,269
- unfinalized races: 215
- win training rows: 498,926
- place training rows: 498,926
- ranking training rows: 498,926

History snapshot policy:

- `feature_snapshot_mode = pre_day`
- `historical_source_race_date < current_race_date`
- same-day results are not used, even if race number is earlier
- current race rows are fully scored before any result from that date is added to history

Resume stores yearly history state files under `outputs/model_feature_dataset_v2_checkpoint/history_state_after_YYYY.pkl`.