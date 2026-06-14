# Feature Set Design

`config/feature_sets.yaml` is generated from an allow-list. Leakage columns, target columns, payout columns, split columns, and raw result columns are not included.

- market_free numeric: 66
- market_free categorical: 19
- market_aware numeric: 73
- market_aware categorical: 23

Phase 1 does not include time-series odds features. `market_aware` only adds current normalized market columns and market availability flags.