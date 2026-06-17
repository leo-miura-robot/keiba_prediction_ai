# Codex Task: Current C1R0 Visualization Web App MVP v1

## 0. Goal

Build a read-only web application that visualizes the current horse-racing prediction model results.

The first page must show the overall ROI and related summary metrics.

Another page must allow the user to select a date from a calendar, select a race, and compare:

- model-selected horses
- actual place-paid horses
- prediction-time place odds
- final place payout
- prediction probability
- expected value
- tier
- actual result

Use the current saved prediction outputs and Phase 6C paper-trading data.

This task is a goal-oriented task. Do not stop after the first non-destructive approach fails. Investigate the repository, identify available data sources, implement adapters, run tests, and continue until the MVP works or all safe approaches are exhausted.

---

# 1. Important Current Limitation

The current ROLLING_10Y C1R0 model cannot yet generate new predictions directly from raw pre-race CSV because the fitted market baseline parameters were not saved.

Therefore this MVP must be based on already-saved prediction outputs.

Available candidate sources include:

```text
outputs/place_market_offset_champion_challenger_phase5c_v1/predictions/
outputs/place_market_offset_forward_paper_phase6c_v2/
outputs/place_market_offset_forward_paper_phase6c_v2_official_champion_fixture/
```

The application must clearly distinguish:

```text
historical backtest
retrospective validation
fixture
true forward paper trading
```

Do not present fixtures or retrospective rows as real forward performance.

---

# 2. Recommended Technology

Use:

```text
Streamlit
Python
Pandas
Plotly
SQLite read-only access
Parquet read-only access
```

Reason:

- existing project is Python-based
- direct access to Parquet and SQLite
- rapid MVP development
- easy local execution
- later migration to FastAPI/React remains possible

Do not introduce a large frontend framework for this MVP.

---

# 3. Application Structure

Recommended layout:

```text
webapp/
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Race_Calendar.py
│   ├── 3_Analysis.py
│   └── 4_Model_Info.py
├── components/
│   ├── metrics.py
│   ├── charts.py
│   ├── race_table.py
│   └── filters.py
├── data/
│   ├── repository.py
│   ├── phase6c_repository.py
│   ├── parquet_repository.py
│   ├── normalization.py
│   └── schema.py
└── README.md
```

A smaller structure is acceptable if it remains clean and testable.

---

# 4. Data Source Discovery

First inspect the repository and identify the actual available files and schemas.

Search:

```text
prediction_runs
predictions
prediction_tiers
settlements
probability_market
probability_raw
probability_calibrated
expected_value
fuku_odds_low
fuku_odds_high
fuku_pay
target_place_paid
KakuteiJyuni
race_id
entry_id
Umaban
horse_name
KettoNum
JyoCD
RaceNum
race_date
retrospective_only
fixture
odds_snapshot_type
```

Candidate data sources:

```text
Phase 6C SQLite databases
Phase 5C prediction Parquet files
Phase 6A/6B evaluation outputs
race result and payout tables in existing DBs
```

Create a discovery report:

```text
outputs/current_model_webapp_mvp_v1/data_source_inventory.json
```

The report must contain:

```text
source path
source type
row count
date range
available columns
strategy
validation year
fixture flag
retrospective flag
settlement availability
horse-name availability
```

---

# 5. Normalized Internal Schema

Normalize all supported sources into one internal dataframe contract.

Minimum columns:

```text
race_date
race_id
entry_id
JyoCD
RaceNum
Umaban
horse_name
KettoNum

strategy
source_type
validation_year
fixture
retrospective_only
odds_snapshot_type
prediction_created_at
odds_observed_at

probability_market
probability_raw
probability_calibrated
expected_value
tier
selected_for_bet

fuku_odds_low
fuku_odds_high
actual_finish_position
target_place_paid
fuku_pay

stake_yen
payout_yen
profit_yen
```

Rules:

```text
stake_yen = 100 if selected_for_bet else 0
payout_yen = fuku_pay if selected_for_bet and target_place_paid == 1 else 0
profit_yen = payout_yen - stake_yen
```

Confirm the actual payout unit before applying this formula.

If `fuku_pay` is already expressed as payout per 100 yen, use it directly.

Do not infer place-paid status from `finish_position <= 3`.

Official place success definition:

```text
target_place_paid == 1
```

or, when target is absent:

```text
fuku_pay > 0
```

This is required because 5–7 horse races may not pay third place.

---

# 6. Dashboard Page

The first displayed page must be the Dashboard.

Show large KPI cards:

```text
Overall ROI
Total Profit
Total Stake
Total Payout
Number of Bets
Number of Hits
Hit Rate
Number of Races
Date Range
```

ROI:

```text
ROI = total_payout_yen / total_stake_yen * 100
```

If total stake is zero, display `N/A`.

Default filters:

```text
strategy = ROLLING_10Y
exclude fixture = true
include retrospective = true
tier = CORE
```

Provide filters:

```text
date range
source type
strategy
tier
racecourse
odds band
popularity band, if available
retrospective/forward
```

Charts:

1. cumulative profit over time
2. rolling or cumulative ROI over time
3. monthly ROI
4. bet count by month
5. ROI by tier
6. ROI by racecourse
7. ROI by odds band
8. hit rate by expected-value band

Important:

- show bet count with ROI
- avoid presenting ROI without sample size
- identify high-payout dependence where possible
- clearly label fixture and retrospective rows

---

# 7. Race Calendar Page

Provide a calendar/date selector.

Minimum acceptable UI:

```text
month selector
date selector
dates with available race data highlighted or listed
```

If a calendar component is stable and lightweight, use it.

Otherwise use:

```text
st.date_input
+
available-date list
```

After selecting a date, show race cards or a race list:

```text
racecourse
race number
surface/distance if available
number of runners
number of selected horses
number of actual place-paid horses
race stake
race payout
race profit
race ROI
```

Selecting a race opens race details.

---

# 8. Race Detail View

Show a table with all runners in the selected race.

Required columns:

```text
馬番
馬名
予想対象
Tier
市場確率
Raw確率
補正後確率
複勝オッズ下限
複勝オッズ上限
EV
実着順
複勝圏内
複勝払戻
購入額
収支
```

Highlight:

```text
model-selected horses
actual place-paid horses
correct model selections
missed place-paid horses
false-positive selections
```

Use readable labels:

```text
予想馬
実際の複勝圏内馬
的中
不的中
対象外
```

Show race summary:

```text
selected horse count
actual place-paid horse count
hit count
stake
payout
profit
ROI
```

Charts:

1. calibrated probability by horse
2. EV by horse
3. market probability vs raw probability vs calibrated probability
4. selected vs actual place-paid comparison

---

# 9. Prediction and Actual Horse Lists

Above the full runner table, display two simple lists.

Example:

```text
予想した馬
- 3番 馬A: EV 1.12 / 複勝 3.6–4.2
- 7番 馬B: EV 1.04 / 複勝 3.8–4.5

実際の複勝圏内馬
- 2番 馬C: 払戻 180円
- 3番 馬A: 払戻 420円
- 9番 馬D: 払戻 250円
```

Use `target_place_paid` or `fuku_pay > 0`, not simple top-three rank.

---

# 10. Analysis Page

Provide grouped analysis.

Minimum:

```text
year
month
racecourse
distance band
surface
odds band
popularity band
tier
EV band
```

Metrics:

```text
bets
hits
hit rate
stake
payout
profit
ROI
average odds
average EV
```

Allow CSV download of the currently filtered table.

Do not modify source data.

---

# 11. Model Information Page

Display:

```text
Champion strategy
ROLLING_10Y

CatBoost artifact path
models/place_market_offset_champion_challenger_phase5c_v1/
ROLLING_10Y/validation_2026/model.cbm

CatBoost SHA256
4c6f1b9e236391bd84b9d75a14f7ea8ea3fe44761737bb645b8f21d74ed38256

Official Platt artifact path
outputs/place_market_offset_official_calibrators_phase6a_v1/
rolling_10y_platt_phase6a_v1.json

Platt SHA256
ffee1efc19c38f3a76a1efa93488153429e8463bc65f693b331617274e208e98
```

Also show:

```text
current prediction limitation:
BLOCKED_MISSING_MARKET_PARAMETERS

live raw prediction:
not available

saved historical prediction visualization:
available
```

Clearly state that the app currently visualizes saved predictions and does not generate new raw predictions.

---

# 12. Source Labels

Every row and every summary must carry a source classification.

Recommended values:

```text
BACKTEST
RETROSPECTIVE_VALIDATION
FORWARD_PAPER
FIXTURE
UNKNOWN
```

Default dashboard behavior:

```text
exclude FIXTURE
include BACKTEST and RETROSPECTIVE_VALIDATION
include FORWARD_PAPER when available
```

Provide a visible source filter.

Never silently mix fixture rows into ROI.

---

# 13. Horse Name Resolution

If prediction outputs do not contain horse names:

1. inspect existing result/source DBs for `race_id`, `entry_id`, `KettoNum`, `Umaban`
2. create a read-only lookup adapter
3. join horse names without modifying original data

Fallback order:

```text
horse_name
KettoNum-based lookup
race_id + Umaban lookup
display `馬番X / KettoNum`
```

Do not block the entire app because horse names are missing.

Record lookup coverage:

```text
horse_name_resolved_rate
```

---

# 14. Racecourse Name Resolution

Map `JyoCD` to Japanese racecourse names using an existing project mapping if present.

Do not create conflicting mappings.

Fallback:

```text
競馬場コード XX
```

---

# 15. Configuration

Create:

```text
config/current_model_webapp_mvp_v1.yaml
```

Example:

```yaml
app:
  title: "競馬予想AI ダッシュボード"
  default_strategy: "ROLLING_10Y"
  default_tier: "CORE"
  exclude_fixture_by_default: true

data_sources:
  phase6c_roots:
    - outputs/place_market_offset_forward_paper_phase6c_v2
  prediction_parquet_globs:
    - outputs/place_market_offset_champion_challenger_phase5c_v1/predictions/ROLLING_10Y/*.parquet

betting:
  stake_per_bet_yen: 100
  official_place_target_column: target_place_paid
```

Paths must be configurable and not hard-coded throughout the UI.

---

# 16. Read-Only Safety

The application must:

```text
open SQLite in read-only mode
never update predictions
never update settlements
never retrain
never refit
never create bets
never write to source output files
```

The app may write only:

```text
cache
temporary normalized files
webapp-specific logs
```

under a dedicated app output/cache directory.

---

# 17. Performance

Use Streamlit caching:

```python
@st.cache_data
```

for:

```text
Parquet loading
SQLite queries
normalization
aggregations
```

Do not load every large DB table if not needed.

Load only required columns.

---

# 18. Error Handling

Show clear UI warnings for:

```text
no data found
no bets under current filters
missing settlement data
missing horse names
missing payout data
fixture-only data
retrospective-only data
unsupported schema
```

Do not crash the whole app because one optional source is missing.

---

# 19. Tests

Create tests for:

1. ROI calculation
2. zero-stake ROI handling
3. fixed 100-yen stake calculation
4. payout calculation
5. target_place_paid priority
6. 5–7 horse race third-place handling
7. fixture exclusion
8. source-type classification
9. tier filtering
10. date filtering
11. race filtering
12. duplicate-row handling
13. SQLite read-only mode
14. horse-name fallback
15. empty-data handling
16. normalized schema
17. race-level summary
18. dashboard summary

Use small fixtures.

---

# 20. Commands

Install dependencies only if missing.

Suggested:

```powershell
pip install streamlit plotly pyarrow pyyaml
```

Run:

```powershell
streamlit run webapp/app.py
```

Create a wrapper:

```text
scripts/start_current_model_webapp.ps1
```

Usage:

```powershell
.\scripts\start_current_model_webapp.ps1
```

The wrapper should:

```text
verify Python
verify required modules
verify config
start Streamlit
```

Do not silently install packages in the wrapper unless the project convention already permits it.

---

# 21. Documentation

Create:

```text
docs/current_model_webapp_mvp_v1.md
webapp/README.md
```

Document:

```text
what data is visualized
what data is excluded
how ROI is calculated
how place success is determined
how to start the app
how to add future Phase 6C data
current live-prediction limitation
```

---

# 22. Autonomous Work Policy

This is a goal-oriented task.

If one data source does not contain all required columns:

```text
inspect another saved source
implement a source adapter
join by stable keys
continue with a safe fallback
```

Do not stop after the first missing optional field.

Stop only when:

```text
all available saved prediction sources are unusable
ROI cannot be calculated from any valid source
required keys cannot be normalized safely
a destructive operation would be necessary
retraining/refit would be necessary
```

Do not ask the user for intermediate confirmation unless:

```text
a destructive operation is required
source data would be modified
a new training/refit is required
business/UI direction must be chosen between materially different products
```

---

# 23. Acceptance Criteria

The task passes when:

1. `streamlit run webapp/app.py` starts successfully
2. Dashboard shows overall ROI
3. Dashboard shows stake, payout, profit, bets, hits, hit rate
4. Fixture rows are excluded by default
5. Calendar/date selection works
6. Race selection works
7. Predicted horses are shown
8. Actual place-paid horses are shown
9. Place odds are shown
10. Actual place payout is shown
11. Race-level profit and ROI are shown
12. Full runner comparison table is shown
13. At least one saved historical source loads successfully
14. tests pass
15. source files remain unchanged
16. commit/push are not performed

---

# 24. Final Status

Use one of:

```text
CURRENT_MODEL_WEBAPP_MVP_PASSED
PARTIAL_WEBAPP_MVP_MISSING_OPTIONAL_FIELDS
BLOCKED_NO_VALID_PREDICTION_DATA
BLOCKED_NO_SETTLEMENT_OR_PAYOUT_DATA
BLOCKED_SCHEMA_NORMALIZATION
MULTIPLE_BLOCKERS
```

Booleans:

```text
dashboard_working
overall_roi_visible
calendar_working
race_selection_working
predicted_horses_visible
actual_place_horses_visible
odds_visible
payout_visible
fixture_excluded_by_default
read_only_verified
ready_for_local_use
```

---

# 25. Final Report

Report:

1. final status
2. web framework
3. start command
4. loaded data sources
5. date range
6. normalized row count
7. number of races
8. number of bets
9. overall ROI
10. fixture exclusion result
11. calendar result
12. race-detail result
13. horse-name resolution rate
14. available and missing fields
15. read-only verification
16. pytest result
17. created/modified files
18. `git status --short`

Do not commit or push.
