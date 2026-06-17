# Latest Model Validation jrvltsql DB Audit v1

Target DB:

`C:\Users\leole\jrvltsql\data\quickstart_20260608_20260617_20260617_100814\keiba.db`

Validation dates:

- `2026-06-13`
- `2026-06-14`

Audit output:

`outputs/latest_model_validation_on_jrvltsql_20260608_audit_v1/`

Final assessment:

`CALIBRATION_ISSUE`

The validation did not use the short DB alone for history features. It combined the existing long-term history source from 2006 with the new short DB rows. The pre-day history cutoff checks passed for both validation dates.

The blocking audit issue is that the existing validation script reconstructed/refit calibrators from OOF predictions instead of loading immutable calibrator artifacts. CatBoost was not retrained, feature schema parity passed, market inputs were present, and target logic passed.

Use:

- Model limit judgement: `false`
- Probability diagnostic: `true`
- ROI judgement: `false`

