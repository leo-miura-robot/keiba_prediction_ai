import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1 import summarize, load_yaml, load_config
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

cfg = load_config(Path('config/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.yaml'))
base_cfg = load_yaml(Path(cfg['base_c1r0_config']))
out = Path(cfg['output_root'])
phase1 = Path(cfg['phase1_output_root'])

# Load BASE
base_pred = pd.read_parquet(phase1 / 'ablation_oof_predictions.parquet')
base_pred = base_pred[base_pred['model_key'].eq(cfg['base_model_key']) & base_pred['Year'].between(2020, 2024)].copy()
base_pred['probability_raw'] = base_pred['probability']

def calibrate_oof(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    train_data = x[x['Year'].between(2020, 2024)]
    if train_data.empty:
        return x
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(train_data['probability_raw'], train_data['actual_place'].to_numpy(int))
    x['probability_calibrated'] = iso.predict(x['probability_raw'])
    return x

base_pred = calibrate_oof(base_pred)
all_oof = [base_pred]

groups = ['T10', 'J10', 'H10', 'J5', 'J20', 'H5', 'H20', 'C1R0_300_rate_smoothed_phase4_v1']
for name in groups:
    parts = []
    for fold in base_cfg['folds']:
        pp = out / 'predictions' / name / f"{fold['name']}.parquet"
        if pp.exists():
            parts.append(pd.read_parquet(pp))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        # Fix the catastrophic in-fold calibration from train_oof by recalculating properly:
        df = calibrate_oof(df)
        all_oof.append(df)

final_pred = pd.concat(all_oof, ignore_index=True, sort=False)
fm, fr, fe, fro = summarize(final_pred, {**base_cfg, 'epsilon': 1e-6})

m = pd.merge(fro, fe, on=['model_key', 'period', 'Year'])

print('| モデル | 年度 | 確率列 | 状態 | EV>=1件数 | 購入額(円) | 払戻額(円) | ROI(%) | 除外1/3/5/10後ROI(%) | EV-ROI Spearman |')
print('|---|---|---|---|---:|---:|---:|---:|---|---:|')

for name in [cfg['base_model_key']] + groups:
    sub = m[m['model_key'] == name].sort_values('Year')
    for _, r in sub.iterrows():
        b = r['bets']
        cost = b * 100
        ret = int(round(cost * r['roi'] / 100.0)) if not pd.isna(r['roi']) else 0
        roi_val = r['roi'] if not pd.isna(r['roi']) else 0.0
        rem = f"{r['top1_removed_roi']:.1f} / {r['top3_removed_roi']:.1f} / {r['top5_removed_roi']:.1f} / {r['top10_removed_roi']:.1f}"
        print(f"| {name.replace('C1R0_fixed300_ablation_drop_person_codes', 'BASE(drop_person)')} | {r['Year']} | probability_calibrated | calibrated | {b} | {cost} | {ret} | {roi_val:.1f} | {rem} | {r['ev_roi_spearman']:.3f} |")

print("\n--- 2025/2026 ---\n")

base_diag = pd.read_parquet(phase1 / 'predictions' / 'drop_person_codes' / 'final_2025_2026.parquet')
base_diag['probability_raw'] = base_diag['probability']
base_diag['probability_calibrated'] = base_diag['final_probability']
dm, dr, de, dro = summarize(base_diag, {**base_cfg, 'epsilon': 1e-6})
m2 = pd.merge(dro, de, on=['model_key', 'period', 'Year'])

print('| モデル | 年度 | 確率列 | 状態 | EV>=1件数 | 購入額(円) | 払戻額(円) | ROI(%) | 除外1/3/5/10後ROI(%) | EV-ROI Spearman |')
print('|---|---|---|---|---:|---:|---:|---:|---|---:|')

for _, r in m2.sort_values('Year').iterrows():
    b = r['bets']
    cost = b * 100
    ret = int(round(cost * r['roi'] / 100.0))
    roi_val = r['roi'] if not pd.isna(r['roi']) else 0.0
    rem = f"{r['top1_removed_roi']:.1f} / {r['top3_removed_roi']:.1f} / {r['top5_removed_roi']:.1f} / {r['top10_removed_roi']:.1f}"
    print(f"| {r['model_key'].replace('C1R0_fixed300_ablation_drop_person_codes', 'BASE(drop_person)')} | {r['Year']} | probability_calibrated | calibrated | {b} | {cost} | {ret} | {roi_val:.1f} | {rem} | {r['ev_roi_spearman']:.3f} |")
