import pandas as pd
from pathlib import Path

out = Path('outputs/place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1')

roi = pd.read_csv(out / 'rate_smoothing_roi_diagnostic.csv')
ev = pd.read_csv(out / 'rate_smoothing_ev_stability.csv')

m = pd.merge(roi, ev, on=['model_key', 'period', 'Year'])

print('| モデル | 年度 | 確率列 | 状態 | EV>=1件数 | 購入額(円) | 払戻額(円) | ROI(%) | 除外1/3/5/10後ROI(%) | EV-ROI Spearman |')
print('|---|---|---|---|---:|---:|---:|---:|---|---:|')

for name in ['C1R0_fixed300_ablation_drop_person_codes', 'T10', 'J10', 'H10', 'J5', 'J20', 'H5', 'H20', 'C1R0_300_rate_smoothed_phase4_v1']:
    sub = m[m['model_key'] == name].sort_values('Year')
    for _, r in sub.iterrows():
        b = r['bets']
        cost = b * 100
        ret = int(round(cost * r['roi'] / 100.0))
        rem = f"{r['top1_removed_roi']:.1f} / {r['top3_removed_roi']:.1f} / {r['top5_removed_roi']:.1f} / {r['top10_removed_roi']:.1f}"
        print(f"| {name.replace('C1R0_fixed300_ablation_drop_person_codes', 'BASE(drop_person)')} | {r['Year']} | probability_calibrated | calibrated | {b} | {cost} | {ret} | {r['roi']:.1f} | {rem} | {r['ev_roi_spearman']:.3f} |")
