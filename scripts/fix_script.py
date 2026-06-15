import re

with open('scripts/run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update summarize to use probability_calibrated
code = code.replace('h["final_probability"] = h["probability_raw"]', 'h["final_probability"] = h["probability_calibrated"]')
code = code.replace('d = add_eval_columns(g, "probability_raw")', 'd = add_eval_columns(g, "probability_calibrated")')

# 2. Update train_oof to fit calibration
train_oof_fix = '''        print(f"[train] {name} {fold['name']}: {reason}", flush=True)
        model = train_fixed_model(train, n2, c2, params, mp)
        raw, residual, prob = predict_parts(model, valid, n2, c2)
        
        from sklearn.isotonic import IsotonicRegression
        _, _, train_prob = predict_parts(model, train, n2, c2)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(train_prob, train["actual_place"].le(3).astype(int))
        prob_calib = iso.predict(prob)
        
        valid["model_key"] = name
        valid["probability_raw"] = prob
        valid["probability_calibrated"] = prob_calib
        valid["probability_used_for_selection"] = prob
        valid["is_calibrated"] = True
        valid["calibration_method"] = "isotonic"'''

code = re.sub(r'print\(f"\[train\] \{name\}.*?valid\["calibration_method"\] = "none"', train_oof_fix, code, flags=re.DOTALL)

# 3. Update train_final to fit calibration
train_final_fix = '''        print(f"[train_final] {name} {year}: {reason}", flush=True)
        model = train_fixed_model(train, n2, c2, params, mp)
        raw, residual, prob = predict_parts(model, valid, n2, c2)
        
        from sklearn.isotonic import IsotonicRegression
        _, _, train_prob = predict_parts(model, train, n2, c2)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(train_prob, train["actual_place"].le(3).astype(int))
        prob_calib = iso.predict(prob)
        
        valid["model_key"] = name
        valid["probability_raw"] = prob
        valid["probability_calibrated"] = prob_calib
        valid["probability_used_for_selection"] = prob
        valid["is_calibrated"] = True
        valid["calibration_method"] = "isotonic"'''

code = re.sub(r'print\(f"\[train_final\] \{name\}.*?valid\["calibration_method"\] = "none"', train_final_fix, code, flags=re.DOTALL)

# 4. Update manifest
code = code.replace('"is_calibrated": False,', '"is_calibrated": True,')
code = code.replace('"calibration_method": "none",', '"calibration_method": "isotonic",')

# 5. Fix 2025/2026 diagnostic to only test selected model
fix_2025 = '''    base_diag = pd.read_parquet(phase1 / "predictions" / "drop_person_codes" / "final_2025_2026.parquet")
    base_diag["probability_raw"] = base_diag["probability"]
    base_diag["probability_calibrated"] = base_diag["final_probability"]
    
    if selected_key == cfg["base_model_key"]:
        final_2025_2026 = base_diag.copy()
        pvc = pd.read_csv(phase1 / "selected_model_pvc_summary.csv")
        shap = pd.read_csv(phase1 / "selected_model_shap_summary.csv")
        shap_add = pd.read_csv(phase1 / "selected_model_shap_additivity.csv")
    else:
        integ_diag, plogs = train_final(integ_name, promising, 10.0, market_pred, base_cfg, cfg, numeric, cat, params, out, model_root, resume)
        all_param_logs.extend(plogs)
        final_2025_2026 = integ_diag.copy()
        pvc, shap, shap_add = fi_shap(integ_name, promising, 10.0, market_pred, base_cfg, cfg, numeric, cat, out, model_root, int(cfg["shap_sample_per_year"]), int(cfg["random_seed"]))
        
    dm, dr, de, dro = summarize(final_2025_2026, {**base_cfg, "epsilon": 1e-6})
    atomic_write_csv(out / "phase4_2025_2026_diagnostic.csv", dm)
    atomic_write_csv(out / "phase4_2025_2026_ev.csv", de)
    atomic_write_csv(out / "phase4_2025_2026_roi.csv", dro)'''

code = re.sub(r'    base_diag = pd\.read_parquet.*?atomic_write_csv\(out / "phase4_2025_2026_diagnostic\.csv", dm\)', fix_2025, code, flags=re.DOTALL)

with open('scripts/run_place_market_offset_catboost_c1r0_rate_smoothing_phase4_v1.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Script updated successfully.")
