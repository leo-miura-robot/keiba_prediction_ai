from __future__ import annotations

from pathlib import Path


LEAKAGE_COLUMNS = {
    "NyusenJyuni", "KakuteiJyuni", "Time", "ChakusaCD", "HaronTimeL3",
    "Jyuni1c", "Jyuni2c", "Jyuni3c", "Jyuni4c", "tan_pay", "fuku_pay",
    "is_win_paid", "is_place_paid", "target_win_rank", "target_ren_rank",
    "target_top3_rank", "target_win_paid", "target_place_paid", "target_place_by_rule",
    "race_has_result", "race_has_win_payout", "race_has_place_payout", "race_is_finalized",
    "eligible_for_win_training", "eligible_for_place_training", "eligible_for_ranking_training",
    "win_training_exclusion_reason", "place_training_exclusion_reason", "ranking_training_exclusion_reason",
    "data_split",
}

CURRENT_MARKET_COLUMNS = {
    "tan_odds", "tan_ninki", "fuku_odds_low", "fuku_odds_high", "fuku_ninki",
    "TanVote", "FukuVote", "win_odds_available", "place_odds_available",
    "win_votes_available", "place_votes_available",
}

RAW_MARKET_COLUMNS = {"TanOdds", "TanNinki", "FukuOddsLow", "FukuOddsHigh", "FukuNinki"}

MARKET_HISTORY_COLUMNS = {
    "horse_last3_avg_ninki", "horse_last5_avg_ninki",
    "horse_last3_avg_tan_odds", "horse_last5_avg_tan_odds",
}

BASE_NUMERIC = [
    "Year", "MonthDay", "Kaiji", "Nichiji", "RaceNum", "Wakuban", "Umaban", "Barei", "Futan",
    "BaTaijyu", "ZogenSa", "Kyori", "TorokuTosu", "SyussoTosu", "place_rank_limit",
    "horse_days_since_last", "horse_past_starts", "horse_last1_avg_finish",
    "horse_last3_avg_finish", "horse_last5_avg_finish", "horse_last3_win_rate",
    "horse_last5_win_rate", "horse_last3_ren_rate", "horse_last5_ren_rate",
    "horse_last3_top3_rate", "horse_last5_top3_rate", "horse_last3_place_paid_rate",
    "horse_last5_place_paid_rate", "horse_last3_avg_haron_l3", "horse_last5_avg_haron_l3",
    "horse_last3_avg_time", "horse_last5_avg_time", "horse_distance_diff_last",
    "horse_futan_diff_last", "horse_body_weight_diff_last", "horse_jyo_past_starts",
    "horse_jyo_win_rate", "horse_jyo_top3_rate", "horse_surface_past_starts",
    "horse_surface_win_rate", "horse_surface_top3_rate", "horse_dist_band_past_starts",
    "horse_dist_band_win_rate", "horse_dist_band_top3_rate", "horse_baba_past_starts",
    "horse_baba_win_rate", "horse_baba_top3_rate", "jockey_past_starts",
    "jockey_win_rate", "jockey_ren_rate", "jockey_top3_rate", "trainer_past_starts",
    "trainer_win_rate", "trainer_ren_rate", "trainer_top3_rate",
    "jockey_jyo_past_starts", "jockey_jyo_win_rate", "jockey_jyo_top3_rate",
    "jockey_dist_band_past_starts", "jockey_dist_band_win_rate", "jockey_dist_band_top3_rate",
    "horse_jockey_past_starts", "horse_jockey_win_rate", "horse_jockey_top3_rate",
]

BASE_CATEGORICAL = [
    "JyoCD", "YoubiCD", "GradeCD", "SyubetuCD", "JyokenCD1", "JyokenCD2", "JyokenCD3",
    "JyokenCD4", "JyokenCD5", "TrackCD", "CourseKubunCD", "TenkoCD", "SibaBabaCD",
    "DirtBabaCD", "SexCD", "ChokyosiCode", "KisyuCode", "ZogenFugo",
    "place_bet_available_by_rule",
]

MARKET_HISTORY_NUMERIC = ["horse_last3_avg_ninki", "horse_last5_avg_ninki"]
CURRENT_MARKET_NUMERIC = ["tan_odds", "tan_ninki", "fuku_odds_low", "fuku_odds_high", "fuku_ninki", "TanVote", "FukuVote"]
CURRENT_MARKET_CATEGORICAL = ["win_odds_available", "place_odds_available", "win_votes_available", "place_votes_available"]


def feature_sets() -> dict[str, dict[str, list[str]]]:
    market_history_numeric = BASE_NUMERIC + MARKET_HISTORY_NUMERIC
    market_history_categorical = BASE_CATEGORICAL.copy()
    return {
        "market_free": {"numeric": BASE_NUMERIC, "categorical": BASE_CATEGORICAL},
        "market_history": {"numeric": market_history_numeric, "categorical": market_history_categorical},
        "market_aware": {
            "numeric": market_history_numeric + CURRENT_MARKET_NUMERIC,
            "categorical": market_history_categorical + CURRENT_MARKET_CATEGORICAL,
        },
    }


def validate_feature_sets() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sets = feature_sets()
    for name, groups in sets.items():
        columns = groups["numeric"] + groups["categorical"]
        duplicated = sorted({c for c in columns if columns.count(c) > 1})
        leaked = sorted(set(columns) & LEAKAGE_COLUMNS)
        raw_market = sorted(set(columns) & RAW_MARKET_COLUMNS)
        if duplicated:
            rows.append({"check_name": f"{name}_duplicates", "status": "fail", "details": ",".join(duplicated)})
        if leaked:
            rows.append({"check_name": f"{name}_leakage", "status": "fail", "details": ",".join(leaked)})
        if raw_market:
            rows.append({"check_name": f"{name}_raw_market", "status": "fail", "details": ",".join(raw_market)})
    free_cols = set(sets["market_free"]["numeric"] + sets["market_free"]["categorical"])
    hist_cols = set(sets["market_history"]["numeric"] + sets["market_history"]["categorical"])
    aware_cols = set(sets["market_aware"]["numeric"] + sets["market_aware"]["categorical"])
    free_market = sorted(free_cols & (CURRENT_MARKET_COLUMNS | MARKET_HISTORY_COLUMNS | RAW_MARKET_COLUMNS))
    hist_current = sorted(hist_cols & (CURRENT_MARKET_COLUMNS | RAW_MARKET_COLUMNS))
    if free_market:
        rows.append({"check_name": "market_free_no_market", "status": "fail", "details": ",".join(free_market)})
    if hist_current:
        rows.append({"check_name": "market_history_no_current_market", "status": "fail", "details": ",".join(hist_current)})
    if not hist_cols <= aware_cols:
        rows.append({"check_name": "market_aware_contains_market_history", "status": "fail", "details": ",".join(sorted(hist_cols - aware_cols))})
    if not rows:
        rows.append({"check_name": "all_feature_set_checks", "status": "pass", "details": ""})
    return rows


def write_feature_set_yaml(path: Path) -> None:
    lines: list[str] = []
    for set_name, groups in feature_sets().items():
        lines.append(f"{set_name}:")
        for group_name, columns in groups.items():
            lines.append(f"  {group_name}:")
            for column in columns:
                lines.append(f"    - {column}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
