from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_place_market_offset_catboost_v1 import cat_indices, prepare_x


FOLDS = [
    ("fold_2020", 2020, [2016, 2017, 2018, 2019]),
    ("fold_2021", 2021, [2016, 2017, 2018, 2019, 2020]),
    ("fold_2022", 2022, [2016, 2017, 2018, 2019, 2020, 2021]),
    ("fold_2023", 2023, [2016, 2017, 2018, 2019, 2020, 2021, 2022]),
    ("fold_2024", 2024, [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]),
]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, default=str))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    data = tmp.read_bytes()
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_info() -> dict[str, Any]:
    try:
        return {
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
            "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True).splitlines(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.any():
            total += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / len(y))
    return float(total)


def ev_spearman(df: pd.DataFrame, p_col: str = "final_probability") -> float:
    d = df.copy()
    d["ev"] = d[p_col] * pd.to_numeric(d["fuku_odds_low"], errors="coerce")
    bins = [-np.inf, .85, .90, .95, 1.00, 1.02, 1.05, 1.10, np.inf]
    d["ev_band_order"] = pd.cut(d["ev"], bins, labels=False, right=False)
    rows = []
    for k, g in d.groupby("ev_band_order", dropna=True):
        if len(g):
            rows.append((int(k), float(pd.to_numeric(g["fuku_pay"], errors="coerce").fillna(0).sum() / (len(g) * 100) * 100)))
    if len(rows) < 2:
        return np.nan
    arr = np.array(rows, dtype=float)
    return float(spearmanr(arr[:, 0], arr[:, 1], nan_policy="omit").statistic)


def load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def group_for_feature(feature: str) -> str:
    if feature in {"p_market", "market_logit"}:
        return "market_baseline"
    if feature.startswith("horse_last") or feature in {"horse_days_since_last", "horse_distance_diff_last", "horse_futan_diff_last", "horse_body_weight_diff_last", "horse_past_starts"}:
        return "horse_recent_form"
    if feature.startswith("horse_jyo_") or feature.startswith("horse_surface_") or feature.startswith("horse_dist_band_") or feature.startswith("horse_baba_"):
        return "horse_course_suitability"
    if feature.startswith("jockey_jyo_") or feature.startswith("jockey_dist_band_"):
        return "jockey_course_suitability"
    if feature.startswith("jockey_"):
        return "jockey_overall"
    if feature.startswith("trainer_"):
        return "trainer"
    if feature.startswith("horse_jockey_"):
        return "horse_jockey_pair"
    if feature in {"JyoCD"}:
        return "venue_identity"
    if feature in {"TrackCD", "CourseKubunCD", "SibaBabaCD", "DirtBabaCD", "TenkoCD"}:
        return "course_context"
    if feature in {"Kyori"}:
        return "distance"
    if feature in {"Wakuban", "Umaban", "Futan", "BaTaijyu", "ZogenSa", "ZogenFugo", "Barei", "SexCD"}:
        return "weight_and_gate"
    if feature in {"Year", "MonthDay", "Kaiji", "Nichiji", "RaceNum", "TorokuTosu", "SyussoTosu", "place_rank_limit", "YoubiCD", "GradeCD", "SyubetuCD"} or feature.startswith("Jyoken"):
        return "race_metadata"
    if any(s in feature.lower() for s in ["sire", "dam", "pedigree", "blood", "lineage", "ketto", "hansyoku"]):
        return "pedigree"
    return "other"


def pool_for(df: pd.DataFrame, features: list[str], cat: list[str], baseline_col: str = "market_logit") -> Pool:
    x = prepare_x(df, [c for c in features if c not in cat], [c for c in features if c in cat])
    x = x[features]
    cats = [x.columns.get_loc(c) for c in features if c in cat]
    baseline = df[baseline_col].to_numpy(float) if baseline_col in df.columns else None
    return Pool(x, df["actual_place"].to_numpy(int) if "actual_place" in df else None, cat_features=cats, baseline=baseline)


def predict_final(model: CatBoostClassifier, df: pd.DataFrame, features: list[str], cat: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict(pool_for(df, features, cat), prediction_type="RawFormulaVal"), dtype=float)
    return np.clip(expit(raw), 1e-6, 1 - 1e-6)


def residual_raw(model: CatBoostClassifier, df: pd.DataFrame, features: list[str], cat: list[str]) -> np.ndarray:
    x = prepare_x(df, [c for c in features if c not in cat], [c for c in features if c in cat])
    x = x[features]
    cats = [x.columns.get_loc(c) for c in features if c in cat]
    return np.asarray(model.predict(Pool(x, cat_features=cats), prediction_type="RawFormulaVal"), dtype=float)


def artifact_inventory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows.append({
            "path": str(p),
            "exists": p.exists(),
            "kind": "directory" if p.exists() and p.is_dir() else "file",
            "size_bytes": p.stat().st_size if p.exists() and p.is_file() else np.nan,
            "mtime": pd.Timestamp.fromtimestamp(p.stat().st_mtime).isoformat() if p.exists() else "",
            "sha256": sha256_file(p) if p.exists() and p.is_file() else "",
        })
    return pd.DataFrame(rows)


def feature_schema_inventory(feature_df: pd.DataFrame, feature_inventory: pd.DataFrame, features: list[str], cat: list[str], model_features: list[str]) -> pd.DataFrame:
    inventory_names = set(feature_inventory["column_name"].astype(str)) if "column_name" in feature_inventory.columns else set(feature_inventory.iloc[:, 0].astype(str))
    model_set = set(model_features)
    rows = []
    all_cols = sorted(set(feature_df.columns) | set(features) | model_set)
    for c in all_cols:
        s = feature_df[c] if c in feature_df.columns else pd.Series(dtype=float)
        non_null_years = sorted(feature_df.loc[s.notna(), "Year"].unique()) if c in feature_df.columns and "Year" in feature_df.columns else []
        rows.append({
            "column_name": c,
            "dtype": str(s.dtype) if c in feature_df.columns else "",
            "exists_in_parquet": c in feature_df.columns,
            "exists_in_feature_inventory": c in inventory_names,
            "included_in_c1_config": c in features,
            "included_in_saved_model": c in model_set,
            "numeric_or_categorical": "categorical" if c in cat else ("numeric" if c in features else ""),
            "null_rate_2020_2024": float(s.isna().mean()) if c in feature_df.columns else np.nan,
            "unique_count_2020_2024": int(s.nunique(dropna=True)) if c in feature_df.columns else 0,
            "first_year_non_null": int(min(non_null_years)) if non_null_years else np.nan,
            "last_year_non_null": int(max(non_null_years)) if non_null_years else np.nan,
            "proposed_group": group_for_feature(c),
            "notes": "",
        })
    return pd.DataFrame(rows)


def course_structure_audit(columns: list[str], features: list[str]) -> pd.DataFrame:
    concepts = ["turn_direction", "right_turn", "left_turn", "inner", "outer", "inner_outer", "straight", "straight_length", "elevation", "height_difference", "slope", "gradient", "final_slope", "steep_slope", "corner_count", "corner_radius", "first_corner_distance", "course_width", "small_turn", "small_course", "course_id"]
    rows = []
    lower_cols = {c.lower(): c for c in columns}
    for concept in concepts:
        matches = [orig for low, orig in lower_cols.items() if concept.replace("_", "") in low.replace("_", "")]
        if matches:
            status = "explicit" if any(m in features for m in matches) else "not_in_model"
        elif concept in {"inner", "outer", "inner_outer"} and "CourseKubunCD" in columns:
            status = "ambiguous"
            matches = ["CourseKubunCD"]
        elif concept in {"course_id"} and {"JyoCD", "TrackCD", "Kyori"}.issubset(set(columns)):
            status = "encoded"
            matches = ["JyoCD", "TrackCD", "Kyori"]
        elif concept in {"small_turn", "small_course", "turn_direction"} and "JyoCD" in columns:
            status = "indirect_only"
            matches = ["JyoCD"]
        else:
            status = "absent"
        rows.append({"concept": concept, "status": status, "matched_columns": ",".join(matches), "included_in_c1": any(m in features for m in matches), "notes": ""})
    return pd.DataFrame(rows)


def pedigree_audit(columns: list[str], features: list[str], code_files: list[Path]) -> pd.DataFrame:
    terms = ["sire", "dam", "damsire", "broodmare_sire", "father", "mother", "pedigree", "bloodline", "lineage", "Ketto", "Hansyoku", "Bamei", "父", "母", "血統", "種牡馬"]
    code_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in code_files if p.exists())
    rows = []
    for term in terms:
        cols = [c for c in columns if term.lower() in c.lower()]
        feat = [c for c in features if term.lower() in c.lower()]
        if feat:
            status = "implemented"
        elif cols:
            status = "not_in_model" if term not in {"Bamei", "Ketto"} else "raw_only"
        elif term.lower() in code_text.lower():
            status = "unknown"
        else:
            status = "absent"
        rows.append({"term": term, "status": status, "matched_columns": ",".join(cols), "included_features": ",".join(feat), "notes": "KettoNum/Bamei are identity/raw fields, not pedigree-performance features." if term in {"Ketto", "Bamei"} else ""})
    return pd.DataFrame(rows)


def importance_tables(models: dict[str, CatBoostClassifier], val: pd.DataFrame, features: list[str], cat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pvc_rows, lfc_rows = [], []
    for fold, model in models.items():
        year = int(fold.split("_")[1])
        d = val[(val["fold"] == fold) & (val["model_key"] == "C1_market_offset_fundamental")].copy()
        pvc = model.get_feature_importance(type="PredictionValuesChange")
        for f, v in zip(features, pvc):
            pvc_rows.append({"fold": fold, "year": year, "feature": f, "group": group_for_feature(f), "importance": float(v)})
        pool = pool_for(d, features, cat)
        lfc = model.get_feature_importance(data=pool, type="LossFunctionChange")
        for f, v in zip(features, lfc):
            lfc_rows.append({"fold": fold, "year": year, "feature": f, "group": group_for_feature(f), "importance": float(v)})
    return pd.DataFrame(pvc_rows), pd.DataFrame(lfc_rows)


def summarize_importance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["feature", "group"], as_index=False).agg(
        weighted_mean=("importance", "mean"),
        unweighted_mean=("importance", "mean"),
        median=("importance", "median"),
        min=("importance", "min"),
        max=("importance", "max"),
        std=("importance", "std"),
        fold_count=("fold", "nunique"),
    )
    out["rank_weighted_mean"] = out["weighted_mean"].rank(ascending=False, method="min").astype(int)
    out["rank_median"] = out["median"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank_weighted_mean")


def sample_by_year(df: pd.DataFrame, per_year: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("Year", sort=True):
        parts.append(g.sample(min(len(g), per_year), random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()


def shap_tables(models: dict[str, CatBoostClassifier], val: pd.DataFrame, features: list[str], cat: list[str], sample_per_year: int, seed: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    samples = sample_by_year(val[val["model_key"] == "C1_market_offset_fundamental"], sample_per_year, seed)
    rows = []
    add_rows = []
    for fold, model in models.items():
        d = samples[samples["fold"] == fold].copy()
        if d.empty:
            continue
        shap = np.asarray(model.get_feature_importance(data=pool_for(d, features, cat), type="ShapValues"))
        vals = shap[:, :-1]
        expected = shap[:, -1]
        rr = residual_raw(model, d, features, cat)
        shap_sum = expected + vals.sum(axis=1)
        add_rows.append({
            "fold": fold,
            "rows": len(d),
            "mean_abs_error_residual_raw": float(np.abs(shap_sum - rr).mean()),
            "max_abs_error_residual_raw": float(np.abs(shap_sum - rr).max()),
            "p999_abs_error_residual_raw": float(np.quantile(np.abs(shap_sum - rr), 0.999)),
            "mean_abs_error_final_logit": float(np.abs((d["market_logit"].to_numpy(float) + shap_sum) - d["final_probability_raw"].to_numpy(float)).mean()) if "final_probability_raw" in d else np.nan,
        })
        for i, f in enumerate(features):
            s = vals[:, i]
            sd = d[["Year", "JyoCD", "TrackCD", "Kyori"]].copy()
            sd["_shap"] = s
            base = {"feature": f, "group": group_for_feature(f), "fold": fold}
            rows.append({**base, "scope": "global", "scope_value": "2020_2024", **shap_stats(s)})
            for y, g in sd.groupby("Year"):
                rows.append({**base, "scope": "year", "scope_value": str(int(y)), **shap_stats(g["_shap"].to_numpy(float))})
            for jyo, g in sd.groupby("JyoCD"):
                rows.append({**base, "scope": "jyo", "scope_value": str(jyo), **shap_stats(g["_shap"].to_numpy(float))})
            naka = sd["JyoCD"].astype(str).str.zfill(2).eq("06")
            if naka.any():
                rows.append({**base, "scope": "nakayama", "scope_value": "all", **shap_stats(sd.loc[naka, "_shap"].to_numpy(float))})
                turf = naka & sd["TrackCD"].astype(str).isin(["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"])
                dirt = naka & sd["TrackCD"].astype(str).isin(["23", "24", "25", "26", "27", "28", "29"])
                if turf.any():
                    rows.append({**base, "scope": "nakayama_turf", "scope_value": "turf", **shap_stats(sd.loc[turf, "_shap"].to_numpy(float))})
                if dirt.any():
                    rows.append({**base, "scope": "nakayama_dirt", "scope_value": "dirt", **shap_stats(sd.loc[dirt, "_shap"].to_numpy(float))})
                for kyori, g in sd[naka].groupby("Kyori"):
                    rows.append({**base, "scope": "nakayama_distance", "scope_value": str(kyori), **shap_stats(g["_shap"].to_numpy(float))})
    all_rows = pd.DataFrame(rows)
    tables = {
        "shap_global_2020_2024.csv": all_rows[all_rows["scope"] == "global"].groupby(["feature", "group"], as_index=False).agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), median_abs_shap=("median_abs_shap", "mean"), p90_abs_shap=("p90_abs_shap", "mean"), p99_abs_shap=("p99_abs_shap", "mean"), positive_share=("positive_share", "mean"), sample_rows=("sample_rows", "sum")).sort_values("mean_abs_shap", ascending=False),
        "shap_by_year.csv": all_rows[all_rows["scope"] == "year"],
        "shap_by_jyo.csv": all_rows[all_rows["scope"] == "jyo"],
        "shap_nakayama.csv": all_rows[all_rows["scope"] == "nakayama"],
        "shap_nakayama_turf.csv": all_rows[all_rows["scope"] == "nakayama_turf"],
        "shap_nakayama_dirt.csv": all_rows[all_rows["scope"] == "nakayama_dirt"],
        "shap_nakayama_by_distance.csv": all_rows[all_rows["scope"] == "nakayama_distance"],
    }
    return tables, pd.DataFrame(add_rows)


def shap_stats(s: np.ndarray) -> dict[str, Any]:
    s = np.asarray(s, dtype=float)
    return {
        "mean_abs_shap": float(np.abs(s).mean()),
        "mean_signed_shap": float(s.mean()),
        "median_abs_shap": float(np.median(np.abs(s))),
        "p90_abs_shap": float(np.quantile(np.abs(s), 0.90)),
        "p99_abs_shap": float(np.quantile(np.abs(s), 0.99)),
        "positive_share": float((s > 0).mean()),
        "sample_rows": int(len(s)),
    }


def group_map(features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"feature": f, "group": group_for_feature(f)} for f in features])


def permute_group(df: pd.DataFrame, group: str, cols: list[str], seed: int) -> pd.DataFrame:
    d = df.copy()
    rng = np.random.default_rng(seed)
    if group == "market_baseline":
        idx = rng.permutation(len(d))
        d["market_logit"] = d["market_logit"].to_numpy()[idx]
        return d
    if not cols:
        return d
    race_level_cols = {"JyoCD", "Kyori", "TrackCD", "CourseKubunCD", "SibaBabaCD", "DirtBabaCD", "TenkoCD", "Year", "MonthDay", "Kaiji", "Nichiji", "RaceNum", "SyussoTosu", "place_rank_limit"}
    if any(c in race_level_cols for c in cols):
        races = d["race_id"].drop_duplicates().to_numpy()
        shuffled = races.copy()
        rng.shuffle(shuffled)
        mapper = dict(zip(races, shuffled))
        src = d.set_index("race_id")
        for c in cols:
            vals = d["race_id"].map(mapper).map(src[~src.index.duplicated(keep="first")][c])
            d[c] = vals.to_numpy()
    else:
        idx = rng.permutation(len(d))
        for c in cols:
            d[c] = d[c].to_numpy()[idx]
    return d


def permutation_importance(models: dict[str, CatBoostClassifier], val: pd.DataFrame, features: list[str], cat: list[str], sample_n: int, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fmap = group_map(features)
    groups = {g: sorted(fmap.loc[fmap["group"] == g, "feature"]) for g in sorted(fmap["group"].unique())}
    rows = []
    for fold, model in models.items():
        d = val[(val["fold"] == fold) & (val["model_key"] == "C1_market_offset_fundamental")].copy()
        d = d.sample(min(len(d), sample_n), random_state=seed)
        base_p = predict_final(model, d, features, cat)
        y = d["actual_place"].to_numpy(int)
        base = metric_values(y, base_p)
        base_sp = ev_spearman(d.assign(final_probability=base_p))
        base_ev_count = int((base_p * d["fuku_odds_low"].to_numpy(float) >= 1.0).sum())
        for group, cols in groups.items():
            for r in range(repeats):
                pdg = permute_group(d, group, cols, seed + r + int(fold[-4:]))
                p = predict_final(model, pdg, features, cat)
                m = metric_values(y, p)
                rows.append({
                    "fold": fold,
                    "year": int(fold[-4:]),
                    "repeat": r,
                    "group": group,
                    "features": ",".join(cols),
                    "feature_count": len(cols),
                    "logloss_delta": m["logloss"] - base["logloss"],
                    "brier_delta": m["brier"] - base["brier"],
                    "ece_delta": m["ece"] - base["ece"],
                    "ev_roi_spearman_delta": ev_spearman(d.assign(final_probability=p)) - base_sp,
                    "mean_abs_p_change": float(np.abs(p - base_p).mean()),
                    "ev_ge_1_count_delta": int((p * d["fuku_odds_low"].to_numpy(float) >= 1.0).sum() - base_ev_count),
                })
    by = pd.DataFrame(rows)
    summary = by.groupby("group", as_index=False).agg(logloss_delta_mean=("logloss_delta", "mean"), brier_delta_mean=("brier_delta", "mean"), ece_delta_mean=("ece_delta", "mean"), ev_roi_spearman_delta_mean=("ev_roi_spearman_delta", "mean"), mean_abs_p_change=("mean_abs_p_change", "mean"), ev_ge_1_count_delta_mean=("ev_ge_1_count_delta", "mean"), feature_count=("feature_count", "max"))
    return by, summary.sort_values(["logloss_delta_mean", "brier_delta_mean"], ascending=False)


def market_vs_residual(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df["model_key"] == "C1_market_offset_fundamental"].copy()
    d["residual_raw"] = d["catboost_residual_score"]
    d["final_logit"] = d["market_logit"] + d["residual_raw"]
    d["p_recalc"] = expit(d["final_logit"])
    rows = []
    for year, g in d.groupby("Year"):
        rows.append({
            "Year": int(year),
            "rows": len(g),
            "market_logit_mean": g["market_logit"].mean(),
            "market_logit_std": g["market_logit"].std(),
            "residual_raw_mean": g["residual_raw"].mean(),
            "residual_raw_std": g["residual_raw"].std(),
            "final_logit_mean": g["final_logit"].mean(),
            "final_logit_std": g["final_logit"].std(),
            "p_market_p_final_pearson": pearsonr(g["p_market"], g["final_probability"]).statistic,
            "p_market_p_final_spearman": spearmanr(g["p_market"], g["final_probability"]).statistic,
            "abs_residual_raw_p50": g["residual_raw"].abs().quantile(.5),
            "abs_residual_raw_p90": g["residual_raw"].abs().quantile(.9),
            "residual_raises_market_count": int((g["residual_raw"] > 0).sum()),
            "residual_lowers_market_count": int((g["residual_raw"] < 0).sum()),
            "baseline_logloss": log_loss(g["actual_place"], np.clip(g["p_market"], 1e-6, 1 - 1e-6), labels=[0, 1]),
            "c1_logloss": log_loss(g["actual_place"], np.clip(g["final_probability"], 1e-6, 1 - 1e-6), labels=[0, 1]),
            "baseline_brier": brier_score_loss(g["actual_place"], g["p_market"]),
            "c1_brier": brier_score_loss(g["actual_place"], g["final_probability"]),
        })
    by = pd.DataFrame(rows)
    by["logloss_improvement"] = by["baseline_logloss"] - by["c1_logloss"]
    by["brier_improvement"] = by["baseline_brier"] - by["c1_brier"]
    return by, by.mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})


def distribution_by_year(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df[df["model_key"] == "C1_market_offset_fundamental"].copy()
    d["residual_raw"] = d["catboost_residual_score"]
    d["final_logit"] = d["market_logit"] + d["residual_raw"]
    d["ev"] = d["final_probability"] * d["fuku_odds_low"]
    d["p_diff"] = d["final_probability"] - d["p_market"]
    d["required_probability"] = 1.0 / pd.to_numeric(d["fuku_odds_low"], errors="coerce").clip(lower=1.0)
    d["required_final_logit"] = logit(np.clip(d["required_probability"], 1e-6, 1 - 1e-6))
    d["ev_logit_margin"] = d["final_logit"] - d["required_final_logit"]
    metrics = ["p_market", "market_logit", "residual_raw", "final_probability", "fuku_odds_low", "ev", "p_diff", "ev_logit_margin"]
    rows = []
    qs = [.01, .05, .10, .25, .50, .75, .90, .95, .99]
    for year, g in d.groupby("Year"):
        for metric in metrics:
            s = pd.to_numeric(g[metric], errors="coerce")
            row = {"Year": int(year), "metric": metric, "min": s.min(), "max": s.max()}
            for q in qs:
                row[f"p{int(q*100):02d}"] = s.quantile(q)
            rows.append(row)
    count_rows = []
    cross_rows = []
    margin_rows = []
    for year, g in d.groupby("Year"):
        ev = g["ev"] >= 1.0
        market_ev = g["p_market"] * g["fuku_odds_low"] >= 1.0
        count_rows.append({"Year": int(year), "eligible_rows": len(g), "races": g["race_id"].nunique(), "ev_ge_1_count": int(ev.sum()), "ev_ge_1_rate": float(ev.mean()), "market_only_ev_ge_1_count": int(market_ev.sum())})
        cross_rows.append({"Year": int(year), "market_only_ev_ge_1_count": int(market_ev.sum()), "c1_ev_ge_1_count": int(ev.sum()), "ev_lt_1_to_ge_1_by_residual": int((~market_ev & ev).sum()), "ev_ge_1_to_lt_1_by_residual": int((market_ev & ~ev).sum()), "margin_near_zero_count_abs_lt_0_02": int((g["ev_logit_margin"].abs() < .02).sum())})
        s = g["ev_logit_margin"]
        margin_rows.append({"Year": int(year), "min": s.min(), "p01": s.quantile(.01), "p05": s.quantile(.05), "p50": s.quantile(.5), "p95": s.quantile(.95), "p99": s.quantile(.99), "max": s.max()})
    return pd.DataFrame(rows), pd.DataFrame(count_rows), pd.DataFrame(cross_rows), pd.DataFrame(margin_rows)


def model_fold_metadata(models: dict[str, CatBoostClassifier], features: list[str], cat: list[str], manifest: dict[str, Any], fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, model in models.items():
        fm = fold_metrics[(fold_metrics["fold"] == fold) & (fold_metrics["model_key"] == "C1_market_offset_fundamental")]
        _, year, train_years = next(x for x in FOLDS if x[0] == fold)
        path = Path("models/place_market_offset_catboost_v1/C1/folds") / fold / "model.cbm"
        rows.append({
            "fold": fold,
            "validation_year": year,
            "train_years": ",".join(map(str, train_years)),
            "tree_count": model.tree_count_,
            "best_iteration": int(fm["best_iteration"].iloc[0]) if len(fm) else np.nan,
            "learning_rate": manifest.get("fingerprint", {}).get("catboost_config_hash", ""),
            "feature_count": len(features),
            "categorical_count": len([f for f in features if f in cat]),
            "feature_names_hash": hashlib.sha256("\n".join(features).encode()).hexdigest(),
            "model_file_hash": sha256_file(path) if path.exists() else "",
        })
    return pd.DataFrame(rows)


def load_feature_years(dataset_dir: Path, years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for y in years:
        p = dataset_dir / f"year={y}" / "data.parquet"
        parts.append(pd.read_parquet(p, columns=columns))
    return pd.concat(parts, ignore_index=True)


def write_reports(out: Path, docs_path: Path, manifest: dict[str, Any], pvc: pd.DataFrame, lfc: pd.DataFrame, shap_global: pd.DataFrame, perm: pd.DataFrame, ev_counts: pd.DataFrame, course: pd.DataFrame, pedigree: pd.DataFrame) -> None:
    top_pvc = pvc.head(15).to_markdown(index=False)
    top_lfc = lfc.head(15).to_markdown(index=False)
    top_shap = shap_global.head(15).to_markdown(index=False)
    top_perm = perm.head(15).to_markdown(index=False)
    counts = ev_counts.to_markdown(index=False)
    report = [
        "# Place Market Offset Feature Audit V1",
        "",
        "## Scope",
        "- Target model: `C1_market_offset_fundamental`",
        "- No retraining, no DB read, no feature dataset rebuild, no 2025/2026 adjustment.",
        f"- Missing required handover: `{manifest['missing_required_files']}`",
        "",
        "## Top PredictionValuesChange",
        top_pvc,
        "",
        "## Top LossFunctionChange",
        top_lfc,
        "",
        "## Top SHAP",
        top_shap,
        "",
        "## Group Permutation",
        top_perm,
        "",
        "## EV Count Shift",
        counts,
        "",
        "## Course Structure",
        course.to_markdown(index=False),
        "",
        "## Pedigree",
        pedigree.to_markdown(index=False),
    ]
    atomic_write_text(out / "audit_report.md", "\n".join(report) + "\n")
    atomic_write_text(docs_path, "\n".join(report) + "\n")
    atomic_write_text(out / "ev_definition_audit.md", "\n".join([
        "# EV Definition Audit",
        "",
        "- EV formula: `final_probability * fuku_odds_low`.",
        "- `final_probability` is the C1 final probability from saved predictions.",
        "- C1 logit formula: `final_logit = market_logit + catboost_residual_score`.",
        "- EV>=1 is equivalent to `final_logit >= logit(1 / fuku_odds_low)` when odds are positive.",
        "- 2025/2026 were read only as diagnostic holdout outputs; no calibration, model, or threshold selection was changed.",
    ]) + "\n")


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    cfg = load_yaml(config_path)
    out = Path(cfg["output_root"])
    if out.exists():
        raise SystemExit(f"output directory already exists; refusing overwrite: {out}")
    out.mkdir(parents=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    source_cfg = load_yaml(Path(cfg["source_config"]))
    source_out = Path(cfg["source_output_dir"])
    source_model = Path(cfg["source_model_dir"])
    dataset_dir = Path(cfg["model_feature_dataset_dir"])
    source_manifest = json.loads((source_out / "manifest.json").read_text(encoding="utf-8"))
    model_manifest = source_manifest
    missing = [p for p in [cfg["required_handover"]] if not Path(p).exists()]
    artifacts = [
        Path(cfg["required_handover"]), Path(cfg["task_markdown"]), Path(cfg["source_config"]), Path(cfg["feature_set_yaml"]),
        source_out / "manifest.json", source_out / "residual_oof_predictions.parquet", source_out / "final_predictions_2025.parquet", source_out / "final_predictions_2026.parquet",
        dataset_dir / "manifest.json", dataset_dir / "feature_inventory.csv", dataset_dir / "feature_set_validation.csv",
    ] + sorted((source_model / "C1" / "folds").glob("fold_*/model.cbm"))
    inv = artifact_inventory(artifacts)
    inv_hash = atomic_write_csv(out / "input_artifact_inventory.csv", inv)
    feature_inventory = pd.read_csv(dataset_dir / "feature_inventory.csv")
    val = pd.read_parquet(source_out / "residual_oof_predictions.parquet")
    val = val[val["model_key"].eq(cfg["target_model_key"])].copy()
    final = pd.concat([pd.read_parquet(source_out / "final_predictions_2025.parquet"), pd.read_parquet(source_out / "final_predictions_2026.parquet")], ignore_index=True)
    final = final[final["model_key"].eq(cfg["target_model_key"])].copy()
    all_pred = pd.concat([val, final], ignore_index=True, sort=False)
    models = {p.parent.name: load_model(p) for p in sorted((source_model / "C1" / "folds").glob("fold_*/model.cbm"))}
    first_model = models["fold_2020"]
    features = list(first_model.feature_names_)
    cat = [c for c in source_manifest["C1_features"][1] if c in features]
    fmap = group_map(features)
    feature_cols = sorted(set(features + ["Year"]))
    feature_df = load_feature_years(dataset_dir, [2020, 2021, 2022, 2023, 2024], columns=[c for c in feature_cols if c != "p_market" and c != "market_logit" and c in pd.read_parquet(dataset_dir / "year=2020" / "data.parquet").columns])
    schema = feature_schema_inventory(feature_df, feature_inventory, features, cat, features)
    course = course_structure_audit(list(feature_df.columns), features)
    pedigree = pedigree_audit(list(feature_df.columns), features, [Path("scripts/build_model_features_v2_1_2.py"), Path("src/features/history_builder_v2_1_2.py"), Path("config/feature_sets_v2_1_2.yaml")])
    pvc_by, lfc_by = importance_tables(models, val, features, cat)
    pvc_summary = summarize_importance(pvc_by)
    lfc_summary = summarize_importance(lfc_by)
    shap_outputs, shap_add = shap_tables(models, val, features, cat, int(cfg["shap_sample_per_year"]), int(cfg["random_seed"]))
    perm_by, perm_summary = permutation_importance(models, val, features, cat, int(cfg["permutation_sample_per_fold"]), int(cfg["permutation_repeats"]), int(cfg["random_seed"]))
    market_by, market_summary = market_vs_residual(all_pred)
    dist, ev_counts, crossing, margin = distribution_by_year(all_pred)
    fold_meta = model_fold_metadata(models, features, cat, model_manifest, pd.read_csv(source_out / "fold_metrics.csv"))
    diag = perm_by.assign(period="validation_2020_2024").head(0)
    counter = pd.DataFrame([{"status": "not_executed", "reason": "Saved market baseline fold models are not available; no DB read or retraining fallback was allowed."}])
    hashes = {}
    tables = {
        "feature_schema_inventory.csv": schema,
        "feature_group_map.csv": fmap,
        "course_structure_feature_audit.csv": course,
        "pedigree_feature_audit.csv": pedigree,
        "catboost_pvc_by_fold.csv": pvc_by,
        "catboost_pvc_summary.csv": pvc_summary,
        "catboost_lfc_by_fold.csv": lfc_by,
        "catboost_lfc_summary.csv": lfc_summary,
        "shap_additivity_check.csv": shap_add,
        "market_vs_residual_contribution_by_year.csv": market_by,
        "market_vs_residual_contribution_summary.csv": market_summary,
        "permutation_importance_by_fold_repeat.csv": perm_by,
        "permutation_importance_summary_2020_2024.csv": perm_summary,
        "permutation_importance_2025_2026_diagnostic.csv": diag,
        "ev_component_distribution_by_year.csv": dist,
        "ev_threshold_counts_by_year.csv": ev_counts,
        "ev_crossing_decomposition_by_year.csv": crossing,
        "ev_margin_distribution_by_year.csv": margin,
        "model_fold_metadata.csv": fold_meta,
        "counterfactual_previous_fold_summary.csv": counter,
    }
    tables.update(shap_outputs)
    for name, df in tables.items():
        hashes[name] = atomic_write_csv(out / name, df)
    manifest = {
        "version": cfg["version"],
        "source_model": cfg["target_model_key"],
        "source_manifest_hash": sha256_file(source_out / "manifest.json"),
        "input_artifact_inventory_hash": inv_hash,
        "output_hashes": hashes,
        "missing_required_files": missing,
        "db_read": False,
        "retraining": False,
        "feature_dataset_rebuild": False,
        "random_split": False,
        "used_2025_2026_for_adjustment": False,
        "git": git_info(),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out / "audit_manifest.json", manifest)
    write_reports(out, Path(cfg["docs_output"]), manifest, pvc_summary, lfc_summary, shap_outputs["shap_global_2020_2024.csv"], perm_summary, ev_counts, course, pedigree)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_place_market_offset_feature_importance_v1.yaml")
    args = parser.parse_args()
    manifest = run(Path(args.config))
    print(json.dumps({"output_root": "outputs/place_market_offset_feature_audit_v1", "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
