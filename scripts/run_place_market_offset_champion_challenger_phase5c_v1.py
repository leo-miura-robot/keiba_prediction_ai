from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_place_market_offset_year_strategy_phase5b_v2 as phase5b  # noqa: E402

KEYS = ["entry_id", "race_id", "race_date", "Year"]
PAIR = ("ROLLING_10Y", "ROLLING_15Y")
LIMITS = [1, 3, 5, 10]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["Year"] = pd.to_numeric(out["Year"], errors="raise").astype(int)
    return out


def merged_pair(pred: pd.DataFrame) -> pd.DataFrame:
    base_cols = KEYS + ["actual_place", "probability_raw", "catboost_residual_score"]
    left = canonical_keys(pred[pred["strategy"].eq(PAIR[0])][base_cols].copy())
    right = canonical_keys(pred[pred["strategy"].eq(PAIR[1])][base_cols].copy())
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError("Duplicate champion/challenger prediction keys")
    merged = left.merge(right, on=KEYS, suffixes=("_10y", "_15y"), how="outer", indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Prediction key mismatch: {merged['_merge'].value_counts().to_dict()}")
    if not merged["actual_place_10y"].equals(merged["actual_place_15y"]):
        raise ValueError("Target mismatch between champion and challenger")
    merged["actual_place"] = merged["actual_place_10y"].astype(int)
    return merged.drop(columns=["_merge"])


def metrics_frame(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for (strategy, year), g in pred.groupby(["strategy", "Year"]):
        rows.append({"strategy": strategy, "Year": int(year), **phase5b.metrics_for_frame(g, cfg), **phase5b.residual_metrics(g)})
    return pd.DataFrame(rows)


def combined_metrics(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for strategy, g in pred.groupby("strategy"):
        rows.append({"strategy": strategy, "Year": "2025_2026", **phase5b.metrics_for_frame(g, cfg), **phase5b.residual_metrics(g)})
    return pd.DataFrame(rows)


def direct_bootstrap(merged: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, g in list(merged.groupby("Year")) + [("2025_2026", merged)]:
        races = np.array(sorted(g["race_id"].unique()))
        race_index = {race: i for i, race in enumerate(races)}
        idx = np.array([race_index[r] for r in g["race_id"]], dtype=np.int64)
        y = g["actual_place"].to_numpy(int)
        p10 = g["probability_raw_10y"].to_numpy(float)
        p15 = g["probability_raw_15y"].to_numpy(float)
        row_ll = -(y * np.log(np.clip(p10, 1e-15, 1 - 1e-15)) + (1 - y) * np.log(np.clip(1 - p10, 1e-15, 1))) - (
            -(y * np.log(np.clip(p15, 1e-15, 1 - 1e-15)) + (1 - y) * np.log(np.clip(1 - p15, 1e-15, 1)))
        )
        row_br = (p10 - y) ** 2 - (p15 - y) ** 2
        race_ll = np.bincount(idx, weights=row_ll, minlength=len(races))
        race_br = np.bincount(idx, weights=row_br, minlength=len(races))
        race_n = np.bincount(idx, minlength=len(races))
        for metric, race_sum in [("logloss", race_ll), ("brier", race_br)]:
            draws = np.empty(iterations, dtype=float)
            for i in range(iterations):
                sample = rng.integers(0, len(races), len(races))
                draws[i] = race_sum[sample].sum() / race_n[sample].sum()
            point = float(race_sum.sum() / race_n.sum())
            rows.append(
                {
                    "Year": label,
                    "metric": metric,
                    "delta_10y_minus_15y": point,
                    "bootstrap_mean": float(draws.mean()),
                    "ci95_lower": float(np.percentile(draws, 2.5)),
                    "ci95_upper": float(np.percentile(draws, 97.5)),
                    "champion_10y_better_probability": float((draws < 0).mean()),
                    "races": int(len(races)),
                    "rows": int(len(g)),
                    "n_bootstrap": int(iterations),
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def probability_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in list(merged.groupby("Year")) + [("2025_2026", merged)]:
        diff = (g["probability_raw_10y"] - g["probability_raw_15y"]).abs()
        rows.append(
            {
                "Year": label,
                "rows": int(len(g)),
                "mean_abs_probability_diff": float(diff.mean()),
                "p50_abs_probability_diff": float(diff.quantile(0.50)),
                "p90_abs_probability_diff": float(diff.quantile(0.90)),
                "p95_abs_probability_diff": float(diff.quantile(0.95)),
                "p99_abs_probability_diff": float(diff.quantile(0.99)),
                "max_abs_probability_diff": float(diff.max()),
                "pearson_correlation": float(g["probability_raw_10y"].corr(g["probability_raw_15y"], method="pearson")),
                "spearman_correlation": float(g["probability_raw_10y"].corr(g["probability_raw_15y"], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def ranking_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in list(merged.groupby("Year")) + [("2025_2026", merged)]:
        top1_same = top3_overlap = rank_corr_sum = rank_corr_n = 0
        for _race, r in g.groupby("race_id"):
            a = r.sort_values("probability_raw_10y", ascending=False)["entry_id"].tolist()
            b = r.sort_values("probability_raw_15y", ascending=False)["entry_id"].tolist()
            top1_same += int(a[:1] == b[:1])
            top3_overlap += len(set(a[:3]).intersection(b[:3])) / max(1, min(3, len(set(a).union(b))))
            corr = spearmanr(r["probability_raw_10y"], r["probability_raw_15y"]).correlation
            if not math.isnan(corr):
                rank_corr_sum += float(corr)
                rank_corr_n += 1
        races = g["race_id"].nunique()
        rows.append(
            {
                "Year": label,
                "races": int(races),
                "top1_agreement_rate": float(top1_same / races) if races else math.nan,
                "top3_set_overlap_mean": float(top3_overlap / races) if races else math.nan,
                "race_rank_spearman_mean": float(rank_corr_sum / rank_corr_n) if rank_corr_n else math.nan,
            }
        )
    return pd.DataFrame(rows)


def error_win_loss(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in list(merged.groupby("Year")) + [("2025_2026", merged)]:
        y = g["actual_place"].to_numpy(int)
        e10 = -(y * np.log(np.clip(g["probability_raw_10y"], 1e-15, 1 - 1e-15)) + (1 - y) * np.log(np.clip(1 - g["probability_raw_10y"], 1e-15, 1)))
        e15 = -(y * np.log(np.clip(g["probability_raw_15y"], 1e-15, 1 - 1e-15)) + (1 - y) * np.log(np.clip(1 - g["probability_raw_15y"], 1e-15, 1)))
        d = e10 - e15
        tmp = g[["race_id"]].copy()
        tmp["delta"] = d
        race_delta = tmp.groupby("race_id")["delta"].mean()
        rows.append(
            {
                "Year": label,
                "champion_10y_better_rows": int((d < 0).sum()),
                "challenger_15y_better_rows": int((d > 0).sum()),
                "tie_rows": int(np.isclose(d, 0).sum()),
                "champion_10y_better_races": int((race_delta < 0).sum()),
                "challenger_15y_better_races": int((race_delta > 0).sum()),
                "tie_races": int(np.isclose(race_delta, 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def roi_tables(pred: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_rows = []
    rr_rows = []
    pz_rows = []
    for (strategy, year), g in pred.groupby(["strategy", "Year"]):
        picks = phase5b.ev_picks(g, cfg)
        payout = pd.to_numeric(picks[cfg["payout_column"]], errors="coerce").fillna(0)
        roi_rows.append(
            {
                "strategy": strategy,
                "Year": int(year),
                "bet_count": int(len(picks)),
                "race_count_with_bet": int(picks["race_id"].nunique()),
                "stake": int(len(picks) * cfg["stake_yen"]),
                "payout": float(payout.sum()),
                "roi": phase5b.roi_value(picks, cfg),
                "hit_count": int((payout > 0).sum()),
                "hit_rate": float((payout > 0).mean()) if len(picks) else math.nan,
                "average_odds": float(picks[cfg["odds_column"]].mean()) if len(picks) else math.nan,
                "average_predicted_probability": float(picks["probability_raw"].mean()) if len(picks) else math.nan,
                "average_ev": float(picks["ev"].mean()) if len(picks) else math.nan,
            }
        )
        _normal, rr, pz = phase5b.stress_roi_rows(g, cfg, LIMITS)
        rr_rows.append(rr)
        pz_rows.append(pz)
    combined = []
    for strategy, g in pred.groupby("strategy"):
        picks = phase5b.ev_picks(g, cfg)
        payout = pd.to_numeric(picks[cfg["payout_column"]], errors="coerce").fillna(0)
        combined.append(
            {
                "strategy": strategy,
                "Year": "2025_2026",
                "bet_count": int(len(picks)),
                "race_count_with_bet": int(picks["race_id"].nunique()),
                "stake": int(len(picks) * cfg["stake_yen"]),
                "payout": float(payout.sum()),
                "roi": phase5b.roi_value(picks, cfg),
                "hit_count": int((payout > 0).sum()),
                "hit_rate": float((payout > 0).mean()) if len(picks) else math.nan,
                "average_odds": float(picks[cfg["odds_column"]].mean()) if len(picks) else math.nan,
                "average_predicted_probability": float(picks["probability_raw"].mean()) if len(picks) else math.nan,
                "average_ev": float(picks["ev"].mean()) if len(picks) else math.nan,
            }
        )
    return pd.DataFrame(roi_rows), pd.DataFrame(combined), pd.concat(rr_rows, ignore_index=True), pd.concat(pz_rows, ignore_index=True)


def bet_overlap(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, g in list(pred.groupby("Year")) + [("2025_2026", pred)]:
        picks = {s: phase5b.ev_picks(d, cfg).copy() for s, d in g.groupby("strategy")}
        a = picks.get(PAIR[0], pd.DataFrame())
        b = picks.get(PAIR[1], pd.DataFrame())
        keys = ["entry_id", "race_id", "race_date", "Year"]
        aset = set(map(tuple, canonical_keys(a)[keys].to_numpy())) if len(a) else set()
        bset = set(map(tuple, canonical_keys(b)[keys].to_numpy())) if len(b) else set()
        common = aset & bset
        only_a = aset - bset
        only_b = bset - aset

        def roi_for(sub: pd.DataFrame, keyset: set[tuple[Any, ...]]) -> float:
            if not keyset or sub.empty:
                return math.nan
            keyed = canonical_keys(sub)
            mask = pd.Series(map(tuple, keyed[keys].to_numpy()), index=keyed.index).isin(keyset)
            return phase5b.roi_value(keyed[mask], cfg)

        rows.append(
            {
                "Year": label,
                "common_bets": int(len(common)),
                "champion_10y_only_bets": int(len(only_a)),
                "challenger_15y_only_bets": int(len(only_b)),
                "jaccard_similarity": float(len(common) / len(aset | bset)) if (aset | bset) else math.nan,
                "common_bet_roi_10y": roi_for(a, common),
                "common_bet_roi_15y": roi_for(b, common),
                "champion_10y_only_roi": roi_for(a, only_a),
                "challenger_15y_only_roi": roi_for(b, only_b),
            }
        )
    return pd.DataFrame(rows)


def segment_comparison(pred: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    d = pred.copy()
    d["surface"] = np.where(pd.to_numeric(d.get("TrackCD"), errors="coerce").fillna(0).between(10, 22), "turf", "dirt_or_other")
    d["distance_bucket"] = pd.cut(pd.to_numeric(d["Kyori"], errors="coerce"), [0, 1399, 1799, 2199, 10000], labels=["short", "mile", "middle", "long"])
    d["field_size_bucket"] = pd.cut(pd.to_numeric(d["SyussoTosu"], errors="coerce"), [0, 7, 11, 15, 99], labels=["small", "medium", "large", "max"])
    d["odds_bucket"] = pd.cut(pd.to_numeric(d[cfg["odds_column"]], errors="coerce"), [0, 1.5, 3, 6, 9999], labels=["low", "mid", "high", "long"])
    d["probability_bucket"] = pd.cut(pd.to_numeric(d["probability_raw"], errors="coerce"), [0, 0.15, 0.25, 0.40, 1.0], labels=["low", "mid", "high", "very_high"])
    dims = ["Year", "JyoCD", "surface", "distance_bucket", "field_size_bucket", "odds_bucket", "probability_bucket"]
    rows = []
    for dim in dims:
        for value, g in d.groupby(dim, dropna=False):
            if g["strategy"].nunique() < 2 or len(g) < 200:
                continue
            row = {"segment": dim, "value": str(value), "row_count": int(len(g)), "race_count": int(g["race_id"].nunique())}
            for strategy in PAIR:
                s = g[g["strategy"].eq(strategy)]
                if s.empty:
                    row[f"{strategy}_logloss"] = math.nan
                    row[f"{strategy}_brier"] = math.nan
                else:
                    y = s["actual_place"].to_numpy(int)
                    p = s["probability_raw"].to_numpy(float)
                    row[f"{strategy}_logloss"] = float(log_loss(y, p, labels=[0, 1]))
                    row[f"{strategy}_brier"] = float(brier_score_loss(y, p))
            row["delta_logloss_10y_minus_15y"] = row["ROLLING_10Y_logloss"] - row["ROLLING_15Y_logloss"]
            row["delta_brier_10y_minus_15y"] = row["ROLLING_10Y_brier"] - row["ROLLING_15Y_brier"]
            rows.append(row)
    return pd.DataFrame(rows)


def write_policy(out: Path) -> None:
    policy = {
        "champion": "ROLLING_10Y",
        "challenger": "ROLLING_15Y",
        "diagnostic_years_are_not_selection_data": [2025, 2026],
        "do_not_change_within_this_task": True,
        "minimum_forward_observation": {
            "months": 6,
            "races": 1000,
            "ev_ge_1_candidates_per_strategy": 200,
        },
        "promotion_conditions": [
            "race_paired_bootstrap_logloss_ci95_upper_below_zero",
            "brier_not_worse",
            "worst_month_not_materially_worse",
            "residual_p99_not_more_than_10_percent_worse",
            "roi_auxiliary_only",
        ],
    }
    (out / "champion_challenger_policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    schema = {
        "required_columns": [
            "prediction_generated_at",
            "model_version",
            "strategy",
            "entry_id",
            "race_id",
            "race_date",
            "probability_raw",
            "market_logit",
            "residual_raw",
            "odds_available_at_prediction",
            "EV_at_prediction",
            "feature_hash",
            "model_hash",
            "data_cutoff_date",
        ]
    }
    (out / "forward_prediction_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")


def postprocess(cfg: dict[str, Any]) -> None:
    out = Path(cfg["output_root"])
    pred = pd.read_parquet(out / "phase5b_predictions.parquet")
    pred = pred[pred["strategy"].isin(PAIR) & pred["Year"].isin([2025, 2026])].copy()
    pred.to_parquet(out / "phase5c_predictions.parquet", index=False)
    metrics = metrics_frame(pred, cfg)
    combined = combined_metrics(pred, cfg)
    merged = merged_pair(pred)
    metrics.to_csv(out / "metrics_2025_2026_by_strategy.csv", index=False)
    combined.to_csv(out / "metrics_2025_2026_combined.csv", index=False)
    direct_bootstrap(merged, int(cfg["bootstrap_iterations"]), int(cfg["random_seed"])).to_csv(out / "direct_pairwise_bootstrap.csv", index=False)
    probability_agreement(merged).to_csv(out / "probability_agreement.csv", index=False)
    ranking_agreement(merged).to_csv(out / "ranking_agreement.csv", index=False)
    error_win_loss(merged).to_csv(out / "error_win_loss.csv", index=False)
    roi_year, roi_combined, rr, pz = roi_tables(pred, cfg)
    roi_year.to_csv(out / "roi_by_strategy_year.csv", index=False)
    roi_combined.to_csv(out / "roi_combined.csv", index=False)
    rr.to_csv(out / "roi_row_removed.csv", index=False)
    pz.to_csv(out / "roi_payout_zeroed_stress.csv", index=False)
    bet_overlap(pred, cfg).to_csv(out / "bet_overlap.csv", index=False)
    segment_comparison(pred, cfg).to_csv(out / "segment_comparison.csv", index=False)
    write_policy(out)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "phase": "Phase 5C",
            "champion_strategy": "ROLLING_10Y",
            "challenger_strategy": "ROLLING_15Y",
            "diagnostic_only_years": [2025, 2026],
            "champion_changed": False,
            "output_files_phase5c": sorted(p.name for p in out.glob("*") if p.is_file()),
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(out)


def write_report(out: Path) -> None:
    metrics = pd.read_csv(out / "metrics_2025_2026_by_strategy.csv")
    combined = pd.read_csv(out / "metrics_2025_2026_combined.csv")
    boot = pd.read_csv(out / "direct_pairwise_bootstrap.csv")
    agree = pd.read_csv(out / "probability_agreement.csv")
    ranking = pd.read_csv(out / "ranking_agreement.csv")
    roi = pd.read_csv(out / "roi_combined.csv")
    overlap = pd.read_csv(out / "bet_overlap.csv")
    text = "\n".join(
        [
            "# Phase 5C Champion-Challenger Results",
            "",
            "Champion remains `ROLLING_10Y`. Challenger is `ROLLING_15Y`. 2025/2026 are diagnostic only.",
            "",
            "## Metrics By Year",
            metrics.to_markdown(index=False),
            "",
            "## Combined Metrics",
            combined.to_markdown(index=False),
            "",
            "## Direct Race-Paired Bootstrap",
            boot.to_markdown(index=False),
            "",
            "## Probability Agreement",
            agree.to_markdown(index=False),
            "",
            "## Ranking Agreement",
            ranking.to_markdown(index=False),
            "",
            "## Combined ROI",
            roi.to_markdown(index=False),
            "",
            "## Bet Overlap",
            overlap.to_markdown(index=False),
            "",
            "## Decision",
            "Champion is not changed by this diagnostic task.",
            "",
        ]
    )
    Path("docs/place_market_offset_champion_challenger_phase5c_v1_results.md").write_text(text, encoding="utf-8")


def run(config: Path, resume: bool, skip_train: bool = False) -> int:
    cfg = load_yaml(config)
    out = Path(cfg["output_root"])
    if not skip_train:
        phase5b.run(
            config,
            [cfg["champion_strategy"], cfg["challenger_strategy"]],
            [2025, 2026],
            resume=resume,
            smoke_rows_per_year=None,
            parity_check=False,
            output_root=str(out),
            model_root=str(cfg["model_root"]),
            reference_mode="corrected",
        )
    postprocess(cfg)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/place_market_offset_champion_challenger_phase5c_v1.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(Path(args.config), resume=args.resume, skip_train=args.skip_train)


if __name__ == "__main__":
    raise SystemExit(main())
