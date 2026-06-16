import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import polars as pl
import hashlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_full_runner_dataset_o1_fixed as base_builder
from src.features.history_builder_v2_1_2 import build_pre_day_history_features_v2_1, new_state
from src.features.target_builder import add_target_columns

def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def setup_logger(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase5_data_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def main():
    out_dir = Path("data/derived/history_extension_2006_phase5_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "generation.log")

    db_2006_2015 = Path(r"D:\keiba\new_jra_2006-2015_fixed_20260616_015045\keiba.db")
    db_2016_2026 = Path(r"D:\keiba\new_jra_2016-2026_fixed\keiba.db")

    manifest = {
        "source_db_paths": [str(db_2006_2015), str(db_2016_2026)],
        "source_db_sha256": [sha256_file(db_2006_2015), sha256_file(db_2016_2026)],
        "source_date_min": "2006-01-01",
        "source_date_max": "2026-12-31",
        "feature_builder_version": "v2_1_2",
        "feature_generation_start_year": 2006,
        "model_training_start_year": 2006,
        "current_race_excluded": True,
        "same_day_future_excluded": True,
        "created_at": datetime.now().isoformat(),
        "row_counts_by_year": {},
        "race_counts_by_year": {}
    }

    base_dir = out_dir / "base_runner"
    base_dir.mkdir(exist_ok=True)

    logger.info("Fetching 2006-2015 base runners")
    base_builder.DB_PATH = db_2006_2015
    for year in range(2006, 2016):
        out_path = base_dir / f"year={year}.parquet"
        if not out_path.exists():
            df = base_builder.fetch_year(year, logger)
            if df is not None and df.height > 0:
                df.write_parquet(out_path, compression="zstd")

    logger.info("Fetching 2016-2026 base runners")
    base_builder.DB_PATH = db_2016_2026
    for year in range(2016, 2027):
        out_path = base_dir / f"year={year}.parquet"
        if not out_path.exists():
            df = base_builder.fetch_year(year, logger)
            if df is not None and df.height > 0:
                df.write_parquet(out_path, compression="zstd")

    logger.info("Concatenating canonical race rows")
    canonical_frames = []
    for year in range(2006, 2027):
        out_path = base_dir / f"year={year}.parquet"
        if out_path.exists():
            df = pl.read_parquet(out_path)
            manifest["row_counts_by_year"][year] = df.height
            manifest["race_counts_by_year"][year] = df["race_id"].n_unique()
            canonical_frames.append(df)
            
    canonical = pl.concat(canonical_frames, how="diagonal_relaxed")
    canonical_out = out_dir / "canonical_race_rows_2006_2026.parquet"
    canonical.write_parquet(canonical_out, compression="zstd")
    
    logger.info("Building history features")
    state = new_state()
    history_frames = []
    
    for year in range(2006, 2027):
        logger.info(f"Building history features for {year}")
        base_df = canonical.filter(pl.col("Year") == year)
        if base_df.height == 0:
            continue
        
        labeled = add_target_columns(base_df)
        features, state, audit_counts, _ = build_pre_day_history_features_v2_1(labeled, logger, state)
        history_frames.append(features)

    history_combined = pl.concat(history_frames, how="diagonal_relaxed")
    history_out = out_dir / "history_features_2006_2026.parquet"
    history_combined.write_parquet(history_out, compression="zstd")
    
    manifest["schema_hash"] = hashlib.sha256(str(history_combined.schema).encode("utf-8")).hexdigest()
    
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Done generating Phase 5 history features.")

if __name__ == "__main__":
    main()
