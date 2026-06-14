from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.catboost_analysis import analyze_all
from src.models.catboost_config import load_yaml_config


CONFIG_PATH = Path("config/catboost_baseline_v1_0_1.yaml")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--target", choices=["win", "place"])
    parser.add_argument("--feature-set", choices=["market_free", "market_history", "market_aware"])
    args = parser.parse_args()
    config = load_yaml_config(CONFIG_PATH)
    output_root = Path(config["output_root"])
    model_root = Path(config["model_root"])
    if args.all:
        targets = ["win", "place"]
        feature_sets = ["market_free", "market_history", "market_aware"]
    else:
        if not args.target or not args.feature_set:
            raise SystemExit("--target and --feature-set are required unless --all is used")
        targets = [args.target]
        feature_sets = [args.feature_set]
    started = time.time()
    result = analyze_all(output_root, model_root, targets, feature_sets, write=True)
    summary = {
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "targets": targets,
        "feature_sets": feature_sets,
        **result,
    }
    (output_root / "analysis_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
