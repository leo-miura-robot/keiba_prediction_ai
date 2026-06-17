from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.data.repository import load_config, write_inventory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/current_model_webapp_mvp_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    path = write_inventory(config)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
