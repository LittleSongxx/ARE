#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbge.baselines import run_baselines
from sbge.config import SBGEConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--maps-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SBGEConfig(seed=args.seed)
    if args.maps_dir:
        config = config.with_overrides(maps_dir=args.maps_dir)
    if args.smoke:
        config = config.smoke(seed=args.seed)
        episodes = 2
    else:
        episodes = args.episodes
    summary = run_baselines(config, episodes=episodes, output_dir=args.output_dir)
    print(f"baseline_rows={len(summary['rows'])}")


if __name__ == "__main__":
    main()
