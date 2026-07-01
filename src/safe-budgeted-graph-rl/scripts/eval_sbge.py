#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbge.config import SBGEConfig
from sbge.eval import evaluate_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maps-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-file")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--map-limit", type=int)
    parser.add_argument("--experiment-name", default="sbge")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SBGEConfig(seed=args.seed, device=args.device)
    if args.maps_dir:
        config = config.with_overrides(maps_dir=args.maps_dir)
    config = config.with_overrides(
        split_file=args.split_file,
        split_name=args.split,
        map_limit=args.map_limit,
        experiment_name=args.experiment_name,
    )
    summary = evaluate_checkpoint(config, args.checkpoint, args.episodes, output_dir=args.output_dir)
    print(f"mean_explored_rate={summary['mean_explored_rate']:.4f}")
    print(f"mean_episode_cost={summary['mean_episode_cost']:.4f}")


if __name__ == "__main__":
    main()
