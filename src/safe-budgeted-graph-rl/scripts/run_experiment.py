#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbge.config import DEFAULT_SPLIT_FILE, SBGEConfig
from sbge.experiment import run_experiment
from sbge.splits import create_map_split, save_map_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--maps-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--map-limit", type=int)
    parser.add_argument("--experiment-name", default="sbge")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SBGEConfig(seed=args.seeds[0], device=args.device, experiment_name=args.experiment_name)
    if args.maps_dir:
        config = config.with_overrides(maps_dir=args.maps_dir)
    split_path = Path(args.split_file).expanduser().resolve()
    if not split_path.exists():
        save_map_split(create_map_split(config.maps_dir, seed=0), split_path)
    config = config.with_overrides(split_file=split_path, map_limit=args.map_limit)
    train_episodes = args.train_episodes
    eval_episodes = args.eval_episodes
    if args.smoke:
        config = config.smoke(seed=args.seeds[0]).with_overrides(
            split_file=split_path,
            map_limit=args.map_limit if args.map_limit is not None else 4,
            experiment_name=args.experiment_name,
        )
        train_episodes = 2
        eval_episodes = 1
    summary = run_experiment(
        config,
        seeds=args.seeds,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        output_dir=args.output_dir,
    )
    print(f"aggregate_methods={len(summary['aggregate'])}")
    print(f"aggregate_summary={Path(args.output_dir or config.result_dir / config.experiment_name) / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
