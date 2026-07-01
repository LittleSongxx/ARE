#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sbge.config import DEFAULT_MAPS_DIR, DEFAULT_SPLIT_FILE
from sbge.splits import create_map_split, save_map_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default=str(DEFAULT_MAPS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    split = create_map_split(
        args.maps_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    output = save_map_split(split, args.output)
    print(f"split_path={output}")
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
