#!/usr/bin/env python3
"""
Analyze ARiADNE training maps and materialize easy/medium/hard curriculum buckets.

Example:
source /root/miniconda3/etc/profile.d/conda.sh && conda activate ros_conda && cd /root/ros_ws/ARE/src && \
python -m ARiADNE.scripts.score_map_difficulty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ARiADNE.map_difficulty import (  # noqa: E402
    DEFAULT_BUCKET_NAMES,
    build_difficulty_dataset,
    default_output_dir,
    write_difficulty_outputs,
)
from ARiADNE.parameter import MAPS_DIR  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default=str(MAPS_DIR))
    parser.add_argument("--output-dir")
    parser.add_argument("--corridor-clearance-cells", dest="corridor_clearance_cells", type=float, default=3.0)
    parser.add_argument("--downsample-factor", dest="downsample_factor", type=int, default=2)
    parser.add_argument("--max-maps", dest="max_maps", type=int)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--bucket-names", default=",".join(DEFAULT_BUCKET_NAMES))
    return parser.parse_args()


def _parse_bucket_names(value: str) -> tuple[str, ...]:
    bucket_names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not bucket_names:
        raise ValueError("bucket_names must contain at least one non-empty name")
    return bucket_names


def main_cli():
    args = parse_args()
    bucket_names = _parse_bucket_names(args.bucket_names)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()

    records = build_difficulty_dataset(
        maps_dir=args.maps_dir,
        corridor_clearance_cells=args.corridor_clearance_cells,
        downsample_factor=args.downsample_factor,
        bucket_names=bucket_names,
        max_maps=args.max_maps,
    )
    outputs = write_difficulty_outputs(
        records,
        output_dir,
        link_mode=args.link_mode,
        clear_output=args.clear_output,
    )

    bucket_counts: dict[str, int] = {}
    for record in records:
        bucket = str(record["difficulty_bucket"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print(f"map_count={len(records)}")
    print(f"output_dir={outputs['output_dir']}")
    print(f"difficulty_scores_csv={outputs['csv_path']}")
    print(f"difficulty_manifest_json={outputs['json_path']}")
    print(f"bucket_root={outputs['bucket_root']}")
    for bucket_name in bucket_names:
        print(f"bucket_{bucket_name}={bucket_counts.get(bucket_name, 0)}")


if __name__ == "__main__":
    main_cli()
