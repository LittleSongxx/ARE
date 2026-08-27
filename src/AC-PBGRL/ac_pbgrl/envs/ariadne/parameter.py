from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MAPS_DIR = Path(os.environ.get("ACPBGRL_MAPS_DIR", PACKAGE_ROOT / "maps")).expanduser().resolve()
RESULT_BUCKET_EPISODES = 100
gifs_path = str(Path(os.environ.get("ACPBGRL_DATA_ROOT", PACKAGE_ROOT / ".runtime")) / "gifs")

CELL_SIZE = 0.4
NODE_RESOLUTION = 4.0
FRONTIER_CELL_SIZE = 2 * CELL_SIZE
FREE = 255
OCCUPIED = 1
UNKNOWN = 127
SENSOR_RANGE = 16
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 2
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION
K_SIZE = 25
NODE_PADDING_SIZE = 512
