from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = PACKAGE_ROOT.parent / "large-scale-DRL-exploration"


def _resolve_maps_dir() -> Path:
    override = os.environ.get("NEW_CODE_MAPS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        PACKAGE_ROOT / "maps",
        BASELINE_ROOT / "maps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


MAPS_DIR = _resolve_maps_dir()

# saving path
FOLDER_NAME = "ariadne1_ground_truth_critic"
model_path = str(PACKAGE_ROOT / "model" / FOLDER_NAME)
train_path = str(PACKAGE_ROOT / "train" / FOLDER_NAME)
gifs_path = str(PACKAGE_ROOT / "gifs" / FOLDER_NAME)
monitor_path = str(Path(train_path) / "monitor")
eval_path = str(PACKAGE_ROOT / "eval" / FOLDER_NAME)
checkpoint_path = str(Path(model_path) / "checkpoint.pth")
checkpoint_final_path = str(Path(model_path) / "checkpoint_final.pth")
checkpoint_interrupted_path = str(Path(model_path) / "checkpoint_interrupted.pth")

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 100
MONITOR_WINDOW = 10
MONITOR_SNAPSHOT_INTERVAL = 10

# auto eval
AUTO_EVAL = True
AUTO_EVAL_EPISODES = 1
AUTO_EVAL_GREEDY = True
AUTO_EVAL_DEVICE = "cpu"

# map and planning resolution
CELL_SIZE = 0.4
NODE_RESOLUTION = 4.0
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 16
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 2

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODES = 5000
INFINITE_TRAINING = False
MAX_EPISODE_STEP = 120

# Active defaults preserved from the current new_code branch.
REPLAY_SIZE = 40000
MINIMUM_BUFFER_SIZE = 8000
BATCH_SIZE = 384
LR = 1e-5
NUM_META_AGENT = 63
TRAIN_UPDATES_PER_ITER = 8
GAMMA = 1

# network parameters
NODE_INPUT_DIM = 4
EMBEDDING_DIM = 120

# Graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 360

# GPU usage
USE_GPU = True
NUM_GPU = 3
USE_GPU_GLOBAL = True
