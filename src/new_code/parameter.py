from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = PACKAGE_ROOT.parent / "large-scale-DRL-exploration"
RUN_SESSION_ENV_VAR = "NEW_CODE_RUN_SESSION"


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
MODEL_ROOT = PACKAGE_ROOT / "model" / FOLDER_NAME
TRAIN_ROOT = PACKAGE_ROOT / "train" / FOLDER_NAME
GIFS_ROOT = PACKAGE_ROOT / "gifs" / FOLDER_NAME
EVAL_ROOT = PACKAGE_ROOT / "eval" / FOLDER_NAME


def build_run_session() -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M")


def get_run_session() -> str:
    run_session = os.environ.get(RUN_SESSION_ENV_VAR)
    if run_session:
        return run_session
    run_session = build_run_session()
    os.environ[RUN_SESSION_ENV_VAR] = run_session
    return run_session


def _append_run_session(base_path: Path, run_session: str | None = None) -> Path:
    return base_path / (run_session or get_run_session())


def get_model_path(run_session: str | None = None) -> Path:
    return _append_run_session(MODEL_ROOT, run_session)


def get_train_path(run_session: str | None = None) -> Path:
    return _append_run_session(TRAIN_ROOT, run_session)


def get_gifs_path(run_session: str | None = None) -> Path:
    return _append_run_session(GIFS_ROOT, run_session)


def get_monitor_path(run_session: str | None = None) -> Path:
    return get_train_path(run_session) / "monitor"


def get_eval_path(run_session: str | None = None) -> Path:
    return _append_run_session(EVAL_ROOT, run_session)


def get_checkpoint_path(run_session: str | None = None) -> Path:
    return get_model_path(run_session) / "checkpoint.pth"


def get_checkpoint_final_path(run_session: str | None = None) -> Path:
    return get_model_path(run_session) / "checkpoint_final.pth"


def get_checkpoint_interrupted_path(run_session: str | None = None) -> Path:
    return get_model_path(run_session) / "checkpoint_interrupted.pth"


def iter_checkpoint_candidates(run_session: str | None = None) -> list[Path]:
    if run_session is not None:
        return [
            get_checkpoint_interrupted_path(run_session),
            get_checkpoint_final_path(run_session),
            get_checkpoint_path(run_session),
        ]

    session_dirs = []
    if MODEL_ROOT.exists():
        session_dirs = sorted([path for path in MODEL_ROOT.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    candidates = []
    for session_dir in session_dirs:
        session = session_dir.name
        candidates.extend(iter_checkpoint_candidates(session))
    if not candidates:
        candidates.extend(iter_checkpoint_candidates(get_run_session()))
    return candidates


def get_latest_checkpoint_path() -> Path:
    for candidate in iter_checkpoint_candidates():
        if candidate.exists():
            return candidate
    return get_checkpoint_path()


def get_run_session_from_checkpoint(checkpoint_file: str | Path) -> str | None:
    checkpoint_file = Path(checkpoint_file).resolve()
    try:
        relative = checkpoint_file.relative_to(MODEL_ROOT.resolve())
    except ValueError:
        return None
    if not relative.parts:
        return None
    return relative.parts[0]


def ensure_output_dirs(run_session: str | None = None) -> None:
    get_model_path(run_session).mkdir(parents=True, exist_ok=True)
    get_train_path(run_session).mkdir(parents=True, exist_ok=True)
    get_gifs_path(run_session).mkdir(parents=True, exist_ok=True)
    get_monitor_path(run_session).mkdir(parents=True, exist_ok=True)
    get_eval_path(run_session).mkdir(parents=True, exist_ok=True)


# Legacy path aliases for existing imports. These resolve against the current
# NEW_CODE_RUN_SESSION environment variable.
model_path = str(get_model_path())
train_path = str(get_train_path())
gifs_path = str(get_gifs_path())
monitor_path = str(get_monitor_path())
eval_path = str(get_eval_path())
checkpoint_path = str(get_checkpoint_path())
checkpoint_final_path = str(get_checkpoint_final_path())
checkpoint_interrupted_path = str(get_checkpoint_interrupted_path())

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
