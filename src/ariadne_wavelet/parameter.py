from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
import os
import re


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
BASELINE_ROOT = SRC_ROOT / "large-scale-DRL-exploration"


def _resolve_maps_dir() -> Path:
    override = os.environ.get("ARIADNE_MAPS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        PACKAGE_ROOT / "maps",
        BASELINE_ROOT / "maps",
        SRC_ROOT / "maps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


MAPS_DIR = _resolve_maps_dir()


# saving path
FOLDER_NAME = "ariadne_wavelet_run1"
SMOKE_FOLDER_NAME = f"{FOLDER_NAME}_smoke"
model_path = PACKAGE_ROOT / "model"
train_path = PACKAGE_ROOT / "train"
gifs_path = PACKAGE_ROOT / "gifs"
checkpoint_path = model_path / "checkpoint.pth"
RESULT_BUCKET_EPISODES = 100  # folders like episodes_100, episodes_200


# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False

# Original local-machine default kept for rollback/reference:
# SAVE_IMG_GAP = 20  # every N episodes, save train visuals and run one auto eval
# Active default for the offline 3 x A40 server:
# reduce PNG/GIF/eval frequency to avoid unnecessary I/O while long training runs.
SAVE_IMG_GAP = 100


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
# Previous active settings before switching to the 3 x A40 server profile:
# MAX_EPISODES=5000, MAX_EPISODE_STEP=128, REPLAY_SIZE=8000,
# MINIMUM_BUFFER_SIZE=2048, BATCH_SIZE=128, LR=1e-5, GAMMA=1,
# NUM_META_AGENT=8, TRAIN_UPDATES_PER_ITER=6, USE_GPU=False,
# USE_GPU_GLOBAL=True, NUM_GPU=0
MAX_EPISODES = 5000
MAX_EPISODE_STEP = 120
REPLAY_SIZE = 40000
MINIMUM_BUFFER_SIZE = 8000
BATCH_SIZE = 384
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 63
TRAIN_UPDATES_PER_ITER = 8


# network parameters
NODE_INPUT_DIM = 5
EMBEDDING_DIM = 128


# graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 360


# inference graph parameters
THR_TO_WAYPOINT = 1
THR_NEXT_WAYPOINT = 5
THR_GRAPH_HARD_UPDATE = 8
CLUSTER_RANGE = 10
ENABLE_DSTARLITE = False
USE_CURRENT_LOCATION_FOR_CONNECTIVITY = False


# wavelet parameters
USE_WAVELET_FEATURE = True
WAVELET_SCALES = (1, 2, 4)
WAVELET_DTH_ALPHA = 1.0
WAVELET_DTH_MAX_MULT = 2.0


# GPU usage
USE_GPU = True  # enable GPU inference inside Ray workers
USE_GPU_GLOBAL = True  # the learner uses CUDA and nn.DataParallel across visible GPUs
NUM_GPU = 3  # reserve 3 GPUs for Ray workers; keep CUDA_VISIBLE_DEVICES aligned with the same 3 cards


@dataclass(frozen=True)
class RuntimeConfig:
    max_episodes: int = MAX_EPISODES
    num_meta_agent: int = NUM_META_AGENT
    max_episode_step: int = MAX_EPISODE_STEP
    minimum_buffer_size: int = MINIMUM_BUFFER_SIZE
    batch_size: int = BATCH_SIZE
    replay_size: int = REPLAY_SIZE
    save_img_gap: int = SAVE_IMG_GAP
    summary_window: int = SUMMARY_WINDOW
    train_updates_per_iter: int = TRAIN_UPDATES_PER_ITER
    result_bucket_episodes: int = RESULT_BUCKET_EPISODES
    load_model: bool = LOAD_MODEL
    use_gpu: bool = USE_GPU
    use_gpu_global: bool = USE_GPU_GLOBAL
    num_gpu: int = NUM_GPU
    enable_training_monitor: bool = True
    monitor_window: int = 10
    monitor_snapshot_interval: int = 10
    enable_auto_eval: bool = True
    auto_eval_episodes: int = 1
    auto_eval_greedy: bool = True
    auto_eval_device: str = "cpu"
    run_name: str = FOLDER_NAME
    run_session: str | None = None

    def with_overrides(self, **kwargs: object) -> "RuntimeConfig":
        return replace(self, **kwargs)


def build_run_session(run_name: str = FOLDER_NAME) -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M") + _get_run_suffix(run_name)


def _get_run_suffix(run_name: str | None) -> str:
    if run_name in (None, "", FOLDER_NAME):
        return ""
    if run_name == SMOKE_FOLDER_NAME:
        return "_smoke"
    return f"_{run_name}"


def _append_run_session(base_path: Path, runtime_config: RuntimeConfig | None = None) -> Path:
    if runtime_config is not None and runtime_config.run_session:
        return base_path / runtime_config.run_session
    return base_path


def get_model_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(model_path, runtime_config)


def get_train_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(train_path, runtime_config)


def get_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(gifs_path, runtime_config)


def get_monitor_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_train_path(runtime_config) / "monitor"


def get_checkpoint_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint.pth"


def get_latest_checkpoint_path(run_name: str = FOLDER_NAME) -> Path:
    if not model_path.exists():
        return model_path / "checkpoint.pth"

    session_dirs = sorted(
        [path for path in model_path.iterdir() if path.is_dir() and (path / "checkpoint.pth").exists()],
        key=lambda path: path.name,
    )
    filtered_dirs = [path for path in session_dirs if _get_run_suffix(run_name) == "" and not path.name.endswith("_smoke")]
    if run_name == SMOKE_FOLDER_NAME:
        filtered_dirs = [path for path in session_dirs if path.name.endswith("_smoke")]
    elif run_name not in (FOLDER_NAME, "", None):
        filtered_dirs = [path for path in session_dirs if path.name.endswith(_get_run_suffix(run_name))]

    if filtered_dirs:
        return filtered_dirs[-1] / "checkpoint.pth"
    if session_dirs:
        return session_dirs[-1] / "checkpoint.pth"
    return model_path / "checkpoint.pth"


def get_run_identity_from_checkpoint(checkpoint_file: str | Path) -> tuple[str, str | None]:
    checkpoint_file = Path(checkpoint_file).resolve()
    model_root = model_path.resolve()
    try:
        relative = checkpoint_file.relative_to(model_root)
    except ValueError:
        return FOLDER_NAME, None

    if not relative.parts:
        return FOLDER_NAME, None

    run_session = relative.parts[0]
    match = re.fullmatch(r"\d{4}_\d{4}_\d{4}(?:_(.+))?", run_session)
    if not match:
        return FOLDER_NAME, run_session
    suffix = match.group(1)
    if suffix == "smoke":
        return SMOKE_FOLDER_NAME, run_session
    if suffix:
        return suffix, run_session
    return FOLDER_NAME, run_session


def ensure_output_dirs(runtime_config: RuntimeConfig | None = None) -> None:
    get_model_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_train_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_gifs_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_monitor_path(runtime_config).mkdir(parents=True, exist_ok=True)
