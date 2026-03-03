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
result_path = PACKAGE_ROOT / "result"
checkpoint_path = model_path / "checkpoint.pth"
RESULT_BUCKET_EPISODES = 100  # folders like episodes_100, episodes_200
USE_FIXED_EVAL_MAPS = True
EVAL_BENCHMARK_MAPS = (
    "img_1714.png",
    "img_4474.png",
    "326.png",
    "img_965.png",
    "img_869.png",
)


# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False

# Original local-machine default kept for rollback/reference:
# SAVE_IMG_GAP = 20  # every N episodes, save train visuals and run one auto eval
# Active default for the offline 3 x A40 server:
# reduce PNG/GIF/eval frequency to avoid unnecessary I/O while long training runs.
SAVE_IMG_GAP = 100
GIF_FRAME_RATE = 1.0


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
MAX_EPISODES = 15000
MAX_EPISODE_STEP = 120
REPLAY_SIZE = 80000
MINIMUM_BUFFER_SIZE = 32768
BATCH_SIZE = 3584
LR = 1e-5
GAMMA = 1
# 20-core server with Numba-JIT sensor/collision: workers are CPU-light,
# so oversubscription is fine — Ray schedules up to 20 concurrently,
# the rest queue but drain fast (~100x speedup on Bresenham loops).
NUM_META_AGENT = 63
TRAIN_UPDATES_PER_ITER = 12


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
# Tuned for the current server allocation: two visible A40 45GB GPUs
# (CUDA_VISIBLE_DEVICES=0,3). The larger learner batch is meant to keep each
# card above ~30GB without pushing too close to the 45GB limit.
# Rollout workers stay on CPU: their batch=1 inference is latency-sensitive and
# does not benefit from sharing learner GPUs. The learner uses CUDA and
# nn.DataParallel across the GPUs currently visible to this process.
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0  # 0 = auto-detect visible learner GPUs from CUDA_VISIBLE_DEVICES / torch


@dataclass(frozen=True)
class RuntimeConfig:
    max_episodes: int = MAX_EPISODES
    num_meta_agent: int = NUM_META_AGENT
    ray_num_cpus: int | None = None
    ray_worker_num_cpus: int | None = None
    worker_num_threads: int | None = None
    max_episode_step: int = MAX_EPISODE_STEP
    minimum_buffer_size: int = MINIMUM_BUFFER_SIZE
    batch_size: int = BATCH_SIZE
    replay_size: int = REPLAY_SIZE
    save_img_gap: int = SAVE_IMG_GAP
    gif_frame_rate: float = GIF_FRAME_RATE
    summary_window: int = SUMMARY_WINDOW
    train_updates_per_iter: int = TRAIN_UPDATES_PER_ITER
    result_bucket_episodes: int = RESULT_BUCKET_EPISODES
    load_model: bool = LOAD_MODEL
    resume_from: str | None = None
    use_gpu: bool = USE_GPU
    use_gpu_global: bool = USE_GPU_GLOBAL
    num_gpu: int = NUM_GPU
    enable_training_monitor: bool = True
    monitor_window: int = 10
    monitor_snapshot_interval: int = 10
    enable_auto_eval: bool = True
    auto_eval_episodes: int = len(EVAL_BENCHMARK_MAPS)
    auto_eval_greedy: bool = True
    auto_eval_device: str = "cpu"
    use_fixed_eval_maps: bool = USE_FIXED_EVAL_MAPS
    eval_benchmark_maps: tuple[str, ...] = tuple(EVAL_BENCHMARK_MAPS)
    run_name: str = FOLDER_NAME
    run_session: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gif_frame_rate", max(float(self.gif_frame_rate), 1e-3))

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


def _require_run_session(runtime_config: RuntimeConfig | None) -> str:
    if runtime_config is None or not str(runtime_config.run_session or "").strip():
        raise ValueError("Artifact paths require runtime_config.run_session to be set")
    return str(runtime_config.run_session).strip()


def _append_run_session(base_path: Path, runtime_config: RuntimeConfig | None = None) -> Path:
    return base_path / _require_run_session(runtime_config)


def is_smoke_run(runtime_config_or_run_name: RuntimeConfig | str | None = None) -> bool:
    if isinstance(runtime_config_or_run_name, RuntimeConfig):
        run_name = runtime_config_or_run_name.run_name
    else:
        run_name = runtime_config_or_run_name
    return str(run_name or FOLDER_NAME).strip() == SMOKE_FOLDER_NAME


def get_result_root() -> Path:
    return result_path


def get_result_split(runtime_config: RuntimeConfig | None = None) -> str:
    return "smoke" if is_smoke_run(runtime_config) else "train"


def get_result_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(get_result_root() / get_result_split(runtime_config) / "gifs", runtime_config)


def get_result_eval_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(get_result_root() / get_result_split(runtime_config) / "eval", runtime_config)


def get_monitor_state_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_eval_path(runtime_config) / ".state"


def get_model_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(model_path, runtime_config)


def get_train_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(get_result_root() / get_result_split(runtime_config) / "tensorboard", runtime_config)


def get_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_gifs_path(runtime_config)


def get_monitor_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(get_result_root() / get_result_split(runtime_config) / "monitor", runtime_config)


def get_checkpoint_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint.pth"


def get_checkpoint_final_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint_final.pth"


def get_checkpoint_interrupted_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint_interrupted.pth"


_CHECKPOINT_NAMES = ("checkpoint_interrupted.pth", "checkpoint_final.pth", "checkpoint.pth")


def _session_has_any_checkpoint(session_dir: Path) -> bool:
    return any((session_dir / name).is_file() for name in _CHECKPOINT_NAMES)


def iter_checkpoint_candidates(run_name: str = FOLDER_NAME) -> list[Path]:
    """Return candidate checkpoint paths ordered by priority (newest session
    first, within each session: interrupted > final > regular), matching the
    convention used by ``new_code``.
    """
    if not model_path.exists():
        return [model_path / "checkpoint.pth"]

    session_dirs = sorted(
        [
            path
            for path in model_path.iterdir()
            if path.is_dir()
            and not path.name.endswith("_smoke")
            and _session_has_any_checkpoint(path)
        ],
        key=lambda path: path.name,
        reverse=True,
    )

    # Apply run-name filter when the caller asks for a specific run suffix.
    if run_name not in (FOLDER_NAME, SMOKE_FOLDER_NAME, "", None):
        suffix = _get_run_suffix(run_name)
        filtered = [p for p in session_dirs if p.name.endswith(suffix)]
        if filtered:
            session_dirs = filtered

    candidates: list[Path] = []
    for session_dir in session_dirs:
        for name in _CHECKPOINT_NAMES:
            candidates.append(session_dir / name)
    return candidates or [model_path / "checkpoint.pth"]


def get_latest_checkpoint_path(run_name: str = FOLDER_NAME) -> Path:
    """Return the most recent existing checkpoint across all sessions.

    Search order per session (newest first):
    ``checkpoint_interrupted.pth`` → ``checkpoint_final.pth`` → ``checkpoint.pth``
    """
    for candidate in iter_checkpoint_candidates(run_name):
        if candidate.exists():
            return candidate
    return model_path / "checkpoint.pth"


def get_run_identity_from_checkpoint(checkpoint_file: str | Path) -> tuple[str, str | None]:
    checkpoint_file = Path(checkpoint_file).resolve()
    model_root = model_path.resolve()
    try:
        relative = checkpoint_file.relative_to(model_root)
    except ValueError:
        return FOLDER_NAME, checkpoint_file.stem

    if not relative.parts:
        return FOLDER_NAME, checkpoint_file.stem

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


def resolve_resume_checkpoint(checkpoint_file: str | Path) -> tuple[Path, str]:
    checkpoint_file = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise ValueError(f"Resume checkpoint does not exist or is not a file: {checkpoint_file}")

    model_root = model_path.resolve()
    try:
        relative = checkpoint_file.relative_to(model_root)
    except ValueError as exc:
        raise ValueError(f"Resume checkpoint must be inside {model_root}") from exc

    if len(relative.parts) != 2 or relative.parts[1] not in _CHECKPOINT_NAMES:
        raise ValueError(
            f"Resume checkpoint must match model/<run_session>/<checkpoint_name>.pth: {checkpoint_file}"
        )

    run_name, run_session = get_run_identity_from_checkpoint(checkpoint_file)
    if run_session is None or is_smoke_run(run_name):
        raise ValueError(f"Resume checkpoint must be a normal training checkpoint: {checkpoint_file}")
    return checkpoint_file, run_session


def ensure_result_dirs(runtime_config: RuntimeConfig | None = None) -> None:
    get_result_gifs_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_result_eval_path(runtime_config).mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(runtime_config: RuntimeConfig | None = None) -> None:
    ensure_result_dirs(runtime_config)
    get_train_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_monitor_path(runtime_config).mkdir(parents=True, exist_ok=True)
    if not is_smoke_run(runtime_config):
        get_model_path(runtime_config).mkdir(parents=True, exist_ok=True)
