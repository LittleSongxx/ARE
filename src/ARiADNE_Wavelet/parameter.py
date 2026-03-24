from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import re


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = SRC_ROOT.parent


def _resolve_maps_dir() -> Path:
    override = os.environ.get("ARIADNE_MAPS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    candidates = [
        PACKAGE_ROOT / "maps",
        SRC_ROOT / "maps",
        WORKSPACE_ROOT / "maps",
        SRC_ROOT / "ARiADNE_curr" / "maps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[2]


MAPS_DIR = _resolve_maps_dir()


# saving path
FOLDER_NAME = "ariadne_wavelet"
SMOKE_FOLDER_NAME = f"{FOLDER_NAME}_smoke"
RESULT_ROOT = PACKAGE_ROOT / "result"
RESULT_BUCKET_EPISODES = 100


# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 100
SAVE_MODEL_GAP = 32
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
MAX_EPISODES = 12000
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 100000
MINIMUM_BUFFER_SIZE = 10000
BATCH_SIZE = 512
LR = 3e-5
GAMMA = 1
NUM_META_AGENT = 48
TRAIN_UPDATES_PER_ITER = 10


# network parameters
NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128


# graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 360


# GPU usage
# Rollout workers are always pinned to CPU. Only learner may use CUDA.
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 3


# wavelet history branch parameters
ENABLE_WAVELET_HISTORY = True
HISTORY_LEN = 8
HISTORY_FEATURE_SET = (
    "utility_sum",
    "utility_mean",
    "frontier_ratio",
    "visited_ratio",
    "frontier_count",
    "frontier_count_delta",
    "reward_proxy",
    "explored_delta",
    "travel_delta",
    "action_repeat",
    "oscillation",
    "utility_to_cost",
)
HISTORY_INPUT_DIM = len(HISTORY_FEATURE_SET)
HISTORY_WAVELET_LEVELS = 2
HISTORY_EMBED_DIM = 64
HISTORY_ENCODER_MODE = "wavelet_split"


# wavelet utility auxiliary loss parameters
ENABLE_WAVELET_UTILITY_LOSS = True
UTILITY_TARGET_TYPE = "n_step_return"
UTILITY_TARGET_HORIZON = 4
UTILITY_TARGET_GAMMA = 1.0
UTILITY_LOSS_MODE = "spatial2d"
UTILITY_LOSS_WEIGHT = 1.0
UTILITY_PATCH_SIZE = 5
UTILITY_PATCH_SIGMA = 0.5
UTILITY_AUX_LOSS_TYPE = "smoothl1"
UTILITY_AUX_BASE_WEIGHT = 0.10
UTILITY_AUX_WAVELET_WEIGHT = 0.05
UTILITY_WAVELET_LEVELS = 2
UTILITY_WAVELET_RHO = 1.0
REPLAY_CHANNELS = 18

_VALID_HISTORY_ENCODER_MODES = {"mlp_only", "wavelet_shared", "wavelet_split"}
_VALID_UTILITY_TARGET_TYPES = {"td_bootstrap", "n_step_return"}
_VALID_UTILITY_LOSS_MODES = {"basic", "spatial2d"}
_VALID_AUX_LOSS_TYPES = {"mse", "smoothl1"}


def _normalize_history_feature_set(raw_features: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if raw_features is None:
        return HISTORY_FEATURE_SET
    if isinstance(raw_features, str):
        values = [feature.strip() for feature in raw_features.split(",") if feature.strip()]
    else:
        values = [str(feature).strip() for feature in raw_features if str(feature).strip()]
    return tuple(values) if values else HISTORY_FEATURE_SET


def _normalize_choice(value: str, valid_values: set[str], default_value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in valid_values:
        return normalized
    return default_value


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
    save_model_gap: int = SAVE_MODEL_GAP
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
    enable_auto_eval: bool = False
    auto_eval_episodes: int = 1
    auto_eval_greedy: bool = True
    auto_eval_device: str = "cpu"
    run_name: str = FOLDER_NAME
    run_session: str | None = None

    enable_wavelet_history: bool = ENABLE_WAVELET_HISTORY
    history_len: int = HISTORY_LEN
    history_input_dim: int = HISTORY_INPUT_DIM
    history_wavelet_levels: int = HISTORY_WAVELET_LEVELS
    history_embed_dim: int = HISTORY_EMBED_DIM
    history_feature_set: tuple[str, ...] | str = HISTORY_FEATURE_SET
    history_encoder_mode: str = HISTORY_ENCODER_MODE

    enable_wavelet_utility_loss: bool = ENABLE_WAVELET_UTILITY_LOSS
    utility_target_type: str = UTILITY_TARGET_TYPE
    utility_target_horizon: int = UTILITY_TARGET_HORIZON
    utility_target_gamma: float = UTILITY_TARGET_GAMMA
    utility_loss_mode: str = UTILITY_LOSS_MODE
    utility_loss_weight: float = UTILITY_LOSS_WEIGHT
    utility_patch_size: int = UTILITY_PATCH_SIZE
    utility_patch_sigma: float = UTILITY_PATCH_SIGMA
    utility_aux_loss_type: str = UTILITY_AUX_LOSS_TYPE
    utility_aux_base_weight: float = UTILITY_AUX_BASE_WEIGHT
    utility_aux_wavelet_weight: float = UTILITY_AUX_WAVELET_WEIGHT
    utility_wavelet_levels: int = UTILITY_WAVELET_LEVELS
    utility_wavelet_rho: float = UTILITY_WAVELET_RHO
    replay_channels: int = REPLAY_CHANNELS

    def __post_init__(self) -> None:
        history_feature_set = _normalize_history_feature_set(self.history_feature_set)
        object.__setattr__(self, "history_feature_set", history_feature_set)
        object.__setattr__(self, "gif_frame_rate", max(float(self.gif_frame_rate), 1e-3))
        object.__setattr__(self, "history_len", max(int(self.history_len), 1))
        history_input_dim = int(self.history_input_dim)
        if history_input_dim <= 0:
            history_input_dim = len(history_feature_set)
        object.__setattr__(self, "history_input_dim", max(history_input_dim, 1))
        object.__setattr__(self, "history_wavelet_levels", max(int(self.history_wavelet_levels), 1))
        object.__setattr__(self, "history_embed_dim", max(int(self.history_embed_dim), 8))
        object.__setattr__(
            self,
            "history_encoder_mode",
            _normalize_choice(self.history_encoder_mode, _VALID_HISTORY_ENCODER_MODES, HISTORY_ENCODER_MODE),
        )
        object.__setattr__(
            self,
            "utility_target_type",
            _normalize_choice(self.utility_target_type, _VALID_UTILITY_TARGET_TYPES, UTILITY_TARGET_TYPE),
        )
        object.__setattr__(
            self,
            "utility_loss_mode",
            _normalize_choice(self.utility_loss_mode, _VALID_UTILITY_LOSS_MODES, UTILITY_LOSS_MODE),
        )
        object.__setattr__(
            self,
            "utility_aux_loss_type",
            _normalize_choice(self.utility_aux_loss_type, _VALID_AUX_LOSS_TYPES, UTILITY_AUX_LOSS_TYPE),
        )
        object.__setattr__(self, "utility_target_horizon", max(int(self.utility_target_horizon), 1))
        object.__setattr__(self, "utility_target_gamma", float(self.utility_target_gamma))
        object.__setattr__(self, "utility_loss_weight", max(float(self.utility_loss_weight), 0.0))
        utility_patch_size = max(int(self.utility_patch_size), 3)
        if utility_patch_size % 2 == 0:
            utility_patch_size += 1
        object.__setattr__(self, "utility_patch_size", utility_patch_size)
        object.__setattr__(self, "utility_patch_sigma", max(float(self.utility_patch_sigma), 1e-4))
        object.__setattr__(self, "utility_aux_base_weight", max(float(self.utility_aux_base_weight), 0.0))
        object.__setattr__(self, "utility_aux_wavelet_weight", max(float(self.utility_aux_wavelet_weight), 0.0))
        object.__setattr__(self, "utility_wavelet_levels", max(int(self.utility_wavelet_levels), 1))
        object.__setattr__(self, "utility_wavelet_rho", max(float(self.utility_wavelet_rho), 0.0))
        object.__setattr__(self, "replay_channels", max(int(self.replay_channels), REPLAY_CHANNELS))

    def with_overrides(self, **kwargs: object) -> "RuntimeConfig":
        return replace(self, **kwargs)


def _get_run_suffix(run_name: str | None) -> str:
    if run_name in (None, "", FOLDER_NAME):
        return ""
    if run_name == SMOKE_FOLDER_NAME:
        return "_smoke"
    return f"_{run_name}"


def build_run_session(run_name: str = FOLDER_NAME) -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M") + _get_run_suffix(run_name)


def is_smoke_run(runtime_config_or_run_name: RuntimeConfig | str | None = None) -> bool:
    if isinstance(runtime_config_or_run_name, RuntimeConfig):
        run_name = runtime_config_or_run_name.run_name
    else:
        run_name = runtime_config_or_run_name
    return str(run_name or FOLDER_NAME).strip() == SMOKE_FOLDER_NAME


def _require_run_session(runtime_config: RuntimeConfig | None) -> str:
    if runtime_config is None or not str(runtime_config.run_session or "").strip():
        raise ValueError("Artifact paths require runtime_config.run_session to be set")
    return str(runtime_config.run_session).strip()


def get_result_session_root(runtime_config: RuntimeConfig | None = None) -> Path:
    return RESULT_ROOT / _require_run_session(runtime_config)


def get_result_train_root(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_session_root(runtime_config) / "train"


def get_result_test_root(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_session_root(runtime_config) / "test"


def get_model_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_train_root(runtime_config) / "model"


def get_train_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_train_root(runtime_config) / "tensorboard"


def get_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_train_root(runtime_config) / "gifs"


def get_monitor_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_train_root(runtime_config) / "monitor"


def get_result_eval_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_test_root(runtime_config) / "eval"


def get_result_eval_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_test_root(runtime_config) / "gifs"


def get_checkpoint_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint.pth"


def get_checkpoint_final_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint_final.pth"


def get_checkpoint_interrupted_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_model_path(runtime_config) / "checkpoint_interrupted.pth"


_CHECKPOINT_NAMES = ("checkpoint_interrupted.pth", "checkpoint_final.pth", "checkpoint.pth")


def _session_has_any_checkpoint(session_dir: Path) -> bool:
    model_dir = session_dir / "train" / "model"
    return any((model_dir / name).is_file() for name in _CHECKPOINT_NAMES)


def iter_checkpoint_candidates(run_name: str = FOLDER_NAME) -> list[Path]:
    if not RESULT_ROOT.exists():
        return [RESULT_ROOT / "checkpoint.pth"]

    session_dirs = sorted(
        [
            path
            for path in RESULT_ROOT.iterdir()
            if path.is_dir() and not path.name.endswith("_smoke") and _session_has_any_checkpoint(path)
        ],
        key=lambda path: path.name,
        reverse=True,
    )

    if run_name not in (FOLDER_NAME, SMOKE_FOLDER_NAME, "", None):
        suffix = _get_run_suffix(run_name)
        filtered = [p for p in session_dirs if p.name.endswith(suffix)]
        if filtered:
            session_dirs = filtered

    candidates: list[Path] = []
    for session_dir in session_dirs:
        model_dir = session_dir / "train" / "model"
        for name in _CHECKPOINT_NAMES:
            candidates.append(model_dir / name)

    return candidates or [RESULT_ROOT / "checkpoint.pth"]


def get_latest_checkpoint_path(run_name: str = FOLDER_NAME) -> Path:
    for candidate in iter_checkpoint_candidates(run_name):
        if candidate.exists():
            return candidate
    return RESULT_ROOT / "checkpoint.pth"


def get_run_identity_from_checkpoint(checkpoint_file: str | Path) -> tuple[str, str | None]:
    checkpoint_file = Path(checkpoint_file).resolve()
    try:
        relative = checkpoint_file.relative_to(RESULT_ROOT.resolve())
    except ValueError:
        return FOLDER_NAME, checkpoint_file.stem

    if len(relative.parts) < 3:
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

    try:
        relative = checkpoint_file.relative_to(RESULT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Resume checkpoint must be inside {RESULT_ROOT.resolve()}") from exc

    if len(relative.parts) != 4:
        raise ValueError(
            f"Resume checkpoint must match result/<run_session>/train/model/<checkpoint_name>.pth: {checkpoint_file}"
        )
    run_session, split, model_dir, checkpoint_name = relative.parts
    if split != "train" or model_dir != "model" or checkpoint_name not in _CHECKPOINT_NAMES:
        raise ValueError(
            f"Resume checkpoint must match result/<run_session>/train/model/<checkpoint_name>.pth: {checkpoint_file}"
        )

    run_name, parsed_session = get_run_identity_from_checkpoint(checkpoint_file)
    if parsed_session is None or is_smoke_run(run_name):
        raise ValueError(f"Resume checkpoint must be a normal training checkpoint: {checkpoint_file}")
    return checkpoint_file, parsed_session


def ensure_output_dirs(runtime_config: RuntimeConfig | None = None) -> None:
    get_gifs_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_train_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_monitor_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_result_eval_path(runtime_config).mkdir(parents=True, exist_ok=True)
    get_result_eval_gifs_path(runtime_config).mkdir(parents=True, exist_ok=True)
    if not is_smoke_run(runtime_config):
        get_model_path(runtime_config).mkdir(parents=True, exist_ok=True)
