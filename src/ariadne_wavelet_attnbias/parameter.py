from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field, replace
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
        SRC_ROOT / "ariadne_wavelet" / "maps",
        BASELINE_ROOT / "maps",
        SRC_ROOT / "maps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


MAPS_DIR = _resolve_maps_dir()


# saving path
FOLDER_NAME = "ariadne_wavelet_attnbias_run1"
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


# Stage-2 attention-bias parameters
USE_ATTENTION_BIAS = True
ATTN_BIAS_BETA = 0.5
ATTN_BIAS_MODE = "diff"
ATTN_BIAS_APPLY_ENCODER = True
ATTN_BIAS_APPLY_DECODER = False


# Optional RL strategies
USE_N_STEP_RETURN = False
N_STEP = 3

USE_REWARD_DECOMPOSITION = False
R_INFO_W = 1.0
R_DIST_W = 1.0
R_SAFE_W = 0.0
R_TERMINAL_BONUS = 20.0

USE_PRIVILEGED_DISTILL = False
DISTILL_LAMBDA = 0.1
DISTILL_TAU = 1.0
DISTILL_WARMUP_UPDATES = 1000

USE_CURRICULUM = False
CURRICULUM_MILESTONES = [0, 2000, 5000]
CURRICULUM_LEVELS = ["easy", "medium", "hard"]
CURRICULUM_MODE = "dir"
CURRICULUM_DIRS = {"easy": "easy", "medium": "medium", "hard": "hard"}
CURRICULUM_PATTERNS = {"easy": ["*easy*"], "medium": ["*medium*"], "hard": ["*hard*"]}
USE_CURRICULUM_IN_EVAL = False


# GPU usage
USE_GPU = True  # enable GPU inference inside Ray workers
USE_GPU_GLOBAL = True  # the learner uses CUDA and nn.DataParallel across visible GPUs
NUM_GPU = 3  # reserve 3 GPUs for Ray workers; keep CUDA_VISIBLE_DEVICES aligned with the same 3 cards


def _normalize_curriculum_milestones(values: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    milestones = tuple(int(value) for value in (values or (0,)))
    if not milestones:
        milestones = (0,)
    if milestones[0] != 0:
        raise ValueError("CURRICULUM_MILESTONES must start at 0")
    if any(curr < prev for prev, curr in zip(milestones, milestones[1:])):
        raise ValueError("CURRICULUM_MILESTONES must be non-decreasing")
    return milestones


def _normalize_curriculum_levels(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    levels = tuple(str(value).strip() for value in (values or ("easy",)))
    if not levels or any(level == "" for level in levels):
        raise ValueError("CURRICULUM_LEVELS must contain non-empty strings")
    return levels


def _normalize_curriculum_dirs(values: dict[str, str] | tuple[tuple[str, str], ...] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        values = {}
    if isinstance(values, tuple):
        items = values
    else:
        items = tuple(values.items())
    normalized = []
    for key, value in items:
        normalized.append((str(key).strip(), str(value).strip()))
    return tuple(normalized)


def _normalize_curriculum_patterns(
    values: dict[str, list[str] | tuple[str, ...]] | tuple[tuple[str, tuple[str, ...]], ...] | None
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if values is None:
        values = {}
    if isinstance(values, tuple):
        items = values
    else:
        items = tuple(values.items())
    normalized = []
    for key, patterns in items:
        if isinstance(patterns, tuple):
            pattern_values = patterns
        else:
            pattern_values = tuple(patterns)
        normalized.append((str(key).strip(), tuple(str(pattern).strip() for pattern in pattern_values if str(pattern).strip())))
    return tuple(normalized)


@dataclass(frozen=True)
class RLOptions:
    use_n_step_return: bool = USE_N_STEP_RETURN
    n_step: int = N_STEP
    use_reward_decomposition: bool = USE_REWARD_DECOMPOSITION
    r_info_w: float = R_INFO_W
    r_dist_w: float = R_DIST_W
    r_safe_w: float = R_SAFE_W
    r_terminal_bonus: float = R_TERMINAL_BONUS
    use_privileged_distill: bool = USE_PRIVILEGED_DISTILL
    distill_lambda: float = DISTILL_LAMBDA
    distill_tau: float = DISTILL_TAU
    distill_warmup_updates: int = DISTILL_WARMUP_UPDATES
    use_curriculum: bool = USE_CURRICULUM
    curriculum_milestones: tuple[int, ...] = tuple(CURRICULUM_MILESTONES)
    curriculum_levels: tuple[str, ...] = tuple(CURRICULUM_LEVELS)
    curriculum_mode: str = CURRICULUM_MODE
    curriculum_dirs: tuple[tuple[str, str], ...] = tuple(CURRICULUM_DIRS.items())
    curriculum_patterns: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
        (key, tuple(value)) for key, value in CURRICULUM_PATTERNS.items()
    )
    use_curriculum_in_eval: bool = USE_CURRICULUM_IN_EVAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "use_n_step_return", bool(self.use_n_step_return))
        object.__setattr__(self, "n_step", max(int(self.n_step), 1))
        object.__setattr__(self, "use_reward_decomposition", bool(self.use_reward_decomposition))
        object.__setattr__(self, "r_info_w", float(self.r_info_w))
        object.__setattr__(self, "r_dist_w", float(self.r_dist_w))
        object.__setattr__(self, "r_safe_w", float(self.r_safe_w))
        object.__setattr__(self, "r_terminal_bonus", float(self.r_terminal_bonus))
        object.__setattr__(self, "use_privileged_distill", bool(self.use_privileged_distill))
        object.__setattr__(self, "distill_lambda", float(self.distill_lambda))
        object.__setattr__(self, "distill_tau", float(self.distill_tau))
        object.__setattr__(self, "distill_warmup_updates", max(int(self.distill_warmup_updates), 0))
        object.__setattr__(self, "use_curriculum", bool(self.use_curriculum))
        object.__setattr__(
            self,
            "curriculum_milestones",
            _normalize_curriculum_milestones(self.curriculum_milestones),
        )
        object.__setattr__(
            self,
            "curriculum_levels",
            _normalize_curriculum_levels(self.curriculum_levels),
        )
        if len(self.curriculum_milestones) != len(self.curriculum_levels):
            raise ValueError("CURRICULUM_MILESTONES and CURRICULUM_LEVELS must have the same length")
        curriculum_mode = str(self.curriculum_mode).strip().lower()
        if curriculum_mode not in {"dir", "pattern"}:
            raise ValueError(f"Unsupported curriculum mode: {self.curriculum_mode}")
        object.__setattr__(self, "curriculum_mode", curriculum_mode)
        object.__setattr__(
            self,
            "curriculum_dirs",
            _normalize_curriculum_dirs(self.curriculum_dirs),
        )
        object.__setattr__(
            self,
            "curriculum_patterns",
            _normalize_curriculum_patterns(self.curriculum_patterns),
        )
        object.__setattr__(self, "use_curriculum_in_eval", bool(self.use_curriculum_in_eval))

    def curriculum_dirs_map(self) -> dict[str, str]:
        return dict(self.curriculum_dirs)

    def curriculum_patterns_map(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(values) for key, values in self.curriculum_patterns}


def get_rl_options() -> RLOptions:
    return RLOptions(
        use_n_step_return=USE_N_STEP_RETURN,
        n_step=N_STEP,
        use_reward_decomposition=USE_REWARD_DECOMPOSITION,
        r_info_w=R_INFO_W,
        r_dist_w=R_DIST_W,
        r_safe_w=R_SAFE_W,
        r_terminal_bonus=R_TERMINAL_BONUS,
        use_privileged_distill=USE_PRIVILEGED_DISTILL,
        distill_lambda=DISTILL_LAMBDA,
        distill_tau=DISTILL_TAU,
        distill_warmup_updates=DISTILL_WARMUP_UPDATES,
        use_curriculum=USE_CURRICULUM,
        curriculum_milestones=tuple(CURRICULUM_MILESTONES),
        curriculum_levels=tuple(CURRICULUM_LEVELS),
        curriculum_mode=CURRICULUM_MODE,
        curriculum_dirs=tuple(CURRICULUM_DIRS.items()),
        curriculum_patterns=tuple((key, tuple(value)) for key, value in CURRICULUM_PATTERNS.items()),
        use_curriculum_in_eval=USE_CURRICULUM_IN_EVAL,
    )


def get_curriculum_level_index(episode_index: int, rl_options: RLOptions | None = None) -> int:
    options = rl_options or get_rl_options()
    index = 0
    episode_index = int(episode_index)
    for level_index, milestone in enumerate(options.curriculum_milestones):
        if episode_index >= milestone:
            index = level_index
        else:
            break
    return index


def get_curriculum_level(episode_index: int, rl_options: RLOptions | None = None) -> str:
    options = rl_options or get_rl_options()
    return options.curriculum_levels[get_curriculum_level_index(episode_index, options)]


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
    rl_options: RLOptions = field(default_factory=get_rl_options)

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


def configure_attention_bias(
    *,
    use_attention_bias: bool | None = None,
    attn_bias_beta: float | None = None,
    attn_bias_mode: str | None = None,
    attn_bias_apply_encoder: bool | None = None,
    attn_bias_apply_decoder: bool | None = None,
) -> None:
    global USE_ATTENTION_BIAS
    global ATTN_BIAS_BETA
    global ATTN_BIAS_MODE
    global ATTN_BIAS_APPLY_ENCODER
    global ATTN_BIAS_APPLY_DECODER

    if use_attention_bias is not None:
        USE_ATTENTION_BIAS = bool(use_attention_bias)

    if attn_bias_beta is not None:
        ATTN_BIAS_BETA = float(attn_bias_beta)

    if attn_bias_mode is not None:
        normalized_mode = str(attn_bias_mode).strip().lower()
        if normalized_mode not in {"diff", "open", "hybrid"}:
            raise ValueError(f"Unsupported attention bias mode: {attn_bias_mode}")
        ATTN_BIAS_MODE = normalized_mode

    if attn_bias_apply_encoder is not None:
        ATTN_BIAS_APPLY_ENCODER = bool(attn_bias_apply_encoder)

    if attn_bias_apply_decoder is not None:
        ATTN_BIAS_APPLY_DECODER = bool(attn_bias_apply_decoder)


def configure_rl_options(
    *,
    use_n_step_return: bool | None = None,
    n_step: int | None = None,
    use_reward_decomposition: bool | None = None,
    r_info_w: float | None = None,
    r_dist_w: float | None = None,
    r_safe_w: float | None = None,
    r_terminal_bonus: float | None = None,
    use_privileged_distill: bool | None = None,
    distill_lambda: float | None = None,
    distill_tau: float | None = None,
    distill_warmup_updates: int | None = None,
    use_curriculum: bool | None = None,
    curriculum_milestones: tuple[int, ...] | list[int] | None = None,
    curriculum_levels: tuple[str, ...] | list[str] | None = None,
    curriculum_mode: str | None = None,
    curriculum_dirs: dict[str, str] | tuple[tuple[str, str], ...] | None = None,
    curriculum_patterns: dict[str, list[str] | tuple[str, ...]] | tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    use_curriculum_in_eval: bool | None = None,
) -> None:
    global USE_N_STEP_RETURN
    global N_STEP
    global USE_REWARD_DECOMPOSITION
    global R_INFO_W
    global R_DIST_W
    global R_SAFE_W
    global R_TERMINAL_BONUS
    global USE_PRIVILEGED_DISTILL
    global DISTILL_LAMBDA
    global DISTILL_TAU
    global DISTILL_WARMUP_UPDATES
    global USE_CURRICULUM
    global CURRICULUM_MILESTONES
    global CURRICULUM_LEVELS
    global CURRICULUM_MODE
    global CURRICULUM_DIRS
    global CURRICULUM_PATTERNS
    global USE_CURRICULUM_IN_EVAL

    if use_n_step_return is not None:
        USE_N_STEP_RETURN = bool(use_n_step_return)
    if n_step is not None:
        N_STEP = max(int(n_step), 1)
    if use_reward_decomposition is not None:
        USE_REWARD_DECOMPOSITION = bool(use_reward_decomposition)
    if r_info_w is not None:
        R_INFO_W = float(r_info_w)
    if r_dist_w is not None:
        R_DIST_W = float(r_dist_w)
    if r_safe_w is not None:
        R_SAFE_W = float(r_safe_w)
    if r_terminal_bonus is not None:
        R_TERMINAL_BONUS = float(r_terminal_bonus)
    if use_privileged_distill is not None:
        USE_PRIVILEGED_DISTILL = bool(use_privileged_distill)
    if distill_lambda is not None:
        DISTILL_LAMBDA = float(distill_lambda)
    if distill_tau is not None:
        DISTILL_TAU = float(distill_tau)
    if distill_warmup_updates is not None:
        DISTILL_WARMUP_UPDATES = max(int(distill_warmup_updates), 0)
    if use_curriculum is not None:
        USE_CURRICULUM = bool(use_curriculum)
    if curriculum_milestones is not None:
        CURRICULUM_MILESTONES = list(_normalize_curriculum_milestones(curriculum_milestones))
    if curriculum_levels is not None:
        CURRICULUM_LEVELS = list(_normalize_curriculum_levels(curriculum_levels))
    if curriculum_mode is not None:
        normalized_mode = str(curriculum_mode).strip().lower()
        if normalized_mode not in {"dir", "pattern"}:
            raise ValueError(f"Unsupported curriculum mode: {curriculum_mode}")
        CURRICULUM_MODE = normalized_mode
    if curriculum_dirs is not None:
        CURRICULUM_DIRS = dict(_normalize_curriculum_dirs(curriculum_dirs))
    if curriculum_patterns is not None:
        CURRICULUM_PATTERNS = {
            key: list(values) for key, values in _normalize_curriculum_patterns(curriculum_patterns)
        }
    if use_curriculum_in_eval is not None:
        USE_CURRICULUM_IN_EVAL = bool(use_curriculum_in_eval)

    options = get_rl_options()
    if len(options.curriculum_milestones) != len(options.curriculum_levels):
        raise ValueError("CURRICULUM_MILESTONES and CURRICULUM_LEVELS must have the same length")
