from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
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

    candidates = (
        PACKAGE_ROOT / "maps",
        BASELINE_ROOT / "maps",
        SRC_ROOT / "maps",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


MAPS_DIR = _resolve_maps_dir()


# saving path
FOLDER_NAME = "ARiADNE_wavelet_run1"
SMOKE_FOLDER_NAME = f"{FOLDER_NAME}_smoke"
model_path = PACKAGE_ROOT / "model"
result_path = PACKAGE_ROOT / "result"
checkpoint_path = model_path / "checkpoint.pth"
RESULT_BUCKET_EPISODES = 100
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
MAX_EPISODES = 15000
MAX_EPISODE_STEP = 120
REPLAY_SIZE = 80000
MINIMUM_BUFFER_SIZE = 32768
BATCH_SIZE = 3584
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 63
TRAIN_UPDATES_PER_ITER = 12


# network parameters
BASE_NODE_INPUT_DIM = 4
CRITIC_EXTRA_INPUT_DIM = 1
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
USE_WAVELET_FEATURE = False
WAVELET_FEATURE_MODE = "scalar"
WAVELET_LOCAL_POOL = "none"
WAVELET_LOCAL_POOL_RADIUS_CELLS = 1
WAVELET_SCALES_AUTO = False
WAVELET_SCALES = (1, 2, 4)
WAVELET_SCALES_AUTO_MULTS = (1, 2, 4)
WAVELET_NORM_METHOD = "minmax"
WAVELET_CLIP_PERCENTILE = 99.0
WAVELET_FIXED_CLIP_VALUE = 1.0
WAVELET_EPS = 1e-6

USE_WAVELET_ATTN_BIAS = False
WAVELET_ATTN_BIAS_TYPE = "sim_exp"
WAVELET_ATTN_BIAS_BETA = 0.5
WAVELET_ATTN_BIAS_SIGMA = 0.25
WAVELET_ATTN_BIAS_APPLY_ON_MASKED_EDGES_ONLY = True

WAVELET_SKIP_UTILITY_UPDATES = False
WAVELET_SKIP_THRESH = 0.2
WAVELET_SKIP_UTILITY_LOW = float(MIN_UTILITY)
WAVELET_SKIP_MAX_AGE_STEPS = 3
WAVELET_SKIP_NEAR_ROBOT_RADIUS = 2.0 * NODE_RESOLUTION

WAVELET_GUIDED_NODE_SAMPLING = False
WAVELET_NODE_KEEP_THRESH = 0.35
WAVELET_NODE_KEEP_PROB_LOW = 0.25
WAVELET_NODE_MIN_KEEP = 32
WAVELET_NODE_ALWAYS_KEEP_CURRENT_AND_NEIGHBORS = True

WAVELET_ADAPTIVE_DTH = False
WAVELET_DTH_ALPHA = 1.0
WAVELET_DTH_MAX_MULT = 2.0


# GPU usage
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0


_FEATURE_MODES = {"scalar", "scales", "scales_orient"}
_POOL_MODES = {"none", "mean", "max"}
_NORM_METHODS = {"minmax", "percentile", "log_percentile", "fixed_clip"}
_ATTN_BIAS_TYPES = {"sim_exp", "neg_l1", "neg_l2"}
_CHECKPOINT_NAMES = ("checkpoint_interrupted.pth", "checkpoint_final.pth", "checkpoint.pth")
_MODEL_CONFIG_KEYS = (
    "use_wavelet_feature",
    "wavelet_feature_mode",
    "wavelet_local_pool",
    "wavelet_local_pool_radius_cells",
    "wavelet_scales_auto",
    "wavelet_scales",
    "wavelet_scales_auto_mults",
    "wavelet_norm_method",
    "wavelet_clip_percentile",
    "wavelet_fixed_clip_value",
    "wavelet_eps",
    "use_wavelet_attn_bias",
    "wavelet_attn_bias_type",
    "wavelet_attn_bias_beta",
    "wavelet_attn_bias_sigma",
    "wavelet_attn_bias_apply_on_masked_edges_only",
)


def _normalize_int_tuple(values: tuple[int, ...] | list[int] | None, fallback: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(max(int(value), 1) for value in (values or fallback))
    if not normalized:
        return tuple(fallback)
    return normalized


def _normalize_feature_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _FEATURE_MODES:
        raise ValueError(f"Unsupported WAVELET_FEATURE_MODE: {value}")
    return normalized


def _normalize_pool_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _POOL_MODES:
        raise ValueError(f"Unsupported WAVELET_LOCAL_POOL: {value}")
    return normalized


def _normalize_norm_method(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _NORM_METHODS:
        raise ValueError(f"Unsupported WAVELET_NORM_METHOD: {value}")
    return normalized


def _normalize_attn_bias_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _ATTN_BIAS_TYPES:
        raise ValueError(f"Unsupported WAVELET_ATTN_BIAS_TYPE: {value}")
    return normalized


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
    use_wavelet_feature: bool = USE_WAVELET_FEATURE
    wavelet_feature_mode: str = WAVELET_FEATURE_MODE
    wavelet_local_pool: str = WAVELET_LOCAL_POOL
    wavelet_local_pool_radius_cells: int = WAVELET_LOCAL_POOL_RADIUS_CELLS
    wavelet_scales_auto: bool = WAVELET_SCALES_AUTO
    wavelet_scales: tuple[int, ...] = tuple(WAVELET_SCALES)
    wavelet_scales_auto_mults: tuple[int, ...] = tuple(WAVELET_SCALES_AUTO_MULTS)
    wavelet_norm_method: str = WAVELET_NORM_METHOD
    wavelet_clip_percentile: float = WAVELET_CLIP_PERCENTILE
    wavelet_fixed_clip_value: float = WAVELET_FIXED_CLIP_VALUE
    wavelet_eps: float = WAVELET_EPS
    use_wavelet_attn_bias: bool = USE_WAVELET_ATTN_BIAS
    wavelet_attn_bias_type: str = WAVELET_ATTN_BIAS_TYPE
    wavelet_attn_bias_beta: float = WAVELET_ATTN_BIAS_BETA
    wavelet_attn_bias_sigma: float = WAVELET_ATTN_BIAS_SIGMA
    wavelet_attn_bias_apply_on_masked_edges_only: bool = WAVELET_ATTN_BIAS_APPLY_ON_MASKED_EDGES_ONLY
    wavelet_skip_utility_updates: bool = WAVELET_SKIP_UTILITY_UPDATES
    wavelet_skip_thresh: float = WAVELET_SKIP_THRESH
    wavelet_skip_utility_low: float = WAVELET_SKIP_UTILITY_LOW
    wavelet_skip_max_age_steps: int = WAVELET_SKIP_MAX_AGE_STEPS
    wavelet_skip_near_robot_radius: float = WAVELET_SKIP_NEAR_ROBOT_RADIUS
    wavelet_guided_node_sampling: bool = WAVELET_GUIDED_NODE_SAMPLING
    wavelet_node_keep_thresh: float = WAVELET_NODE_KEEP_THRESH
    wavelet_node_keep_prob_low: float = WAVELET_NODE_KEEP_PROB_LOW
    wavelet_node_min_keep: int = WAVELET_NODE_MIN_KEEP
    wavelet_node_always_keep_current_and_neighbors: bool = WAVELET_NODE_ALWAYS_KEEP_CURRENT_AND_NEIGHBORS
    wavelet_adaptive_dth: bool = WAVELET_ADAPTIVE_DTH
    wavelet_dth_alpha: float = WAVELET_DTH_ALPHA
    wavelet_dth_max_mult: float = WAVELET_DTH_MAX_MULT

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_episodes", max(int(self.max_episodes), 1))
        object.__setattr__(self, "num_meta_agent", max(int(self.num_meta_agent), 1))
        object.__setattr__(self, "max_episode_step", max(int(self.max_episode_step), 1))
        object.__setattr__(self, "minimum_buffer_size", max(int(self.minimum_buffer_size), 1))
        object.__setattr__(self, "batch_size", max(int(self.batch_size), 1))
        object.__setattr__(self, "replay_size", max(int(self.replay_size), 1))
        object.__setattr__(self, "save_img_gap", max(int(self.save_img_gap), 1))
        object.__setattr__(self, "summary_window", max(int(self.summary_window), 1))
        object.__setattr__(self, "train_updates_per_iter", max(int(self.train_updates_per_iter), 1))
        object.__setattr__(self, "result_bucket_episodes", max(int(self.result_bucket_episodes), 1))
        object.__setattr__(self, "auto_eval_episodes", max(int(self.auto_eval_episodes), 1))
        object.__setattr__(self, "enable_training_monitor", bool(self.enable_training_monitor))
        object.__setattr__(self, "enable_auto_eval", bool(self.enable_auto_eval))
        object.__setattr__(self, "auto_eval_greedy", bool(self.auto_eval_greedy))
        object.__setattr__(self, "use_fixed_eval_maps", bool(self.use_fixed_eval_maps))
        object.__setattr__(self, "use_wavelet_feature", bool(self.use_wavelet_feature))
        object.__setattr__(self, "wavelet_feature_mode", _normalize_feature_mode(self.wavelet_feature_mode))
        object.__setattr__(self, "wavelet_local_pool", _normalize_pool_mode(self.wavelet_local_pool))
        object.__setattr__(
            self,
            "wavelet_local_pool_radius_cells",
            max(int(self.wavelet_local_pool_radius_cells), 0),
        )
        object.__setattr__(self, "wavelet_scales_auto", bool(self.wavelet_scales_auto))
        object.__setattr__(
            self,
            "wavelet_scales",
            _normalize_int_tuple(self.wavelet_scales, tuple(WAVELET_SCALES)),
        )
        object.__setattr__(
            self,
            "wavelet_scales_auto_mults",
            _normalize_int_tuple(self.wavelet_scales_auto_mults, tuple(WAVELET_SCALES_AUTO_MULTS)),
        )
        object.__setattr__(self, "wavelet_norm_method", _normalize_norm_method(self.wavelet_norm_method))
        object.__setattr__(self, "wavelet_clip_percentile", float(self.wavelet_clip_percentile))
        object.__setattr__(self, "wavelet_fixed_clip_value", float(self.wavelet_fixed_clip_value))
        object.__setattr__(self, "wavelet_eps", max(float(self.wavelet_eps), 1e-12))
        object.__setattr__(self, "use_wavelet_attn_bias", bool(self.use_wavelet_attn_bias))
        object.__setattr__(
            self,
            "wavelet_attn_bias_type",
            _normalize_attn_bias_type(self.wavelet_attn_bias_type),
        )
        object.__setattr__(self, "wavelet_attn_bias_beta", float(self.wavelet_attn_bias_beta))
        object.__setattr__(self, "wavelet_attn_bias_sigma", max(float(self.wavelet_attn_bias_sigma), 1e-6))
        object.__setattr__(
            self,
            "wavelet_attn_bias_apply_on_masked_edges_only",
            bool(self.wavelet_attn_bias_apply_on_masked_edges_only),
        )
        object.__setattr__(self, "wavelet_skip_utility_updates", bool(self.wavelet_skip_utility_updates))
        object.__setattr__(self, "wavelet_skip_thresh", float(self.wavelet_skip_thresh))
        object.__setattr__(self, "wavelet_skip_utility_low", float(self.wavelet_skip_utility_low))
        object.__setattr__(self, "wavelet_skip_max_age_steps", max(int(self.wavelet_skip_max_age_steps), 0))
        object.__setattr__(
            self,
            "wavelet_skip_near_robot_radius",
            max(float(self.wavelet_skip_near_robot_radius), 0.0),
        )
        object.__setattr__(self, "wavelet_guided_node_sampling", bool(self.wavelet_guided_node_sampling))
        object.__setattr__(self, "wavelet_node_keep_thresh", float(self.wavelet_node_keep_thresh))
        object.__setattr__(self, "wavelet_node_keep_prob_low", float(self.wavelet_node_keep_prob_low))
        object.__setattr__(self, "wavelet_node_min_keep", max(int(self.wavelet_node_min_keep), 1))
        object.__setattr__(
            self,
            "wavelet_node_always_keep_current_and_neighbors",
            bool(self.wavelet_node_always_keep_current_and_neighbors),
        )
        object.__setattr__(self, "wavelet_adaptive_dth", bool(self.wavelet_adaptive_dth))
        object.__setattr__(self, "wavelet_dth_alpha", float(self.wavelet_dth_alpha))
        object.__setattr__(self, "wavelet_dth_max_mult", max(float(self.wavelet_dth_max_mult), 1.0))

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


def resolve_wavelet_scales(runtime_config: RuntimeConfig | None = None) -> tuple[int, ...]:
    runtime_config = runtime_config or RuntimeConfig()
    if runtime_config.wavelet_scales_auto:
        base_scale = max(1, int(round(NODE_RESOLUTION / CELL_SIZE)))
        return tuple(
            sorted(
                {
                    max(1, int(round(base_scale * mult)))
                    for mult in runtime_config.wavelet_scales_auto_mults
                }
            )
        )
    return runtime_config.wavelet_scales


def get_wavelet_feature_dim(runtime_config: RuntimeConfig | None = None) -> int:
    runtime_config = runtime_config or RuntimeConfig()
    if not runtime_config.use_wavelet_feature:
        return 0
    scales = resolve_wavelet_scales(runtime_config)
    if runtime_config.wavelet_feature_mode == "scalar":
        return 1
    if runtime_config.wavelet_feature_mode == "scales":
        return len(scales)
    return len(scales) * 3


def get_node_input_dim(runtime_config: RuntimeConfig | None = None) -> int:
    return BASE_NODE_INPUT_DIM + get_wavelet_feature_dim(runtime_config)


def get_critic_node_input_dim(runtime_config: RuntimeConfig | None = None) -> int:
    return BASE_NODE_INPUT_DIM + CRITIC_EXTRA_INPUT_DIM + get_wavelet_feature_dim(runtime_config)


def checkpoint_model_config(runtime_config: RuntimeConfig | None = None) -> dict[str, object]:
    runtime_config = runtime_config or RuntimeConfig()
    config_dict = asdict(runtime_config)
    model_config = {
        key: config_dict[key]
        for key in _MODEL_CONFIG_KEYS
    }
    model_config.update(
        {
            "resolved_wavelet_scales": resolve_wavelet_scales(runtime_config),
            "node_input_dim": get_node_input_dim(runtime_config),
            "critic_node_input_dim": get_critic_node_input_dim(runtime_config),
        }
    )
    return model_config


def ensure_checkpoint_compatible(
    checkpoint_payload: dict[str, object],
    runtime_config: RuntimeConfig | None = None,
) -> None:
    runtime_config = runtime_config or RuntimeConfig()
    checkpoint_config = checkpoint_payload.get("config_snapshot")
    if checkpoint_config is None:
        checkpoint_node_dim = checkpoint_payload.get("node_input_dim")
        checkpoint_critic_dim = checkpoint_payload.get("critic_node_input_dim")
        if checkpoint_node_dim is not None and int(checkpoint_node_dim) != get_node_input_dim(runtime_config):
            raise ValueError(
                "Checkpoint actor input dim does not match runtime config: "
                f"{checkpoint_node_dim} != {get_node_input_dim(runtime_config)}"
            )
        if checkpoint_critic_dim is not None and int(checkpoint_critic_dim) != get_critic_node_input_dim(runtime_config):
            raise ValueError(
                "Checkpoint critic input dim does not match runtime config: "
                f"{checkpoint_critic_dim} != {get_critic_node_input_dim(runtime_config)}"
            )
        return

    expected = checkpoint_model_config(runtime_config)
    for key, expected_value in expected.items():
        if checkpoint_config.get(key) != expected_value:
            raise ValueError(
                "Checkpoint model config mismatch for "
                f"{key}: {checkpoint_config.get(key)!r} != {expected_value!r}"
            )


def runtime_config_from_checkpoint(
    checkpoint_payload: dict[str, object],
    base_config: RuntimeConfig | None = None,
) -> RuntimeConfig:
    base_config = base_config or RuntimeConfig()
    checkpoint_config = checkpoint_payload.get("config_snapshot")
    if not isinstance(checkpoint_config, dict):
        return base_config
    overrides = {key: checkpoint_config[key] for key in _MODEL_CONFIG_KEYS if key in checkpoint_config}
    if not overrides:
        return base_config
    return base_config.with_overrides(**overrides)


NODE_INPUT_DIM = get_node_input_dim()
CRITIC_NODE_INPUT_DIM = get_critic_node_input_dim()


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


def _session_has_any_checkpoint(session_dir: Path) -> bool:
    return any((session_dir / name).is_file() for name in _CHECKPOINT_NAMES)


def iter_checkpoint_candidates(run_name: str = FOLDER_NAME) -> list[Path]:
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

    if run_name not in (FOLDER_NAME, SMOKE_FOLDER_NAME, "", None):
        suffix = _get_run_suffix(run_name)
        filtered = [path for path in session_dirs if path.name.endswith(suffix)]
        if filtered:
            session_dirs = filtered

    candidates: list[Path] = []
    for session_dir in session_dirs:
        for name in _CHECKPOINT_NAMES:
            candidates.append(session_dir / name)
    return candidates or [model_path / "checkpoint.pth"]


def get_latest_checkpoint_path(run_name: str = FOLDER_NAME) -> Path:
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


# Recommended presets
# Training:
# RuntimeConfig(
#     use_wavelet_feature=True,
#     wavelet_feature_mode="scales_orient",
#     wavelet_local_pool="mean",
#     wavelet_local_pool_radius_cells=1,
#     wavelet_scales_auto=True,
#     wavelet_scales_auto_mults=(1, 2, 4),
#     wavelet_norm_method="log_percentile",
#     wavelet_clip_percentile=99.0,
#     use_wavelet_attn_bias=True,
#     wavelet_attn_bias_type="sim_exp",
#     wavelet_attn_bias_beta=0.5,
#     wavelet_attn_bias_sigma=0.25,
#     wavelet_attn_bias_apply_on_masked_edges_only=True,
# )
#
# Inference:
# RuntimeConfig(
#     wavelet_skip_utility_updates=True,
#     wavelet_skip_thresh=0.2,
#     wavelet_skip_utility_low=float(MIN_UTILITY),
#     wavelet_skip_max_age_steps=3,
#     wavelet_skip_near_robot_radius=2.0 * NODE_RESOLUTION,
#     wavelet_guided_node_sampling=True,
#     wavelet_node_keep_thresh=0.35,
#     wavelet_node_keep_prob_low=0.25,
#     wavelet_node_min_keep=32,
#     wavelet_node_always_keep_current_and_neighbors=True,
#     wavelet_adaptive_dth=True,
# )
