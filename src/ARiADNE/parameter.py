from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import re


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent


def _resolve_maps_dir() -> Path:
    override = os.environ.get("ARIADNE_MAPS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    candidates = (
        PACKAGE_ROOT / "maps",
        SRC_ROOT / "maps",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
# The offline training server completes rollouts quickly, so reduce the
# visualization / evaluation cadence to keep disk I/O out of the hot path.
SAVE_IMG_GAP = 100
GIF_FRAME_RATE = 0.3


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
# Server profile:
# - 2 x A40 45 GB visible to the learner
# - 500 GB system memory, but keep some margin for Ray workers and buffers
# - Intel Xeon Gold 6342; rollouts stay CPU-only and are oversubscribed
#   through Ray, while the learner consumes the visible GPUs via DataParallel.
MAX_EPISODES = 12000
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 100000
MINIMUM_BUFFER_SIZE = 10000
BATCH_SIZE = 2048
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 40
TRAIN_UPDATES_PER_ITER = 10

# RL optimization parameters
ENABLE_NSTEP = True
N_STEP = 1
ENABLE_PER = True
PER_ALPHA = 0.6
PER_BETA0 = 0.4
PER_BETA_STEPS = 100000
PER_EPS = 1e-6
ENABLE_ADAPTIVE_ENTROPY_TARGET = True
ENTROPY_TARGET_SCALE = 0.05
ENABLE_SOFT_TARGET_UPDATE = True
TAU = 0.0
POLICY_DELAY = 1
REPLAY_RATIO = 0.0
ENABLE_REWARD_DECOMPOSITION = True
REWARD_INFO_WEIGHT = 1.0
REWARD_DIST_WEIGHT = 1.0
REWARD_SAFE_WEIGHT = 0.0
REWARD_TERMINAL_BONUS = 20.0
ENABLE_CURRICULUM = True
# 两档 coarse curriculum:
# - 0~4999: 纯 easy
# - 5000~7999: easy/hard 混采过渡
# - 8000+: 纯 hard
CURRICULUM_MILESTONES = (0, 5000)
CURRICULUM_LEVELS = ("easy", "hard")
CURRICULUM_SOURCE = None
# 进入新难度阶段后的混采窗口长度；0 表示直接硬切换。
CURRICULUM_MIX_WINDOW = 3000
USE_CURRICULUM_IN_EVAL = False


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
# 是否启用小波特征。开启后，节点输入会额外携带局部纹理/边缘能量信息。
USE_WAVELET_FEATURE = True
# 小波特征编码方式：
# - scalar: 单个综合强度
# - scales: 每个尺度一个强度
# - scales_orient: 每个尺度保留三个方向响应，信息最完整
WAVELET_FEATURE_MODE = "scales_orient"
# 是否对局部小波图做邻域池化，降低单点噪声。
WAVELET_LOCAL_POOL = "mean"
# 局部池化半径，单位为栅格数。
WAVELET_LOCAL_POOL_RADIUS_CELLS = 1
# 是否根据地图/节点分辨率自动推导尺度。
WAVELET_SCALES_AUTO = True
# 手动指定的小波尺度；当自动尺度关闭时生效。
WAVELET_SCALES = (1, 2, 4)
# 自动尺度相对基础尺度的倍率。
WAVELET_SCALES_AUTO_MULTS = (1, 2, 4)
# 小波图归一化方式；log_percentile 对异常值更稳。
WAVELET_NORM_METHOD = "log_percentile"
# 百分位裁剪上限，用于抑制极端响应。
WAVELET_CLIP_PERCENTILE = 99.0
# 固定裁剪模式下的裁剪值；当前默认模式下仅作保留配置。
WAVELET_FIXED_CLIP_VALUE = 1.0
# 防止归一化除零的数值稳定项。
WAVELET_EPS = 1e-6

# 是否把小波相似度转成 attention bias，引导图注意力更关注结构相近节点。
USE_WAVELET_ATTN_BIAS = True
# bias 计算形式。
WAVELET_ATTN_BIAS_TYPE = "sim_exp"
# bias 衰减/放大强度。
WAVELET_ATTN_BIAS_BETA = 0.5
# 高斯相似度里的 sigma，越小越强调局部差异。
WAVELET_ATTN_BIAS_SIGMA = 0.25
# 对 attention bias 做数值裁剪，0 表示不裁剪。
WAVELET_ATTN_BIAS_CLAMP = 0.0
# 是否只在原本被 mask 的边上施加 bias，避免改动已知连通关系。
WAVELET_ATTN_BIAS_APPLY_ON_MASKED_EDGES_ONLY = True

# 是否跳过变化很小的 utility 更新，减少重复计算。
WAVELET_SKIP_UTILITY_UPDATES = True
# 小波变化低于该阈值时可跳过更新。
WAVELET_SKIP_THRESH = 0.2
# 仅对 utility 不高的节点应用跳过策略，避免漏掉高价值 frontier。
WAVELET_SKIP_UTILITY_LOW = float(MIN_UTILITY)
# 连续最多跳过多少步后强制刷新。
WAVELET_SKIP_MAX_AGE_STEPS = 3
# 机器人附近半径内不跳过，保证近场决策足够及时。
WAVELET_SKIP_NEAR_ROBOT_RADIUS = 2.0 * NODE_RESOLUTION

# 是否用小波响应指导节点保留/裁剪，优先保留结构变化明显的区域。
WAVELET_GUIDED_NODE_SAMPLING = True
# 节点保留阈值；越高越偏向保留高响应节点。
WAVELET_NODE_KEEP_THRESH = 0.35
# 低响应节点的最小保留概率，避免采样过于激进。
WAVELET_NODE_KEEP_PROB_LOW = 0.25
# 无论如何都至少保留的节点数。
WAVELET_NODE_MIN_KEEP = 32
# 始终保留当前节点及其邻居，避免局部图断裂。
WAVELET_NODE_ALWAYS_KEEP_CURRENT_AND_NEIGHBORS = True

# 是否根据小波响应自适应调整建图距离阈值。
WAVELET_ADAPTIVE_DTH = True
# 自适应阈值的响应放大系数。
WAVELET_DTH_ALPHA = 1.0
# 自适应阈值的最大放大倍数。
WAVELET_DTH_MAX_MULT = 2.0


# GPU usage
USE_GPU = False
USE_GPU_GLOBAL = True
# learner 默认只使用两张可见 GPU。
NUM_GPU = 2


_FEATURE_MODES = {"scalar", "scales", "scales_orient"}
_POOL_MODES = {"none", "mean", "max"}
_NORM_METHODS = {"minmax", "percentile", "log_percentile", "fixed_clip"}
_ATTN_BIAS_TYPES = {"sim_exp", "neg_l1", "neg_l2", "diff", "product", "rbf"}
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
    "wavelet_attn_bias_clamp",
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


def _normalize_curriculum_milestones(
    values: tuple[int, ...] | list[int] | None,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    milestones = tuple(int(value) for value in (values or fallback))
    if not milestones:
        milestones = tuple(fallback) or (0,)
    if milestones[0] != 0:
        raise ValueError("CURRICULUM_MILESTONES must start at 0")
    if any(curr < prev for prev, curr in zip(milestones, milestones[1:])):
        raise ValueError("CURRICULUM_MILESTONES must be non-decreasing")
    return milestones


def _normalize_curriculum_levels(
    values: tuple[str, ...] | list[str] | None,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    raw_values = values if values is not None else fallback
    levels = tuple(str(value).strip() for value in raw_values if str(value).strip())
    if not levels:
        raise ValueError("CURRICULUM_LEVELS must contain at least one non-empty level")
    return levels


def _normalize_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return str(Path(normalized).expanduser().resolve())


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
    enable_nstep: bool = ENABLE_NSTEP
    n_step: int = N_STEP
    enable_per: bool = ENABLE_PER
    per_alpha: float = PER_ALPHA
    per_beta0: float = PER_BETA0
    per_beta_steps: int = PER_BETA_STEPS
    per_eps: float = PER_EPS
    enable_adaptive_entropy_target: bool = ENABLE_ADAPTIVE_ENTROPY_TARGET
    entropy_target_scale: float = ENTROPY_TARGET_SCALE
    enable_soft_target_update: bool = ENABLE_SOFT_TARGET_UPDATE
    tau: float = TAU
    policy_delay: int = POLICY_DELAY
    replay_ratio: float = REPLAY_RATIO
    enable_reward_decomposition: bool = ENABLE_REWARD_DECOMPOSITION
    reward_info_weight: float = REWARD_INFO_WEIGHT
    reward_dist_weight: float = REWARD_DIST_WEIGHT
    reward_safe_weight: float = REWARD_SAFE_WEIGHT
    reward_terminal_bonus: float = REWARD_TERMINAL_BONUS
    enable_curriculum: bool = ENABLE_CURRICULUM
    curriculum_milestones: tuple[int, ...] = tuple(CURRICULUM_MILESTONES)
    curriculum_levels: tuple[str, ...] = tuple(CURRICULUM_LEVELS)
    curriculum_source: str | None = CURRICULUM_SOURCE
    curriculum_mix_window: int = CURRICULUM_MIX_WINDOW
    use_curriculum_in_eval: bool = USE_CURRICULUM_IN_EVAL
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
    wavelet_attn_bias_clamp: float = WAVELET_ATTN_BIAS_CLAMP
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
        object.__setattr__(self, "gif_frame_rate", max(float(self.gif_frame_rate), 1e-3))
        object.__setattr__(self, "summary_window", max(int(self.summary_window), 1))
        object.__setattr__(self, "train_updates_per_iter", max(int(self.train_updates_per_iter), 1))
        object.__setattr__(self, "enable_nstep", bool(self.enable_nstep))
        object.__setattr__(self, "n_step", max(int(self.n_step), 1))
        object.__setattr__(self, "enable_per", bool(self.enable_per))
        object.__setattr__(self, "per_alpha", min(max(float(self.per_alpha), 0.0), 1.0))
        object.__setattr__(self, "per_beta0", min(max(float(self.per_beta0), 0.0), 1.0))
        object.__setattr__(self, "per_beta_steps", max(int(self.per_beta_steps), 1))
        object.__setattr__(self, "per_eps", max(float(self.per_eps), 1e-12))
        object.__setattr__(self, "enable_adaptive_entropy_target", bool(self.enable_adaptive_entropy_target))
        object.__setattr__(self, "entropy_target_scale", float(self.entropy_target_scale))
        object.__setattr__(self, "enable_soft_target_update", bool(self.enable_soft_target_update))
        object.__setattr__(self, "tau", max(float(self.tau), 0.0))
        object.__setattr__(self, "policy_delay", max(int(self.policy_delay), 1))
        object.__setattr__(self, "replay_ratio", max(float(self.replay_ratio), 0.0))
        object.__setattr__(self, "enable_reward_decomposition", bool(self.enable_reward_decomposition))
        object.__setattr__(self, "reward_info_weight", float(self.reward_info_weight))
        object.__setattr__(self, "reward_dist_weight", float(self.reward_dist_weight))
        object.__setattr__(self, "reward_safe_weight", float(self.reward_safe_weight))
        object.__setattr__(self, "reward_terminal_bonus", float(self.reward_terminal_bonus))
        object.__setattr__(self, "enable_curriculum", bool(self.enable_curriculum))
        object.__setattr__(
            self,
            "curriculum_milestones",
            _normalize_curriculum_milestones(self.curriculum_milestones, tuple(CURRICULUM_MILESTONES)),
        )
        object.__setattr__(
            self,
            "curriculum_levels",
            _normalize_curriculum_levels(self.curriculum_levels, tuple(CURRICULUM_LEVELS)),
        )
        if len(self.curriculum_milestones) != len(self.curriculum_levels):
            raise ValueError("CURRICULUM_MILESTONES and CURRICULUM_LEVELS must have the same length")
        resolved_curriculum_source = _normalize_optional_path(self.curriculum_source)
        if self.enable_curriculum and resolved_curriculum_source is None:
            auto_curriculum_source = get_latest_curriculum_source(self.run_name)
            if auto_curriculum_source is None:
                raise ValueError(
                    "CURRICULUM_SOURCE is not set and no difficulty manifest was found under "
                    f"{result_path / 'map_difficulty'} or {result_path / 'map_difficulty_smoke'}"
                )
            resolved_curriculum_source = str(auto_curriculum_source)
        object.__setattr__(self, "curriculum_source", resolved_curriculum_source)
        object.__setattr__(self, "curriculum_mix_window", max(int(self.curriculum_mix_window), 0))
        object.__setattr__(self, "use_curriculum_in_eval", bool(self.use_curriculum_in_eval))
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
        object.__setattr__(self, "wavelet_attn_bias_clamp", max(float(self.wavelet_attn_bias_clamp), 0.0))
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


def get_curriculum_level_index(episode_index: int, runtime_config: RuntimeConfig | None = None) -> int:
    runtime_config = runtime_config or RuntimeConfig()
    index = 0
    for level_index, milestone in enumerate(runtime_config.curriculum_milestones):
        if int(episode_index) >= int(milestone):
            index = level_index
        else:
            break
    return index


def get_curriculum_level(episode_index: int, runtime_config: RuntimeConfig | None = None) -> str:
    runtime_config = runtime_config or RuntimeConfig()
    return runtime_config.curriculum_levels[get_curriculum_level_index(episode_index, runtime_config)]


def _iter_curriculum_source_candidates(run_name: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    map_difficulty_root = result_path / "map_difficulty"
    smoke_root = result_path / "map_difficulty_smoke"

    if run_name == SMOKE_FOLDER_NAME and smoke_root.exists():
        candidates.append(smoke_root)

    if map_difficulty_root.is_dir():
        session_dirs = sorted(
            [path for path in map_difficulty_root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )
        candidates.extend(session_dirs)

    if smoke_root.exists() and smoke_root not in candidates:
        candidates.append(smoke_root)

    return candidates


def get_latest_curriculum_source(run_name: str | None = None) -> Path | None:
    for candidate in _iter_curriculum_source_candidates(run_name):
        if (candidate / "difficulty_manifest.json").is_file():
            return candidate.resolve()
        bucket_root = candidate / "buckets"
        if bucket_root.is_dir():
            return candidate.resolve()
        if candidate.name == "buckets" and candidate.is_dir():
            return candidate.resolve()
    return None


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


def get_result_run_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(get_result_root(), runtime_config)


def get_result_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_run_path(runtime_config) / "gifs"


def get_result_eval_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_run_path(runtime_config) / "eval"


def get_monitor_state_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_monitor_path(runtime_config) / ".state"


def get_model_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return _append_run_session(model_path, runtime_config)


def get_train_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_run_path(runtime_config) / "train"


def get_gifs_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_result_gifs_path(runtime_config)


def get_monitor_path(runtime_config: RuntimeConfig | None = None) -> Path:
    return get_train_path(runtime_config) / "monitor"


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
