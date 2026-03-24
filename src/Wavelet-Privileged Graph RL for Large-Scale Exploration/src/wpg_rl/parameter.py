from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import re


# Editable defaults
# 实验名与结果目录。改 run_name 主要影响结果落盘位置，不改算法本身。
FOLDER_NAME = "wpg_rl"
SMOKE_FOLDER_NAME = "wpg_rl_smoke"
RUN_SESSION = ""

# 结果写盘频率；主要影响日志/图片/checkpoint 密度，不直接改变策略行为。
SUMMARY_WINDOW = 300
LOAD_MODEL = False
SAVE_IMG_GAP = 300
SAVE_MODEL_GAP = 300
RESULT_BUCKET_EPISODES = 300

# 稀疏图几何尺度。这组参数最直接影响“点怎么长、边怎么连、路径会不会抖”。
CELL_SIZE = 0.4  # 地图每格的物理尺寸；改它会整体改变坐标/距离换算。
NODE_RESOLUTION = 4  # 图节点采样步长；调大图更稀更直，调小图更密更容易出现并行点列。
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # frontier 下采样粒度；调大更平滑，但会弱化细小前沿。

FREE = 255
OCCUPIED = 1
UNKNOWN = 127

SENSOR_RANGE = 20.0  # 单步可观测半径；调大更“远视”，局部图和 utility 覆盖范围也会变大。
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # 节点统计 frontier 的半径；调大更偏全局，调小更偏局部。
MIN_UTILITY = 3  # 小于等于该值的 frontier 直接视为无效；调大可更激进地剪枝候选点。
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # 局部规划窗口；调大上下文更大，但图更重。

# 训练超参；主要决定收敛速度和稳定性。
MAX_EPISODES = 12000
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 80000
MINIMUM_BUFFER_SIZE = 8000
BATCH_SIZE = 512

LR = 3e-5
GAMMA = 1.0
NUM_META_AGENT = 48
TRAIN_UPDATES_PER_ITER = 10
TARGET_Q_UPDATE_INTERVAL = 64
POLICY_GRAD_CLIP = 100.0
Q_GRAD_CLIP = 20000.0

# 观测/动作张量形状。改这里通常意味着网络输入维度或显存占用会变。
NODE_INPUT_DIM = 4  # actor 节点特征维度：相对坐标 + utility + visited。
CRITIC_NODE_INPUT_DIM = NODE_INPUT_DIM + 1  # critic 额外看一维特权信息。
EMBEDDING_DIM = 128  # 图编码隐藏维度；调大表示力更强，但算力/显存开销更高。

K_SIZE = 25  # 单步候选动作槽位上限；过小会截断邻居，过大主要是更耗算。
NODE_PADDING_SIZE = 360  # 单张图最大节点数；调大更稳但注意 attention 成本会上升。

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 3

# 模型结构开关。默认值不变时，baseline 路线保持不变。
USE_LF_ATTENTION_HF_RESIDUAL = True
WAVELET_SCALES = (1, 2, 4)  # 多尺度图平滑步数；更大更偏全局拓扑。
WAVELET_FUSE_DIM = 128  # 多尺度特征融合层宽度；调大容量更强但更重。

WAVELET_LF_QK = True  # True 时 Q/K 各自用低频分支，表达更灵活。

# 特权 critic / distillation。主要影响 actor 如何借助 GT 图学习表征。
USE_PRIVILEGED_WAVELET_DISTILLATION = True
WAVELET_DISTILL_WEIGHT = 0.1
WAVELET_DISTILL_LF_WEIGHT = 1.0
WAVELET_DISTILL_HF_WEIGHT = 1.0
WAVELET_DISTILL_WARMUP_UPDATES = 1000
WAVELET_DISTILL_RAMP_UPDATES = 2000

ENABLE_TRAINING_MONITOR = True
MONITOR_WINDOW = 10
MONITOR_SNAPSHOT_INTERVAL = 10

# 自动评测。固定测试集大小、评测频率和是否贪心都在这里控制。
ENABLE_AUTO_EVAL = True
AUTO_EVAL_MAP_COUNT = 10
AUTO_EVAL_INTERVAL = 100
AUTO_EVAL_GREEDY = True

# B/C/D 走廊优化实验开关。默认全关，保持原版逻辑。
ENABLE_CORRIDOR_GRAPH_COMPRESSION = True
ENABLE_CORRIDOR_EDGE_PRUNING = True
ENABLE_SMOOTHNESS_REWARD = False
CORRIDOR_MAX_WIDTH = 1.5 * NODE_RESOLUTION  # 认定为“窄走廊”的最大宽度；调大影响范围更广。
CORRIDOR_MIN_LENGTH = 2.0 * NODE_RESOLUTION  # 认定为“走廊”的最短长度；调小会更激进介入。
SMOOTHNESS_TURN_PENALTY = 0.25  # 转角惩罚；调大路径更直，但可能压制必要转弯。
SMOOTHNESS_LATERAL_PENALTY = 0.1  # 侧向偏移惩罚；调大可抑制横跳，但过大会影响绕障。


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
RESULT_ROOT = PROJECT_ROOT / "result"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return str(default)
    return str(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_scales(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(default)
    scales = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        scale = max(int(token), 1)
        if scale not in scales:
            scales.append(scale)
    return tuple(scales or default)


def _resolve_maps_dir() -> Path:
    # 地图目录查找优先级：显式环境变量 > 项目内常见位置。
    candidates = [
        os.environ.get("WPG_RL_MAPS_DIR"),
        os.environ.get("ARIADNE_MAPS_DIR"),
        str(PROJECT_ROOT / "maps"),
        str(SRC_ROOT / "maps"),
        str(PROJECT_ROOT.parent / "maps"),
        str(PROJECT_ROOT.parent.parent / "maps"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return (PROJECT_ROOT / "maps").resolve()


# 环境变量覆盖入口；命令行和 Ray worker 最终都会落到这里。
FOLDER_NAME = _env_str("WPG_RL_RUN_NAME", FOLDER_NAME).strip() or "wpg_rl"
RUN_SESSION = _env_str("WPG_RL_RUN_SESSION", RUN_SESSION).strip()

SUMMARY_WINDOW = _env_int("WPG_RL_SUMMARY_WINDOW", SUMMARY_WINDOW)
LOAD_MODEL = _env_bool("WPG_RL_LOAD_MODEL", LOAD_MODEL)
SAVE_IMG_GAP = _env_int("WPG_RL_SAVE_IMG_GAP", SAVE_IMG_GAP)
SAVE_MODEL_GAP = _env_int("WPG_RL_SAVE_MODEL_GAP", SAVE_MODEL_GAP)
RESULT_BUCKET_EPISODES = _env_int("WPG_RL_RESULT_BUCKET_EPISODES", RESULT_BUCKET_EPISODES)

CELL_SIZE = _env_float("WPG_RL_CELL_SIZE", CELL_SIZE)
NODE_RESOLUTION = _env_float("WPG_RL_NODE_RESOLUTION", NODE_RESOLUTION)
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

SENSOR_RANGE = _env_float("WPG_RL_SENSOR_RANGE", SENSOR_RANGE)
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = _env_int("WPG_RL_MIN_UTILITY", MIN_UTILITY)
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

MAX_EPISODES = _env_int("WPG_RL_MAX_EPISODES", MAX_EPISODES)
MAX_EPISODE_STEP = _env_int("WPG_RL_MAX_EPISODE_STEP", MAX_EPISODE_STEP)
REPLAY_SIZE = _env_int("WPG_RL_REPLAY_SIZE", REPLAY_SIZE)
MINIMUM_BUFFER_SIZE = _env_int("WPG_RL_MINIMUM_BUFFER_SIZE", MINIMUM_BUFFER_SIZE)
BATCH_SIZE = _env_int("WPG_RL_BATCH_SIZE", BATCH_SIZE)
LR = _env_float("WPG_RL_LR", LR)
GAMMA = _env_float("WPG_RL_GAMMA", GAMMA)
NUM_META_AGENT = _env_int("WPG_RL_NUM_META_AGENT", NUM_META_AGENT)
TRAIN_UPDATES_PER_ITER = _env_int("WPG_RL_TRAIN_UPDATES_PER_ITER", TRAIN_UPDATES_PER_ITER)
TARGET_Q_UPDATE_INTERVAL = _env_int("WPG_RL_TARGET_Q_UPDATE_INTERVAL", TARGET_Q_UPDATE_INTERVAL)
POLICY_GRAD_CLIP = _env_float("WPG_RL_POLICY_GRAD_CLIP", POLICY_GRAD_CLIP)
Q_GRAD_CLIP = _env_float("WPG_RL_Q_GRAD_CLIP", Q_GRAD_CLIP)

NODE_INPUT_DIM = _env_int("WPG_RL_NODE_INPUT_DIM", NODE_INPUT_DIM)
CRITIC_NODE_INPUT_DIM = _env_int("WPG_RL_CRITIC_NODE_INPUT_DIM", CRITIC_NODE_INPUT_DIM)
EMBEDDING_DIM = _env_int("WPG_RL_EMBEDDING_DIM", EMBEDDING_DIM)

K_SIZE = _env_int("WPG_RL_K_SIZE", K_SIZE)
NODE_PADDING_SIZE = _env_int("WPG_RL_NODE_PADDING_SIZE", NODE_PADDING_SIZE)

USE_GPU = _env_bool("WPG_RL_USE_GPU", USE_GPU)
USE_GPU_GLOBAL = _env_bool("WPG_RL_USE_GPU_GLOBAL", USE_GPU_GLOBAL)
NUM_GPU = _env_int("WPG_RL_NUM_GPU", NUM_GPU)

USE_LF_ATTENTION_HF_RESIDUAL = _env_bool(
    "WPG_RL_USE_LF_ATTENTION_HF_RESIDUAL",
    USE_LF_ATTENTION_HF_RESIDUAL,
)
USE_PRIVILEGED_WAVELET_DISTILLATION = _env_bool(
    "WPG_RL_USE_PRIVILEGED_WAVELET_DISTILLATION",
    USE_PRIVILEGED_WAVELET_DISTILLATION,
)
WAVELET_SCALES = _env_scales("WPG_RL_WAVELET_SCALES", WAVELET_SCALES)
WAVELET_FUSE_DIM = _env_int("WPG_RL_WAVELET_FUSE_DIM", WAVELET_FUSE_DIM)
WAVELET_LF_QK = _env_bool("WPG_RL_WAVELET_LF_QK", WAVELET_LF_QK)
WAVELET_DISTILL_WEIGHT = _env_float("WPG_RL_WAVELET_DISTILL_WEIGHT", WAVELET_DISTILL_WEIGHT)
WAVELET_DISTILL_LF_WEIGHT = _env_float("WPG_RL_WAVELET_DISTILL_LF_WEIGHT", WAVELET_DISTILL_LF_WEIGHT)
WAVELET_DISTILL_HF_WEIGHT = _env_float("WPG_RL_WAVELET_DISTILL_HF_WEIGHT", WAVELET_DISTILL_HF_WEIGHT)
WAVELET_DISTILL_WARMUP_UPDATES = _env_int(
    "WPG_RL_WAVELET_DISTILL_WARMUP_UPDATES",
    WAVELET_DISTILL_WARMUP_UPDATES,
)
WAVELET_DISTILL_RAMP_UPDATES = _env_int(
    "WPG_RL_WAVELET_DISTILL_RAMP_UPDATES",
    WAVELET_DISTILL_RAMP_UPDATES,
)
ENABLE_AUTO_EVAL = _env_bool("WPG_RL_ENABLE_AUTO_EVAL", ENABLE_AUTO_EVAL)
AUTO_EVAL_MAP_COUNT = _env_int("WPG_RL_AUTO_EVAL_MAP_COUNT", AUTO_EVAL_MAP_COUNT)
AUTO_EVAL_INTERVAL = _env_int("WPG_RL_AUTO_EVAL_INTERVAL", AUTO_EVAL_INTERVAL)
AUTO_EVAL_GREEDY = _env_bool("WPG_RL_AUTO_EVAL_GREEDY", AUTO_EVAL_GREEDY)
ENABLE_CORRIDOR_GRAPH_COMPRESSION = _env_bool(
    "WPG_RL_ENABLE_CORRIDOR_GRAPH_COMPRESSION",
    ENABLE_CORRIDOR_GRAPH_COMPRESSION,
)
ENABLE_CORRIDOR_EDGE_PRUNING = _env_bool(
    "WPG_RL_ENABLE_CORRIDOR_EDGE_PRUNING",
    ENABLE_CORRIDOR_EDGE_PRUNING,
)
ENABLE_SMOOTHNESS_REWARD = _env_bool("WPG_RL_ENABLE_SMOOTHNESS_REWARD", ENABLE_SMOOTHNESS_REWARD)
CORRIDOR_MAX_WIDTH = _env_float("WPG_RL_CORRIDOR_MAX_WIDTH", CORRIDOR_MAX_WIDTH)
CORRIDOR_MIN_LENGTH = _env_float("WPG_RL_CORRIDOR_MIN_LENGTH", CORRIDOR_MIN_LENGTH)
SMOOTHNESS_TURN_PENALTY = _env_float("WPG_RL_SMOOTHNESS_TURN_PENALTY", SMOOTHNESS_TURN_PENALTY)
SMOOTHNESS_LATERAL_PENALTY = _env_float("WPG_RL_SMOOTHNESS_LATERAL_PENALTY", SMOOTHNESS_LATERAL_PENALTY)

MAPS_DIR = _resolve_maps_dir()


def _get_run_suffix(run_name: str | None) -> str:
    value = str(run_name or FOLDER_NAME).strip()
    if value in {"", FOLDER_NAME}:
        return ""
    if value == SMOKE_FOLDER_NAME:
        return "_smoke"
    return f"_{value}"


def build_run_session(run_name: str = FOLDER_NAME) -> str:
    return datetime.now().strftime("%Y_%m%d_%H%M") + _get_run_suffix(run_name)


def is_smoke_run(runtime_config_or_run_name: RuntimeConfig | str | None = None) -> bool:
    if isinstance(runtime_config_or_run_name, RuntimeConfig):
        run_name = runtime_config_or_run_name.run_name
    else:
        run_name = runtime_config_or_run_name
    return str(run_name or FOLDER_NAME).strip() == SMOKE_FOLDER_NAME


def _require_run_session(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> str:
    if run_session is not None:
        session = str(run_session).strip()
    elif runtime_config is not None:
        session = str(runtime_config.run_session or "").strip()
    else:
        session = str(RUN_SESSION or "").strip()
    if not session:
        raise ValueError("run_session is required for artifact paths")
    return session


def get_result_session_root(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return RESULT_ROOT / _require_run_session(runtime_config, run_session)


def get_result_train_root(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "train"


def get_result_test_root(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "test"


def get_model_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "model"


def get_train_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "tensorboard"


def get_gifs_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "gifs"


def get_monitor_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "monitor"


def get_result_eval_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_test_root(runtime_config, run_session) / "eval"


def get_result_eval_gifs_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_test_root(runtime_config, run_session) / "gifs"


def get_checkpoint_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_model_path(runtime_config, run_session) / "checkpoint.pth"


def get_checkpoint_final_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_model_path(runtime_config, run_session) / "checkpoint_final.pth"


def get_checkpoint_interrupted_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_model_path(runtime_config, run_session) / "checkpoint_interrupted.pth"


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
        filtered = [path for path in session_dirs if path.name.endswith(suffix)]
        if filtered:
            session_dirs = filtered

    candidates = []
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


def ensure_output_dirs(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> None:
    target_runtime = runtime_config
    if target_runtime is None and run_session is not None:
        target_runtime = RuntimeConfig(run_session=run_session)
    get_gifs_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_train_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_monitor_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_eval_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_eval_gifs_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    if not is_smoke_run(target_runtime):
        get_model_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)


_DEFAULT_SESSION = RUN_SESSION or build_run_session(FOLDER_NAME)
model_path = str(get_model_path(run_session=_DEFAULT_SESSION))
train_path = str(get_train_path(run_session=_DEFAULT_SESSION))
gifs_path = str(get_gifs_path(run_session=_DEFAULT_SESSION))


# 统一的运行时配置入口；CLI/env 覆盖最终都会收敛到这里。
@dataclass(frozen=True)
class RuntimeConfig:
    run_name: str = FOLDER_NAME
    run_session: str | None = None
    maps_dir: str | None = None
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
    summary_window: int = SUMMARY_WINDOW
    train_updates_per_iter: int = TRAIN_UPDATES_PER_ITER
    result_bucket_episodes: int = RESULT_BUCKET_EPISODES
    load_model: bool = LOAD_MODEL
    resume_from: str | None = None
    use_gpu: bool = USE_GPU
    use_gpu_global: bool = USE_GPU_GLOBAL
    num_gpu: int = NUM_GPU
    enable_training_monitor: bool = ENABLE_TRAINING_MONITOR
    monitor_window: int = MONITOR_WINDOW
    monitor_snapshot_interval: int = MONITOR_SNAPSHOT_INTERVAL
    enable_auto_eval: bool = ENABLE_AUTO_EVAL
    auto_eval_map_count: int = AUTO_EVAL_MAP_COUNT
    auto_eval_interval: int = AUTO_EVAL_INTERVAL
    auto_eval_greedy: bool = AUTO_EVAL_GREEDY
    enable_corridor_graph_compression: bool = ENABLE_CORRIDOR_GRAPH_COMPRESSION
    enable_corridor_edge_pruning: bool = ENABLE_CORRIDOR_EDGE_PRUNING
    enable_smoothness_reward: bool = ENABLE_SMOOTHNESS_REWARD
    corridor_max_width: float = CORRIDOR_MAX_WIDTH
    corridor_min_length: float = CORRIDOR_MIN_LENGTH
    smoothness_turn_penalty: float = SMOOTHNESS_TURN_PENALTY
    smoothness_lateral_penalty: float = SMOOTHNESS_LATERAL_PENALTY
    use_lf_attention_hf_residual: bool = USE_LF_ATTENTION_HF_RESIDUAL
    use_privileged_wavelet_distillation: bool = USE_PRIVILEGED_WAVELET_DISTILLATION
    wavelet_scales: tuple[int, ...] = WAVELET_SCALES
    wavelet_fuse_dim: int = WAVELET_FUSE_DIM
    wavelet_lf_qk: bool = WAVELET_LF_QK
    wavelet_distill_weight: float = WAVELET_DISTILL_WEIGHT
    wavelet_distill_lf_weight: float = WAVELET_DISTILL_LF_WEIGHT
    wavelet_distill_hf_weight: float = WAVELET_DISTILL_HF_WEIGHT
    wavelet_distill_warmup_updates: int = WAVELET_DISTILL_WARMUP_UPDATES
    wavelet_distill_ramp_updates: int = WAVELET_DISTILL_RAMP_UPDATES

    def __post_init__(self) -> None:
        scales = []
        for scale in self.wavelet_scales:
            value = max(int(scale), 1)
            if value not in scales:
                scales.append(value)
        object.__setattr__(self, "wavelet_scales", tuple(scales or (1, 2, 4)))
        object.__setattr__(self, "wavelet_fuse_dim", max(int(self.wavelet_fuse_dim), 8))
        object.__setattr__(self, "max_episodes", max(int(self.max_episodes), 1))
        object.__setattr__(self, "num_meta_agent", max(int(self.num_meta_agent), 1))
        object.__setattr__(self, "max_episode_step", max(int(self.max_episode_step), 1))
        object.__setattr__(self, "minimum_buffer_size", max(int(self.minimum_buffer_size), 1))
        object.__setattr__(self, "batch_size", max(int(self.batch_size), 1))
        object.__setattr__(self, "replay_size", max(int(self.replay_size), self.batch_size))
        object.__setattr__(self, "save_img_gap", max(int(self.save_img_gap), 1))
        object.__setattr__(self, "save_model_gap", max(int(self.save_model_gap), 1))
        object.__setattr__(self, "summary_window", max(int(self.summary_window), 1))
        object.__setattr__(self, "train_updates_per_iter", max(int(self.train_updates_per_iter), 1))
        object.__setattr__(self, "result_bucket_episodes", max(int(self.result_bucket_episodes), 1))
        object.__setattr__(self, "num_gpu", max(int(self.num_gpu), 0))
        object.__setattr__(self, "monitor_window", max(int(self.monitor_window), 1))
        object.__setattr__(self, "monitor_snapshot_interval", max(int(self.monitor_snapshot_interval), 1))
        object.__setattr__(self, "auto_eval_map_count", max(int(self.auto_eval_map_count), 0))
        object.__setattr__(self, "auto_eval_interval", max(int(self.auto_eval_interval), 1))
        object.__setattr__(self, "corridor_max_width", max(float(self.corridor_max_width), CELL_SIZE))
        object.__setattr__(self, "corridor_min_length", max(float(self.corridor_min_length), CELL_SIZE))
        object.__setattr__(self, "smoothness_turn_penalty", max(float(self.smoothness_turn_penalty), 0.0))
        object.__setattr__(self, "smoothness_lateral_penalty", max(float(self.smoothness_lateral_penalty), 0.0))
        object.__setattr__(self, "wavelet_distill_weight", max(float(self.wavelet_distill_weight), 0.0))
        object.__setattr__(self, "wavelet_distill_lf_weight", max(float(self.wavelet_distill_lf_weight), 0.0))
        object.__setattr__(self, "wavelet_distill_hf_weight", max(float(self.wavelet_distill_hf_weight), 0.0))
        object.__setattr__(
            self,
            "wavelet_distill_warmup_updates",
            max(int(self.wavelet_distill_warmup_updates), 0),
        )
        object.__setattr__(
            self,
            "wavelet_distill_ramp_updates",
            max(int(self.wavelet_distill_ramp_updates), 0),
        )

    def with_overrides(self, **kwargs: object) -> "RuntimeConfig":
        return replace(self, **kwargs)


def apply_runtime_config(runtime_config: RuntimeConfig) -> RuntimeConfig:
    run_session = runtime_config.run_session or build_run_session(runtime_config.run_name)
    runtime_config = runtime_config.with_overrides(run_session=run_session)

    env_updates = {
        "WPG_RL_RUN_NAME": runtime_config.run_name,
        "WPG_RL_RUN_SESSION": run_session,
        "WPG_RL_MAX_EPISODES": str(runtime_config.max_episodes),
        "WPG_RL_NUM_META_AGENT": str(runtime_config.num_meta_agent),
        "WPG_RL_MAX_EPISODE_STEP": str(runtime_config.max_episode_step),
        "WPG_RL_MINIMUM_BUFFER_SIZE": str(runtime_config.minimum_buffer_size),
        "WPG_RL_BATCH_SIZE": str(runtime_config.batch_size),
        "WPG_RL_REPLAY_SIZE": str(runtime_config.replay_size),
        "WPG_RL_SAVE_IMG_GAP": str(runtime_config.save_img_gap),
        "WPG_RL_SAVE_MODEL_GAP": str(runtime_config.save_model_gap),
        "WPG_RL_SUMMARY_WINDOW": str(runtime_config.summary_window),
        "WPG_RL_TRAIN_UPDATES_PER_ITER": str(runtime_config.train_updates_per_iter),
        "WPG_RL_RESULT_BUCKET_EPISODES": str(runtime_config.result_bucket_episodes),
        "WPG_RL_LOAD_MODEL": "1" if runtime_config.load_model else "0",
        "WPG_RL_USE_GPU": "1" if runtime_config.use_gpu else "0",
        "WPG_RL_USE_GPU_GLOBAL": "1" if runtime_config.use_gpu_global else "0",
        "WPG_RL_NUM_GPU": str(runtime_config.num_gpu),
        "WPG_RL_ENABLE_AUTO_EVAL": "1" if runtime_config.enable_auto_eval else "0",
        "WPG_RL_AUTO_EVAL_MAP_COUNT": str(runtime_config.auto_eval_map_count),
        "WPG_RL_AUTO_EVAL_INTERVAL": str(runtime_config.auto_eval_interval),
        "WPG_RL_AUTO_EVAL_GREEDY": "1" if runtime_config.auto_eval_greedy else "0",
        "WPG_RL_ENABLE_CORRIDOR_GRAPH_COMPRESSION": (
            "1" if runtime_config.enable_corridor_graph_compression else "0"
        ),
        "WPG_RL_ENABLE_CORRIDOR_EDGE_PRUNING": "1" if runtime_config.enable_corridor_edge_pruning else "0",
        "WPG_RL_ENABLE_SMOOTHNESS_REWARD": "1" if runtime_config.enable_smoothness_reward else "0",
        "WPG_RL_CORRIDOR_MAX_WIDTH": str(runtime_config.corridor_max_width),
        "WPG_RL_CORRIDOR_MIN_LENGTH": str(runtime_config.corridor_min_length),
        "WPG_RL_SMOOTHNESS_TURN_PENALTY": str(runtime_config.smoothness_turn_penalty),
        "WPG_RL_SMOOTHNESS_LATERAL_PENALTY": str(runtime_config.smoothness_lateral_penalty),
        "WPG_RL_USE_LF_ATTENTION_HF_RESIDUAL": "1" if runtime_config.use_lf_attention_hf_residual else "0",
        "WPG_RL_USE_PRIVILEGED_WAVELET_DISTILLATION": (
            "1" if runtime_config.use_privileged_wavelet_distillation else "0"
        ),
        "WPG_RL_WAVELET_SCALES": ",".join(str(scale) for scale in runtime_config.wavelet_scales),
        "WPG_RL_WAVELET_FUSE_DIM": str(runtime_config.wavelet_fuse_dim),
        "WPG_RL_WAVELET_LF_QK": "1" if runtime_config.wavelet_lf_qk else "0",
        "WPG_RL_WAVELET_DISTILL_WEIGHT": str(runtime_config.wavelet_distill_weight),
        "WPG_RL_WAVELET_DISTILL_LF_WEIGHT": str(runtime_config.wavelet_distill_lf_weight),
        "WPG_RL_WAVELET_DISTILL_HF_WEIGHT": str(runtime_config.wavelet_distill_hf_weight),
        "WPG_RL_WAVELET_DISTILL_WARMUP_UPDATES": str(runtime_config.wavelet_distill_warmup_updates),
        "WPG_RL_WAVELET_DISTILL_RAMP_UPDATES": str(runtime_config.wavelet_distill_ramp_updates),
    }
    if runtime_config.ray_num_cpus is not None:
        env_updates["WPG_RL_RAY_NUM_CPUS"] = str(runtime_config.ray_num_cpus)
    if runtime_config.ray_worker_num_cpus is not None:
        env_updates["WPG_RL_RAY_WORKER_NUM_CPUS"] = str(runtime_config.ray_worker_num_cpus)
    if runtime_config.worker_num_threads is not None:
        env_updates["WPG_RL_WORKER_NUM_THREADS"] = str(runtime_config.worker_num_threads)
    if runtime_config.maps_dir:
        env_updates["WPG_RL_MAPS_DIR"] = str(Path(runtime_config.maps_dir).expanduser().resolve())

    for key, value in env_updates.items():
        os.environ[key] = value

    globals().update(
        {
            "FOLDER_NAME": runtime_config.run_name,
            "RUN_SESSION": run_session,
            "MAX_EPISODES": runtime_config.max_episodes,
            "NUM_META_AGENT": runtime_config.num_meta_agent,
            "MAX_EPISODE_STEP": runtime_config.max_episode_step,
            "MINIMUM_BUFFER_SIZE": runtime_config.minimum_buffer_size,
            "BATCH_SIZE": runtime_config.batch_size,
            "REPLAY_SIZE": runtime_config.replay_size,
            "SAVE_IMG_GAP": runtime_config.save_img_gap,
            "SAVE_MODEL_GAP": runtime_config.save_model_gap,
            "SUMMARY_WINDOW": runtime_config.summary_window,
            "TRAIN_UPDATES_PER_ITER": runtime_config.train_updates_per_iter,
            "RESULT_BUCKET_EPISODES": runtime_config.result_bucket_episodes,
            "LOAD_MODEL": runtime_config.load_model,
            "USE_GPU": runtime_config.use_gpu,
            "USE_GPU_GLOBAL": runtime_config.use_gpu_global,
            "NUM_GPU": runtime_config.num_gpu,
            "ENABLE_AUTO_EVAL": runtime_config.enable_auto_eval,
            "AUTO_EVAL_MAP_COUNT": runtime_config.auto_eval_map_count,
            "AUTO_EVAL_INTERVAL": runtime_config.auto_eval_interval,
            "AUTO_EVAL_GREEDY": runtime_config.auto_eval_greedy,
            "ENABLE_CORRIDOR_GRAPH_COMPRESSION": runtime_config.enable_corridor_graph_compression,
            "ENABLE_CORRIDOR_EDGE_PRUNING": runtime_config.enable_corridor_edge_pruning,
            "ENABLE_SMOOTHNESS_REWARD": runtime_config.enable_smoothness_reward,
            "CORRIDOR_MAX_WIDTH": runtime_config.corridor_max_width,
            "CORRIDOR_MIN_LENGTH": runtime_config.corridor_min_length,
            "SMOOTHNESS_TURN_PENALTY": runtime_config.smoothness_turn_penalty,
            "SMOOTHNESS_LATERAL_PENALTY": runtime_config.smoothness_lateral_penalty,
            "USE_LF_ATTENTION_HF_RESIDUAL": runtime_config.use_lf_attention_hf_residual,
            "USE_PRIVILEGED_WAVELET_DISTILLATION": runtime_config.use_privileged_wavelet_distillation,
            "WAVELET_SCALES": runtime_config.wavelet_scales,
            "WAVELET_FUSE_DIM": runtime_config.wavelet_fuse_dim,
            "WAVELET_LF_QK": runtime_config.wavelet_lf_qk,
            "WAVELET_DISTILL_WEIGHT": runtime_config.wavelet_distill_weight,
            "WAVELET_DISTILL_LF_WEIGHT": runtime_config.wavelet_distill_lf_weight,
            "WAVELET_DISTILL_HF_WEIGHT": runtime_config.wavelet_distill_hf_weight,
            "WAVELET_DISTILL_WARMUP_UPDATES": runtime_config.wavelet_distill_warmup_updates,
            "WAVELET_DISTILL_RAMP_UPDATES": runtime_config.wavelet_distill_ramp_updates,
            "MAPS_DIR": _resolve_maps_dir(),
        }
    )

    globals()["model_path"] = str(get_model_path(runtime_config))
    globals()["train_path"] = str(get_train_path(runtime_config))
    globals()["gifs_path"] = str(get_gifs_path(runtime_config))
    return runtime_config
