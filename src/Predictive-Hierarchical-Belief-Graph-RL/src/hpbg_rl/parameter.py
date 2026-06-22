from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import re


# Editable defaults
# 实验名与结果目录。改 run_name 主要影响结果落盘位置，不改算法本身。
FOLDER_NAME = "hpbg_rl"
SMOKE_FOLDER_NAME = "hpbg_rl_smoke"
RUN_SESSION = ""

# 结果写盘频率；主要影响日志/图片/checkpoint 密度，不直接改变策略行为。
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 100
SAVE_MODEL_GAP = 32
RESULT_BUCKET_EPISODES = 100

# 稀疏图几何尺度。这组参数最直接影响“点怎么长、边怎么连、路径会不会抖”。
CELL_SIZE = 0.4  # 地图每格的物理尺寸；改它会整体改变坐标/距离换算。
NODE_RESOLUTION = 4.0  # 图节点采样步长；调大图更稀更直，调小图更密更容易出现并行点列。
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # frontier 下采样粒度；调大更平滑，但会弱化细小前沿。

FREE = 255
OCCUPIED = 1
UNKNOWN = 127

SENSOR_RANGE = 16.0  # 单步可观测半径；HPBG-off 默认与 Cao baseline 对齐。
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # 节点统计 frontier 的半径；调大更偏全局，调小更偏局部。
MIN_UTILITY = 2  # 小于等于该值的 frontier 直接视为无效；HPBG-off 默认与 Cao baseline 对齐。
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # 局部规划窗口；调大上下文更大，但图更重。

# 训练超参；主要决定收敛速度和稳定性。
MAX_EPISODES = 10000
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128

LR = 1e-5
GAMMA = 1.0
NUM_META_AGENT = 16
TRAIN_UPDATES_PER_ITER = 8
TARGET_Q_UPDATE_INTERVAL = 64
POLICY_GRAD_CLIP = 100.0
Q_GRAD_CLIP = 20000.0

# HPBG 主线开关。actor 只使用在线 belief/prediction/hierarchy；privileged 信号只进入 critic、reward 或辅助蒸馏。
USE_HPBG = True
USE_BELIEF_STATE = True
USE_MAP_PREDICTION = True
USE_HIERARCHICAL_GRAPH = True
USE_EXPERT_REWARD = True
USE_BELIEF_DISTILLATION = True
HPBG_RISK_WEIGHT = 0.35
HPBG_BELIEF_EMA_ALPHA = 0.35
HPBG_CLUSTER_RESOLUTION = 3.0 * NODE_RESOLUTION
HPBG_CLUSTER_EDGE_HOPS = 1
HPBG_EXPERT_REWARD_WEIGHT = 0.25
HPBG_EXPERT_POTENTIAL_WEIGHT = 1.0
HPBG_ORACLE_GAIN_WEIGHT = 0.0
HPBG_BELIEF_DISTILL_WEIGHT = 0.05
HPBG_BELIEF_DISTILL_WARMUP_UPDATES = 1000
HPBG_BELIEF_DISTILL_RAMP_UPDATES = 2000

# 观测/动作张量形状。actor 只接收在线 belief/prediction 派生特征；critic 额外接收训练期 privileged 特征。
BASE_NODE_INPUT_DIM = 4  # 相对坐标 + utility + visited。
HPBG_ACTOR_BELIEF_DIM = 4  # predicted utility + uncertainty + risk-aware utility + cluster prior。
HPBG_CRITIC_PRIVILEGED_DIM = 3  # explored sign + oracle utility + expert potential。
NODE_INPUT_DIM = BASE_NODE_INPUT_DIM + HPBG_ACTOR_BELIEF_DIM
CRITIC_NODE_INPUT_DIM = NODE_INPUT_DIM + HPBG_CRITIC_PRIVILEGED_DIM
EMBEDDING_DIM = 128  # 图编码隐藏维度；调大表示力更强，但算力/显存开销更高。

HPBG_PREDICTED_UTILITY_INDEX = BASE_NODE_INPUT_DIM
HPBG_UNCERTAINTY_INDEX = BASE_NODE_INPUT_DIM + 1
HPBG_RISK_AWARE_UTILITY_INDEX = BASE_NODE_INPUT_DIM + 2
HPBG_CLUSTER_PRIOR_INDEX = BASE_NODE_INPUT_DIM + 3
HPBG_CRITIC_EXPLORED_INDEX = NODE_INPUT_DIM
HPBG_CRITIC_ORACLE_UTILITY_INDEX = NODE_INPUT_DIM + 1
HPBG_CRITIC_EXPERT_POTENTIAL_INDEX = NODE_INPUT_DIM + 2

K_SIZE = 25  # 单步候选动作槽位上限；过小会截断邻居，过大主要是更耗算。
NODE_PADDING_SIZE = 360  # 单张图最大节点数；调大更稳但注意 attention 成本会上升。

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0

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

# 自动评测。固定验证集大小、评测频率和是否贪心都在这里控制。
ENABLE_AUTO_EVAL = True
AUTO_EVAL_MAP_COUNT = 10
AUTO_EVAL_INTERVAL = 100
AUTO_EVAL_GREEDY = True

# 地图 split / 公平评估协议。maps_dir 只作为兼容入口；训练、验证、测试会被拆到互斥 split。
TRAIN_MAPS_DIR = ""
VAL_MAPS_DIR = ""
TEST_MAPS_DIR = ""
SPLIT_MANIFEST_PATH = ""
SPLIT_SEED = 2024
VAL_MAP_COUNT = AUTO_EVAL_MAP_COUNT
TEST_MAP_COUNT = AUTO_EVAL_MAP_COUNT
EVAL_BUDGET_MODE = "fixed_steps"
LARGE_SCALE_MIN_FREE_AREA = 0
LARGE_SCALE_MIN_SIDE = 0
RUN_FINAL_TEST = False
ALLOW_TRAIN_SPLIT_EVAL = False

# B/C/D 走廊优化实验开关。默认全关，保持原版逻辑。
ENABLE_CORRIDOR_GRAPH_COMPRESSION = False
ENABLE_CORRIDOR_EDGE_PRUNING = False
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
        os.environ.get("HPBG_RL_MAPS_DIR"),
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
FOLDER_NAME = _env_str("HPBG_RL_RUN_NAME", FOLDER_NAME).strip() or "hpbg_rl"
RUN_SESSION = _env_str("HPBG_RL_RUN_SESSION", RUN_SESSION).strip()

SUMMARY_WINDOW = _env_int("HPBG_RL_SUMMARY_WINDOW", SUMMARY_WINDOW)
LOAD_MODEL = _env_bool("HPBG_RL_LOAD_MODEL", LOAD_MODEL)
SAVE_IMG_GAP = _env_int("HPBG_RL_SAVE_IMG_GAP", SAVE_IMG_GAP)
SAVE_MODEL_GAP = _env_int("HPBG_RL_SAVE_MODEL_GAP", SAVE_MODEL_GAP)
RESULT_BUCKET_EPISODES = _env_int("HPBG_RL_RESULT_BUCKET_EPISODES", RESULT_BUCKET_EPISODES)

CELL_SIZE = _env_float("HPBG_RL_CELL_SIZE", CELL_SIZE)
NODE_RESOLUTION = _env_float("HPBG_RL_NODE_RESOLUTION", NODE_RESOLUTION)
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

SENSOR_RANGE = _env_float("HPBG_RL_SENSOR_RANGE", SENSOR_RANGE)
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = _env_int("HPBG_RL_MIN_UTILITY", MIN_UTILITY)
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

MAX_EPISODES = _env_int("HPBG_RL_MAX_EPISODES", MAX_EPISODES)
MAX_EPISODE_STEP = _env_int("HPBG_RL_MAX_EPISODE_STEP", MAX_EPISODE_STEP)
REPLAY_SIZE = _env_int("HPBG_RL_REPLAY_SIZE", REPLAY_SIZE)
MINIMUM_BUFFER_SIZE = _env_int("HPBG_RL_MINIMUM_BUFFER_SIZE", MINIMUM_BUFFER_SIZE)
BATCH_SIZE = _env_int("HPBG_RL_BATCH_SIZE", BATCH_SIZE)
LR = _env_float("HPBG_RL_LR", LR)
GAMMA = _env_float("HPBG_RL_GAMMA", GAMMA)
NUM_META_AGENT = _env_int("HPBG_RL_NUM_META_AGENT", NUM_META_AGENT)
TRAIN_UPDATES_PER_ITER = _env_int("HPBG_RL_TRAIN_UPDATES_PER_ITER", TRAIN_UPDATES_PER_ITER)
TARGET_Q_UPDATE_INTERVAL = _env_int("HPBG_RL_TARGET_Q_UPDATE_INTERVAL", TARGET_Q_UPDATE_INTERVAL)
POLICY_GRAD_CLIP = _env_float("HPBG_RL_POLICY_GRAD_CLIP", POLICY_GRAD_CLIP)
Q_GRAD_CLIP = _env_float("HPBG_RL_Q_GRAD_CLIP", Q_GRAD_CLIP)

USE_HPBG = _env_bool("HPBG_RL_USE_HPBG", USE_HPBG)
USE_BELIEF_STATE = _env_bool("HPBG_RL_USE_BELIEF_STATE", USE_BELIEF_STATE)
USE_MAP_PREDICTION = _env_bool("HPBG_RL_USE_MAP_PREDICTION", USE_MAP_PREDICTION)
USE_HIERARCHICAL_GRAPH = _env_bool("HPBG_RL_USE_HIERARCHICAL_GRAPH", USE_HIERARCHICAL_GRAPH)
USE_EXPERT_REWARD = _env_bool("HPBG_RL_USE_EXPERT_REWARD", USE_EXPERT_REWARD)
USE_BELIEF_DISTILLATION = _env_bool("HPBG_RL_USE_BELIEF_DISTILLATION", USE_BELIEF_DISTILLATION)
HPBG_RISK_WEIGHT = _env_float("HPBG_RL_RISK_WEIGHT", HPBG_RISK_WEIGHT)
HPBG_BELIEF_EMA_ALPHA = _env_float("HPBG_RL_BELIEF_EMA_ALPHA", HPBG_BELIEF_EMA_ALPHA)
HPBG_CLUSTER_RESOLUTION = _env_float("HPBG_RL_CLUSTER_RESOLUTION", HPBG_CLUSTER_RESOLUTION)
HPBG_CLUSTER_EDGE_HOPS = _env_int("HPBG_RL_CLUSTER_EDGE_HOPS", HPBG_CLUSTER_EDGE_HOPS)
HPBG_EXPERT_REWARD_WEIGHT = _env_float("HPBG_RL_EXPERT_REWARD_WEIGHT", HPBG_EXPERT_REWARD_WEIGHT)
HPBG_EXPERT_POTENTIAL_WEIGHT = _env_float("HPBG_RL_EXPERT_POTENTIAL_WEIGHT", HPBG_EXPERT_POTENTIAL_WEIGHT)
HPBG_ORACLE_GAIN_WEIGHT = _env_float("HPBG_RL_ORACLE_GAIN_WEIGHT", HPBG_ORACLE_GAIN_WEIGHT)
HPBG_BELIEF_DISTILL_WEIGHT = _env_float("HPBG_RL_BELIEF_DISTILL_WEIGHT", HPBG_BELIEF_DISTILL_WEIGHT)
HPBG_BELIEF_DISTILL_WARMUP_UPDATES = _env_int(
    "HPBG_RL_BELIEF_DISTILL_WARMUP_UPDATES",
    HPBG_BELIEF_DISTILL_WARMUP_UPDATES,
)
HPBG_BELIEF_DISTILL_RAMP_UPDATES = _env_int(
    "HPBG_RL_BELIEF_DISTILL_RAMP_UPDATES",
    HPBG_BELIEF_DISTILL_RAMP_UPDATES,
)
EMBEDDING_DIM = _env_int("HPBG_RL_EMBEDDING_DIM", EMBEDDING_DIM)

K_SIZE = _env_int("HPBG_RL_K_SIZE", K_SIZE)
NODE_PADDING_SIZE = _env_int("HPBG_RL_NODE_PADDING_SIZE", NODE_PADDING_SIZE)

USE_GPU = _env_bool("HPBG_RL_USE_GPU", USE_GPU)
USE_GPU_GLOBAL = _env_bool("HPBG_RL_USE_GPU_GLOBAL", USE_GPU_GLOBAL)
NUM_GPU = _env_int("HPBG_RL_NUM_GPU", NUM_GPU)

USE_LF_ATTENTION_HF_RESIDUAL = _env_bool(
    "HPBG_RL_USE_LF_ATTENTION_HF_RESIDUAL",
    USE_LF_ATTENTION_HF_RESIDUAL,
)
USE_PRIVILEGED_WAVELET_DISTILLATION = _env_bool(
    "HPBG_RL_USE_PRIVILEGED_WAVELET_DISTILLATION",
    USE_PRIVILEGED_WAVELET_DISTILLATION,
)
WAVELET_SCALES = _env_scales("HPBG_RL_WAVELET_SCALES", WAVELET_SCALES)
WAVELET_FUSE_DIM = _env_int("HPBG_RL_WAVELET_FUSE_DIM", WAVELET_FUSE_DIM)
WAVELET_LF_QK = _env_bool("HPBG_RL_WAVELET_LF_QK", WAVELET_LF_QK)
WAVELET_DISTILL_WEIGHT = _env_float("HPBG_RL_WAVELET_DISTILL_WEIGHT", WAVELET_DISTILL_WEIGHT)
WAVELET_DISTILL_LF_WEIGHT = _env_float("HPBG_RL_WAVELET_DISTILL_LF_WEIGHT", WAVELET_DISTILL_LF_WEIGHT)
WAVELET_DISTILL_HF_WEIGHT = _env_float("HPBG_RL_WAVELET_DISTILL_HF_WEIGHT", WAVELET_DISTILL_HF_WEIGHT)
WAVELET_DISTILL_WARMUP_UPDATES = _env_int(
    "HPBG_RL_WAVELET_DISTILL_WARMUP_UPDATES",
    WAVELET_DISTILL_WARMUP_UPDATES,
)
WAVELET_DISTILL_RAMP_UPDATES = _env_int(
    "HPBG_RL_WAVELET_DISTILL_RAMP_UPDATES",
    WAVELET_DISTILL_RAMP_UPDATES,
)
ENABLE_AUTO_EVAL = _env_bool("HPBG_RL_ENABLE_AUTO_EVAL", ENABLE_AUTO_EVAL)
AUTO_EVAL_MAP_COUNT = _env_int("HPBG_RL_AUTO_EVAL_MAP_COUNT", AUTO_EVAL_MAP_COUNT)
AUTO_EVAL_INTERVAL = _env_int("HPBG_RL_AUTO_EVAL_INTERVAL", AUTO_EVAL_INTERVAL)
AUTO_EVAL_GREEDY = _env_bool("HPBG_RL_AUTO_EVAL_GREEDY", AUTO_EVAL_GREEDY)
TRAIN_MAPS_DIR = _env_str("HPBG_RL_TRAIN_MAPS_DIR", TRAIN_MAPS_DIR).strip()
VAL_MAPS_DIR = _env_str("HPBG_RL_VAL_MAPS_DIR", VAL_MAPS_DIR).strip()
TEST_MAPS_DIR = _env_str("HPBG_RL_TEST_MAPS_DIR", TEST_MAPS_DIR).strip()
SPLIT_MANIFEST_PATH = _env_str("HPBG_RL_SPLIT_MANIFEST_PATH", SPLIT_MANIFEST_PATH).strip()
SPLIT_SEED = _env_int("HPBG_RL_SPLIT_SEED", SPLIT_SEED)
VAL_MAP_COUNT = _env_int("HPBG_RL_VAL_MAP_COUNT", VAL_MAP_COUNT)
TEST_MAP_COUNT = _env_int("HPBG_RL_TEST_MAP_COUNT", TEST_MAP_COUNT)
EVAL_BUDGET_MODE = _env_str("HPBG_RL_EVAL_BUDGET_MODE", EVAL_BUDGET_MODE).strip() or "fixed_steps"
LARGE_SCALE_MIN_FREE_AREA = _env_int("HPBG_RL_LARGE_SCALE_MIN_FREE_AREA", LARGE_SCALE_MIN_FREE_AREA)
LARGE_SCALE_MIN_SIDE = _env_int("HPBG_RL_LARGE_SCALE_MIN_SIDE", LARGE_SCALE_MIN_SIDE)
RUN_FINAL_TEST = _env_bool("HPBG_RL_RUN_FINAL_TEST", RUN_FINAL_TEST)
ALLOW_TRAIN_SPLIT_EVAL = _env_bool("HPBG_RL_ALLOW_TRAIN_SPLIT_EVAL", ALLOW_TRAIN_SPLIT_EVAL)
ENABLE_CORRIDOR_GRAPH_COMPRESSION = _env_bool(
    "HPBG_RL_ENABLE_CORRIDOR_GRAPH_COMPRESSION",
    ENABLE_CORRIDOR_GRAPH_COMPRESSION,
)
ENABLE_CORRIDOR_EDGE_PRUNING = _env_bool(
    "HPBG_RL_ENABLE_CORRIDOR_EDGE_PRUNING",
    ENABLE_CORRIDOR_EDGE_PRUNING,
)
ENABLE_SMOOTHNESS_REWARD = _env_bool("HPBG_RL_ENABLE_SMOOTHNESS_REWARD", ENABLE_SMOOTHNESS_REWARD)
CORRIDOR_MAX_WIDTH = _env_float("HPBG_RL_CORRIDOR_MAX_WIDTH", CORRIDOR_MAX_WIDTH)
CORRIDOR_MIN_LENGTH = _env_float("HPBG_RL_CORRIDOR_MIN_LENGTH", CORRIDOR_MIN_LENGTH)
SMOOTHNESS_TURN_PENALTY = _env_float("HPBG_RL_SMOOTHNESS_TURN_PENALTY", SMOOTHNESS_TURN_PENALTY)
SMOOTHNESS_LATERAL_PENALTY = _env_float("HPBG_RL_SMOOTHNESS_LATERAL_PENALTY", SMOOTHNESS_LATERAL_PENALTY)

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


def get_result_validation_root(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "validation"


def get_result_protocol_root(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "protocol"


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


def get_protocol_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_protocol_root(runtime_config, run_session)


def get_result_validation_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_validation_root(runtime_config, run_session) / "eval"


def get_result_validation_gifs_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_validation_root(runtime_config, run_session) / "gifs"


def get_result_eval_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_validation_path(runtime_config, run_session)


def get_result_eval_gifs_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_validation_gifs_path(runtime_config, run_session)


def get_result_test_eval_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
    return get_result_test_root(runtime_config, run_session) / "eval"


def get_result_test_gifs_path(runtime_config: RuntimeConfig | None = None, run_session: str | None = None) -> Path:
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
    get_protocol_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_validation_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_validation_gifs_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_test_eval_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
    get_result_test_gifs_path(target_runtime, run_session).mkdir(parents=True, exist_ok=True)
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
    train_maps_dir: str | None = TRAIN_MAPS_DIR or None
    val_maps_dir: str | None = VAL_MAPS_DIR or None
    test_maps_dir: str | None = TEST_MAPS_DIR or None
    split_manifest_path: str | None = SPLIT_MANIFEST_PATH or None
    split_seed: int = SPLIT_SEED
    val_map_count: int | None = VAL_MAP_COUNT
    test_map_count: int | None = TEST_MAP_COUNT
    eval_budget_mode: str = EVAL_BUDGET_MODE
    large_scale_min_free_area: int = LARGE_SCALE_MIN_FREE_AREA
    large_scale_min_side: int = LARGE_SCALE_MIN_SIDE
    run_final_test: bool = RUN_FINAL_TEST
    allow_train_split_eval: bool = ALLOW_TRAIN_SPLIT_EVAL
    enable_corridor_graph_compression: bool = ENABLE_CORRIDOR_GRAPH_COMPRESSION
    enable_corridor_edge_pruning: bool = ENABLE_CORRIDOR_EDGE_PRUNING
    enable_smoothness_reward: bool = ENABLE_SMOOTHNESS_REWARD
    corridor_max_width: float = CORRIDOR_MAX_WIDTH
    corridor_min_length: float = CORRIDOR_MIN_LENGTH
    smoothness_turn_penalty: float = SMOOTHNESS_TURN_PENALTY
    smoothness_lateral_penalty: float = SMOOTHNESS_LATERAL_PENALTY
    use_lf_attention_hf_residual: bool = USE_LF_ATTENTION_HF_RESIDUAL
    use_privileged_wavelet_distillation: bool = USE_PRIVILEGED_WAVELET_DISTILLATION
    use_hpbg: bool = USE_HPBG
    use_belief_state: bool = USE_BELIEF_STATE
    use_map_prediction: bool = USE_MAP_PREDICTION
    use_hierarchical_graph: bool = USE_HIERARCHICAL_GRAPH
    use_expert_reward: bool = USE_EXPERT_REWARD
    use_belief_distillation: bool = USE_BELIEF_DISTILLATION
    hpbg_risk_weight: float = HPBG_RISK_WEIGHT
    hpbg_belief_ema_alpha: float = HPBG_BELIEF_EMA_ALPHA
    hpbg_cluster_resolution: float = HPBG_CLUSTER_RESOLUTION
    hpbg_cluster_edge_hops: int = HPBG_CLUSTER_EDGE_HOPS
    hpbg_expert_reward_weight: float = HPBG_EXPERT_REWARD_WEIGHT
    hpbg_expert_potential_weight: float = HPBG_EXPERT_POTENTIAL_WEIGHT
    hpbg_oracle_gain_weight: float = HPBG_ORACLE_GAIN_WEIGHT
    hpbg_belief_distill_weight: float = HPBG_BELIEF_DISTILL_WEIGHT
    hpbg_belief_distill_warmup_updates: int = HPBG_BELIEF_DISTILL_WARMUP_UPDATES
    hpbg_belief_distill_ramp_updates: int = HPBG_BELIEF_DISTILL_RAMP_UPDATES
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
        object.__setattr__(self, "train_maps_dir", str(self.train_maps_dir).strip() or None if self.train_maps_dir else None)
        object.__setattr__(self, "val_maps_dir", str(self.val_maps_dir).strip() or None if self.val_maps_dir else None)
        object.__setattr__(self, "test_maps_dir", str(self.test_maps_dir).strip() or None if self.test_maps_dir else None)
        object.__setattr__(
            self,
            "split_manifest_path",
            str(self.split_manifest_path).strip() or None if self.split_manifest_path else None,
        )
        object.__setattr__(self, "split_seed", int(self.split_seed))
        object.__setattr__(self, "val_map_count", max(int(self.val_map_count), 0) if self.val_map_count is not None else None)
        object.__setattr__(
            self,
            "test_map_count",
            max(int(self.test_map_count), 0) if self.test_map_count is not None else None,
        )
        object.__setattr__(self, "eval_budget_mode", str(self.eval_budget_mode or "fixed_steps").strip() or "fixed_steps")
        object.__setattr__(self, "large_scale_min_free_area", max(int(self.large_scale_min_free_area), 0))
        object.__setattr__(self, "large_scale_min_side", max(int(self.large_scale_min_side), 0))
        object.__setattr__(self, "corridor_max_width", max(float(self.corridor_max_width), CELL_SIZE))
        object.__setattr__(self, "corridor_min_length", max(float(self.corridor_min_length), CELL_SIZE))
        object.__setattr__(self, "smoothness_turn_penalty", max(float(self.smoothness_turn_penalty), 0.0))
        object.__setattr__(self, "smoothness_lateral_penalty", max(float(self.smoothness_lateral_penalty), 0.0))
        object.__setattr__(self, "hpbg_risk_weight", max(float(self.hpbg_risk_weight), 0.0))
        object.__setattr__(self, "hpbg_belief_ema_alpha", min(max(float(self.hpbg_belief_ema_alpha), 0.0), 1.0))
        object.__setattr__(self, "hpbg_cluster_resolution", max(float(self.hpbg_cluster_resolution), CELL_SIZE))
        object.__setattr__(self, "hpbg_cluster_edge_hops", max(int(self.hpbg_cluster_edge_hops), 0))
        object.__setattr__(self, "hpbg_expert_reward_weight", max(float(self.hpbg_expert_reward_weight), 0.0))
        object.__setattr__(self, "hpbg_expert_potential_weight", max(float(self.hpbg_expert_potential_weight), 0.0))
        object.__setattr__(self, "hpbg_oracle_gain_weight", max(float(self.hpbg_oracle_gain_weight), 0.0))
        object.__setattr__(self, "hpbg_belief_distill_weight", max(float(self.hpbg_belief_distill_weight), 0.0))
        object.__setattr__(
            self,
            "hpbg_belief_distill_warmup_updates",
            max(int(self.hpbg_belief_distill_warmup_updates), 0),
        )
        object.__setattr__(
            self,
            "hpbg_belief_distill_ramp_updates",
            max(int(self.hpbg_belief_distill_ramp_updates), 0),
        )
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
        "HPBG_RL_RUN_NAME": runtime_config.run_name,
        "HPBG_RL_RUN_SESSION": run_session,
        "HPBG_RL_MAX_EPISODES": str(runtime_config.max_episodes),
        "HPBG_RL_NUM_META_AGENT": str(runtime_config.num_meta_agent),
        "HPBG_RL_MAX_EPISODE_STEP": str(runtime_config.max_episode_step),
        "HPBG_RL_MINIMUM_BUFFER_SIZE": str(runtime_config.minimum_buffer_size),
        "HPBG_RL_BATCH_SIZE": str(runtime_config.batch_size),
        "HPBG_RL_REPLAY_SIZE": str(runtime_config.replay_size),
        "HPBG_RL_SAVE_IMG_GAP": str(runtime_config.save_img_gap),
        "HPBG_RL_SAVE_MODEL_GAP": str(runtime_config.save_model_gap),
        "HPBG_RL_SUMMARY_WINDOW": str(runtime_config.summary_window),
        "HPBG_RL_TRAIN_UPDATES_PER_ITER": str(runtime_config.train_updates_per_iter),
        "HPBG_RL_RESULT_BUCKET_EPISODES": str(runtime_config.result_bucket_episodes),
        "HPBG_RL_LOAD_MODEL": "1" if runtime_config.load_model else "0",
        "HPBG_RL_USE_GPU": "1" if runtime_config.use_gpu else "0",
        "HPBG_RL_USE_GPU_GLOBAL": "1" if runtime_config.use_gpu_global else "0",
        "HPBG_RL_NUM_GPU": str(runtime_config.num_gpu),
        "HPBG_RL_ENABLE_AUTO_EVAL": "1" if runtime_config.enable_auto_eval else "0",
        "HPBG_RL_AUTO_EVAL_MAP_COUNT": str(runtime_config.auto_eval_map_count),
        "HPBG_RL_AUTO_EVAL_INTERVAL": str(runtime_config.auto_eval_interval),
        "HPBG_RL_AUTO_EVAL_GREEDY": "1" if runtime_config.auto_eval_greedy else "0",
        "HPBG_RL_SPLIT_SEED": str(runtime_config.split_seed),
        "HPBG_RL_EVAL_BUDGET_MODE": runtime_config.eval_budget_mode,
        "HPBG_RL_LARGE_SCALE_MIN_FREE_AREA": str(runtime_config.large_scale_min_free_area),
        "HPBG_RL_LARGE_SCALE_MIN_SIDE": str(runtime_config.large_scale_min_side),
        "HPBG_RL_RUN_FINAL_TEST": "1" if runtime_config.run_final_test else "0",
        "HPBG_RL_ALLOW_TRAIN_SPLIT_EVAL": "1" if runtime_config.allow_train_split_eval else "0",
        "HPBG_RL_ENABLE_CORRIDOR_GRAPH_COMPRESSION": (
            "1" if runtime_config.enable_corridor_graph_compression else "0"
        ),
        "HPBG_RL_ENABLE_CORRIDOR_EDGE_PRUNING": "1" if runtime_config.enable_corridor_edge_pruning else "0",
        "HPBG_RL_ENABLE_SMOOTHNESS_REWARD": "1" if runtime_config.enable_smoothness_reward else "0",
        "HPBG_RL_CORRIDOR_MAX_WIDTH": str(runtime_config.corridor_max_width),
        "HPBG_RL_CORRIDOR_MIN_LENGTH": str(runtime_config.corridor_min_length),
        "HPBG_RL_SMOOTHNESS_TURN_PENALTY": str(runtime_config.smoothness_turn_penalty),
        "HPBG_RL_SMOOTHNESS_LATERAL_PENALTY": str(runtime_config.smoothness_lateral_penalty),
        "HPBG_RL_USE_LF_ATTENTION_HF_RESIDUAL": "1" if runtime_config.use_lf_attention_hf_residual else "0",
        "HPBG_RL_USE_PRIVILEGED_WAVELET_DISTILLATION": (
            "1" if runtime_config.use_privileged_wavelet_distillation else "0"
        ),
        "HPBG_RL_USE_HPBG": "1" if runtime_config.use_hpbg else "0",
        "HPBG_RL_USE_BELIEF_STATE": "1" if runtime_config.use_belief_state else "0",
        "HPBG_RL_USE_MAP_PREDICTION": "1" if runtime_config.use_map_prediction else "0",
        "HPBG_RL_USE_HIERARCHICAL_GRAPH": "1" if runtime_config.use_hierarchical_graph else "0",
        "HPBG_RL_USE_EXPERT_REWARD": "1" if runtime_config.use_expert_reward else "0",
        "HPBG_RL_USE_BELIEF_DISTILLATION": "1" if runtime_config.use_belief_distillation else "0",
        "HPBG_RL_RISK_WEIGHT": str(runtime_config.hpbg_risk_weight),
        "HPBG_RL_BELIEF_EMA_ALPHA": str(runtime_config.hpbg_belief_ema_alpha),
        "HPBG_RL_CLUSTER_RESOLUTION": str(runtime_config.hpbg_cluster_resolution),
        "HPBG_RL_CLUSTER_EDGE_HOPS": str(runtime_config.hpbg_cluster_edge_hops),
        "HPBG_RL_EXPERT_REWARD_WEIGHT": str(runtime_config.hpbg_expert_reward_weight),
        "HPBG_RL_EXPERT_POTENTIAL_WEIGHT": str(runtime_config.hpbg_expert_potential_weight),
        "HPBG_RL_ORACLE_GAIN_WEIGHT": str(runtime_config.hpbg_oracle_gain_weight),
        "HPBG_RL_BELIEF_DISTILL_WEIGHT": str(runtime_config.hpbg_belief_distill_weight),
        "HPBG_RL_BELIEF_DISTILL_WARMUP_UPDATES": str(runtime_config.hpbg_belief_distill_warmup_updates),
        "HPBG_RL_BELIEF_DISTILL_RAMP_UPDATES": str(runtime_config.hpbg_belief_distill_ramp_updates),
        "HPBG_RL_WAVELET_SCALES": ",".join(str(scale) for scale in runtime_config.wavelet_scales),
        "HPBG_RL_WAVELET_FUSE_DIM": str(runtime_config.wavelet_fuse_dim),
        "HPBG_RL_WAVELET_LF_QK": "1" if runtime_config.wavelet_lf_qk else "0",
        "HPBG_RL_WAVELET_DISTILL_WEIGHT": str(runtime_config.wavelet_distill_weight),
        "HPBG_RL_WAVELET_DISTILL_LF_WEIGHT": str(runtime_config.wavelet_distill_lf_weight),
        "HPBG_RL_WAVELET_DISTILL_HF_WEIGHT": str(runtime_config.wavelet_distill_hf_weight),
        "HPBG_RL_WAVELET_DISTILL_WARMUP_UPDATES": str(runtime_config.wavelet_distill_warmup_updates),
        "HPBG_RL_WAVELET_DISTILL_RAMP_UPDATES": str(runtime_config.wavelet_distill_ramp_updates),
    }
    if runtime_config.ray_num_cpus is not None:
        env_updates["HPBG_RL_RAY_NUM_CPUS"] = str(runtime_config.ray_num_cpus)
    if runtime_config.ray_worker_num_cpus is not None:
        env_updates["HPBG_RL_RAY_WORKER_NUM_CPUS"] = str(runtime_config.ray_worker_num_cpus)
    if runtime_config.worker_num_threads is not None:
        env_updates["HPBG_RL_WORKER_NUM_THREADS"] = str(runtime_config.worker_num_threads)
    if runtime_config.val_map_count is not None:
        env_updates["HPBG_RL_VAL_MAP_COUNT"] = str(runtime_config.val_map_count)
    if runtime_config.test_map_count is not None:
        env_updates["HPBG_RL_TEST_MAP_COUNT"] = str(runtime_config.test_map_count)
    if runtime_config.maps_dir:
        env_updates["HPBG_RL_MAPS_DIR"] = str(Path(runtime_config.maps_dir).expanduser().resolve())
    if runtime_config.train_maps_dir:
        env_updates["HPBG_RL_TRAIN_MAPS_DIR"] = str(Path(runtime_config.train_maps_dir).expanduser().resolve())
    if runtime_config.val_maps_dir:
        env_updates["HPBG_RL_VAL_MAPS_DIR"] = str(Path(runtime_config.val_maps_dir).expanduser().resolve())
    if runtime_config.test_maps_dir:
        env_updates["HPBG_RL_TEST_MAPS_DIR"] = str(Path(runtime_config.test_maps_dir).expanduser().resolve())
    if runtime_config.split_manifest_path:
        env_updates["HPBG_RL_SPLIT_MANIFEST_PATH"] = str(Path(runtime_config.split_manifest_path).expanduser().resolve())

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
            "TRAIN_MAPS_DIR": runtime_config.train_maps_dir or "",
            "VAL_MAPS_DIR": runtime_config.val_maps_dir or "",
            "TEST_MAPS_DIR": runtime_config.test_maps_dir or "",
            "SPLIT_MANIFEST_PATH": runtime_config.split_manifest_path or "",
            "SPLIT_SEED": runtime_config.split_seed,
            "VAL_MAP_COUNT": runtime_config.val_map_count,
            "TEST_MAP_COUNT": runtime_config.test_map_count,
            "EVAL_BUDGET_MODE": runtime_config.eval_budget_mode,
            "LARGE_SCALE_MIN_FREE_AREA": runtime_config.large_scale_min_free_area,
            "LARGE_SCALE_MIN_SIDE": runtime_config.large_scale_min_side,
            "RUN_FINAL_TEST": runtime_config.run_final_test,
            "ALLOW_TRAIN_SPLIT_EVAL": runtime_config.allow_train_split_eval,
            "ENABLE_CORRIDOR_GRAPH_COMPRESSION": runtime_config.enable_corridor_graph_compression,
            "ENABLE_CORRIDOR_EDGE_PRUNING": runtime_config.enable_corridor_edge_pruning,
            "ENABLE_SMOOTHNESS_REWARD": runtime_config.enable_smoothness_reward,
            "CORRIDOR_MAX_WIDTH": runtime_config.corridor_max_width,
            "CORRIDOR_MIN_LENGTH": runtime_config.corridor_min_length,
            "SMOOTHNESS_TURN_PENALTY": runtime_config.smoothness_turn_penalty,
            "SMOOTHNESS_LATERAL_PENALTY": runtime_config.smoothness_lateral_penalty,
            "USE_LF_ATTENTION_HF_RESIDUAL": runtime_config.use_lf_attention_hf_residual,
            "USE_PRIVILEGED_WAVELET_DISTILLATION": runtime_config.use_privileged_wavelet_distillation,
            "USE_HPBG": runtime_config.use_hpbg,
            "USE_BELIEF_STATE": runtime_config.use_belief_state,
            "USE_MAP_PREDICTION": runtime_config.use_map_prediction,
            "USE_HIERARCHICAL_GRAPH": runtime_config.use_hierarchical_graph,
            "USE_EXPERT_REWARD": runtime_config.use_expert_reward,
            "USE_BELIEF_DISTILLATION": runtime_config.use_belief_distillation,
            "HPBG_RISK_WEIGHT": runtime_config.hpbg_risk_weight,
            "HPBG_BELIEF_EMA_ALPHA": runtime_config.hpbg_belief_ema_alpha,
            "HPBG_CLUSTER_RESOLUTION": runtime_config.hpbg_cluster_resolution,
            "HPBG_CLUSTER_EDGE_HOPS": runtime_config.hpbg_cluster_edge_hops,
            "HPBG_EXPERT_REWARD_WEIGHT": runtime_config.hpbg_expert_reward_weight,
            "HPBG_EXPERT_POTENTIAL_WEIGHT": runtime_config.hpbg_expert_potential_weight,
            "HPBG_ORACLE_GAIN_WEIGHT": runtime_config.hpbg_oracle_gain_weight,
            "HPBG_BELIEF_DISTILL_WEIGHT": runtime_config.hpbg_belief_distill_weight,
            "HPBG_BELIEF_DISTILL_WARMUP_UPDATES": runtime_config.hpbg_belief_distill_warmup_updates,
            "HPBG_BELIEF_DISTILL_RAMP_UPDATES": runtime_config.hpbg_belief_distill_ramp_updates,
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
