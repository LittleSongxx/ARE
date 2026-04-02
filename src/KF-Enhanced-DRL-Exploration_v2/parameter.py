from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import os
import re


# =====================================================================
# KF-Enhanced DRL Exploration -- 可配置参数
# 所有功能开关和可调参数集中在此，修改这里即可控制全局行为。
# =====================================================================

# -- 图稀疏化 (Graph Rarefaction) -----------------------------------
ENABLE_GRAPH_RAREFACTION = True   # 节点数超 NODE_PADDING_SIZE 时自动稀疏化

# -- KF 动态优势估计 (KRPO, arXiv:2505.07527) -----------------------
ENABLE_KF_REWARD_BASELINE = True  # 用 KF 跟踪 reward baseline
KF_REWARD_PROCESS_NOISE = 0.005   # reward baseline KF 过程噪声
KF_REWARD_MEASUREMENT_NOISE = 0.5 # reward baseline KF 观测噪声

# -- KF 节点 utility 预测 (KARNet, arXiv:2305.14644) ----------------
ENABLE_KF_UTILITY_PREDICTION = True  # 每个节点用 KF 预测 utility 演化
KF_UTILITY_INITIAL_VARIANCE = 10.0   # utility KF 初始方差
KF_UTILITY_PROCESS_NOISE = 0.5       # utility KF 过程噪声
KF_UTILITY_MEASUREMENT_NOISE = 2.0   # utility KF 观测噪声

# -- KF 位置去噪 (Sim-to-Real, arXiv:2303.07243) -------------------
ENABLE_POSITION_KF = False        # 用 KF 滤波机器人位置（部署时开启）
KF_POSITION_PROCESS_NOISE = 0.01  # 位置 KF 过程噪声
KF_POSITION_MEASUREMENT_NOISE = 0.1  # 位置 KF 观测噪声

# -- KF 目标Q网络软更新 (LKTD, arXiv:2403.13178) -------------------
ENABLE_KF_TARGET_SOFT_UPDATE = False  # 用 EMA 软更新替代硬拷贝（False = 原始每N步硬拷贝）
KF_TARGET_TAU = 0.005                 # EMA 系数 tau: target = (1-tau)*target + tau*online

# -- KF 不确定性 exploration bonus (LKTD, arXiv:2403.13178) ---------
ENABLE_KF_EXPLORATION_BONUS = False   # 用节点 utility KF 不确定性作为 exploration bonus
KF_EXPLORATION_BONUS_WEIGHT = 0.5     # bonus = weight * uncertainty（叠加到 utility 上）

# -- Successor Feature 跨环境迁移 (MB-SF RL, arXiv:2310.10818) ------
ENABLE_SUCCESSOR_FEATURES = False     # 启用 SF 分解（开启后 checkpoint 不兼容旧模型）
SF_FEATURE_DIM = 96                   # successor feature 维度（EMBEDDING_DIM 中分出）
SF_REWARD_HEAD_DIM = 32               # reward weight 头维度
SF_KF_PROCESS_NOISE = 0.01            # SF 迁移时 KF 过程噪声
SF_KF_MEASUREMENT_NOISE = 0.1         # SF 迁移时 KF 观测噪声

# -- Domain Randomization (Sim-to-Real, arXiv:2303.07243) -----------
POSITION_NOISE_STD = 0.0          # 位置高斯噪声标准差（0 = 关闭）
SENSOR_NOISE_PROB = 0.0           # 信念图随机翻转概率（0 = 关闭）

# =====================================================================
# 以下为原始参数，与 large-scale-DRL-exploration 保持一致
# =====================================================================

# saving path
FOLDER_NAME = "kf_enhanced_drl_exploration_v2"
SMOKE_FOLDER_NAME = "kf_enhanced_drl_exploration_v2_smoke"
RUN_SESSION = ""

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 100
SAVE_MODEL_GAP = 32
RESULT_BUCKET_EPISODES = 100

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
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
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

# network parameters
NODE_INPUT_DIM = 4
CRITIC_NODE_INPUT_DIM = NODE_INPUT_DIM + 1
EMBEDDING_DIM = 128

# graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 360

# GPU usage
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0

# training monitor
ENABLE_TRAINING_MONITOR = True
MONITOR_WINDOW = 10
MONITOR_SNAPSHOT_INTERVAL = 10

# auto evaluation
ENABLE_AUTO_EVAL = True
AUTO_EVAL_MAP_COUNT = 10
AUTO_EVAL_INTERVAL = 100
AUTO_EVAL_GREEDY = True


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT
SRC_ROOT = PROJECT_ROOT
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


def _resolve_maps_dir() -> Path:
    candidates = [
        os.environ.get("LARGE_DRL_MAPS_DIR"),
        os.environ.get("ARIADNE_MAPS_DIR"),
        str(PROJECT_ROOT / "maps"),
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


FOLDER_NAME = _env_str("LARGE_DRL_RUN_NAME", FOLDER_NAME).strip() or "kf_enhanced_drl_exploration_v2"
RUN_SESSION = _env_str("LARGE_DRL_RUN_SESSION", RUN_SESSION).strip()

SUMMARY_WINDOW = _env_int("LARGE_DRL_SUMMARY_WINDOW", SUMMARY_WINDOW)
LOAD_MODEL = _env_bool("LARGE_DRL_LOAD_MODEL", LOAD_MODEL)
SAVE_IMG_GAP = _env_int("LARGE_DRL_SAVE_IMG_GAP", SAVE_IMG_GAP)
SAVE_MODEL_GAP = _env_int("LARGE_DRL_SAVE_MODEL_GAP", SAVE_MODEL_GAP)
RESULT_BUCKET_EPISODES = _env_int("LARGE_DRL_RESULT_BUCKET_EPISODES", RESULT_BUCKET_EPISODES)

CELL_SIZE = _env_float("LARGE_DRL_CELL_SIZE", CELL_SIZE)
NODE_RESOLUTION = _env_float("LARGE_DRL_NODE_RESOLUTION", NODE_RESOLUTION)
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

SENSOR_RANGE = _env_float("LARGE_DRL_SENSOR_RANGE", SENSOR_RANGE)
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = _env_int("LARGE_DRL_MIN_UTILITY", MIN_UTILITY)
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

MAX_EPISODES = _env_int("LARGE_DRL_MAX_EPISODES", MAX_EPISODES)
MAX_EPISODE_STEP = _env_int("LARGE_DRL_MAX_EPISODE_STEP", MAX_EPISODE_STEP)
REPLAY_SIZE = _env_int("LARGE_DRL_REPLAY_SIZE", REPLAY_SIZE)
MINIMUM_BUFFER_SIZE = _env_int("LARGE_DRL_MINIMUM_BUFFER_SIZE", MINIMUM_BUFFER_SIZE)
BATCH_SIZE = _env_int("LARGE_DRL_BATCH_SIZE", BATCH_SIZE)
LR = _env_float("LARGE_DRL_LR", LR)
GAMMA = _env_float("LARGE_DRL_GAMMA", GAMMA)
NUM_META_AGENT = _env_int("LARGE_DRL_NUM_META_AGENT", NUM_META_AGENT)
TRAIN_UPDATES_PER_ITER = _env_int("LARGE_DRL_TRAIN_UPDATES_PER_ITER", TRAIN_UPDATES_PER_ITER)
TARGET_Q_UPDATE_INTERVAL = _env_int("LARGE_DRL_TARGET_Q_UPDATE_INTERVAL", TARGET_Q_UPDATE_INTERVAL)
POLICY_GRAD_CLIP = _env_float("LARGE_DRL_POLICY_GRAD_CLIP", POLICY_GRAD_CLIP)
Q_GRAD_CLIP = _env_float("LARGE_DRL_Q_GRAD_CLIP", Q_GRAD_CLIP)

NODE_INPUT_DIM = _env_int("LARGE_DRL_NODE_INPUT_DIM", NODE_INPUT_DIM)
CRITIC_NODE_INPUT_DIM = _env_int("LARGE_DRL_CRITIC_NODE_INPUT_DIM", CRITIC_NODE_INPUT_DIM)
EMBEDDING_DIM = _env_int("LARGE_DRL_EMBEDDING_DIM", EMBEDDING_DIM)

K_SIZE = _env_int("LARGE_DRL_K_SIZE", K_SIZE)
NODE_PADDING_SIZE = _env_int("LARGE_DRL_NODE_PADDING_SIZE", NODE_PADDING_SIZE)

USE_GPU = _env_bool("LARGE_DRL_USE_GPU", USE_GPU)
USE_GPU_GLOBAL = _env_bool("LARGE_DRL_USE_GPU_GLOBAL", USE_GPU_GLOBAL)
NUM_GPU = _env_int("LARGE_DRL_NUM_GPU", NUM_GPU)

ENABLE_TRAINING_MONITOR = _env_bool("LARGE_DRL_ENABLE_TRAINING_MONITOR", ENABLE_TRAINING_MONITOR)
MONITOR_WINDOW = _env_int("LARGE_DRL_MONITOR_WINDOW", MONITOR_WINDOW)
MONITOR_SNAPSHOT_INTERVAL = _env_int("LARGE_DRL_MONITOR_SNAPSHOT_INTERVAL", MONITOR_SNAPSHOT_INTERVAL)

ENABLE_AUTO_EVAL = _env_bool("LARGE_DRL_ENABLE_AUTO_EVAL", ENABLE_AUTO_EVAL)
AUTO_EVAL_MAP_COUNT = _env_int("LARGE_DRL_AUTO_EVAL_MAP_COUNT", AUTO_EVAL_MAP_COUNT)
AUTO_EVAL_INTERVAL = _env_int("LARGE_DRL_AUTO_EVAL_INTERVAL", AUTO_EVAL_INTERVAL)
AUTO_EVAL_GREEDY = _env_bool("LARGE_DRL_AUTO_EVAL_GREEDY", AUTO_EVAL_GREEDY)

ENABLE_GRAPH_RAREFACTION = _env_bool("KF_ENABLE_GRAPH_RAREFACTION", ENABLE_GRAPH_RAREFACTION)
ENABLE_KF_REWARD_BASELINE = _env_bool("KF_ENABLE_REWARD_BASELINE", ENABLE_KF_REWARD_BASELINE)
KF_REWARD_PROCESS_NOISE = _env_float("KF_REWARD_PROCESS_NOISE", KF_REWARD_PROCESS_NOISE)
KF_REWARD_MEASUREMENT_NOISE = _env_float("KF_REWARD_MEASUREMENT_NOISE", KF_REWARD_MEASUREMENT_NOISE)
ENABLE_KF_UTILITY_PREDICTION = _env_bool("KF_ENABLE_UTILITY_PREDICTION", ENABLE_KF_UTILITY_PREDICTION)
KF_UTILITY_INITIAL_VARIANCE = _env_float("KF_UTILITY_INITIAL_VARIANCE", KF_UTILITY_INITIAL_VARIANCE)
KF_UTILITY_PROCESS_NOISE = _env_float("KF_UTILITY_PROCESS_NOISE", KF_UTILITY_PROCESS_NOISE)
KF_UTILITY_MEASUREMENT_NOISE = _env_float("KF_UTILITY_MEASUREMENT_NOISE", KF_UTILITY_MEASUREMENT_NOISE)
ENABLE_POSITION_KF = _env_bool("KF_ENABLE_POSITION_KF", ENABLE_POSITION_KF)
KF_POSITION_PROCESS_NOISE = _env_float("KF_POSITION_PROCESS_NOISE", KF_POSITION_PROCESS_NOISE)
KF_POSITION_MEASUREMENT_NOISE = _env_float("KF_POSITION_MEASUREMENT_NOISE", KF_POSITION_MEASUREMENT_NOISE)
ENABLE_KF_TARGET_SOFT_UPDATE = _env_bool("KF_ENABLE_TARGET_SOFT_UPDATE", ENABLE_KF_TARGET_SOFT_UPDATE)
KF_TARGET_TAU = _env_float("KF_TARGET_TAU", KF_TARGET_TAU)
ENABLE_KF_EXPLORATION_BONUS = _env_bool("KF_ENABLE_EXPLORATION_BONUS", ENABLE_KF_EXPLORATION_BONUS)
KF_EXPLORATION_BONUS_WEIGHT = _env_float("KF_EXPLORATION_BONUS_WEIGHT", KF_EXPLORATION_BONUS_WEIGHT)
ENABLE_SUCCESSOR_FEATURES = _env_bool("KF_ENABLE_SUCCESSOR_FEATURES", ENABLE_SUCCESSOR_FEATURES)
SF_FEATURE_DIM = _env_int("KF_SF_FEATURE_DIM", SF_FEATURE_DIM)
SF_REWARD_HEAD_DIM = _env_int("KF_SF_REWARD_HEAD_DIM", SF_REWARD_HEAD_DIM)
SF_KF_PROCESS_NOISE = _env_float("KF_SF_KF_PROCESS_NOISE", SF_KF_PROCESS_NOISE)
SF_KF_MEASUREMENT_NOISE = _env_float("KF_SF_KF_MEASUREMENT_NOISE", SF_KF_MEASUREMENT_NOISE)
POSITION_NOISE_STD = _env_float("KF_POSITION_NOISE_STD", POSITION_NOISE_STD)
SENSOR_NOISE_PROB = _env_float("KF_SENSOR_NOISE_PROB", SENSOR_NOISE_PROB)

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


def is_smoke_run(runtime_config_or_run_name: "RuntimeConfig | str | None" = None) -> bool:
    if isinstance(runtime_config_or_run_name, RuntimeConfig):
        run_name = runtime_config_or_run_name.run_name
    else:
        run_name = runtime_config_or_run_name
    return str(run_name or FOLDER_NAME).strip() == SMOKE_FOLDER_NAME


def _require_run_session(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> str:
    if run_session is not None:
        session = str(run_session).strip()
    elif runtime_config is not None:
        session = str(runtime_config.run_session or "").strip()
    else:
        session = str(RUN_SESSION or "").strip()
    if not session:
        raise ValueError("run_session is required for artifact paths")
    return session


def get_result_session_root(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return RESULT_ROOT / _require_run_session(runtime_config, run_session)


def get_result_train_root(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "train"


def get_result_test_root(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_session_root(runtime_config, run_session) / "test"


def get_model_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "model"


def get_train_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "tensorboard"


def get_gifs_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "gifs"


def get_monitor_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_train_root(runtime_config, run_session) / "monitor"


def get_result_eval_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_test_root(runtime_config, run_session) / "eval"


def get_result_eval_gifs_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_result_test_root(runtime_config, run_session) / "gifs"


def get_checkpoint_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_model_path(runtime_config, run_session) / "checkpoint.pth"


def get_checkpoint_final_path(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> Path:
    return get_model_path(runtime_config, run_session) / "checkpoint_final.pth"


def get_checkpoint_interrupted_path(
    runtime_config: "RuntimeConfig | None" = None,
    run_session: str | None = None,
) -> Path:
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


def ensure_output_dirs(runtime_config: "RuntimeConfig | None" = None, run_session: str | None = None) -> None:
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

    def __post_init__(self) -> None:
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

    def with_overrides(self, **kwargs: object) -> "RuntimeConfig":
        return replace(self, **kwargs)


def apply_runtime_config(runtime_config: RuntimeConfig) -> RuntimeConfig:
    run_session = runtime_config.run_session or build_run_session(runtime_config.run_name)
    runtime_config = runtime_config.with_overrides(run_session=run_session)

    env_updates = {
        "LARGE_DRL_RUN_NAME": runtime_config.run_name,
        "LARGE_DRL_RUN_SESSION": run_session,
        "LARGE_DRL_MAX_EPISODES": str(runtime_config.max_episodes),
        "LARGE_DRL_NUM_META_AGENT": str(runtime_config.num_meta_agent),
        "LARGE_DRL_MAX_EPISODE_STEP": str(runtime_config.max_episode_step),
        "LARGE_DRL_MINIMUM_BUFFER_SIZE": str(runtime_config.minimum_buffer_size),
        "LARGE_DRL_BATCH_SIZE": str(runtime_config.batch_size),
        "LARGE_DRL_REPLAY_SIZE": str(runtime_config.replay_size),
        "LARGE_DRL_SAVE_IMG_GAP": str(runtime_config.save_img_gap),
        "LARGE_DRL_SAVE_MODEL_GAP": str(runtime_config.save_model_gap),
        "LARGE_DRL_SUMMARY_WINDOW": str(runtime_config.summary_window),
        "LARGE_DRL_TRAIN_UPDATES_PER_ITER": str(runtime_config.train_updates_per_iter),
        "LARGE_DRL_RESULT_BUCKET_EPISODES": str(runtime_config.result_bucket_episodes),
        "LARGE_DRL_LOAD_MODEL": "1" if runtime_config.load_model else "0",
        "LARGE_DRL_USE_GPU": "1" if runtime_config.use_gpu else "0",
        "LARGE_DRL_USE_GPU_GLOBAL": "1" if runtime_config.use_gpu_global else "0",
        "LARGE_DRL_NUM_GPU": str(runtime_config.num_gpu),
        "LARGE_DRL_ENABLE_TRAINING_MONITOR": "1" if runtime_config.enable_training_monitor else "0",
        "LARGE_DRL_MONITOR_WINDOW": str(runtime_config.monitor_window),
        "LARGE_DRL_MONITOR_SNAPSHOT_INTERVAL": str(runtime_config.monitor_snapshot_interval),
        "LARGE_DRL_ENABLE_AUTO_EVAL": "1" if runtime_config.enable_auto_eval else "0",
        "LARGE_DRL_AUTO_EVAL_MAP_COUNT": str(runtime_config.auto_eval_map_count),
        "LARGE_DRL_AUTO_EVAL_INTERVAL": str(runtime_config.auto_eval_interval),
        "LARGE_DRL_AUTO_EVAL_GREEDY": "1" if runtime_config.auto_eval_greedy else "0",
    }
    if runtime_config.ray_num_cpus is not None:
        env_updates["LARGE_DRL_RAY_NUM_CPUS"] = str(runtime_config.ray_num_cpus)
    if runtime_config.ray_worker_num_cpus is not None:
        env_updates["LARGE_DRL_RAY_WORKER_NUM_CPUS"] = str(runtime_config.ray_worker_num_cpus)
    if runtime_config.worker_num_threads is not None:
        env_updates["LARGE_DRL_WORKER_NUM_THREADS"] = str(runtime_config.worker_num_threads)
    if runtime_config.maps_dir:
        env_updates["LARGE_DRL_MAPS_DIR"] = str(Path(runtime_config.maps_dir).expanduser().resolve())

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
            "ENABLE_TRAINING_MONITOR": runtime_config.enable_training_monitor,
            "MONITOR_WINDOW": runtime_config.monitor_window,
            "MONITOR_SNAPSHOT_INTERVAL": runtime_config.monitor_snapshot_interval,
            "ENABLE_AUTO_EVAL": runtime_config.enable_auto_eval,
            "AUTO_EVAL_MAP_COUNT": runtime_config.auto_eval_map_count,
            "AUTO_EVAL_INTERVAL": runtime_config.auto_eval_interval,
            "AUTO_EVAL_GREEDY": runtime_config.auto_eval_greedy,
            "MAPS_DIR": _resolve_maps_dir(),
        }
    )

    globals()["model_path"] = str(get_model_path(runtime_config))
    globals()["train_path"] = str(get_train_path(runtime_config))
    globals()["gifs_path"] = str(get_gifs_path(runtime_config))
    return runtime_config
