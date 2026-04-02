#!/usr/bin/env bash
# ── KF-Enhanced DRL Exploration v2 训练启动脚本 ────────────────────
#
# 相比 v1，v2 新增以下可配置功能：
#   - KF 目标Q网络软更新 (EMA, LKTD)
#   - KF 不确定性 exploration bonus
#   - Successor Feature 跨环境迁移 + KF 自适应 loss 加权
#   - KF 位置去噪（部署时开启）
#   - Domain Randomization 噪声注入
#
# Usage:
#   bash start_train_kf_enhanced_v2.sh
#   bash start_train_kf_enhanced_v2.sh --smoke
#   bash start_train_kf_enhanced_v2.sh --resume-from /path/to/checkpoint.pth
#   LARGE_DRL_MAX_EPISODES=8000 bash start_train_kf_enhanced_v2.sh
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_PREFIX="/home/user/songensheng/conda_envs/ros_conda"
CODE_DIR="/home/user/songensheng/KF-Enhanced-DRL-Exploration_v2"

# ── 训练配置（留空 = 使用 parameter.py 默认值）──────────────────────
CUDA_DEVICES="0,1,2,3"
MAPS_DIR=""
RESUME_FROM=""

MAX_EPISODES=""
NUM_META_AGENT=""
BATCH_SIZE=""
SAVE_MODEL_GAP=""
NUM_GPU=""

RAY_NUM_CPUS=""
RAY_WORKER_NUM_CPUS=""

# ── KF 功能开关（留空 = 使用 parameter.py 默认值）──────────────────
# 图稀疏化
KF_ENABLE_GRAPH_RAREFACTION=""

# KF 动态优势估计 (KRPO)
KF_ENABLE_REWARD_BASELINE=""
KF_REWARD_PROCESS_NOISE=""
KF_REWARD_MEASUREMENT_NOISE=""

# KF 节点 utility 预测 (KARNet)
KF_ENABLE_UTILITY_PREDICTION=""
KF_UTILITY_INITIAL_VARIANCE=""
KF_UTILITY_PROCESS_NOISE=""
KF_UTILITY_MEASUREMENT_NOISE=""

# KF 位置去噪 (Sim-to-Real)
KF_ENABLE_POSITION_KF=""
KF_POSITION_PROCESS_NOISE=""
KF_POSITION_MEASUREMENT_NOISE=""

# KF 目标Q网络软更新 (LKTD)
KF_ENABLE_TARGET_SOFT_UPDATE=""
KF_TARGET_TAU=""

# KF 不确定性 exploration bonus
KF_ENABLE_EXPLORATION_BONUS=""
KF_EXPLORATION_BONUS_WEIGHT=""

# Successor Feature 跨环境迁移 (MB-SF RL)
KF_ENABLE_SUCCESSOR_FEATURES=""
KF_SF_FEATURE_DIM=""
KF_SF_REWARD_HEAD_DIM=""
KF_SF_KF_PROCESS_NOISE=""
KF_SF_KF_MEASUREMENT_NOISE=""

# Domain Randomization
KF_POSITION_NOISE_STD=""
KF_SENSOR_NOISE_PROB=""
# ──────────────────────────────────────────────────────────────────────

fail() { printf '[train-kf-v2] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "${ENV_PREFIX}/bin/python" ]] || fail "python not found: ${ENV_PREFIX}/bin/python"
[[ -f "${CODE_DIR}/scripts/train.py" ]] || fail "train.py not found: ${CODE_DIR}/scripts/train.py"

# shellcheck disable=SC1091
source "${ENV_PREFIX}/bin/activate"

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

: "${CUDA_VISIBLE_DEVICES:=${CUDA_DEVICES}}"
export CUDA_VISIBLE_DEVICES

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kf-drl-v2-${USER:-user}-$$}"
export MPLCONFIGDIR

export_if_set() { if [[ -n "$2" ]]; then export "$1=$2"; fi; }

# 基础训练参数
export_if_set LARGE_DRL_MAPS_DIR              "${MAPS_DIR}"
export_if_set LARGE_DRL_MAX_EPISODES          "${MAX_EPISODES}"
export_if_set LARGE_DRL_NUM_META_AGENT        "${NUM_META_AGENT}"
export_if_set LARGE_DRL_BATCH_SIZE            "${BATCH_SIZE}"
export_if_set LARGE_DRL_SAVE_MODEL_GAP        "${SAVE_MODEL_GAP}"
export_if_set LARGE_DRL_NUM_GPU               "${NUM_GPU}"
export_if_set LARGE_DRL_RAY_NUM_CPUS          "${RAY_NUM_CPUS}"
export_if_set LARGE_DRL_RAY_WORKER_NUM_CPUS   "${RAY_WORKER_NUM_CPUS}"

# KF 功能开关与参数
export_if_set KF_ENABLE_GRAPH_RAREFACTION     "${KF_ENABLE_GRAPH_RAREFACTION}"
export_if_set KF_ENABLE_REWARD_BASELINE       "${KF_ENABLE_REWARD_BASELINE}"
export_if_set KF_REWARD_PROCESS_NOISE         "${KF_REWARD_PROCESS_NOISE}"
export_if_set KF_REWARD_MEASUREMENT_NOISE     "${KF_REWARD_MEASUREMENT_NOISE}"
export_if_set KF_ENABLE_UTILITY_PREDICTION    "${KF_ENABLE_UTILITY_PREDICTION}"
export_if_set KF_UTILITY_INITIAL_VARIANCE     "${KF_UTILITY_INITIAL_VARIANCE}"
export_if_set KF_UTILITY_PROCESS_NOISE        "${KF_UTILITY_PROCESS_NOISE}"
export_if_set KF_UTILITY_MEASUREMENT_NOISE    "${KF_UTILITY_MEASUREMENT_NOISE}"
export_if_set KF_ENABLE_POSITION_KF           "${KF_ENABLE_POSITION_KF}"
export_if_set KF_POSITION_PROCESS_NOISE       "${KF_POSITION_PROCESS_NOISE}"
export_if_set KF_POSITION_MEASUREMENT_NOISE   "${KF_POSITION_MEASUREMENT_NOISE}"
export_if_set KF_ENABLE_TARGET_SOFT_UPDATE    "${KF_ENABLE_TARGET_SOFT_UPDATE}"
export_if_set KF_TARGET_TAU                   "${KF_TARGET_TAU}"
export_if_set KF_ENABLE_EXPLORATION_BONUS     "${KF_ENABLE_EXPLORATION_BONUS}"
export_if_set KF_EXPLORATION_BONUS_WEIGHT     "${KF_EXPLORATION_BONUS_WEIGHT}"
export_if_set KF_ENABLE_SUCCESSOR_FEATURES    "${KF_ENABLE_SUCCESSOR_FEATURES}"
export_if_set KF_SF_FEATURE_DIM              "${KF_SF_FEATURE_DIM}"
export_if_set KF_SF_REWARD_HEAD_DIM          "${KF_SF_REWARD_HEAD_DIM}"
export_if_set KF_SF_KF_PROCESS_NOISE         "${KF_SF_KF_PROCESS_NOISE}"
export_if_set KF_SF_KF_MEASUREMENT_NOISE     "${KF_SF_KF_MEASUREMENT_NOISE}"
export_if_set KF_POSITION_NOISE_STD          "${KF_POSITION_NOISE_STD}"
export_if_set KF_SENSOR_NOISE_PROB           "${KF_SENSOR_NOISE_PROB}"

# 自动检测地图目录
if [[ -z "${LARGE_DRL_MAPS_DIR:-}" ]]; then
    for d in "${CODE_DIR}/maps" "${CODE_DIR}/../maps"; do
        [[ -d "$d" ]] && export LARGE_DRL_MAPS_DIR="$d" && break
    done
fi

declare -a cli_args=()
[[ -n "${RESUME_FROM}" ]] && cli_args+=(--resume-from "${RESUME_FROM}")

cd "${CODE_DIR}"
printf '[train-kf-v2] env:   %s\n' "${ENV_PREFIX}"
printf '[train-kf-v2] code:  %s\n' "${CODE_DIR}"
printf '[train-kf-v2] cuda:  %s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[train-kf-v2] maps:  %s\n' "${LARGE_DRL_MAPS_DIR:-auto}"
printf '[train-kf-v2] args:  %s %s\n' "${cli_args[*]+"${cli_args[*]}"}" "$*"

exec python -u scripts/train.py ${cli_args[@]+"${cli_args[@]}"} "$@"
