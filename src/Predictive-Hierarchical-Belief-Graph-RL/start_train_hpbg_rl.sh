#!/usr/bin/env bash
# ── Start HPBG-RL training ─────────────────────────────────────────────────
#
# Core training hyperparameters are read from:
#   src/hpbg_rl/parameter.py
#
# This script only handles environment bootstrap, path wiring, and optional
# workflow flags such as smoke / resume.
#
# Usage:
#   bash start_train_hpbg_rl.sh [extra args forwarded to scripts/train.py]
#
# Examples:
#   bash start_train_hpbg_rl.sh
#   TRAIN_SMOKE="1" bash start_train_hpbg_rl.sh
#   RESUME_FROM="result/hpbg_rl/train/model/checkpoint_final.pth" bash start_train_hpbg_rl.sh
#
# Server defaults:
#   ENV_PREFIX=/home/user/songensheng/conda_envs/ros_conda
#   CODE_DIR=/home/user/songensheng/Predictive-Hierarchical-Belief-Graph-RL
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/home/user/songensheng/conda_envs/ros_conda}"
CODE_DIR="${CODE_DIR:-/home/user/songensheng/Predictive-Hierarchical-Belief-Graph-RL}"

# ── 启动配置（核心训练/算法参数仍建议在 src/hpbg_rl/parameter.py 调整） ──
# 指定物理 GPU。留空则不设置 CUDA_VISIBLE_DEVICES，交给系统当前环境。
CUDA_DEVICES="0,1,2,3"

# 地图目录。留空则使用 parameter.py 中的自动查找逻辑。
MAPS_DIR=""

# matplotlib 缓存目录。留空则自动使用 /tmp/mpl-hpbg-rl-${USER}-$$。
MPL_CACHE_DIR=""

# 工作流参数。留空/0 表示不启用。
TRAIN_SMOKE="0"
RESUME_FROM=""
RUN_NAME=""
RUN_SESSION=""
# ────────────────────────────────────────────────────────────────────────────

fail() {
    printf '[%s] [train-hpbg-rl] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2
    exit 1
}

log() {
    printf '[%s] [train-hpbg-rl] %s\n' "$(date '+%F %T')" "$*"
}

has_flag() {
    local flag="$1"
    shift
    local arg=""
    for arg in "$@"; do
        if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
            return 0
        fi
    done
    return 1
}

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

[[ -x "${ENV_PREFIX}/bin/python" ]] || fail "missing environment python: ${ENV_PREFIX}/bin/python"
if [[ -f "${ENV_PREFIX}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_PREFIX}/bin/activate"
fi
[[ -d "${CODE_DIR}/src/hpbg_rl" ]] || fail "missing package tree: ${CODE_DIR}"

PYTHON_BIN="${ENV_PREFIX}/bin/python"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${MPL_CACHE_DIR}" ]] && [[ -z "${MPLCONFIGDIR:-}" ]]; then
    MPLCONFIGDIR="${MPL_CACHE_DIR}"
fi
: "${MPLCONFIGDIR:=/tmp/mpl-hpbg-rl-${USER:-user}-$$}"
export MPLCONFIGDIR

if [[ -n "${CUDA_DEVICES}" ]] && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

if [[ -n "${MAPS_DIR}" ]] && [[ -z "${HPBG_RL_MAPS_DIR:-}" ]]; then
    export HPBG_RL_MAPS_DIR="${MAPS_DIR}"
fi
if [[ -n "${RUN_NAME}" ]] && [[ -z "${HPBG_RL_RUN_NAME:-}" ]]; then
    export HPBG_RL_RUN_NAME="${RUN_NAME}"
fi
if [[ -n "${RUN_SESSION}" ]] && [[ -z "${HPBG_RL_RUN_SESSION:-}" ]]; then
    export HPBG_RL_RUN_SESSION="${RUN_SESSION}"
fi

declare -a launch_args=()
if is_truthy "${TRAIN_SMOKE}" && ! has_flag "--smoke" "$@"; then
    launch_args+=(--smoke)
fi
if [[ -n "${RESUME_FROM}" ]] && ! has_flag "--resume-from" "$@"; then
    launch_args+=(--resume-from "${RESUME_FROM}")
fi

cd "${CODE_DIR}"
log "env=${ENV_PREFIX}"
log "code=${CODE_DIR}"
log "entry=${CODE_DIR}/scripts/train.py"
log "python=${PYTHON_BIN}"
log "PYTHONPATH=${PYTHONPATH}"
log "MPLCONFIGDIR=${MPLCONFIGDIR}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"
log "HPBG_RL_MAPS_DIR=${HPBG_RL_MAPS_DIR:-auto-detect-in-code}"
log "HPBG_RL_RUN_NAME=${HPBG_RL_RUN_NAME:-auto-from-parameter.py}"
log "HPBG_RL_RUN_SESSION=${HPBG_RL_RUN_SESSION:-auto-from-parameter.py}"
log "TRAIN_SMOKE=${TRAIN_SMOKE} RESUME_FROM=${RESUME_FROM:-<none>}"
if ((${#launch_args[@]} > 0)); then
    printf '[%s] [train-hpbg-rl] defaults: ' "$(date '+%F %T')"
    printf '%q ' "${launch_args[@]}"
    printf '\n'
fi
printf '[%s] [train-hpbg-rl] command: %q -u scripts/train.py ' "$(date '+%F %T')" "${PYTHON_BIN}"
printf '%q ' "${launch_args[@]}" "$@"
printf '\n'
log "handoff_to_python_start"

exec "${PYTHON_BIN}" -u scripts/train.py "${launch_args[@]}" "$@"