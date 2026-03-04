#!/usr/bin/env bash
# ── Start ARiADNE training (offline ros_conda environment) ────────────────
#
# Usage:
#   bash start_train_ariadne.sh [extra args forwarded to train_wavelet]
#
# Examples:
#   bash start_train_ariadne.sh
#   bash start_train_ariadne.sh --smoke
#   bash start_train_ariadne.sh --resume-from /path/to/checkpoint.pth
#   CUDA_VISIBLE_DEVICES=0,1 bash start_train_ariadne.sh
#
# Environment overrides (set before running):
#   CUDA_VISIBLE_DEVICES          default: 0,1
#   ARIADNE_RAY_NUM_CPUS          default: 44
#   ARIADNE_RAY_WORKER_NUM_CPUS   default: 1
#   ARIADNE_WORKER_NUM_THREADS    default: 1
# ───────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_PREFIX="/home/user/songensheng/conda_envs/ros_conda"
PROJECT_ROOT="/home/user/songensheng/wavelet"
TRAIN_MODULE="ARiADNE.scripts.train_wavelet"

fail() {
    printf '[train-ariadne] ERROR: %s\n' "$*" >&2
    exit 1
}

resolve_package_root() {
    if [[ -d "${PROJECT_ROOT}/ARiADNE" ]]; then
        printf '%s\n' "${PROJECT_ROOT}/ARiADNE"
        return 0
    fi
    if [[ -d "${PROJECT_ROOT}/src/ARiADNE" ]]; then
        printf '%s\n' "${PROJECT_ROOT}/src/ARiADNE"
        return 0
    fi
    return 1
}

resolve_pythonpath() {
    if [[ -d "${PROJECT_ROOT}/src" ]]; then
        printf '%s:%s\n' "${PROJECT_ROOT}/src" "${PROJECT_ROOT}"
        return 0
    fi
    printf '%s\n' "${PROJECT_ROOT}"
}

[[ -x "${ENV_PREFIX}/bin/python" ]] || fail "missing environment python: ${ENV_PREFIX}/bin/python"
[[ -f "${ENV_PREFIX}/bin/activate" ]] || fail "missing environment activate script: ${ENV_PREFIX}/bin/activate"

PACKAGE_ROOT="$(resolve_package_root)" || fail "missing package tree: ${PROJECT_ROOT}/ARiADNE or ${PROJECT_ROOT}/src/ARiADNE"
[[ -f "${PACKAGE_ROOT}/scripts/train_wavelet.py" ]] || fail "missing train entry: ${PACKAGE_ROOT}/scripts/train_wavelet.py"

# shellcheck disable=SC1090
source "${ENV_PREFIX}/bin/activate"

PYTHONPATH_ROOTS="$(resolve_pythonpath)"
export PYTHONPATH="${PYTHONPATH_ROOTS}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Use the two allocated A40 cards by default; callers can override as needed.
: "${CUDA_VISIBLE_DEVICES:=0,1}"
: "${ARIADNE_RAY_NUM_CPUS:=44}"
: "${ARIADNE_RAY_WORKER_NUM_CPUS:=1}"
: "${ARIADNE_WORKER_NUM_THREADS:=1}"
export CUDA_VISIBLE_DEVICES
export ARIADNE_RAY_NUM_CPUS
export ARIADNE_RAY_WORKER_NUM_CPUS
export ARIADNE_WORKER_NUM_THREADS

cd "${PROJECT_ROOT}"
printf '[train-ariadne] env:    %s\n' "${ENV_PREFIX}"
printf '[train-ariadne] root:   %s\n' "${PROJECT_ROOT}"
printf '[train-ariadne] pkg:    %s\n' "${PACKAGE_ROOT}"
printf '[train-ariadne] PYTHONPATH=%s\n' "${PYTHONPATH}"
printf '[train-ariadne] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[train-ariadne] ARIADNE_RAY_NUM_CPUS=%s\n' "${ARIADNE_RAY_NUM_CPUS}"
printf '[train-ariadne] ARIADNE_RAY_WORKER_NUM_CPUS=%s\n' "${ARIADNE_RAY_WORKER_NUM_CPUS}"
printf '[train-ariadne] ARIADNE_WORKER_NUM_THREADS=%s\n' "${ARIADNE_WORKER_NUM_THREADS}"

exec python -u -m "${TRAIN_MODULE}" "$@"
