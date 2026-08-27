#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.runtime/ros_python"
WHEELHOUSE="${ACPBGRL_ROS_WHEELHOUSE:-${PROJECT_DIR}/.wheelhouse/ros-noetic-py38}"

[[ -d "${WHEELHOUSE}" ]] || { printf 'Missing ROS wheelhouse: %s\n' "${WHEELHOUSE}" >&2; exit 2; }
mkdir -p "${RUNTIME_DIR}"

# Some minimal ROS Noetic images omit python3.8-venv/ensurepip.  A project-local
# target install is equally isolated, works fully offline, and can be injected
# into roslaunch without changing the container's system Python.
python3 -m pip install --no-index --find-links "${WHEELHOUSE}" \
  --upgrade --target "${RUNTIME_DIR}" numpy==1.24.3 onnxruntime==1.16.3
printf 'ROS Python target ready: %s\n' "${RUNTIME_DIR}"
printf 'ac_pbgrl.launch adds this directory to PYTHONPATH automatically.\n'
