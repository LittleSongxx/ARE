#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${ACPBGRL_DATA_ROOT:-/mnt/songensheng/ac-pbgrl}"
ENV_DIR="${ACPBGRL_ENV_DIR:-${DATA_ROOT}/env}"
BASE_ARCHIVE="${ACPBGRL_BASE_ENV_ARCHIVE:-/home/user/songensheng/ros_conda_packed.tar.gz}"
WHEELHOUSE="${ACPBGRL_WHEELHOUSE:-${PROJECT_DIR}/.wheelhouse/server-py38}"

mkdir -p "${DATA_ROOT}" "$(dirname "${ENV_DIR}")"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  [[ -f "${BASE_ARCHIVE}" ]] || { printf 'Missing base environment: %s\n' "${BASE_ARCHIVE}" >&2; exit 2; }
  mkdir -p "${ENV_DIR}"
  tar -xzf "${BASE_ARCHIVE}" -C "${ENV_DIR}"
  if [[ -x "${ENV_DIR}/bin/conda-unpack" ]]; then
    "${ENV_DIR}/bin/conda-unpack"
  fi
fi

[[ -d "${WHEELHOUSE}" ]] || { printf 'Missing wheelhouse: %s\n' "${WHEELHOUSE}" >&2; exit 2; }
"${ENV_DIR}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" \
  ray==2.10.0 numpy==1.24.3 scipy==1.10.1 scikit-image==0.21.0 \
  matplotlib==3.7.5 tensorboard==2.14.0 PyYAML==6.0.3 \
  h5py==3.11.0 pandas==2.0.3 seaborn==0.13.2 scikit-learn==1.3.2 \
  onnx==1.16.2 onnxruntime==1.16.3 pytest==8.3.5 pytest-cov==5.0.0
"${ENV_DIR}/bin/python" -m pip install --no-index --no-deps --no-build-isolation --editable "${PROJECT_DIR}"
"${ENV_DIR}/bin/python" "${PROJECT_DIR}/scripts/offline/verify_environment.py" --server

printf 'Server environment ready: %s\n' "${ENV_DIR}"
printf 'Run with: ACPBGRL_PYTHON=%s/bin/python %s/run.sh doctor --system server_a40\n' "${ENV_DIR}" "${PROJECT_DIR}"
