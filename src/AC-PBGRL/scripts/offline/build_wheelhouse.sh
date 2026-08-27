#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_DIR}/.wheelhouse}"
SERVER_DIR="${OUTPUT_DIR}/server-py38"
ROS_DIR="${OUTPUT_DIR}/ros-noetic-py38"
mkdir -p "${SERVER_DIR}" "${ROS_DIR}"

python3 -m pip download \
  --dest "${SERVER_DIR}" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 38 \
  --implementation cp \
  --abi cp38 \
  "ray==2.10.0" "numpy==1.24.3" "scipy==1.10.1" "scikit-image==0.21.0" \
  "matplotlib==3.7.5" "tensorboard==2.14.0" "PyYAML==6.0.3" \
  "h5py==3.11.0" "pandas==2.0.3" "seaborn==0.13.2" \
  "scikit-learn==1.3.2" "onnx==1.16.2" "onnxruntime==1.16.3" \
  "pytest==8.3.5" "pytest-cov==5.0.0"

python3 -m pip download \
  --dest "${ROS_DIR}" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 38 \
  --implementation cp \
  --abi cp38 \
  "numpy==1.24.3" "onnxruntime==1.16.3"

(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
printf 'Wheelhouse created at %s\n' "${OUTPUT_DIR}"
