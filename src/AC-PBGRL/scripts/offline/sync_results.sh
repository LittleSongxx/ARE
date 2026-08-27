#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_HOST="${ACPBGRL_SSH_HOST:?Set ACPBGRL_SSH_HOST or configure an SSH alias}"
SSH_USER="${ACPBGRL_SSH_USER:-user}"
REMOTE_DATA_ROOT="${ACPBGRL_REMOTE_DATA_ROOT:-/mnt/songensheng/ac-pbgrl}"
LOCAL_RESULTS="${ACPBGRL_LOCAL_RESULTS:-${PROJECT_DIR}/.runtime/server-results}"
mkdir -p "${LOCAL_RESULTS}"
rsync -az --partial "${SSH_USER}@${SSH_HOST}:${REMOTE_DATA_ROOT}/runs/" "${LOCAL_RESULTS}/runs/"
rsync -az --partial "${SSH_USER}@${SSH_HOST}:${REMOTE_DATA_ROOT}/paper_figures/" "${LOCAL_RESULTS}/paper_figures/" || true
