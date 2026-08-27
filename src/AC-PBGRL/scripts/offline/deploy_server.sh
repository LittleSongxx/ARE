#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_HOST="${ACPBGRL_SSH_HOST:?Set ACPBGRL_SSH_HOST or configure an SSH alias}"
SSH_USER="${ACPBGRL_SSH_USER:-user}"
REMOTE_DIR="${ACPBGRL_REMOTE_DIR:-/home/user/songensheng/AC-PBGRL}"

rsync -az --delete-delay \
  --exclude '.git/' --exclude '.runtime/' --exclude 'runs/' --exclude 'artifacts/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "${PROJECT_DIR}/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

if [[ -d "${PROJECT_DIR}/.wheelhouse" ]]; then
  rsync -az "${PROJECT_DIR}/.wheelhouse/" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/.wheelhouse/"
  ssh "${SSH_USER}@${SSH_HOST}" \
    "cd '${REMOTE_DIR}/.wheelhouse' && sha256sum -c SHA256SUMS --quiet"
fi
printf 'Deployed to %s@%s:%s\n' "${SSH_USER}" "${SSH_HOST}" "${REMOTE_DIR}"
