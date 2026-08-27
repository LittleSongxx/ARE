#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

if [[ -x "${ACPBGRL_PYTHON:-}" ]]; then
    PYTHON_BIN="${ACPBGRL_PYTHON}"
elif [[ -x "/mnt/songensheng/ac-pbgrl/env/bin/python" ]]; then
    PYTHON_BIN="/mnt/songensheng/ac-pbgrl/env/bin/python"
else
    PYTHON_BIN="${PYTHON:-python3}"
fi

exec "${PYTHON_BIN}" -m ac_pbgrl.cli "$@"
