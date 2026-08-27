#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${1:-${PROJECT_DIR}/.external/tare_planner}"
REPOSITORY="https://github.com/caochao39/tare_planner.git"
BRANCH="melodic-noetic"
REVISION="44500592b86138257273e0cab264e6a847ccefc7"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  git clone --branch "${BRANCH}" "${REPOSITORY}" "${TARGET_DIR}"
fi
git -C "${TARGET_DIR}" fetch origin "${BRANCH}"
git -C "${TARGET_DIR}" checkout --detach "${REVISION}"
ACTUAL="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
[[ "${ACTUAL}" == "${REVISION}" ]] || { printf 'Unexpected TARE revision: %s\n' "${ACTUAL}" >&2; exit 3; }
printf 'TARE pinned at %s in %s\n' "${REVISION}" "${TARGET_DIR}"
printf 'Build it in the ROS Noetic workspace following its retained upstream README.\n'
