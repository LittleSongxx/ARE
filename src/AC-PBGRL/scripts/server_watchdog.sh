#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent, single-instance wrapper for the complete paper pipeline.  The
# GPU-aware Python supervisor remains responsible for each training phase; this
# outer watchdog covers SSH logout, driver crashes and host reboot (when invoked
# from cron).  It never kills processes outside its own current child tree.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ACPBGRL_DATA_ROOT:-${PROJECT_DIR}/.runtime}"
PYTHON_BIN="${ACPBGRL_PYTHON:-}"
RESTART_DELAY="${ACPBGRL_WATCHDOG_RESTART_DELAY:-30}"

if [[ ! "${RESTART_DELAY}" =~ ^[0-9]+$ ]]; then
    echo "ACPBGRL_WATCHDOG_RESTART_DELAY must be a non-negative integer" >&2
    exit 2
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ACPBGRL_PYTHON must point to an executable Python interpreter" >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    PIPELINE_ARGS=(paper --gpus auto --gpu-policy prefer-idle)
else
    PIPELINE_ARGS=("$@")
fi

STATE_DIR="${DATA_ROOT}/orchestration"
LOCK_PATH="${STATE_DIR}/paper_watchdog.lock"
WATCHDOG_PID_PATH="${STATE_DIR}/paper_watchdog.pid"
CHILD_PID_PATH="${STATE_DIR}/paper_driver.pid"
HEARTBEAT_PATH="${STATE_DIR}/paper_watchdog.heartbeat"
COMPLETE_PATH="${STATE_DIR}/paper_pipeline.complete"
DRIVER_LOG="${STATE_DIR}/paper_driver.log"
WATCHDOG_LOG="${STATE_DIR}/paper_watchdog.log"
mkdir -p "${STATE_DIR}"

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    exit 0
fi
if [[ -f "${COMPLETE_PATH}" ]]; then
    exit 0
fi

atomic_line() {
    local target="$1"
    local value="$2"
    local temporary="${target}.tmp.$$"
    printf '%s\n' "${value}" >"${temporary}"
    mv -f "${temporary}" "${target}"
}

log_event() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${WATCHDOG_LOG}"
}

child_pid=""
heartbeat_pid=""
stop_requested=0

stop_child_tree() {
    if [[ -z "${child_pid}" ]] || ! kill -0 "${child_pid}" 2>/dev/null; then
        return
    fi
    # The direct child is the paper driver.  Signal its active phase child
    # first so the GPU supervisor can checkpoint at an update boundary.
    local descendants=()
    while read -r descendant; do
        [[ -n "${descendant}" ]] && descendants+=("${descendant}")
    done < <(pgrep -P "${child_pid}" || true)
    if (( ${#descendants[@]} == 0 )); then
        kill -TERM "${child_pid}" 2>/dev/null || true
        return
    fi
    for descendant in "${descendants[@]}"; do
        [[ -n "${descendant}" ]] && kill -TERM "${descendant}" 2>/dev/null || true
    done
}

handle_stop() {
    stop_requested=1
    log_event "event=watchdog_signal child_pid=${child_pid:-none}"
    stop_child_tree
}
trap handle_stop SIGINT SIGTERM SIGHUP

cleanup() {
    if [[ -n "${heartbeat_pid}" ]]; then
        kill "${heartbeat_pid}" 2>/dev/null || true
        wait "${heartbeat_pid}" 2>/dev/null || true
    fi
    rm -f "${WATCHDOG_PID_PATH}"
    if [[ -n "${child_pid}" ]] && [[ -f "${CHILD_PID_PATH}" ]] \
        && [[ "$(<"${CHILD_PID_PATH}")" == "${child_pid}" ]]; then
        rm -f "${CHILD_PID_PATH}"
    fi
}
trap cleanup EXIT

atomic_line "${WATCHDOG_PID_PATH}" "$$"
log_event "event=watchdog_started pid=$$ args=${PIPELINE_ARGS[*]}"

while (( ! stop_requested )); do
    log_event "event=driver_start"
    (
        cd "${PROJECT_DIR}"
        exec env \
            ACPBGRL_DATA_ROOT="${DATA_ROOT}" \
            ACPBGRL_PYTHON="${PYTHON_BIN}" \
            "${PROJECT_DIR}/run.sh" "${PIPELINE_ARGS[@]}"
    ) >>"${DRIVER_LOG}" 2>&1 9>&- &
    child_pid=$!
    atomic_line "${CHILD_PID_PATH}" "${child_pid}"

    (
        while kill -0 "${child_pid}" 2>/dev/null; do
            atomic_line "${HEARTBEAT_PATH}" \
                "time=$(date --iso-8601=seconds) watchdog_pid=$$ driver_pid=${child_pid}"
            sleep 30
        done
    ) 9>&- &
    heartbeat_pid=$!

    set +e
    wait "${child_pid}"
    return_code=$?
    set -e
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
    heartbeat_pid=""
    rm -f "${CHILD_PID_PATH}"
    log_event "event=driver_exit return_code=${return_code}"

    if (( stop_requested )); then
        exit 130
    fi
    if (( return_code == 0 )); then
        atomic_line "${COMPLETE_PATH}" \
            "completed_at=$(date --iso-8601=seconds) watchdog_pid=$$"
        log_event "event=pipeline_complete"
        exit 0
    fi
    log_event "event=restart_wait seconds=${RESTART_DELAY}"
    sleep "${RESTART_DELAY}" &
    wait $! || true
done

exit 130
