from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from ac_pbgrl.config import Config
from .gpu import (
    GPULease,
    GPUInfo,
    calibrate_micro_batch,
    query_gpus,
    query_gpu_processes,
    recommended_micro_batch,
    select_gpus,
)


TEMPORARY_RESOURCE_EXIT = 75


def _descendant_pids(root_pid: int) -> set[int]:
    """Return only this supervisor child's process tree from Linux /proc."""

    parents: dict[int, int] = {}
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return {int(root_pid)}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # Field two can contain spaces and parentheses; splitting after the
            # last ')' leaves state and PPID as the first two tokens.
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat.rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(remainder[1])
        except (OSError, IndexError, ValueError):
            continue
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _selection(config: Config, min_gpus: int, max_gpus: int, policy: str) -> list[GPUInfo]:
    scheduler = config.gpu_scheduler
    return select_gpus(
        query_gpus(),
        policy=policy,
        min_gpus=min_gpus,
        max_gpus=max_gpus,
        min_free_memory_gib=float(scheduler.min_free_memory_gib),
        max_utilization_pct=int(scheduler.max_utilization_pct),
        max_temperature_c=int(scheduler.max_temperature_c),
        idle_used_memory_mib=int(scheduler.idle_used_memory_mib),
        idle_utilization_pct=int(scheduler.idle_utilization_pct),
    )


def supervise_training(
    config: Config,
    *,
    experiment: str,
    system: str | None,
    overrides: Sequence[str],
    min_gpus: int,
    max_gpus: int,
    policy: str,
    wait: bool = True,
    smoke: bool = False,
) -> int:
    data_root = Path(config.project.data_root)
    run_name = str(config.project.run_name)
    if run_name == "auto":
        run_name = f"{experiment}/seed_{int(config.project.seed)}"
    supervisor_root = data_root / "supervisor" / run_name
    supervisor_root.mkdir(parents=True, exist_ok=True)
    event_path = supervisor_root / "events.jsonl"
    lock_root = data_root / "gpu_locks"
    run_root = data_root / "runs" / run_name
    stop_event = threading.Event()
    previous_handlers = {}

    def request_stop(signum, frame) -> None:
        del signum, frame
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    def event(kind: str, **fields) -> None:
        record = {"time": time.time(), "event": kind, **fields}
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def terminate_own_process(process: subprocess.Popen, reason: str) -> None:
        event("terminate_training", reason=reason, pid=process.pid)
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            # The process group was created by this supervisor; no external PID
            # is ever signaled.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def archive_marker(path: Path, kind: str) -> bool:
        if not path.is_file():
            return False
        target = supervisor_root / f"{kind}_{time.time_ns()}.txt"
        path.replace(target)
        return True

    micro_batch_cap: int | None = None

    try:
        while not stop_event.is_set():
            selected = _selection(config, min_gpus, max_gpus, policy)
            if not selected:
                event("waiting_for_gpu", policy=policy)
                if not wait:
                    return TEMPORARY_RESOURCE_EXIT
                stop_event.wait(float(config.gpu_scheduler.wait_seconds))
                continue
            lease = GPULease(lock_root, selected)
            if not lease.acquire():
                stop_event.wait(min(10.0, float(config.gpu_scheduler.wait_seconds)))
                continue
            try:
                micro_batch = recommended_micro_batch(selected, float(config.gpu_scheduler.memory_reserve_gib))
                if micro_batch_cap is not None:
                    micro_batch = min(micro_batch, micro_batch_cap)
                if micro_batch <= 0:
                    event("selection_rejected", reason="insufficient_safe_memory")
                    stop_event.wait(float(config.gpu_scheduler.wait_seconds))
                    continue
                micro_batch, probe_attempts = calibrate_micro_batch(
                    selected,
                    experiment=experiment,
                    system=system,
                    overrides=overrides,
                    candidates=config.gpu_scheduler.micro_batch_candidates,
                    start=micro_batch,
                    reserve_gib=float(config.gpu_scheduler.memory_reserve_gib),
                    smoke=smoke,
                )
                event("memory_probe", selected_micro_batch=micro_batch, attempts=probe_attempts)
                if micro_batch <= 0:
                    if not wait:
                        return TEMPORARY_RESOURCE_EXIT
                    stop_event.wait(float(config.gpu_scheduler.wait_seconds))
                    continue
                indices = [gpu.index for gpu in selected]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in indices)
                environment["ACPBGRL_SELECTED_GPU_INDICES"] = ",".join(str(index) for index in indices)
                environment["ACPBGRL_SELECTED_GPU_UUIDS"] = ",".join(gpu.uuid for gpu in selected)
                environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
                environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={len(selected)}",
                    "-m",
                    "ac_pbgrl.learning.train",
                    "--config",
                    experiment,
                    "--micro-batch",
                    str(micro_batch),
                    "--resume",
                    "auto",
                ]
                if system:
                    command.extend(("--system", system))
                for override in overrides:
                    command.extend(("--set", override))
                if smoke:
                    command.append("--smoke")
                event(
                    "launch",
                    gpu_indices=indices,
                    gpu_uuids=[gpu.uuid for gpu in selected],
                    micro_batch=micro_batch,
                    world_size=len(selected),
                )
                preexisting_processes = query_gpu_processes()
                process = subprocess.Popen(command, env=environment, start_new_session=True)
                pressure_checks = 0
                pressure_triggered = False
                reported_external: dict[str, list[int]] = {}
                while process.poll() is None:
                    if stop_event.wait(float(config.gpu_scheduler.monitor_seconds)):
                        terminate_own_process(process, "supervisor_signal")
                        break
                    current = {gpu.uuid: gpu for gpu in query_gpus()}
                    own_pids = _descendant_pids(process.pid)
                    process_snapshot = query_gpu_processes()
                    new_external = {
                        gpu.uuid: sorted(
                            process_snapshot.get(gpu.uuid, set())
                            - preexisting_processes.get(gpu.uuid, set())
                            - own_pids
                        )
                        for gpu in selected
                    }
                    new_external = {uuid: pids for uuid, pids in new_external.items() if pids}
                    if new_external != reported_external:
                        event("external_gpu_processes", processes=new_external)
                        reported_external = new_external
                    unsafe = [
                        gpu.uuid
                        for gpu in selected
                        if gpu.uuid not in current
                        or current[gpu.uuid].free_gib < float(config.gpu_scheduler.memory_reserve_gib)
                        or current[gpu.uuid].temperature_c > int(config.gpu_scheduler.max_temperature_c)
                    ]
                    pressure_checks = pressure_checks + 1 if unsafe else 0
                    if pressure_checks >= int(config.gpu_scheduler.pressure_grace_checks):
                        event("resource_pressure", gpu_uuids=unsafe, external_processes=new_external)
                        pressure_triggered = True
                        terminate_own_process(process, "resource_pressure")
                        break
                return_code = int(process.returncode if process.returncode is not None else process.wait())
                event("exit", return_code=return_code)
                oom = archive_marker(run_root / "oom.request", "oom")
                restart_requested = archive_marker(run_root / "restart.request", "restart")
                if stop_event.is_set():
                    event("supervisor_stopped")
                    return 130
                if return_code == 0 and not pressure_triggered:
                    return 0
                if oom:
                    candidates = [int(value) for value in config.gpu_scheduler.micro_batch_candidates]
                    lower = [value for value in candidates if value < micro_batch]
                    if not lower:
                        event("oom_exhausted", micro_batch=micro_batch)
                        return return_code or TEMPORARY_RESOURCE_EXIT
                    micro_batch_cap = max(lower)
                    event("micro_batch_backoff", previous=micro_batch, next=micro_batch_cap)
                elif not pressure_triggered and not restart_requested and return_code not in {
                    TEMPORARY_RESOURCE_EXIT,
                    -signal.SIGTERM,
                    137,
                    -9,
                }:
                    return return_code
            finally:
                lease.release()
            if not wait:
                return TEMPORARY_RESOURCE_EXIT
            stop_event.wait(float(config.gpu_scheduler.wait_seconds))
        return 130
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
